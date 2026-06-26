  """
  B-MoE: Block-Mixture-of-Experts Toy Model
  ==========================================
  A minimal MoE transformer with inter-bloc routing, two pre-specialized experts,
  and straight-through gradient estimation for discrete routing decisions.

  Architecture:
  - 4 transformer layers grouped into 2 blocs (Z=2)
  - Each layer has a MoE block with 2 MLP experts (d=64)
  - Inter-bloc routing: lightweight attention over expert outputs to select next expert
  - Residual connections throughout
  - Pre-specialization: Expert_0 → arithmetic sequences, Expert_1 → random tokens

  Training:
  - Next-token prediction on interleaved arithmetic/random datasets
  - CrossEntropy + load balancing loss (alpha=0.01)
  - Straight-through estimator for routing
  """

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import numpy as np
  from collections import defaultdict

  ─── Data Generation ───────────────────────────────────────────────────────────

  def generate_arithmetic_sequence(length: int, vocab_size: int = 50, start_min: int = 0, step_range: tuple = (1, 3)) -> torch.Tensor:
      """Generate a sequence where each token is (prev_token + step) mod vocab_size."""
      seq = [torch.randint(start_min, start_min + step_range[1], (1,)).item()]
      step = torch.randint(step_range[0], step_range[1] + 1, (1,)).item()
      for _ in range(1, length):
          next_val = (seq[-1] + step) % vocab_size
          seq.append(next_val)
      return torch.tensor(seq, dtype=torch.long)

  def generate_random_sequence(length: int, vocab_size: int = 50) -> torch.Tensor:
      """Generate a sequence of uniformly random tokens."""
      return torch.randint(0, vocab_size, (length,))

  def build_dataset(n_samples: int, seq_len: int, dataset_type: str = "arithmetic") -> torch.Tensor:
      """Build a dataset of sequences."""
      sequences = []
      for _ in range(n_samples):
          if dataset_type == "arithmetic":
              seq = generate_arithmetic_sequence(seq_len)
          else:
              seq = generate_random_sequence(seq_len)
          sequences.append(seq)
      return torch.stack(sequences)

  def create_training_pairs(data: torch.Tensor, seq_len: int):
      """Create (input_seq, target_token) pairs for next-token prediction."""
      X_list, Y_list = [], []
      for seq in data:
          for i in range(seq_len - 1):
              X_list.append(seq[:i+1])          # context up to position i
              Y_list.append(seq[i+1])            # next token at position i+1
      return torch.stack(X_list), torch.stack(Y_list)

  ─── Model Components ──────────────────────────────────────────────────────────

  class SimpleMLP(nn.Module):
      """A simple 2-layer MLP with GELU activation."""
      def init(self, d_model: int, hidden_dim: int = 128):
          super().init()
          self.net = nn.Sequential(
              nn.Linear(d_model, hidden_dim),
              nn.GELU(),
              nn.Linear(hidden_dim, d_model),
          )

  def forward(self, x):
      return self.net(x)
  class MoEBlock(nn.Module):
      """Mixture-of-Experts block with 2 experts and a learnable router.

  The router produces logits over experts. During training we use straight-through
  estimation: the routing decision is sampled discretely (via argmax), but gradients
  flow through the soft probabilities.
  """
  def __init__(self, d_model: int, n_experts: int = 2):
      super().__init__()
      self.d_model = d_model
      self.n_experts = n_experts

      # Two expert networks
      self.experts = nn.ModuleList([SimpleMLP(d_model) for _ in range(n_experts)])

      # Router: maps residual stream to expert logits
      self.router = nn.Linear(d_model, n_experts)

  def forward(self, x: torch.Tensor, train: bool = True):
      """Forward pass with MoE routing.

      Returns:
        output: weighted sum of expert outputs (straight-through gradient)
        routing_probs: soft probabilities (for logging / load balancing)
        routing_indices: hard expert indices chosen per token
      """
      B, S, D = x.shape

      # Compute raw router logits
      router_logits = self.router(x)  # (B, S, n_experts)
      routing_probs = F.softmax(router_logits, dim=-1)  # (B, S, n_experts)

      # Hard selection via argmax (discrete decision)
      routing_indices = torch.argmax(routing_probs, dim=-1)  # (B, S)

      # ── Straight-through estimator ────────────────────────────────
      # Forward pass uses the one-hot "hard" selection
      one_hot = F.one_hot(routing_indices, num_classes=self.n_experts).float()  # (B, S, n_experts)

      # Gather expert outputs
      expert_outputs = torch.stack([self.experts[e](x) for e in range(self.n_experts)], dim=-1)  # (B, S, D, n_experts)

      # Weighted combination using straight-through probs
      # Gradients flow through routing_probs (soft), but computation uses one_hot
      weighted_output = (expert_outputs * one_hot.unsqueeze(-1)).sum(dim=-1)  # (B, S, D)

      # Straight-through: detach from one_hot so gradients come from soft probs
      weighted_output = weighted_output + (routing_probs.detach().unsqueeze(-1) * expert_outputs).sum(dim=-1) - \
                        (routing_probs.detach().unsqueeze(-1) * expert_outputs).sum(dim=-1).detach()

      return weighted_output, routing_probs, routing_indices
  class TransformerLayer(nn.Module):
      """Minimal transformer layer with MoE feed-forward block."""
      def init(self, d_model: int, n_heads: int = 4):
          super().init()
          self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
          self.moe_block = MoEBlock(d_model)
          self.norm1 = nn.LayerNorm(d_model)
          self.norm2 = nn.LayerNorm(d_model)

  def forward(self, x: torch.Tensor, train: bool = True):
      # Self-attention residual
      attn_out, _ = self.attn(x, x, x)
      x = self.norm1(x + attn_out)

      # MoE feed-forward residual
      moe_out, rpri, ridx = self.moe_block(x, train=train)
      x = self.norm2(x + moe_out)

      return x, rpri, ridx
  ─── B-MoE Model ───────────────────────────────────────────────────────────────

  class BMoE(nn.Module):
      """Block Mixture-of-Experts model.

  Architecture:
    - Token embedding (vocab_size × d_model)
    - Positional encoding (learnable)
    - Z=2 blocs of transformer layers (Z=2 layers per bloc → 4 total)
    - Each layer has a MoE block with 2 experts
    - Inter-bloc routing: lightweight attention over expert outputs to select next expert
    - Final linear head for next-token prediction
  """
  def __init__(self, vocab_size: int = 50, d_model: int = 64, n_heads: int = 4,
               n_blocs: int = 2, layers_per_bloc: int = 2, n_experts: int = 2,
               d_k_routing: int = 16):
      super().__init__()
      self.d_model = d_model
      self.n_blocs = n_blocs
      self.layers_per_bloc = layers_per_bloc
      self.d_k_routing = d_k_routing

      self.token_embed = nn.Embedding(vocab_size, d_model)
      self.pos_embed = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)  # max seq len 200

      # Build transformer layers grouped into blocs
      self.blocs = nn.ModuleList()
      for _ in range(n_blocs):
          bloc_layers = nn.ModuleList()
          for _ in range(layers_per_bloc):
              bloc_layers.append(TransformerLayer(d_model, n_heads))
          self.blocs.append(bloc_layers)

      # Inter-bloc routing: lightweight attention over expert outputs
      # Takes last-token representations from each bloc and produces routing scores
      self.routing_attn_q = nn.Linear(d_model, d_k_routing)   # query from current bloc output
      self.routing_attn_k = nn.Linear(d_model, d_k_routing)   # key from previous bloc output
      self.routing_gate = nn.Linear(d_k_routing, n_experts)    # score per expert

      # Final prediction head
      self.head = nn.Linear(d_model, vocab_size)

      self._init_weights()

  def _init_weights(self):
      """Initialize weights to encourage early specialization."""
      with torch.no_grad():
          # Pre-specialize experts: Expert_0 gets smaller init (for arithmetic),
          # Expert_1 gets larger init (for random patterns)
          for bloc in self.blocs:
              for layer in bloc:
                  moe = layer.moe_block
                  # Initialize experts differently to encourage specialization
                  mean_init = 0.02 if moe.experts[0] is moe.experts[0] else 0.05
                  for param in moe.experts[0].parameters():
                      param.normal_(mean_init, 0.02)
                  for param in moe.experts[1].parameters():
                      param.normal_(mean_init * 2, 0.03)

  def inter_bloc_routing(self, prev_bloc_output: torch.Tensor, curr_bloc_output: torch.Tensor):
      """Lightweight attention-based routing between blocs.

      Uses the last token representation from each bloc to compute routing scores
      over experts for the next bloc.
      """
      # Last token representations
      prev_last = prev_bloc_output[:, -1, :]  # (B, D)
      curr_last = curr_bloc_output[:, -1, :]  # (B, D)

      # Query-key attention in reduced dimension
      q = self.routing_attn_q(curr_last)     # (B, d_k)
      k = self.routing_attn_k(prev_last)     # (B, d_k)

      # Scaled dot-product score
      score = (q * k).sum(dim=-1, keepdim=True) / np.sqrt(self.d_k_routing)  # (B, 1)
      expert_logits = self.routing_gate(score.squeeze(-1))  # (B, n_experts)

      return expert_logits

  def forward(self, x: torch.Tensor, train: bool = True):
      """Forward pass through all blocs with inter-bloc routing.

      Args:
        x: input tokens of shape (B, S)
        train: if True, use straight-through estimation; if False, use argmax deterministically

      Returns:
        logits: (B*S, vocab_size) prediction logits for each position
        routing_history: dict tracking routing decisions per bloc and dataset type
      """
      B, S = x.shape
      x = self.token_embed(x) + self.pos_embed[:, :S, :]

      routing_history = {
          "bloc_0": defaultdict(list),
          "bloc_1": defaultdict(list),
      }
      dataset_type_tags = getattr(self, "_current_dataset_type", "mixed")

      prev_bloc_output = None

      for b_idx, bloc in enumerate(self.blocs):
          bloc_x = x
          layer_routing_probs_all = []
          layer_routing_indices_all = []

          for layer in bloc:
              layer_out, rpri, ridx = layer(bloc_x, train=train)
              layer_routing_probs_all.append(rpri)
              layer_routing_indices_all.append(ridx)
              bloc_x = layer_out  # pass through residual connections within bloc

          # Record routing decisions for this bloc
          # Aggregate over all layers in the bloc
          for l_idx, (rpri, ridx) in enumerate(zip(layer_routing_probs_all, layer_routing_indices_all)):
              # Average over sequence positions
              avg_probs = rpri.mean(dim=1).cpu().numpy()  # (B, n_experts)
              avg_indices = ridx.cpu().numpy()             # (B,)

              tag_key = f"{dataset_type_tags}_l{l_idx}"
              routing_history[f"bloc_{b_idx}"].extend([
                  {"prob0": avg_probs[i][0], "prob1": avg_probs[i][1], "choice": int(avg_indices[i]), "tag": tag_key}
                  for i in range(B)
              ])

          # Inter-bloc routing: use previous bloc output to influence next bloc's experts
          if prev_bloc_output is not None and b_idx < self.n_blocs - 1:
              routing_logits = self.inter_bloc_routing(prev_bloc_output, bloc_x)
              routing_dist = F.softmax(routing_logits, dim=-1)

              # Apply routing bias to MoE routers in next bloc
              with torch.no_grad():
                  for layer in self.blocs[b_idx + 1]:
                      layer.moe_block.router.weight += 0.1 * routing_dist.unsqueeze(-1) * \
                                                       layer.moe_block.router.weight.sign()

          prev_bloc_output = bloc_x

      # Flatten sequence dimension for prediction head
      x_flat = x.reshape(-1, self.d_model)
      logits = self.head(x_flat)  # (B*S, vocab_size)

      return logits, routing_history
  ─── Training Utilities ────────────────────────────────────────────────────────

  def load_balancing_loss(routing_probs_all):
      """Load balancing loss encouraging uniform expert usage.

  Computes the cosine similarity between the average selection probability
  and a uniform distribution, then takes 1 - cos_sim as the loss.
  """
  avg_probs = []
  for probs_list in routing_probs_all:
      if len(probs_list) > 0:
          avg_probs.append(torch.stack(probs_list).mean(dim=0))

  if not avg_probs:
      return torch.tensor(0.0, device=routing_probs_all[0].device if routing_probs_all else 'cpu')

  avg_probs = torch.stack(avg_probs).mean(dim=0)  # (n_experts,)
  uniform = torch.ones_like(avg_probs) / avg_probs.size(0)
  cos_sim = F.cosine_similarity(avg_probs.unsqueeze(0), uniform.unsqueeze(0), dim=1)[0]
  return 1.0 - cos_sim
  def train_step(model, X_batch, Y_batch, optimizer, dataset_type="arithmetic", alpha=0.01):
      """Single training step with straight-through estimation."""
      model._current_dataset_type = dataset_type
      model.train(True)

  optimizer.zero_grad()

  logits, routing_history = model(X_batch, train=True)

  # Flatten for cross-entropy
  logits_flat = logits.reshape(-1, logits.size(-1))
  Y_flat = Y_batch.reshape(-1)

  ce_loss = F.cross_entropy(logits_flat, Y_flat)

  # Collect routing probabilities for load balancing
  routing_probs_all = []
  for bloc_key in routing_history:
      for entry in routing_history[bloc_key]:
          p0 = torch.tensor(entry["prob0"], device=logits.device)
          p1 = torch.tensor(entry["prob1"], device=logits.device)
          routing_probs_all.append(torch.stack([p0, p1]))

  lb_loss = load_balancing_loss(routing_probs_all)

  total_loss = ce_loss + alpha * lb_loss
  total_loss.backward()

  # Gradient clipping
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

  optimizer.step()

  return total_loss.item(), ce_loss.item(), lb_loss.item(), routing_history
  @torch.no_grad()
  def evaluate(model, X_batch, Y_batch, dataset_type="arithmetic"):
      """Evaluation without gradient computation."""
      model._current_dataset_type = dataset_type
      model.eval()

  logits, _ = model(X_batch, train=False)

  logits_flat = logits.reshape(-1, logits.size(-1))
  Y_flat = Y_batch.reshape(-1)

  ce_loss = F.cross_entropy(logits_flat, Y_flat)
  preds = logits_flat.argmax(dim=-1)
  accuracy = (preds == Y_flat).float().mean().item()

  return ce_loss.item(), accuracy
  ─── Main Training Loop ────────────────────────────────────────────────────────

  def main():
      # Reproducibility
      torch.manual_seed(42)
      np.random.seed(42)

  # Hyperparameters
  VOCAB_SIZE = 50
  D_MODEL = 64
  SEQ_LEN = 30
  BATCH_SIZE = 32
  LR = 1e-3
  N_STEPS = 500
  ALPHA_LB = 0.01
  N_SAMPLES_PER_TYPE = 200

  print("=" * 70)
  print("  B-MoE: Block Mixture-of-Experts Toy Model")
  print("=" * 70)
  print(f"  Architecture: {D_MODEL}d model, "
        f"{N_SAMPLES_PER_TYPE} samples/type, "
        f"vocab={VOCAB_SIZE}, seq_len={SEQ_LEN}")
  print(f"  Training: {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}, α_lb={ALPHA_LB}")
  print("-" * 70)

  # Build datasets
  arith_data = build_dataset(N_SAMPLES_PER_TYPE, SEQ_LEN, "arithmetic")
  rand_data = build_dataset(N_SAMPLES_PER_TYPE, SEQ_LEN, "random")

  arith_X, arith_Y = create_training_pairs(arith_data, SEQ_LEN)
  rand_X, rand_Y = create_training_pairs(rand_data, SEQ_LEN)

  print(f"\nDataset sizes:")
  print(f"  Arithmetic: {arith_X.shape[0]} sequences")
  print(f"  Random:     {rand_X.shape[0]} sequences")

  # Initialize model
  model = BMoE(
      vocab_size=VOCAB_SIZE,
      d_model=D_MODEL,
      n_heads=4,
      n_blocs=2,
      layers_per_bloc=2,
      n_experts=2,
      d_k_routing=16,
  )

  total_params = sum(p.numel() for p in model.parameters())
  moe_params = sum(p.numel() for n, p in model.named_parameters() if "moe_block" in n)
  print(f"\nModel parameters: {total_params:,} (MoE: {moe_params:,})")

  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  # Tracking
  losses_arith = []
  losses_rand = []
  accs_arith = []
  accs_rand = []
  routing_records = {"bloc_0": [], "bloc_1": []}

  step_interval = max(1, N_STEPS // 50)  # log every ~50 steps

  print(f"\nTraining ({N_STEPS} steps)...\n")

  for step in range(N_STEPS):
      # Sample batches from both datasets (interleaved training)
      arith_idx = torch.randint(0, arith_X.shape[0], (BATCH_SIZE,))
      rand_idx = torch.randint(0, rand_X.shape[0], (BATCH_SIZE,))

      X_arith, Y_arith = arith_X[arith_idx], arith_Y[arith_idx]
      X_rand, Y_rand = rand_X[rand_idx], rand_Y[rand_idx]

      # Train on arithmetic batch
      loss_a, ce_a, lb_a, hist_a = train_step(model, X_arith, Y_arith, optimizer, "arithmetic", ALPHA_LB)

      # Train on random batch
      loss_r, ce_r, lb_r, hist_r = train_step(model, X_rand, Y_rand, optimizer, "random", ALPHA_LB)

      losses_arith.append(loss_a)
      losses_rand.append(loss_r)

      # Evaluate periodically
      if step % step_interval == 0 or step == N_STEPS - 1:
          eval_loss_a, eval_acc_a = evaluate(model, X_arith[:64], Y_arith[:64], "arithmetic")
          eval_loss_r, eval_acc_r = evaluate(model, X_rand[:64], Y_rand[:64], "random")

          losses_arith.append(eval_loss_a)
          losses_rand.append(eval_loss_r)
          accs_arith.append(eval_acc_a)
          accs_rand.append(eval_acc_r)

          # Record routing decisions (aggregate over both dataset types)
          for hist in [hist_a, hist_r]:
              for bloc_key in routing_records:
                  entries = hist.get(bloc_key, [])
                  for entry in entries[:32]:  # limit to avoid memory issues
                      routing_records[bloc_key].append({
                          "choice": entry["choice"],
                          "prob0": entry["prob0"],
                          "tag": entry["tag"],
                      })

          log_msg = f"  Step {step:>4d} | " \
                    f"L_arith={loss_a:.4f} L_rand={loss_r:.4f} | " \
                    f"A_arith={eval_acc_a:.3f} A_rand={eval_acc_r:.3f}"
          print(log_msg)

  # ── Final Evaluation ────────────────────────────────────────────
  print("\n" + "=" * 70)
  print("  FINAL EVALUATION")
  print("=" * 70)

  eval_loss_a_final, eval_acc_a_final = evaluate(model, arith_X[:128], arith_Y[:128], "arithmetic")
  eval_loss_r_final, eval_acc_r_final = evaluate(model, rand_X[:128], rand_Y[:128], "random")

  print(f"\n  Arithmetic dataset:  Loss={eval_loss_a_final:.4f}, Accuracy={eval_acc_a_final:.3f}")
  print(f"  Random dataset:      Loss={eval_loss_r_final:.4f}, Accuracy={eval_acc_r_final:.3f}")

  # ── Routing Analysis ────────────────────────────────────────────
  print("\n" + "=" * 70)
  print("  ROUTING ANALYSIS")
  print("=" * 70)

  # Compute per-bloc expert selection frequency by input type
  bloc_routing_stats = {}
  for bloc_key in ["bloc_0", "bloc_1"]:
      records = routing_records[bloc_key]
      if not records:
          continue

      stats = {"arith": defaultdict(int), "rand": defaultdict(int), "total": defaultdict(int)}
      for rec in records:
          tag = rec["tag"]
          choice = rec["choice"]
          if "arith" in tag:
              stats["arith"][choice] += 1
              stats["total"][choice] += 1
          elif "rand" in tag:
              stats["rand"][choice] += 1
              stats["total"][choice] += 1

      bloc_routing_stats[bloc_key] = stats

      print(f"\n  {bloc_key}:")
      for dtype_name in ["arith", "rand", "total"]:
          total_count = sum(stats[dtype_name].values())
          if total_count == 0:
              continue
          freq0 = stats[dtype_name][0] / total_count * 100
          freq1 = stats[dtype_name][1] / total_count * 100
          label = "Arithmetic" if dtype_name == "arith" else ("Random" if dtype_name == "rand" else "Total")
          print(f"    {label:>10s} ({total_count:>4d} samples): "
                f"Expert_0={freq0:5.1f}% | Expert_1={freq1:5.1f}%")

  # ── Plotting ────────────────────────────────────────────────────
  try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt

      fig, axes = plt.subplots(2, 3, figsize=(18, 8))

      # Loss curves per dataset type
      ax = axes[0, 0]
      steps_arith = list(range(0, len(losses_arith), step_interval + 1))[:len(losses_arith)]
      steps_rand = list(range(0, len(losses_rand), step_interval + 1))[:len(losses_rand)]
      ax.plot(steps_arith, losses_arith[:len(steps_arith)], 'b-', label='Arithmetic', linewidth=2)
      ax.plot(steps_rand, losses_rand[:len(steps_rand)], 'r-', label='Random', linewidth=2)
      ax.set_xlabel('Step')
      ax.set_ylabel('Loss')
      ax.set_title('Per-Dataset Loss Curves')
      ax.legend()
      ax.grid(True, alpha=0.3)

      # Accuracy curves
      ax = axes[0, 1]
      n_acc_points = min(len(accs_arith), len(accs_rand))
      if n_acc_points > 0:
          acc_steps = range(n_acc_points)
          ax.plot(acc_steps, accs_arith[:n_acc_points], 'b-', label='Arithmetic', linewidth=2)
          ax.plot(acc_steps, accs_rand[:n_acc_points], 'r-', label='Random', linewidth=2)
      ax.set_xlabel('Evaluation Step')
      ax.set_ylabel('Accuracy')
      ax.set_title('Per-Dataset Accuracy')
      ax.legend()
      ax.grid(True, alpha=0.3)

      # Expert selection frequency - Bloc 0
      ax = axes[0, 2]
      stats_0 = bloc_routing_stats.get("bloc_0", {})
      if "arith" in stats_0 and sum(stats_0["arith"].values()) > 0:
          arith_freq_0 = stats_0["arith"][0] / sum(stats_0["arith"].values()) * 100
          rand_freq_0 = stats_0["rand"][0] / sum(stats_0["rand"].values()) * 100
          x_pos = [0.5, 1.5]
          ax.bar(x_pos, [arith_freq_0, rand_freq_0], color=['blue', 'red'], alpha=0.7, edgecolor='black')
          ax.set_xticks([0.5, 1.5])
          ax.set_xticklabels(['Arithmetic', 'Random'])
      ax.set_ylabel('Selects Expert_0 (%)')
      ax.set_title(f'Bloc 0 — Expert Selection Frequency\n(Expert_0 select rate)')
      ax.set_ylim(0, 105)
      ax.grid(True, alpha=0.3, axis='y')

      # Expert selection frequency - Bloc 1
      ax = axes[1, 0]
      stats_1 = bloc_routing_stats.get("bloc_1", {})
      if "arith" in stats_1 and sum(stats_1["arith"].values()) > 0:
          arith_freq_1 = stats_1["arith"][0] / sum(stats_1["arith"].values()) * 100
          rand_freq_1 = stats_1["rand"][0] / sum(stats_1["rand"].values()) * 100
          x_pos = [0.5, 1.5]
          ax.bar(x_pos, [arith_freq_1, rand_freq_1], color=['blue', 'red'], alpha=0.7, edgecolor='black')
          ax.set_xticks([0.5, 1.5])
          ax.set_xticklabels(['Arithmetic', 'Random'])
      ax.set_ylabel('Selects Expert_0 (%)')
      ax.set_title(f'Bloc 1 — Expert Selection Frequency\n(Expert_0 select rate)')
      ax.set_ylim(0, 105)
      ax.grid(True, alpha=0.3, axis='y')

      # Expert activation correlation with input type
      ax = axes[1, 1]
      dataset_types = ['Arithmetic', 'Random']
      expert_0_freqs = []
      expert_1_freqs = []
      for bloc_key in ["bloc_0", "bloc_1"]:
          s = bloc_routing_stats.get(bloc_key, {})
          if "arith" in s and sum(s["arith"].values()) > 0:
              arith_e0 = s["arith"][0] / sum(s["arith"].values())
              rand_e0 = s["rand"][0] / sum(s["rand"].values())
              expert_0_freqs.append(arith_e0)
              expert_0_freqs.append(rand_e0)

      x_pos_corr = list(range(len(expert_0_freqs)))
      colors_corr = ['blue', 'red'] * (len(dataset_types))
      ax.bar(x_pos_corr, expert_0_freqs, color=['blue', 'red', 'blue', 'red'], alpha=0.7, edgecolor='black')
      ax.set_xticks(list(range(4)))
      ax.set_xticklabels(['B0-Arith', 'B0-Rand', 'B1-Arith', 'B1-Rand'])
      ax.set_ylabel('Fraction selecting Expert_0')
      ax.set_title('Expert Activation Correlation with Input Type')
      ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
      ax.grid(True, alpha=0.3, axis='y')

      # Summary heatmap: routing decisions vs input type
      ax = axes[1, 2]
      matrix_data = []
      row_labels = []
      col_labels = ['Exp_0', 'Exp_1']
      for bloc_key in ["bloc_0", "bloc_1"]:
          s = bloc_routing_stats.get(bloc_key, {})
          for dtype_name in ["arith", "rand"]:
              total_count = sum(s[dtype_name].values())
              if total_count > 0:
                  freq0 = s[dtype_name][0] / total_count
                  freq1 = s[dtype_name][1] / total_count
                  matrix_data.append([freq0, freq1])
                  label = f"{bloc_key} {dtype_name}"
                  row_labels.append(label.replace("_", " ").title())

      matrix_data = np.array(matrix_data)
      im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
      ax.set_xticks(list(range(2)))
      ax.set_xticklabels(col_labels)
      ax.set_yticks(list(range(len(row_labels))))
      ax.set_yticklabels(row_labels)

      # Annotate cells
      for i in range(matrix_data.shape[0]):
          for j in range(matrix_data.shape[1]):
              text_color = 'white' if matrix_data[i, j] > 0.65 else 'black'
              ax.text(j, i, f'{matrix_data[i,j]:.2f}', ha='center', va='center',
                     fontsize=9, color=text_color, fontweight='bold')

      ax.set_title('Routing Decisions vs Input Type\n(Heatmap: fraction selecting Expert_0)')
      plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

      plt.suptitle('B-MoE: Block Mixture-of-Experts — Training Results',
                   fontsize=14, fontweight='bold', y=1.02)
      plt.tight_layout()
      plt.savefig('/Users/sviou/Documents/BMOE research/bmoe_results.png', dpi=150, bbox_inches='tight')
      print("\n  Plot saved to: bmoe_results.png")
      plt.close()

  except ImportError:
      print("\n  ⚠ matplotlib not available — skipping plots")

  # ── Summary Table ───────────────────────────────────────────────
  print("\n" + "=" * 70)
  print("  SUMMARY TABLE: Routing Decisions vs Input Type")
  print("=" * 70)

  header = f"{'Bloc':<8} {'Input Type':<12} {'N Samples':>10} {'Exp_0 %':>9} {'Exp_1 %':>9} {'Bias':>8}"
  print(header)
  print("-" * len(header))

  for bloc_key in ["bloc_0", "bloc_1"]:
      stats = bloc_routing_stats.get(bloc_key, {})
      for dtype_name in ["arith", "rand"]:
          total_count = sum(stats[dtype_name].values())
          if total_count == 0:
              continue
          freq0 = stats[dtype_name][0] / total_count * 100
          freq1 = stats[dtype_name][1] / total_count * 100
          bias = "→ Exp_0" if freq0 > freq1 else ("→ Exp_1" if freq1 > freq0 else "Balanced")
          label_bloc = bloc_key.replace("_", " ").title()
          label_dtype = "Arithmetic" if dtype_name == "arith" else "Random"
          row = f"{label_bloc:<8} {label_dtype:<12} {total_count:>10} {freq0:>8.1f}% {freq1:>8.1f}% {bias:>8}"
          print(row)

  # Specialization score: how differently the model routes arithmetic vs random
  print("\n" + "-" * 70)
  print("  SPECIALIZATION SCORES (|Δ fraction_expert_0 between datasets)|")
  print("-" * 70)

  for bloc_key in ["bloc_0", "bloc_1"]:
      s = bloc_routing_stats.get(bloc_key, {})
      if "arith" in s and "rand" in s:
          n_arith = sum(s["arith"].values())
          n_rand = sum(s["rand"].values())
          if n_arith > 0 and n_rand > 0:
              frac_arith_e0 = s["arith"][0] / n_arith
              frac_rand_e0 = s["rand"][0] / n_rand
              delta = abs(frac_arith_e0 - frac_rand_e0)
              label_bloc = bloc_key.replace("_", " ").title()
              print(f"  {label_bloc:<8}: Δ = {delta:.3f} "
                    f"(arith E0={frac_arith_e0:.2f}, rand E0={frac_rand_e0:.2f}) "
                    f"{'← Specialized!' if delta > 0.15 else '← Weak'}")

  # Expert correlation with input type
  print("\n" + "-" * 70)
  print("  EXPERT ACTIVATION CORRELATION WITH INPUT TYPE")
  print("-" * 70)

  for bloc_key in ["bloc_0", "bloc_1"]:
      s = bloc_routing_stats.get(bloc_key, {})
      label_bloc = bloc_key.replace("_", " ").title()
      arith_entries = [e for e in routing_records[bloc_key] if "arith" in e["tag"]]
      rand_entries = [e for e in routing_records[bloc_key] if "rand" in e["tag"]]

      if arith_entries and rand_entries:
          arith_expert_0_frac = sum(1 for e in arith_entries if e["choice"] == 0) / len(arith_entries)
          rand_expert_0_frac = sum(1 for e in rand_entries if e["choice"] == 0) / len(rand_entries)
          diff = abs(arith_expert_0_frac - rand_expert_0_frac)
          print(f"  {label_bloc:<8}: Arithmetic→Exp0={arith_expert_0_frac:.3f}, "
                f"Random→Exp0={rand_expert_0_frac:.3f}, "
                f"Difference={diff:.3f} "
                f"{'← Clear specialization' if diff > 0.2 else '← Weak'}")

  print("\n" + "=" * 70)
  print("  DONE ✓")
  print("=" * 70)
  if name == "main":
      main()
