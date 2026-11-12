"""sweep_table.py - AP by object size for every lambda run, with change vs lambda=0.
usage: python sweep_table.py [snr ...]      (default: 10 20)
"""
import json
import sys
from pathlib import Path

RUNS = [("0", "lam0"), ("8", "lam8"), ("16", "lam16"), ("32", "lam32"), ("64", "lam64")]
BINS = ("tiny", "small", "large", "all")
snrs = sys.argv[1:] or ["10", "20"]

data = {}
for lam, stem in RUNS:
    p = Path("results") / f"{stem}.json"
    if p.exists():
        data[lam] = json.loads(p.read_text())["ap"]
    else:
        print(f"(skipping lambda={lam}: {p} not found)")
base = data.get("0")
if base is None:
    raise SystemExit("results/lam0.json is needed as the baseline")

for s in snrs:
    print(f"\nSNR {s} dB   AP@0.5 (change vs lambda=0)")
    print(f"{'lambda':>7}" + "".join(f"{b:>19}" for b in BINS))
    for lam, ap in data.items():
        row = f"{lam:>7}"
        for b in BINS:
            v = ap[s][b]
            cell = f"{v:.4f}" if lam == "0" else f"{v:.4f} ({100 * (v / base[s][b] - 1):+.1f}%)"
            row += f"{cell:>19}"
        print(row)