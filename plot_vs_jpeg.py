r"""plot_vs_jpeg.py - small-object AP against SNR for several result files.
usage: python plot_vs_jpeg.py results\lam0.json results\lam32.json results\jpeg.json [--out results\vs_jpeg.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
out = "results/vs_jpeg.png"
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    del args[i:i + 2]
fig, ax = plt.subplots(figsize=(6, 4))
for p in args:
    ap = json.loads(Path(p).read_text())["ap"]
    snrs = sorted(ap, key=float)
    ax.plot([float(s) for s in snrs], [ap[s]["small"] for s in snrs], marker="o", label=Path(p).stem)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("AP small @0.5")
ax.grid(alpha=0.3)
ax.legend()
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)