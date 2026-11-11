"""compare.py - small-object AP of two codecs. usage: python compare.py a.json b.json"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
print(f"{'SNR':>5}{'A small':>10}{'B small':>10}{'change':>9}   {'A tiny':>8}{'B tiny':>8}")
for s in a["ap"]:
    x, y = a["ap"][s], b["ap"][s]
    print(f"{s:>5}{x['small']:>10.4f}{y['small']:>10.4f}{100 * (y['small'] / x['small'] - 1):>8.1f}%"
          f"   {x['tiny']:>8.4f}{y['tiny']:>8.4f}")