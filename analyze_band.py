#!/usr/bin/env python3
"""Summarise a no-trade-band sweep (wf_band_*.json): seed-averaged
TC-inclusive CVaR95 and turnover per (market, cell, base-model, K), and
the improvement of the best K over the raw policy (K=0). Usage:
    python analyze_band.py [results.json]
"""
import json, sys, re
from collections import defaultdict
import numpy as np

Kre = re.compile(r"^(.*)_K([0-9.]+)$")


def summarize(path: str) -> None:
  state = json.load(open(path))
  print(f"# No-trade-band sweep summary: {path}\n")
  for market, tests in state.items():
    for cell, tags in tests.items():
        # group tags by base model
        by_model = defaultdict(dict)
        for tag, seeds in tags.items():
            m = Kre.match(tag)
            if not m:
                continue
            base, K = m.group(1), float(m.group(2))
            cv = np.mean([v["cvar_95"] for v in seeds.values()])
            tu = np.mean([v["turnover"] for v in seeds.values()])
            nseed = len(seeds)
            by_model[base][K] = (cv, tu)
        for base, ks in by_model.items():
            Ks = sorted(ks)
            raw_cv, raw_tu = ks[0.0]
            # best K by CVaR
            bestK = min(Ks, key=lambda k: ks[k][0])
            bcv, btu = ks[bestK]
            dcv = 100 * (bcv - raw_cv) / raw_cv
            dtu = 100 * (btu - raw_tu) / raw_tu
            sweep = "  ".join(f"K{k}:{ks[k][0]:.1f}/t{ks[k][1]:.2f}" for k in Ks)
            print(f"{market:5s} {cell:16s} {base:5s} (n={nseed}): {sweep}")
            print(f"      -> best K={bestK}: CVaR {raw_cv:.1f}->{bcv:.1f} ({dcv:+.1f}%), "
                  f"turnover {raw_tu:.2f}->{btu:.2f} ({dtu:+.1f}%)")
    print()


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "wf_band_results.json")
