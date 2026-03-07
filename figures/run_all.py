#!/usr/bin/env python3
"""Master runner: execute all figure generation scripts."""
import subprocess, sys, os

FIGDIR = os.path.dirname(os.path.abspath(__file__))
scripts = [
    'fig_part1_heston.py',
    'fig_part2_architectures.py',
    'fig_part3_regime_training.py',
    'fig_part4_risk_metrics.py',
    'fig_part5_market_validation.py',
    'fig_part6_regulatory.py',
]

print("=" * 60)
print("GENERATING ALL PUBLICATION-READY FIGURES")
print("=" * 60)

for script in scripts:
    path = os.path.join(FIGDIR, script)
    print(f"\n{'─'*60}")
    print(f"Running: {script}")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable, path], cwd=FIGDIR, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠ {script} exited with code {result.returncode}")

print(f"\n{'='*60}")
print("ALL DONE — Figures saved to:")
print(f"  {FIGDIR}")
print(f"{'='*60}")

# List all generated PDFs
pdfs = sorted([f for f in os.listdir(FIGDIR) if f.endswith('.pdf')])
print(f"\nGenerated {len(pdfs)} figures:")
for p in pdfs:
    size = os.path.getsize(os.path.join(FIGDIR, p))
    print(f"  {p}  ({size/1024:.0f} KB)")
