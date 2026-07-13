# Preparing a clean push to GitHub

Target repo: `https://github.com/prathamkailashya/deep-hedging.git` (history preserved).
**Nothing here is auto-executed** — run these yourself after reviewing.

## Recommended branch strategy

| branch | purpose |
|---|---|
| `main` | stable, reviewed code |
| `research/pass5-improvements` | this session's code + improved models (the PR source) |
| `results/auto` | machine-written result JSONs from Colab (via `push_results.sh`) — kept off the code branch so diffs stay clean |

Keep **code** and **bulky result artefacts** on separate branches; open a PR
from `research/pass5-improvements` → `main`, and cherry-pick result JSONs from
`results/auto` when you want them referenced by the paper.

## 1. Inspect
```bash
git status
git ls-files -ci --exclude-standard | head      # tracked files the new .gitignore now excludes
```

## 2. Stop tracking the excluded bloat (keeps your local copies)
The new `.gitignore` excludes LaTeX/PDF/images/reference/reports/checkpoints.
Remove those from the index **without deleting them on disk**:
```bash
git rm -r --cached $(git ls-files -ci --exclude-standard)
```
(That command untracks exactly what `.gitignore` matches: `reports/*.pdf`,
`reference/*.png`, `experiments/*/figures/*`, `combined_report*.tex`,
`supplementary_material.tex`, `*.tex`, `*.bib`, `poster.pdf`, …)

## 3. Create the research branch and stage code + reproducibility artefacts
```bash
git checkout -b research/pass5-improvements

git add .gitignore README.md PACKAGING.md COLAB_RUN.md \
        run_all.sh run_single_experiment.sh push_results.sh sanity_check.py \
        experiment_manifest.json seeds.json requirements.txt \
        walk_forward_backtest.py walk_forward_bootstrap.py \
        wf_variants_pilot.py wf_band_pilot.py wf_cross_market.py analyze_band.py \
        new_approaches/code/w_dro_t_adaptive.py new_approaches/code/rse_improved.py \
        new_approaches/code/no_trade_band.py \
        new_approaches/experiments/run_full_experiments.py \
        tests/test_walkforward_causality.py \
        src/   new_approaches/code/   new_approaches/experiments/

# small result JSONs are provenance for the paper's tables — include them:
git add -f walk_forward_ci_results.json walk_forward_multiseed_results.json \
        wf_band_results.json wf_band_spy3.json wf_variants_results.json \
        wf_wdrot_adapt.json full_tc_validation_results.json \
        spy_nifty_full_validation_results.json 2>/dev/null || true
```

## 4. Commit and push
```bash
git commit -m "pass 5: causal WDROT walk-forward, block-bootstrap CIs, \
Whalley-Wilmott no-trade band, cross-market transfer, adaptive-DRO/friction \
research variants, Colab reproducibility artefacts"

git push -u origin research/pass5-improvements
```

## 5. Open the PR
```bash
gh pr create --base main --head research/pass5-improvements \
  --title "Pass 5: causal OHLCV walk-forward, bootstrap CIs, no-trade-band overlay, cross-market transfer" \
  --body-file - <<'EOF'
Adds the causal walk-forward battery (WDROT included, per-window PnL),
paired moving-block-bootstrap CVaR CIs, a Whalley-Wilmott no-trade-band
overlay (model-agnostic, inference-only), cross-market transfer (US<->India),
and pilot-validated research variants (adaptive Wasserstein radius,
friction-aware RSE, rich regime features). Full multi-seed runs execute on
Colab (see COLAB_RUN.md) and land on branch results/auto.
EOF
```

## Notes
- `main` history is preserved; this is a new branch + PR, nothing is force-pushed.
- The manuscript (`paper/`, `*.tex`, `*.bib`) stays **local** by design — it is
  compiled on Overleaf, not shipped in the code artefact.
- Never commit a PAT or `.env`; results are pushed with a one-shot token URL
  (see `push_results.sh`).
