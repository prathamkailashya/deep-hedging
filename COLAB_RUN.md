# Comprehensive experiment run on Google Colab (GPU)

This runs the **entire pass-5 suite** and persists results back to GitHub:
in-distribution 10-seed, TC-inclusive, SPY/NIFTY regime stress battery,
causal OHLCV walk-forward (+ block-bootstrap CIs), the improved-model
studies (Whalley–Wilmott no-trade band, adaptive-DRO, RSE friction
frontier, rich regime features), **cross-market transfer** (US⇄India),
and ablations.

**Why Colab.** The only heavy model is W-DRO-T (a second-order/double-backward
gradient through attention). On Apple MPS that serialises to ~18 min per
market-seed; on a Colab CUDA GPU it is ~2–3 min. Everything else is seconds.
Every script auto-detects the device (`walk_forward_backtest.device()` →
`cuda` when no MPS is present). All scripts were validated locally at pilot
scale before this handover.

---

### CELL 1 — Clone the repository
Set **Runtime → Change runtime type → GPU** first.
```python
!git clone -b research/pass5-improvements https://github.com/prathamkailashya/deep-hedging.git
%cd deep-hedging
```
> If the branch is not pushed yet, see `PACKAGING.md` for the exact push commands.

### CELL 2 — Install dependencies
```python
!pip -q install "torch>=2.1" numpy scipy pandas yfinance
```

### CELL 3 — Verify GPU
```python
import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

### CELL 4 — (Assets) none required
The walk-forward pulls SPY / ^NSEI OHLCV live via `yfinance`; Colab has
internet, so no manual downloads. (Optional: cache to Drive — see CELL 10.)

### CELL 5 — Sanity check (fast; gates the run)
```python
!PYTHONPATH=$PWD:$PWD/src:$PWD/new_approaches/code python sanity_check.py
```
Expect `ALL OK` (5 model forwards finite + causality test 2 passed).

### CELL 6 — Pilot (one cell, ~2–3 min on GPU)
```python
!bash run_single_experiment.sh walkforward   # SPY COVID, seed 42, 4 models
```
Confirms training + evaluation on GPU before the long run.

### CELL 7 — Full suite (checkpointed, resume-safe)
Run tiers separately so a disconnect never loses completed work (each step
skips cells already in its JSON). Total ~1.5–2.5 h on a T4.
```python
# a) causal walk-forward battery (5 seeds) + bootstrap CIs
!SEEDS=42,142,242 BIG_SEEDS=42,142,242,342,442 bash run_all.sh core
# b) improved-model studies (band sweep, adaptive-DRO, RSE frontier, features)
!SEEDS=42,142,242 bash run_all.sh improved
# c) cross-market transfer US<->India (+ bootstrap)
!SEEDS=42,142,242 bash run_all.sh transfer
# d) in-distribution 10-seed + TC-inclusive + regime stress
!bash run_all.sh indist
# e) ablations (DRO radius, RSE experts, Bates transfer)
!bash run_all.sh ablation
```
Or everything in one call: `!bash run_all.sh full`.

### CELL 8 — Collect + summarise outputs
```python
!ls -1 *_results.json *_bootstrap.json 2>/dev/null
!python analyze_band.py wf_band_results.json | head -40
!python walk_forward_bootstrap.py --results wf_cross_market_results.json \
        --ref LSTM --models LSTM,3SCH,RSE,WDROT --out wf_cross_market_bootstrap.json | head -40
```

### CELL 9 — Persist results back to GitHub (durable, for further research)
Create a **fine-grained PAT** (repo `prathamkailashya/deep-hedging`,
*Contents: read/write*) and paste it when prompted — it is used only to
build a one-shot push URL and is never written to disk.
```python
import getpass, os
os.environ["GITHUB_TOKEN"] = getpass.getpass("GitHub fine-grained PAT: ")
os.environ["GIT_AUTHOR_NAME"]  = "Pratham Kailasiya"
os.environ["GIT_AUTHOR_EMAIL"] = "g_adhiraj@ma.iitr.ac.in"
!bash push_results.sh          # -> commits only *_results.json/*_bootstrap.json to branch results/auto
```

### CELL 10 — Also snapshot to Google Drive (belt-and-suspenders)
```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/deep-hedging-results
!cp -v *_results.json *_bootstrap.json /content/drive/MyDrive/deep-hedging-results/
# and download locally:
from google.colab import files
for f in ["walk_forward_ci_results.json","walk_forward_ci_bootstrap.json",
          "wf_band_results.json","wf_cross_market_results.json","wf_cross_market_bootstrap.json"]:
    try: files.download(f)
    except Exception as e: print("skip", f, e)
```

---

## Runtime / GPU budget (Colab T4)

| Tier | contents | approx time |
|---|---|---|
| core | walk-forward 5 seeds × 4 models × 2 markets + bootstrap | 45–75 min (W-DRO-T dominates) |
| improved | band sweep + adaptive-DRO (3 seeds SPY) + RSE frontier×3 + features | 25–40 min |
| transfer | SPY⇄NIFTY, 3 seeds, 4 models + band | 25–40 min |
| indist | 10-seed Heston (7 models) + TC + stress | 40–70 min |
| ablation | DRO radius + RSE experts + Bates transfer | 15–30 min |

**GPU memory:** small (< 2 GB; batch 64/256, d_model 64). A free **T4 is
sufficient.** **Colab Pro** helps only by reducing disconnects on the longer
`indist`/`core` tiers (longer sessions, better GPUs); Pro+ is not needed.

## Disconnect recovery
Every script writes its JSON after each cell and resumes by skipping present
keys. After a disconnect: re-run CELL 1–2, then re-issue the same CELL 7
line — it continues where it stopped. Persist early and often with CELL 9.
