#!/usr/bin/env bash
# =====================================================================
# Comprehensive reproducible experiment suite (pass 5).
# Designed for a Colab CUDA GPU but runs anywhere (auto-detects device).
# Each step is independent, logs to results/logs/, and is resume-safe:
# re-running skips cells already present in the output JSON.
#
#   bash run_all.sh            # full suite
#   bash run_all.sh core       # only the causal walk-forward + bootstrap
#   bash run_all.sh improved   # band + adaptive-DRO + RSE frontier
#   bash run_all.sh transfer   # cross-market generalisation
#   bash run_all.sh indist     # in-distribution 10-seed + TC + stress
#   bash run_all.sh ablation   # DRO-radius / RSE-expert / Bates transfer
# =====================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/new_approaches/code"
LOG="$ROOT/results/logs"; mkdir -p "$LOG"
SEEDS="${SEEDS:-42,142,242}"
BIG_SEEDS="${BIG_SEEDS:-42,142,242,342,442}"
run(){ echo "=== $1 ==="; shift; ( "$@" ) 2>&1 | tee "$LOG/$(date +%H%M%S)_$(echo "$1"|tr ' /' '__').log"; }

step_sanity(){ run "sanity" python "$ROOT/sanity_check.py"; }

step_core(){   # causal OHLCV walk-forward battery + block-bootstrap CIs
  run "walk_forward battery"  python "$ROOT/walk_forward_backtest.py" --seeds "$BIG_SEEDS"
  run "walk_forward bootstrap" python "$ROOT/walk_forward_bootstrap.py"
}

step_improved(){  # improved-model studies (all reuse the causal pipeline)
  run "no-trade band sweep"   python "$ROOT/wf_band_pilot.py" --seeds "$SEEDS" --markets SPY,NIFTY --out "$ROOT/wf_band_results.json"
  run "band bootstrap"        python "$ROOT/walk_forward_bootstrap.py" --results "$ROOT/wf_band_results.json" --ref RSE_K0.0 --models RSE_K0.0,RSE_K0.6,RSE_K1.0 --out "$ROOT/wf_band_bootstrap.json"
  run "WDROT adaptive-vs-van" python "$ROOT/wf_variants_pilot.py" --which wdrot --seeds "$SEEDS" --markets SPY --out "$ROOT/wf_wdrot_results.json"
  for k in 1 2 3; do
    run "RSE frontier k=$k"    python "$ROOT/wf_variants_pilot.py" --which rse --only RSE_base,RSE_fric --seeds "$SEEDS" --markets SPY,NIFTY --fric-mult "$k" --out "$ROOT/wf_rse_fric_k$k.json"
  done
  run "RSE rich features"     python "$ROOT/wf_variants_pilot.py" --which rse --only RSE_base,RSE_rich --seeds "$SEEDS" --markets SPY,NIFTY --out "$ROOT/wf_rse_features.json"
}

step_transfer(){  # cross-market generalisation (train US<->India, deploy on crisis)
  run "cross-market transfer" python "$ROOT/wf_cross_market.py" --seeds "$SEEDS"
  run "xmarket bootstrap"     python "$ROOT/walk_forward_bootstrap.py" --results "$ROOT/wf_cross_market_results.json" --ref LSTM --models LSTM,3SCH,RSE,WDROT --out "$ROOT/wf_cross_market_bootstrap.json"
}

step_indist(){    # in-distribution 10-seed + TC-inclusive + regime stress battery
  run "in-distribution 10seed" python "$ROOT/new_approaches/experiments/run_full_experiments.py" --all --r 0
  run "TC-inclusive 10seed"    python "$ROOT/full_tc_validation.py"
  run "SPY/NIFTY stress"       python "$ROOT/spy_nifty_full_validation.py"
}

step_ablation(){  # DRO radius (eps=0..0.2), RSE expert-drop, Bates transfer
  run "ablations" python "$ROOT/jaws_research/experiments/ablation.py" || echo "ablation.py optional; skipping on error"
}

MODE="${1:-full}"
case "$MODE" in
  sanity)   step_sanity ;;
  core)     step_sanity; step_core ;;
  improved) step_sanity; step_improved ;;
  transfer) step_sanity; step_transfer ;;
  indist)   step_sanity; step_indist ;;
  ablation) step_ablation ;;
  full)     step_sanity; step_core; step_improved; step_transfer; step_indist; step_ablation ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac
echo "[run_all] done ($MODE). Results: $ROOT/*.json  Logs: $LOG"
