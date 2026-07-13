#!/usr/bin/env bash
# Debug / smoke a single experiment quickly (1 seed, minimal scope).
#   bash run_single_experiment.sh sanity
#   bash run_single_experiment.sh walkforward   # SPY COVID only, seed 42
#   bash run_single_experiment.sh band          # band sweep, SPY, seed 42
#   bash run_single_experiment.sh transfer      # SPY->NIFTY, seed 42
#   bash run_single_experiment.sh wdrot         # adaptive vs vanilla, SPY, seed 42
set -uo pipefail
export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/new_approaches/code"
case "${1:-sanity}" in
  sanity)      python "$ROOT/sanity_check.py" ;;
  walkforward) python "$ROOT/walk_forward_backtest.py" --markets SPY --tests covid_2020 --seeds 42 --out "$ROOT/_debug_wf.json" ;;
  band)        python "$ROOT/wf_band_pilot.py" --seeds 42 --markets SPY --out "$ROOT/_debug_band.json" ;;
  transfer)    python "$ROOT/wf_cross_market.py" --seeds 42 --out "$ROOT/_debug_xmarket.json" ;;
  wdrot)       python "$ROOT/wf_variants_pilot.py" --which wdrot --seeds 42 --markets SPY --out "$ROOT/_debug_wdrot.json" ;;
  rse)         python "$ROOT/wf_variants_pilot.py" --which rse --seeds 42 --markets SPY,NIFTY --out "$ROOT/_debug_rse.json" ;;
  *) echo "unknown experiment: $1"; exit 2 ;;
esac
