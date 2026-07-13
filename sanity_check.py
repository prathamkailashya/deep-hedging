#!/usr/bin/env python3
"""Fast smoke test — imports, a tiny forward for every model, and the
walk-forward causality regression. Run this before any long job.
Exits non-zero on failure so it can gate a pipeline."""
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src", ROOT / "new_approaches" / "code"):
    sys.path.insert(0, str(p))

import torch, torch.nn.functional as F  # noqa: E402

def main() -> int:
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] torch {torch.__version__}  device={dev}")

    B, T, d = 8, 30, 5
    feats = torch.randn(B, T, d); feats[:, :, 2] = feats[:, :, 2].abs() * 0.3
    feats[:, :, 0] = torch.exp(torch.cumsum(torch.randn(B, T) * 0.01, 1))  # positive price proxy
    prices = 100 + torch.cumsum(torch.randn(B, T + 1) * 0.5, 1)
    payoff = F.relu(prices[:, -1] - 100)

    from src.models.kozyra_models import HedgingLSTM
    from rse import RegimeSwitchingEnsemble
    from rse_improved import RegimeSwitchingEnsembleRich
    from w_dro_t import WDROTransformerHedger
    from w_dro_t_adaptive import WDROTransformerHedgerAdaptive

    checks = {
        "LSTM": lambda: HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                                    delta_scale=1.5)(feats),
        "RSE": lambda: RegimeSwitchingEnsemble(input_dim=5, n_regimes=4)(feats),
        "RSE_rich": lambda: RegimeSwitchingEnsembleRich(input_dim=5, n_regimes=4)(feats),
        "WDROT": lambda: WDROTransformerHedger(epsilon=0.1)(feats),
        "WDROT_adapt": lambda: WDROTransformerHedgerAdaptive(epsilon=0.1, beta=3.0)(feats),
    }
    ok = True
    for name, fn in checks.items():
        try:
            out = fn()
            good = out.shape[:2] == (B, T) and bool(torch.isfinite(out).all())
            print(f"[sanity] {name:12s} out={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}"
                  f"  {'OK' if good else 'FAIL'}")
            ok &= good
        except Exception as e:  # noqa: BLE001
            print(f"[sanity] {name:12s} FAIL: {type(e).__name__}: {e}")
            ok = False

    r = subprocess.run([sys.executable, "-m", "pytest",
                        str(ROOT / "tests" / "test_walkforward_causality.py"), "-q"],
                       capture_output=True, text=True)
    print("[sanity] causality test:", r.stdout.strip().splitlines()[-1] if r.stdout else r.returncode)
    ok &= (r.returncode == 0)

    print("[sanity]", "ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
