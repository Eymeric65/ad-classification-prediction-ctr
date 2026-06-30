"""MLP vs LightGBM on the forward holdout, plus a cross-family ensemble."""
import sys
import numpy as np
sys.path.insert(0, "/workspace/experiments")
from harness import load_everything, eval_split, temporal_indices, lgbm_factory, _metrics
from mlp import mlp_factory

D = load_everything()
CUT = "2022-11-01"
tr, te = temporal_indices(D, CUT)
Yte = D["Y"][te]

def show(label, r):
    print(f"{label:22}  RMSE={r['rmse']:.6f}  all={r['all']:+5.1f}  tail={r['tail']:+5.1f}  "
          f"mid={r['mid']:+5.1f}  yda={r['yda']:+5.1f}  lap={r['lap']:+5.1f}")

print(f"### forward holdout {CUT}  (train {len(tr):,} / test {len(te):,})  device-check below")
import torch; print("cuda:", torch.cuda.is_available())
print(f"{'variant':22}  {'RMSE':>8}  {'all':>5}  {'tail':>5}  {'mid':>5}  {'yda':>5}  {'lap':>5}")
print("-" * 82)

r_lgb = eval_split(D, tr, te, lgbm_factory(D["pcfg"]))
show("lightgbm (baseline)", r_lgb)

# MLP — a few seeds to gauge variance, then average their score predictions (seed-ensemble)
mlp_preds = []
for sd in range(3):
    r = eval_split(D, tr, te, mlp_factory(hidden=(256, 128), dropout=0.2, epochs=80, seed=sd))
    show(f"mlp seed{sd}", r)
    mlp_preds.append(r["Yp"])
Yp_mlp = np.mean(mlp_preds, axis=0)
show("mlp (3-seed avg)", _metrics(D, te, Yte, Yp_mlp))

# cross-family ensemble: blend LGBM + MLP scores
for a in [0.3, 0.5, 0.7]:
    Yp = a * r_lgb["Yp"] + (1 - a) * Yp_mlp
    show(f"ensemble {a:.1f}lgb/{1-a:.1f}mlp", _metrics(D, te, Yte, Yp))
