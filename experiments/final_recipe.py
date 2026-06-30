"""Final recipe: do feat_iw (on the LGBM member) and the LGBM+MLP ensemble STACK?

feat_iw helps the tail/pooled side; the ensemble helps the middle + yda. They attack
different buckets, so combining them should keep both gains. Tests across the 3 cutoffs.
"""
import sys
import numpy as np
sys.path.insert(0, "/workspace/experiments")
from harness import load_everything, eval_split, temporal_indices, lgbm_factory, _metrics
from methods import feature_iw
from mlp import mlp_factory

CUTOFFS = ["2022-08-01", "2022-11-01", "2023-01-01"]
D = load_everything()
lgb = lgbm_factory(D["pcfg"])
NAMES = ["baseline", "iw_lgb", "ens.5(iw+mlp)", "ens.6(iw+mlp)", "ens.5(plain+mlp)"]
agg = {n: [] for n in NAMES}

def line(label, r):
    return (f"{label:18}  RMSE={r['rmse']:.6f}  all={r['all']:+5.1f}  tail={r['tail']:+5.1f}  "
            f"mid={r['mid']:+5.1f}  yda={r['yda']:+5.1f}  lap={r['lap']:+5.1f}")

for cut in CUTOFFS:
    tr, te = temporal_indices(D, cut)
    Yte = D["Y"][te]
    print(f"\n### cutoff {cut}")
    r_base = eval_split(D, tr, te, lgb)
    print(line("baseline", r_base)); agg["baseline"].append(r_base)

    w = feature_iw(D, tr)
    r_iw = eval_split(D, tr, te, lgb, sw_extra=w)
    print(line("iw_lgb", r_iw)); agg["iw_lgb"].append(r_iw)

    Yp_mlp = np.mean([eval_split(D, tr, te, mlp_factory(epochs=80, seed=s))["Yp"]
                      for s in range(3)], axis=0)

    r = _metrics(D, te, Yte, 0.5 * r_iw["Yp"] + 0.5 * Yp_mlp)
    print(line("ens.5(iw+mlp)", r)); agg["ens.5(iw+mlp)"].append(r)
    r = _metrics(D, te, Yte, 0.6 * r_iw["Yp"] + 0.4 * Yp_mlp)
    print(line("ens.6(iw+mlp)", r)); agg["ens.6(iw+mlp)"].append(r)
    r = _metrics(D, te, Yte, 0.5 * r_base["Yp"] + 0.5 * Yp_mlp)
    print(line("ens.5(plain+mlp)", r)); agg["ens.5(plain+mlp)"].append(r)

print("\n=== MEAN ACROSS CUTOFFS ===")
for n in NAMES:
    rs = agg[n]
    m = {k: np.mean([r[k] for r in rs]) for k in ("rmse", "all", "tail", "mid", "yda", "lap")}
    print(f"{n:18}  RMSE={m['rmse']:.6f}  all={m['all']:+5.1f}  tail={m['tail']:+5.1f}  "
          f"mid={m['mid']:+5.1f}  yda={m['yda']:+5.1f}  lap={m['lap']:+5.1f}")
