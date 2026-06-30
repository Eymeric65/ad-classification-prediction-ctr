"""Validate candidate methods across MULTIPLE temporal cutoffs (not just one).

Single-cutoff deltas of ~0.2pp are within noise. This refits the winners at the three
cutoffs VALIDATION.md uses and reports each + the mean, so a real improvement shows as a
consistent sign across cutoffs. Pass a comma-list of methods to keep runtime down.

  python multi_cutoff.py [methods] [cutoffs]
  methods : baseline,feat_iw,mlp,ensemble  (default all)
  cutoffs : 2022-08-01,2022-11-01,2023-01-01 (default)
"""
import sys
import numpy as np
sys.path.insert(0, "/workspace/experiments")
from harness import load_everything, eval_split, temporal_indices, lgbm_factory, _metrics
from methods import feature_iw

METHODS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["baseline", "feat_iw", "mlp", "ensemble"]
CUTOFFS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["2022-08-01", "2022-11-01", "2023-01-01"]
N_MLP_SEEDS = 3

need_mlp = any(m in ("mlp", "ensemble") for m in METHODS)
if need_mlp:
    from mlp import mlp_factory

D = load_everything()
lgb = lgbm_factory(D["pcfg"])

def line(label, r):
    return (f"{label:18}  RMSE={r['rmse']:.6f}  all={r['all']:+5.1f}  tail={r['tail']:+5.1f}  "
            f"mid={r['mid']:+5.1f}  yda={r['yda']:+5.1f}  lap={r['lap']:+5.1f}")

agg = {m: [] for m in METHODS}
for cut in CUTOFFS:
    tr, te = temporal_indices(D, cut)
    Yte = D["Y"][te]
    print(f"\n### cutoff {cut}  (train {len(tr):,} / test {len(te):,})")

    r_lgb = eval_split(D, tr, te, lgb)
    if "baseline" in METHODS:
        print(line("baseline(lgbm)", r_lgb)); agg["baseline"].append(r_lgb)

    if "feat_iw" in METHODS:
        w = feature_iw(D, tr)
        r = eval_split(D, tr, te, lgb, sw_extra=w)
        print(line("feat_iw+recency", r)); agg["feat_iw"].append(r)

    if need_mlp:
        preds = [eval_split(D, tr, te, mlp_factory(epochs=80, seed=s))["Yp"]
                 for s in range(N_MLP_SEEDS)]
        Yp_mlp = np.mean(preds, axis=0)
        if "mlp" in METHODS:
            r = _metrics(D, te, Yte, Yp_mlp)
            print(line("mlp(3seed)", r)); agg["mlp"].append(r)
        if "ensemble" in METHODS:
            r = _metrics(D, te, Yte, 0.5 * r_lgb["Yp"] + 0.5 * Yp_mlp)
            print(line("ens .5lgb/.5mlp", r)); agg["ensemble"].append(r)

print("\n=== MEAN ACROSS CUTOFFS ===")
for m in METHODS:
    rs = agg[m]
    if not rs:
        continue
    mean = {k: np.mean([r[k] for r in rs]) for k in ("rmse", "all", "tail", "mid", "yda", "lap")}
    print(f"{m:18}  RMSE={mean['rmse']:.6f}  all={mean['all']:+5.1f}  tail={mean['tail']:+5.1f}  "
          f"mid={mean['mid']:+5.1f}  yda={mean['yda']:+5.1f}  lap={mean['lap']:+5.1f}")
