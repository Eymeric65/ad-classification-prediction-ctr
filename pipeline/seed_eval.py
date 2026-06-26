"""Multi-seed held-out validation: how stable is the model's skill across splits?

Refits the score-predictor (same config as pipeline/run.py) on several random
train/test splits and reports curve-space skill per seed, then mean +/- std. Skill
is broken out OVERALL, by tail/middle, and WITHIN each platform (pooled skill is
inflated by between-platform level separation -- Simpson's paradox -- so the
within-platform numbers are the honest estimate of prediction-set performance).

  uv run python seed_eval.py [n_seeds]   (default 5)
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = "/Users/tokoemy/Documents/petit-projet-philippe"
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from sklearn.model_selection import train_test_split

from compress_curves import decode, load_codebook, load_settings
from pipeline.run import (assemble, build_estimator, component_weights,
                          load_yaml, sample_weights, _rmse, DATASET_CONFIG, PIPELINE_CONFIG)

ccfg = load_settings()
dcfg = load_yaml(DATASET_CONFIG)
pcfg = load_yaml(PIPELINE_CONFIG)
cb = load_codebook(ccfg)
cw = component_weights(dcfg, cb, ccfg["n_components"])

X, Y, cat, num, pc_cols = assemble(ccfg, dcfg)
# Platform indicator from the __NA__ sentinel: lap ads carry __NA__ for `device`.
platform = np.where(X["device"].to_numpy() == "__NA__", "lap", "yda")
print(f"assembled {len(X):,} ads  ({(platform=='yda').sum():,} yda / "
      f"{(platform=='lap').sum():,} lap)  feat={len(cat)}cat+{len(num)}num\n")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEEDS = list(range(42, 42 + N_SEEDS))


def skill(true, pred, baseline, mask=None):
    base = _rmse(true, baseline, mask)
    mdl = _rmse(true, pred, mask)
    return (1 - mdl / base) * 100 if base else 0.0


def eval_seed(seed):
    Xtr, Xte, Ytr, Yte, ptr, pte = train_test_split(
        X, Y, platform, test_size=dcfg["test_size"], random_state=seed)
    w = sample_weights(Ytr, dcfg, ccfg, cb)
    est = build_estimator(pcfg, cat, num)
    est.fit(Xtr, Ytr, **({} if w is None else {"reg__sample_weight": w}))
    Yp = est.predict(Xte)

    ct, cp = decode(Yte, ccfg, cb), decode(Yp, ccfg, cb)
    cm = decode(np.zeros_like(Yte), ccfg, cb)          # mean-curve baseline
    level = ct.mean(axis=1)
    tail = level >= np.quantile(level, 0.90)
    per_pc = np.sqrt(((Yte - Yp) ** 2).mean(axis=0))
    return dict(
        wRMSE=float(np.sqrt(np.average(per_pc ** 2, weights=cw))),
        all=skill(ct, cp, cm),
        tail=skill(ct, cp, cm, tail),
        mid=skill(ct, cp, cm, ~tail),
        yda=skill(ct, cp, cm, pte == "yda"),
        lap=skill(ct, cp, cm, pte == "lap"),
    )


cols = ["wRMSE", "all", "tail", "mid", "yda", "lap"]
print(f"{'seed':>5} " + " ".join(f"{c:>8}" for c in cols))
rows = []
for s in SEEDS:
    r = eval_seed(s)
    rows.append(r)
    print(f"{s:>5} " + " ".join(f"{r[c]:>8.3f}" for c in cols))

print("-" * (6 + 9 * len(cols)))
arr = {c: np.array([r[c] for r in rows]) for c in cols}
print("mean  " + " ".join(f"{arr[c].mean():>8.3f}" for c in cols))
print("std   " + " ".join(f"{arr[c].std():>8.3f}" for c in cols))
print("\n(skill % = 1 - model_RMSE/mean-curve_RMSE in decoded CTR space; higher=better.")
print(" yda/lap are WITHIN-platform skill -- the honest predictor of prediction-set performance.)")
