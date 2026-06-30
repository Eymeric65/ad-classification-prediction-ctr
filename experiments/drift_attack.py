"""Attack the forward-in-time drift with distribution-shift methods (LightGBM only).

All variants are scored on the SAME 2022-11-01 forward holdout and compared to the
live LightGBM config (baseline). The baseline already includes date recency-decay
(half_life=120d) + tail boost + advertiser encoding.

Variants
--------
  baseline          live config as-is.
  no_recency        config tail boost but NO date recency-decay (isolates recency's value).
  feat_iw           feature-based covariate-shift importance weighting (leakage-safe):
                    within TRAIN, learn p(late | features) where "late" = newest
                    `late_days` of the training window; weight each train ad by the
                    density ratio p/(1-p). A learned generalization of date-decay that
                    keys on the FEATURE drift (campaign_goal, image comp), not just date.
  feat_iw+recency   feat_iw multiplied on top of the config date-decay.
  drop_goal         remove campaign_goal (the biggest covariate-shifter) from features.
  drop_goal+iw      both.

Run: prints a comparison table.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/workspace/experiments")
from harness import (load_everything, eval_split, temporal_indices, lgbm_factory)
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict

D = load_everything()
CUT = "2022-11-01"
tr, te = temporal_indices(D, CUT)
fac = lgbm_factory(D["pcfg"])


def feature_iw(idx_tr, late_days=90, clip=10.0):
    """Leakage-safe importance weights: resemble the newest training ads."""
    date = D["date"].iloc[idx_tr].reset_index(drop=True)
    cut_late = date.max() - pd.Timedelta(days=late_days)
    late = (date >= cut_late).to_numpy().astype(int)
    cat, num = D["cat"], D["num"]
    Xtr = D["X"].iloc[idx_tr]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ])
    Xt = pre.fit_transform(Xtr)
    clf = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbose=-1)
    p = cross_val_predict(clf, Xt, late, cv=3, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    w = p / (1 - p)                       # density ratio late/early
    w = w / np.median(w)
    w = np.clip(w, 1.0 / clip, clip)
    return w, late.mean()


def run(label, **kw):
    r = eval_split(D, tr, te, fac, **kw)
    print(f"{label:18}  RMSE={r['rmse']:.6f}  all={r['all']:+5.1f}  tail={r['tail']:+5.1f}  "
          f"mid={r['mid']:+5.1f}  yda={r['yda']:+5.1f}  lap={r['lap']:+5.1f}")
    return r

print(f"### forward holdout cutoff {CUT}  (train {len(tr):,} / test {len(te):,})")
print(f"{'variant':18}  {'RMSE':>8}  {'all':>5}  {'tail':>5}  {'mid':>5}  {'yda':>5}  {'lap':>5}")
print("-" * 78)

run("baseline")
run("no_recency", use_recency=False)

w_iw, late_frac = feature_iw(tr)
print(f"# feat_iw: late fraction={late_frac:.1%}, iw range [{w_iw.min():.2f},{w_iw.max():.2f}]")
run("feat_iw", sw_override=w_iw)            # importance weight only (no tail boost)
run("feat_iw+recency", sw_extra=w_iw)        # on top of config (tail+date-decay)

# drop campaign_goal
cat_nogoal = [c for c in D["cat"] if c != "campaign_goal"]
D2 = dict(D); D2["cat"] = cat_nogoal
def run2(label, Dx, **kw):
    r = eval_split(Dx, tr, te, fac, **kw)
    print(f"{label:18}  RMSE={r['rmse']:.6f}  all={r['all']:+5.1f}  tail={r['tail']:+5.1f}  "
          f"mid={r['mid']:+5.1f}  yda={r['yda']:+5.1f}  lap={r['lap']:+5.1f}")
    return r
run2("drop_goal", D2)
run2("drop_goal+iw", D2, sw_extra=w_iw)
