"""Forward-in-time (temporal) held-out validation -- the honest geometry for this task.

The real prediction set is a forward holdout: training ends 2023-04-30, the predict
set is 2023-05-01 -> 2023-11-07. The random-split numbers in VALIDATION.md/TUNING.md
let the model train on ads interleaved in time with the test ads, so they can't see
temporal drift. This harness splits the TRAINING ads on `min_date` at a cutoff, trains
only on ads BEFORE it, and evaluates on the slice AFTER -- mimicking the real setup.

For each cutoff it reports, on the future holdout:
  - grader RMSE on raw cumu_ctr (overall / yda / lap), like rmse_eval.py
  - curve-space skill (all / tail / mid / yda / lap), like seed_eval.py
plus a MATCHED RANDOM-SPLIT control (same test fraction, seed 42). The temporal-minus-
random gap is the cost of forward-in-time prediction -- the number the docs flag as
unmeasured. Random > temporal => drift is hurting; ~equal => the random numbers transfer.

  uv run python pipeline/temporal_eval.py [cutoff ...]   (default 2022-11-01)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = "/Users/tokoemy/Documents/petit-projet-philippe"
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from sklearn.model_selection import train_test_split

from compress_curves import decode, load_codebook, load_settings
from dataset import load_ads
from pipeline.run import (build_estimator, component_weights, feature_columns,
                          sample_weights, select_rows, _rmse, load_yaml,
                          DATASET_CONFIG, PIPELINE_CONFIG)

ccfg = load_settings()
dcfg = load_yaml(DATASET_CONFIG)
pcfg = load_yaml(PIPELINE_CONFIG)
cb = load_codebook(ccfg)
cw = component_weights(dcfg, cb, ccfg["n_components"])
id_col = ccfg["columns"]["id"]
pc_cols = [f"pc{i}" for i in range(ccfg["n_components"])]
grid = np.asarray(cb["grid"], float)

# --- assemble features + score targets, KEEPING ad_id + min_date (assemble() drops them) ---
ads = select_rows(load_ads(), dcfg)
scores = pd.read_parquet(ccfg["paths"]["scores"])
df = ads.merge(scores, on=id_col, how="inner").reset_index(drop=True)
cat, num = feature_columns(dcfg)
X = df[cat + num]
Y = df[pc_cols].to_numpy(float)
ids = df[id_col].astype(str).to_numpy()
date = pd.to_datetime(df["min_date"])
platform = np.where(df["device"].to_numpy() == "__NA__", "lap", "yda")

# --- raw laps (grader actuals) + per-ad lifetime for normalized-time mapping ---
laps = pd.read_parquet(ccfg["paths"]["laps"])[[id_col, "delivery_days", "cumu_ctr"]].copy()
laps[id_col] = laps[id_col].astype(str)
span = laps.groupby(id_col)["delivery_days"].agg(["min", "max"])
laps = laps.join(span, on=id_col)
plat_of = dict(zip(ids, platform))

print(f"assembled {len(df):,} ads  ({(platform=='yda').sum():,} yda / "
      f"{(platform=='lap').sum():,} lap)  feat={len(cat)}cat+{len(num)}num")
print(f"min_date spans {date.min().date()} -> {date.max().date()}\n")


def interp_curves_at(curves, pos, day, dmin, dmax):
    """Linear-interpolate each lap row's curve (pos -> curve) at its normalized day."""
    sp = np.where(dmax > dmin, dmax - dmin, 1.0)
    t = (day - dmin) / sp
    idx = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    w = (t - grid[idx]) / (grid[idx + 1] - grid[idx])
    return curves[pos, idx] * (1 - w) + curves[pos, idx + 1] * w


def eval_split(idx_tr, idx_te):
    """Fit on train idx, return grader-RMSE + skill metrics on test idx."""
    w = sample_weights(Y[idx_tr], dcfg, ccfg, cb, dates=date.iloc[idx_tr])  # train-only, no leakage
    est = build_estimator(pcfg, cat, num)
    est.fit(X.iloc[idx_tr], Y[idx_tr], **({} if w is None else {"reg__sample_weight": w}))

    Yte = Y[idx_te]
    Yp = est.predict(X.iloc[idx_te])

    # --- curve-space skill (decoded CTR) ---
    ct, cp = decode(Yte, ccfg, cb), decode(Yp, ccfg, cb)
    cm = decode(np.zeros_like(Yte), ccfg, cb)
    pte = platform[idx_te]
    level = ct.mean(axis=1)
    tail = level >= np.quantile(level, 0.90)

    def skill(mask=None):
        base = _rmse(ct, cm, mask)
        return (1 - _rmse(ct, cp, mask) / base) * 100 if base else 0.0

    # --- grader RMSE on raw cumu_ctr (interp predicted curve to each ad's real days) ---
    te_ids = ids[idx_te]
    pos_of = {a: i for i, a in enumerate(te_ids)}
    L = laps[laps[id_col].isin(set(te_ids))]
    pos = L[id_col].map(pos_of).to_numpy()
    day = L["delivery_days"].to_numpy(float)
    dmin, dmax = L["min"].to_numpy(float), L["max"].to_numpy(float)
    actual = L["cumu_ctr"].to_numpy(float)
    gplat = np.where(L[id_col].map(plat_of).to_numpy() == "yda", "yda", "lap")
    pred = interp_curves_at(cp, pos, day, dmin, dmax)
    base = interp_curves_at(np.repeat(cm.mean(axis=0, keepdims=True), len(te_ids), 0),
                            pos, day, dmin, dmax)

    def grmse(m=slice(None)):
        return float(np.sqrt(np.mean((pred[m] - actual[m]) ** 2)))

    return dict(
        n_te=len(idx_te), n_rows=len(L),
        rmse=grmse(), rmse_yda=grmse(gplat == "yda"), rmse_lap=grmse(gplat == "lap"),
        rmse_base=float(np.sqrt(np.mean((base - actual) ** 2))),
        all=skill(), tail=skill(tail), mid=skill(~tail),
        yda=skill(pte == "yda"), lap=skill(pte == "lap"),
    )


def show(label, r):
    print(f"== {label} ==  (test n={r['n_te']:,} ads, {r['n_rows']:,} lap rows)")
    print(f"   grader RMSE  all={r['rmse']:.6f}  yda={r['rmse_yda']:.6f}  "
          f"lap={r['rmse_lap']:.6f}  baseline={r['rmse_base']:.6f}  "
          f"(skill {(1 - r['rmse']/r['rmse_base'])*100:+.1f}%)")
    print(f"   curve skill  all={r['all']:+.1f}%  tail={r['tail']:+.1f}%  "
          f"mid={r['mid']:+.1f}%  yda={r['yda']:+.1f}%  lap={r['lap']:+.1f}%")


cutoffs = sys.argv[1:] or ["2022-11-01"]
for c in cutoffs:
    cut = pd.Timestamp(c)
    idx = np.arange(len(df))
    tr = idx[(date < cut).to_numpy()]
    te = idx[(date >= cut).to_numpy()]
    if len(tr) == 0 or len(te) == 0:
        print(f"cutoff {c}: empty side (train={len(tr)}, test={len(te)}) -- skipped\n")
        continue
    frac = len(te) / len(df)
    print(f"### cutoff {c}  ->  train {len(tr):,} ads (< {c}) / "
          f"test {len(te):,} ads (>= {c}, {frac:.0%} of data)")
    show("TEMPORAL  (train past, test future)", eval_split(tr, te))
    # matched random control: same test fraction, so the only difference is time-ordering
    rtr, rte = train_test_split(idx, test_size=frac, random_state=42)
    show("RANDOM    (matched test fraction, control)", eval_split(rtr, rte))
    print()

print("Random > temporal => forward-in-time drift is costing skill (the gap is the cost).")
print("~equal           => the random-split numbers in VALIDATION.md transfer to the real submission.")
