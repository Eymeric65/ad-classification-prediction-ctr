"""Plain held-out RMSE on raw cumu_ctr -- exactly the judges' metric.

The grader compares predicted cumu_ctr vs ACTUAL cumu_ctr per (ad, delivery_day)
row and takes sqrt(mean(sq error)). This reproduces that on held-out TRAINING ads
(the only ads with known actuals): train on the rest, predict each held-out ad's
curve via the model, interpolate it to that ad's real delivery_days, and RMSE
against its raw laps. Includes BOTH model error and PCA reconstruction error --
just like the grader will see. Reported per seed + mean/std, with the mean-curve
baseline for context.

  uv run python rmse_eval.py [n_seeds]   (default 5)
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
from pipeline.run import (add_advertiser_feature, build_estimator, feature_columns,
                          load_yaml, sample_weights, DATASET_CONFIG, PIPELINE_CONFIG)

ccfg = load_settings()
dcfg = load_yaml(DATASET_CONFIG)
pcfg = load_yaml(PIPELINE_CONFIG)
cb = load_codebook(ccfg)
id_col = ccfg["columns"]["id"]
pc_cols = [f"pc{i}" for i in range(ccfg["n_components"])]
grid = np.asarray(cb["grid"], float)

# --- assemble features + score targets + ad_id (keep the key, unlike pipeline.run) ---
ads = load_ads()
if dcfg["rows"].get("ann_schema", "any") != "any":
    ads = ads[ads["ann_schema"] == dcfg["rows"]["ann_schema"]]
scores = pd.read_parquet(ccfg["paths"]["scores"])
df = ads.merge(scores, on=id_col, how="inner").reset_index(drop=True)
cat, num = feature_columns(dcfg)
X = df[cat + num]
Y = df[pc_cols].to_numpy(float)
dates = pd.to_datetime(df["min_date"]).reset_index(drop=True)   # for recency weighting
adv = df["advertiser_id"].astype(str).reset_index(drop=True)    # for advertiser encoding
ids = df[id_col].astype(str).to_numpy()
platform = np.where(df["device"].to_numpy() == "__NA__", "lap", "yda")

# --- raw laps (the ACTUALS), with per-ad lifetime for normalized-time mapping ---
laps = pd.read_parquet(ccfg["paths"]["laps"])[[id_col, "delivery_days", "cumu_ctr"]].copy()
laps[id_col] = laps[id_col].astype(str)
span = laps.groupby(id_col)["delivery_days"].agg(["min", "max"])
laps = laps.join(span, on=id_col)
print(f"assembled {len(df):,} ads ({(platform=='yda').sum():,} yda / "
      f"{(platform=='lap').sum():,} lap), {len(laps):,} actual lap rows\n")


def interp_curves_at(curves, pos, day, dmin, dmax):
    """Linear-interpolate each row's curve (via pos -> curve row) at its normalized day."""
    sp = np.where(dmax > dmin, dmax - dmin, 1.0)
    t = (day - dmin) / sp
    idx = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    w = (t - grid[idx]) / (grid[idx + 1] - grid[idx])
    return curves[pos, idx] * (1 - w) + curves[pos, idx + 1] * w


def eval_seed(seed):
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=dcfg["test_size"], random_state=seed)
    w = sample_weights(Y[idx_tr], dcfg, ccfg, cb, dates=dates.iloc[idx_tr])
    Xtr, Xte, num_l = add_advertiser_feature(
        X.iloc[idx_tr], X.iloc[idx_te], Y[idx_tr],
        adv.iloc[idx_tr].to_numpy(), adv.iloc[idx_te].to_numpy(), num, dcfg, ccfg, cb)
    est = build_estimator(pcfg, cat, num_l)
    est.fit(Xtr, Y[idx_tr], **({} if w is None else {"reg__sample_weight": w}))

    te_ids = ids[idx_te]
    pos_of = {a: i for i, a in enumerate(te_ids)}
    curves = decode(est.predict(Xte), ccfg, cb)                      # (n_te, 50)
    mean_curve = decode(np.zeros((1, len(pc_cols))), ccfg, cb)        # (1, 50)

    L = laps[laps[id_col].isin(set(te_ids))]
    pos = L[id_col].map(pos_of).to_numpy()
    day = L["delivery_days"].to_numpy(float)
    dmin, dmax = L["min"].to_numpy(float), L["max"].to_numpy(float)
    actual = L["cumu_ctr"].to_numpy(float)
    plat = np.where(L[id_col].map(dict(zip(ids, platform))).to_numpy() == "yda", "yda", "lap")

    pred = interp_curves_at(curves, pos, day, dmin, dmax)
    base = interp_curves_at(np.repeat(mean_curve, len(te_ids), 0), pos, day, dmin, dmax)

    def rmse(m=slice(None)):
        return float(np.sqrt(np.mean((pred[m] - actual[m]) ** 2)))
    return dict(
        rmse=rmse(), yda=rmse(plat == "yda"), lap=rmse(plat == "lap"),
        baseline=float(np.sqrt(np.mean((base - actual) ** 2))),
        n_rows=len(L),
    )


N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
cols = ["rmse", "yda", "lap", "baseline"]
print(f"{'seed':>5} " + " ".join(f"{c:>10}" for c in cols) + f"{'rows':>10}")
rows = []
for s in range(42, 42 + N):
    r = eval_seed(s)
    rows.append(r)
    print(f"{s:>5} " + " ".join(f"{r[c]:>10.6f}" for c in cols) + f"{r['n_rows']:>10,}")

print("-" * (6 + 11 * len(cols) + 10))
arr = {c: np.array([r[c] for r in rows]) for c in cols}
print("mean  " + " ".join(f"{arr[c].mean():>10.6f}" for c in cols))
print("std   " + " ".join(f"{arr[c].std():>10.6f}" for c in cols))
print(f"\nmodel RMSE = {arr['rmse'].mean():.6f} +/- {arr['rmse'].std():.6f}  "
      f"(baseline mean-curve RMSE = {arr['baseline'].mean():.6f})")
print("This is plain RMSE on raw cumu_ctr, the grader's metric (model + PCA recon error).")
