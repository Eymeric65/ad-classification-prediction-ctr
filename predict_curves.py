"""Fill `cumu_ctr` in the result-group file from the model's curve predictions.

This is the deployment counterpart of `pipeline/run.py`. Where `run.py` holds out a
test split to MEASURE the score-predictor, this script trains on 100% of the data
(no test split, to maximise throughput / coverage), then for every ad in the
prediction set:

  features --[model]--> 8 PCA scores --[decode]--> full cumulative-CTR curve

and resamples that curve onto the result file's delivery-days, writing `cumu_ctr`.

Time mapping: the codebook is fit with `time_axis: normalized_lifetime`, so a curve
lives on a grid t in [0,1]. The result file asks for cumu_ctr at concrete
`delivery_days`; each ad's lifetime is its own [min_day, max_day], so we map
  t = (day - min_day) / (max_day - min_day)
per ad and linearly interpolate the decoded 50-point grid at those t's.

Run from the repo root (after `build_dataset.py` has rebuilt ads_predict.parquet to
cover BOTH prediction platforms):
  uv run python predict_curves.py
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compress_curves import decode, load_codebook, load_settings
from dataset import load_predict
from pipeline.run import (
    ADV_COL, DATASET_CONFIG, PIPELINE_CONFIG, build_estimator, component_weights,
    feature_columns, load_yaml, sample_weights,
)
from pipeline.encoding import (DEFAULT_ALPHA, encode_loo, encode_new,
                               fit_advertiser_encoder)

RESULT_IN = "data/7. result_group_X.csv"
RESULT_OUT = "data/7. result_group_X_filled.csv"


# ---- training (no test split) ----------------------------------------------

def fit_full(ccfg, dcfg, pcfg, cb):
    """Train the score-predictor on EVERY training ad (no holdout)."""
    from dataset import load_ads

    ads = load_ads()
    schema = dcfg["rows"].get("ann_schema", "any")
    if schema != "any":
        ads = ads[ads["ann_schema"] == schema]
    scores = pd.read_parquet(ccfg["paths"]["scores"])
    id_col = ccfg["columns"]["id"]
    pc_cols = [f"pc{i}" for i in range(ccfg["n_components"])]

    df = ads.merge(scores, on=id_col, how="inner")
    if df.empty:
        raise SystemExit("no overlap between ads and scores — run `compress_curves.py fit` first")

    cat, num = feature_columns(dcfg)
    X, Y = df[cat + num].copy(), df[pc_cols].to_numpy(float)

    # Advertiser encoding: fit on ALL training ads, add the leave-one-out column to X. The
    # fitted encoder is returned so main() can encode the prediction set from full-train means.
    enc = None
    if dcfg["features"].get("advertiser"):
        alpha = (dcfg.get("advertiser_encoding") or {}).get("alpha", DEFAULT_ALPHA)
        adv = df["advertiser_id"].astype(str).to_numpy()
        level = decode(Y, ccfg, cb).mean(axis=1)
        enc = fit_advertiser_encoder(adv, level, alpha)
        X[ADV_COL] = encode_loo(enc, adv, level)
        num = num + [ADV_COL]

    est = build_estimator(pcfg, cat, num)
    # same loss weighting as run.py (None -> uniform). dates feed optional recency
    # weighting; here the anchor is the newest training ad (~2023-04-30), the closest
    # analogue to the forward predict set.
    sw = sample_weights(Y, dcfg, ccfg, cb, dates=pd.to_datetime(df["min_date"]))
    fit_params = {} if sw is None else {"reg__sample_weight": sw}
    est.fit(X, Y, **fit_params)
    print(f"  trained {pcfg['model']} on {len(X):,} ads "
          f"({len(cat)} cat + {len(num)} num features -> {len(pc_cols)} PCs)")
    return est, cat, num, pc_cols, enc


# ---- curve -> per-day cumu_ctr ---------------------------------------------

def fill_cumu_ctr(result, curves, ad_pos, grid, mean_curve, id_col="ad_id"):
    """Interpolate each result row's `cumu_ctr` off its ad's decoded curve.

    `ad_pos` maps an ad_id (str) to its row in `curves`; rows whose ad is absent from
    the prediction set fall back to the population mean curve (decode of all-zero scores).
    """
    days = result["delivery_days"].to_numpy(float)
    span = result.groupby(id_col)["delivery_days"].transform("max").to_numpy(float)
    span = np.where(span > 1, span - 1.0, 1.0)          # max_day - min_day (min is 1 here)
    t = (days - 1.0) / span                             # per-row position in [0,1]

    # Shared grid -> the two bracketing grid points + linear weight for every row at once.
    idx = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    w = (t - grid[idx]) / (grid[idx + 1] - grid[idx])

    pos = result[id_col].astype(str).map(ad_pos)
    have = pos.notna().to_numpy()
    pos_arr = pos.fillna(-1).astype(int).to_numpy()

    out = np.empty(len(result), float)
    # ads we have a prediction for: gather only the two needed curve columns per row
    p = pos_arr[have]
    out[have] = curves[p, idx[have]] * (1 - w[have]) + curves[p, idx[have] + 1] * w[have]
    # ads we don't: population mean curve
    if (~have).any():
        out[~have] = mean_curve[idx[~have]] * (1 - w[~have]) + mean_curve[idx[~have] + 1] * w[~have]

    n_missing_ads = result.loc[~have, id_col].nunique()
    if n_missing_ads:
        print(f"  WARNING: {n_missing_ads:,} ad_id(s) absent from the prediction set "
              f"-> filled with the mean curve. Rebuild ads_predict.parquet to cover them.")
    return np.clip(out, 0.0, None)                       # CTR is non-negative


# ---- main -------------------------------------------------------------------

def main():
    ccfg = load_settings()
    dcfg = load_yaml(DATASET_CONFIG)
    pcfg = load_yaml(PIPELINE_CONFIG)
    cb = load_codebook(ccfg)
    id_col = ccfg["columns"]["id"]

    est, cat, num, pc_cols, enc = fit_full(ccfg, dcfg, pcfg, cb)

    # Predict PCA scores for the prediction set, then decode to full curves.
    pred = load_predict().copy()
    if enc is not None:                                  # advertiser encoding from full-train means
        pred[ADV_COL] = encode_new(enc, pred["advertiser_id"].astype(str).to_numpy())
    Yhat = est.predict(pred[cat + num])
    curves = decode(Yhat, ccfg, cb)                     # (n_ads, grid_points), CTR space
    grid = np.asarray(cb["grid"], float)
    mean_curve = decode(np.zeros((1, len(pc_cols))), ccfg, cb)[0]
    ad_pos = {str(a): i for i, a in enumerate(pred[id_col].astype(str))}
    print(f"  predicted curves for {len(pred):,} ads")

    # Load the result template, overwrite cumu_ctr, write it back out.
    result = pd.read_csv(RESULT_IN, index_col=0, low_memory=False)
    result["cumu_ctr"] = fill_cumu_ctr(result, curves, ad_pos, grid, mean_curve, id_col)
    result.to_csv(RESULT_OUT)
    print(f"  wrote {RESULT_OUT}: {len(result):,} rows, "
          f"cumu_ctr in [{result['cumu_ctr'].min():.4g}, {result['cumu_ctr'].max():.4g}] "
          f"(mean {result['cumu_ctr'].mean():.4g})")


if __name__ == "__main__":
    main()
