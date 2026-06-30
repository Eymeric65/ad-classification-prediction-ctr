"""Production submission writer — the overnight-winning recipe.

  cumu_ctr  =  decode( 0.6 * iw_LightGBM_scores  +  0.4 * MLP_scores )

where iw_LightGBM is the live LightGBM config with covariate-shift importance weighting
(methods.feature_iw) multiplied onto its training sample weights, and MLP is the torch
net (mlp.TorchMLP) trained with the live config weights. Both train on 100% of the
training ads; scores are blended in PCA space, decoded to curves, resampled to the
result file's delivery-days. Validated to beat the LightGBM baseline on forward-holdout
grader RMSE and on the yda/middle buckets (see OVERNIGHT_REPORT.md §5).

  PY=<night venv python>
  $PY experiments/predict_ensemble.py            # default 0.6 lgb / 0.4 mlp, with iw
  $PY experiments/predict_ensemble.py 0.5        # override blend weight

Writes data/7. result_group_X_filled_ensemble.csv (does NOT touch the live filled file).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/experiments")
os.chdir("/workspace")

from compress_curves import decode, load_codebook, load_settings
from dataset import load_ads, load_predict
from predict_curves import fill_cumu_ctr, RESULT_IN
from pipeline.run import (ADV_COL, DATASET_CONFIG, PIPELINE_CONFIG, build_estimator,
                          feature_columns, load_yaml, sample_weights, select_rows)
from pipeline.encoding import (DEFAULT_ALPHA, encode_loo, encode_new,
                               fit_advertiser_encoder)
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from mlp import TorchMLP

RESULT_OUT = "data/7. result_group_X_filled_ensemble.csv"
W_LGB = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6


def full_train_iw(X, dates, cat, num, late_days=90, clip=10.0):
    """feature_iw on the full training set: up-weight ads resembling the newest `late_days`."""
    late = (dates >= dates.max() - pd.Timedelta(days=late_days)).to_numpy().astype(int)
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ])
    Xt = pre.fit_transform(X)
    clf = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbose=-1)
    p = np.clip(cross_val_predict(clf, Xt, late, cv=3, method="predict_proba")[:, 1], 1e-3, 1 - 1e-3)
    w = p / (1 - p); w = w / np.median(w)
    return np.clip(w, 1.0 / clip, clip)


def main():
    ccfg = load_settings(); dcfg = load_yaml(DATASET_CONFIG); pcfg = load_yaml(PIPELINE_CONFIG)
    cb = load_codebook(ccfg); id_col = ccfg["columns"]["id"]
    pc_cols = [f"pc{i}" for i in range(ccfg["n_components"])]

    ads = select_rows(load_ads(), dcfg)
    scores = pd.read_parquet(ccfg["paths"]["scores"])
    df = ads.merge(scores, on=id_col, how="inner")
    cat, num0 = feature_columns(dcfg)
    Xtr = df[cat + num0].copy()
    Ytr = df[pc_cols].to_numpy(float)
    dates = pd.to_datetime(df["min_date"])

    # advertiser encoding (leakage-safe: LOO for train, full-mean for predict)
    enc, num = None, list(num0)
    if dcfg["features"].get("advertiser"):
        alpha = (dcfg.get("advertiser_encoding") or {}).get("alpha", DEFAULT_ALPHA)
        adv = df["advertiser_id"].astype(str).to_numpy()
        level = decode(Ytr, ccfg, cb).mean(axis=1)
        enc = fit_advertiser_encoder(adv, level, alpha)
        Xtr[ADV_COL] = encode_loo(enc, adv, level)
        num = num0 + [ADV_COL]

    sw_cfg = sample_weights(Ytr, dcfg, ccfg, cb, dates=dates)        # tail + recency
    sw_cfg = np.ones(len(Xtr)) if sw_cfg is None else sw_cfg
    iw = full_train_iw(df[cat + num0], dates, cat, num0)            # covariate-shift weights
    sw_lgb = sw_cfg * iw; sw_lgb *= len(sw_lgb) / sw_lgb.sum()

    print(f"  training iw-LightGBM on {len(Xtr):,} ads ({len(cat)}cat+{len(num)}num) ...")
    lgb = build_estimator(pcfg, cat, num)
    lgb.fit(Xtr, Ytr, reg__sample_weight=sw_lgb)
    print(f"  training MLP on the same matrix ...")
    mlp = TorchMLP(cat, num, hidden=(256, 128), dropout=0.2, epochs=80, seed=0)
    mlp.fit(Xtr, Ytr, sample_weight=sw_cfg)

    pred = load_predict().copy()
    if enc is not None:
        pred[ADV_COL] = encode_new(enc, pred["advertiser_id"].astype(str).to_numpy())
    Yhat = W_LGB * lgb.predict(pred[cat + num]) + (1 - W_LGB) * mlp.predict(pred[cat + num])

    curves = decode(Yhat, ccfg, cb)
    grid = np.asarray(cb["grid"], float)
    mean_curve = decode(np.zeros((1, len(pc_cols))), ccfg, cb)[0]
    ad_pos = {str(a): i for i, a in enumerate(pred[id_col].astype(str))}
    print(f"  predicted {len(pred):,} ads  (blend {W_LGB:.2f} iw-lgb / {1-W_LGB:.2f} mlp)")

    result = pd.read_csv(RESULT_IN, index_col=0, low_memory=False)
    result["cumu_ctr"] = fill_cumu_ctr(result, curves, ad_pos, grid, mean_curve, id_col)
    result.to_csv(RESULT_OUT)
    print(f"  wrote {RESULT_OUT}: {len(result):,} rows, cumu_ctr mean {result['cumu_ctr'].mean():.4g}")


if __name__ == "__main__":
    main()
