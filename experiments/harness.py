"""Model-agnostic temporal-eval harness.

Re-implements pipeline/temporal_eval.py's geometry (the repo copy has a hardcoded
Mac path and can't run here), but factors the *model* out behind a `model_factory`
so we can drop in MLP / sequence / ensemble models and score them on the EXACT same
forward-in-time holdout the LightGBM baseline is judged on.

A `model_factory(cat, num)` returns an object with:
    fit(X_df, Y_array, sample_weight=None)
    predict(X_df) -> (n, n_components) PCA-score array
That's the only contract; everything around it (advertiser encoding, sample
weighting, decode, grader RMSE, skill) is shared so comparisons are apples-to-apples.

Usage:
    from harness import load_everything, eval_split, temporal_indices, lgbm_factory
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = "/workspace"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sklearn.model_selection import train_test_split

from compress_curves import decode, load_codebook, load_settings
from dataset import load_ads
from pipeline.run import (add_advertiser_feature, build_estimator,
                          component_weights, feature_columns, sample_weights,
                          select_rows, _rmse, load_yaml, DATASET_CONFIG,
                          PIPELINE_CONFIG)

# ---- shared config + data (loaded once) ------------------------------------

def load_everything():
    ccfg = load_settings()
    dcfg = load_yaml(DATASET_CONFIG)
    pcfg = load_yaml(PIPELINE_CONFIG)
    cb = load_codebook(ccfg)
    id_col = ccfg["columns"]["id"]
    pc_cols = [f"pc{i}" for i in range(ccfg["n_components"])]
    grid = np.asarray(cb["grid"], float)

    ads = select_rows(load_ads(), dcfg)
    scores = pd.read_parquet(ccfg["paths"]["scores"])
    df = ads.merge(scores, on=id_col, how="inner").reset_index(drop=True)
    cat, num = feature_columns(dcfg)
    X = df[cat + num]
    Y = df[pc_cols].to_numpy(float)
    ids = df[id_col].astype(str).to_numpy()
    date = pd.to_datetime(df["min_date"])
    adv = df["advertiser_id"].astype(str)
    platform = np.where(df["device"].to_numpy() == "__NA__", "lap", "yda")

    laps = pd.read_parquet(ccfg["paths"]["laps"])[[id_col, "delivery_days", "cumu_ctr"]].copy()
    laps[id_col] = laps[id_col].astype(str)
    span = laps.groupby(id_col)["delivery_days"].agg(["min", "max"])
    laps = laps.join(span, on=id_col)

    return dict(ccfg=ccfg, dcfg=dcfg, pcfg=pcfg, cb=cb, id_col=id_col,
                pc_cols=pc_cols, grid=grid, df=df, X=X, Y=Y, ids=ids, date=date,
                adv=adv, platform=platform, laps=laps, cat=cat, num=num,
                cw=component_weights(dcfg, cb, ccfg["n_components"]))


# ---- model factories --------------------------------------------------------

class SklearnWrap:
    """Wrap the repo's sklearn Pipeline (preprocessing + LGBM/Ridge) to the contract."""
    def __init__(self, pcfg, cat, num):
        self.est = build_estimator(pcfg, cat, num)

    def fit(self, X, Y, sample_weight=None):
        fp = {} if sample_weight is None else {"reg__sample_weight": sample_weight}
        self.est.fit(X, Y, **fp)
        return self

    def predict(self, X):
        return self.est.predict(X)


def lgbm_factory(pcfg):
    return lambda cat, num: SklearnWrap(pcfg, cat, num)


# ---- the eval (geometry identical to temporal_eval.py) ----------------------

def eval_split(D, idx_tr, idx_te, model_factory, use_recency=True, use_adv=True,
               sw_override=None, sw_extra=None):
    """Fit model on idx_tr, return grader-RMSE + skill dict on idx_te.

    sw_override : full per-train-row weight vector to use INSTEAD of the config weights.
    sw_extra    : per-train-row multiplier applied ON TOP of the config weights
                  (renormalized to mean 1). Use for covariate-shift importance weights.
    """
    X, Y, date, adv = D["X"], D["Y"], D["date"], D["adv"]
    dcfg, ccfg, cb = D["dcfg"], D["ccfg"], D["cb"]
    cat, num = D["cat"], D["num"]
    ids, platform, laps = D["ids"], D["platform"], D["laps"]
    id_col, grid = D["id_col"], D["grid"]

    if sw_override is not None:
        w = np.asarray(sw_override, float)
    else:
        w = sample_weights(Y[idx_tr], dcfg, ccfg, cb, dates=date.iloc[idx_tr]) if use_recency else None
    if sw_extra is not None:
        base = np.ones(len(idx_tr)) if w is None else w
        w = base * np.asarray(sw_extra, float)
    if w is not None:
        w = w * (len(w) / w.sum())
    if use_adv:
        Xtr, Xte, num_l = add_advertiser_feature(
            X.iloc[idx_tr], X.iloc[idx_te], Y[idx_tr],
            adv.iloc[idx_tr].to_numpy(), adv.iloc[idx_te].to_numpy(), num, dcfg, ccfg, cb)
    else:
        Xtr, Xte, num_l = X.iloc[idx_tr], X.iloc[idx_te], num

    model = model_factory(cat, num_l)
    model.fit(Xtr, Y[idx_tr], sample_weight=w)
    Yte = Y[idx_te]
    Yp = model.predict(Xte)
    return _metrics(D, idx_te, Yte, Yp)


def _metrics(D, idx_te, Yte, Yp):
    ccfg, cb, grid = D["ccfg"], D["cb"], D["grid"]
    ids, platform, laps, id_col = D["ids"], D["platform"], D["laps"], D["id_col"]

    ct, cp = decode(Yte, ccfg, cb), decode(Yp, ccfg, cb)
    cm = decode(np.zeros_like(Yte), ccfg, cb)
    pte = platform[idx_te]
    level = ct.mean(axis=1)
    tail = level >= np.quantile(level, 0.90)

    def skill(mask=None):
        base = _rmse(ct, cm, mask)
        return (1 - _rmse(ct, cp, mask) / base) * 100 if base else 0.0

    te_ids = ids[idx_te]
    pos_of = {a: i for i, a in enumerate(te_ids)}
    L = laps[laps[id_col].isin(set(te_ids))]
    pos = L[id_col].map(pos_of).to_numpy()
    day = L["delivery_days"].to_numpy(float)
    dmin, dmax = L["min"].to_numpy(float), L["max"].to_numpy(float)
    actual = L["cumu_ctr"].to_numpy(float)
    plat_of = dict(zip(ids, platform))
    gplat = np.where(L[id_col].map(plat_of).to_numpy() == "yda", "yda", "lap")

    sp = np.where(dmax > dmin, dmax - dmin, 1.0)
    t = (day - dmin) / sp
    gi = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    gw = (t - grid[gi]) / (grid[gi + 1] - grid[gi])
    pred = cp[pos, gi] * (1 - gw) + cp[pos, gi + 1] * gw
    cmean = cm.mean(axis=0)
    base = cmean[gi] * (1 - gw) + cmean[gi + 1] * gw

    def grmse(m=slice(None)):
        return float(np.sqrt(np.mean((pred[m] - actual[m]) ** 2)))

    return dict(
        n_te=len(idx_te), n_rows=len(L),
        rmse=grmse(), rmse_yda=grmse(gplat == "yda"), rmse_lap=grmse(gplat == "lap"),
        rmse_base=float(np.sqrt(np.mean((base - actual) ** 2))),
        all=skill(), tail=skill(tail), mid=skill(~tail),
        yda=skill(pte == "yda"), lap=skill(pte == "lap"),
        Yp=Yp,  # keep raw predictions for ensembling
    )


def temporal_indices(D, cutoff):
    date = D["date"]
    idx = np.arange(len(D["df"]))
    cut = pd.Timestamp(cutoff)
    tr = idx[(date < cut).to_numpy()]
    te = idx[(date >= cut).to_numpy()]
    return tr, te


def random_indices(D, frac, seed=42):
    idx = np.arange(len(D["df"]))
    return train_test_split(idx, test_size=frac, random_state=seed)


def show(label, r):
    print(f"== {label} ==  (test n={r['n_te']:,} ads, {r['n_rows']:,} lap rows)")
    print(f"   grader RMSE  all={r['rmse']:.6f}  yda={r['rmse_yda']:.6f}  "
          f"lap={r['rmse_lap']:.6f}  baseline={r['rmse_base']:.6f}  "
          f"(skill {(1 - r['rmse']/r['rmse_base'])*100:+.1f}%)")
    print(f"   curve skill  all={r['all']:+.1f}%  tail={r['tail']:+.1f}%  "
          f"mid={r['mid']:+.1f}%  yda={r['yda']:+.1f}%  lap={r['lap']:+.1f}%")
