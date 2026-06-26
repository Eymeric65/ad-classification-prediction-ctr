"""Predict an ad's PCA curve-scores from its static header features.

This is the modeling step that sits AFTER curve compression: compress_curves.py
turned each ad's cumulative-CTR curve into N PCA scores; here we learn to predict
those scores from the creative's header (categorical attrs + annotation features),
so that at predict time we can `decode` them back into a full curve.

Three configs, each a single responsibility:
  curve_config.yaml    PCA/curve settings + path to the fitted codebook (shared toolchain)
  dataset_config.yaml  which ads to use, which feature groups, per-component RMSE weights
  pipeline_config.yaml  which model to fit and its hyperparameters

Run from the repo root:
  uv run python -m pipeline.run        (or: uv run python pipeline/run.py)
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

# Make the repo-root modules importable whether run as `-m pipeline.run` or as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from compress_curves import decode, load_codebook, load_settings
from dataset import ANN_FLAG_COLS, AREA_COLS, CAT_COLS, DATE_COLS, load_ads

DATASET_CONFIG = "dataset_config.yaml"
PIPELINE_CONFIG = "pipeline_config.yaml"


# ---- config -----------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---- data assembly ----------------------------------------------------------

def feature_columns(dcfg):
    """(categorical cols, numeric cols) selected by dataset_config.features."""
    groups = dcfg["features"]
    cat = list(CAT_COLS) if groups.get("categorical") else []
    num = []
    if groups.get("area"):
        num += AREA_COLS
    if groups.get("ann_flags"):
        num += ANN_FLAG_COLS
    if groups.get("date"):
        num += DATE_COLS
    return cat, num


def select_rows(ads, dcfg):
    """Filter ads by annotation schema per dataset_config.rows."""
    schema = dcfg["rows"].get("ann_schema", "any")
    if schema != "any":
        if "ann_schema" not in ads.columns:
            raise SystemExit("ads.parquet has no 'ann_schema' column; set rows.ann_schema: any")
        ads = ads[ads["ann_schema"] == schema]
    if ads.empty:
        raise SystemExit(f"no ads left after ann_schema={schema!r} filter")
    return ads


def assemble(ccfg, dcfg):
    """Join features to PCA-score targets. Returns (X df, Y array, cat, num, pc_cols)."""
    ads = select_rows(load_ads(), dcfg)
    scores = pd.read_parquet(ccfg["paths"]["scores"])
    id_col = ccfg["columns"]["id"]
    pc_cols = [f"pc{i}" for i in range(ccfg["n_components"])]

    df = ads.merge(scores, on=id_col, how="inner")
    if df.empty:
        raise SystemExit("no overlap between ads and scores — did you run `compress_curves.py fit`?")

    cat, num = feature_columns(dcfg)
    return df[cat + num], df[pc_cols].to_numpy(float), cat, num, pc_cols


# ---- model ------------------------------------------------------------------

def build_estimator(pcfg, cat, num):
    """A preprocessing + regressor Pipeline. Ridge predicts all PCs jointly (multi-output)."""
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
    ])
    name = pcfg["model"]
    if name == "ridge":
        reg = Ridge(**pcfg.get("ridge", {}))           # native multi-output
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor
        reg = MultiOutputRegressor(LGBMRegressor(**pcfg.get("lightgbm", {})))  # one tree-model per PC
    else:
        raise SystemExit(f"unknown model {name!r}; expected 'ridge' or 'lightgbm'")
    return Pipeline([("pre", pre), ("reg", reg)])


# ---- metrics ----------------------------------------------------------------

def sample_weights(Y, dcfg, ccfg, cb):
    """Per-AD training weight, to make the loss attend to the high-CTR tail.

    Weight is a function of each ad's actual curve level (mean cumulative CTR,
    decoded from its PCA scores). This is the clean alternative to oversampling
    outliers: same effect on the loss, no row duplication, continuously tunable.
    Computed from training targets only — never touches the test set.
      none      -> uniform (1.0 everywhere)
      power     -> w = (level / median_level) ** gamma   (gamma>0 favors high CTR)
      threshold -> ads with level >= `threshold` get `boost`, the rest get 1.0
    Returns None for `none` (so we don't pass a weight to .fit at all).
    """
    sw = dcfg.get("sample_weighting") or {}
    scheme = sw.get("scheme", "none")
    if scheme == "none":
        return None
    level = decode(Y, ccfg, cb).mean(axis=1)          # per-ad actual mean cumulative CTR
    if scheme == "power":
        gamma = float(sw.get("power", 1.0))
        w = (level / np.median(level)) ** gamma
    elif scheme == "threshold":
        w = np.where(level >= float(sw.get("threshold", 0.01)),
                     float(sw.get("boost", 5.0)), 1.0)
    elif scheme == "balanced":
        # "Cluster the curves by level, weight each cluster equally" = inverse-density.
        # Bins are EQUAL-WIDTH in CTR space (not quantiles — quantile bins have equal
        # counts, so inverse-frequency would be a no-op). The dense low-CTR band gets
        # many ads / low weight; the sparse tail bins get few ads / high weight.
        bins = int(sw.get("bins", 10))
        edges = np.linspace(level.min(), level.max(), bins + 1)
        idx = np.clip(np.digitize(level, edges[1:-1]), 0, bins - 1)
        counts = np.bincount(idx, minlength=bins)
        w = 1.0 / counts[idx]                          # rarer bin -> higher weight
    else:
        raise SystemExit(f"unknown sample_weighting scheme {scheme!r}")
    w = w * (len(w) / w.sum())                         # normalize to mean 1 FIRST...
    clip = sw.get("clip")
    if clip:                                           # ...so the clip is a ratio cap around 1
        w = np.clip(w, 1.0 / float(clip), float(clip))
        w = w * (len(w) / w.sum())                     # renormalize after clipping
    return w


def component_weights(dcfg, cb, n):
    """Per-PC weights for the aggregate score-space RMSE."""
    w = dcfg["weighting"]
    scheme = w.get("scheme", "equal")
    if scheme == "equal":
        return np.ones(n)
    if scheme == "explained_variance":
        return np.asarray(cb["explained_variance_ratio"][:n], float)
    if scheme == "manual":
        return np.asarray(w["manual"][:n], float)
    raise SystemExit(f"unknown weighting scheme {scheme!r}")


def _rmse(a, b, mask=None):
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    return np.sqrt(d.mean())


def report(y_true, y_pred, weights, pc_cols, ccfg, cb, tail_frac=0.10):
    """Print per-component, weighted score-space, and real curve-space errors.

    Also breaks curve-space error into the high-CTR TAIL (top `tail_frac` of ads by
    actual curve level) vs the rest — the global RMSE is dominated by the boring
    middle, so the tail number is the one that tells us if we predict the ads anyone
    actually cares about.
    """
    per_pc = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))          # RMSE per component
    weighted = np.sqrt(np.average(per_pc ** 2, weights=weights))

    # Real metric: decode predicted vs true scores back to CTR curves and compare.
    # Baseline = predicting the mean curve (all scores = 0) — the floor any model must beat.
    curves_true = decode(y_true, ccfg, cb)
    curves_pred = decode(y_pred, ccfg, cb)
    curves_mean = decode(np.zeros_like(y_true), ccfg, cb)
    curve_rmse = _rmse(curves_true, curves_pred)
    base_rmse = _rmse(curves_true, curves_mean)

    # Rank ads by actual curve level (mean cumulative CTR); top slice = the tail.
    level = curves_true.mean(axis=1)
    thresh = np.quantile(level, 1 - tail_frac)
    tail = level >= thresh
    mid = ~tail

    print("\nper-component RMSE (score space):")
    for name, e, wt in zip(pc_cols, per_pc, weights):
        print(f"  {name:>4}  rmse={e:.4f}   weight={wt:.4f}")
    print(f"\nweighted score-space RMSE : {weighted:.4f}")

    def block(title, m):
        mdl = _rmse(curves_true, curves_pred, m)
        base = _rmse(curves_true, curves_mean, m)
        skill = (1 - mdl / base) * 100 if base else 0.0
        n = int(m.sum()) if m is not None else len(level)
        print(f"  {title:18} n={n:>6}  model={mdl:.6f}  base={base:.6f}  skill={skill:+6.1f}%")

    print("\ncurve-space RMSE (decoded CTR):")
    block("all", None)
    block(f"tail (top {tail_frac:.0%})", tail)
    block("middle (rest)", mid)
    print(f"  tail threshold = mean CTR ≥ {thresh:.4%}")


# ---- main -------------------------------------------------------------------

def run(ccfg=None, dcfg=None, pcfg=None):
    """Train on the configured split and predict the held-out test set.

    Returns a dict with the fitted estimator, the test features/targets/predictions,
    and the configs + codebook — so callers (e.g. the visualizer) reuse one fit.
    """
    ccfg = ccfg or load_settings()                 # curve_config.yaml
    dcfg = dcfg or load_yaml(DATASET_CONFIG)
    pcfg = pcfg or load_yaml(PIPELINE_CONFIG)
    cb = load_codebook(ccfg)

    X, Y, cat, num, pc_cols = assemble(ccfg, dcfg)
    Xtr, Xte, Ytr, Yte = train_test_split(
        X, Y, test_size=dcfg["test_size"], random_state=dcfg["random_state"])

    est = build_estimator(pcfg, cat, num)
    sw = sample_weights(Ytr, dcfg, ccfg, cb)           # None -> uniform
    fit_params = {} if sw is None else {"reg__sample_weight": sw}
    est.fit(Xtr, Ytr, **fit_params)
    Ypred = est.predict(Xte)

    return dict(est=est, Xte=Xte, Yte=Yte, Ypred=Ypred, n_rows=len(X),
                cat=cat, num=num, pc_cols=pc_cols, ccfg=ccfg, dcfg=dcfg, pcfg=pcfg, cb=cb)


def main():
    r = run()
    sw = (r["dcfg"].get("sample_weighting") or {}).get("scheme", "none")
    print(f"model={r['pcfg']['model']}  rows={r['n_rows']:,}  "
          f"features={len(r['cat'])} cat + {len(r['num'])} num  targets={len(r['pc_cols'])} PCs  "
          f"sample_weighting={sw}")
    report(r["Yte"], r["Ypred"], component_weights(r["dcfg"], r["cb"], len(r["pc_cols"])),
           r["pc_cols"], r["ccfg"], r["cb"])


if __name__ == "__main__":
    main()
