"""Reusable drift-mitigation methods (importable, no side effects)."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict


def feature_iw(D, idx_tr, late_days=90, clip=10.0):
    """Leakage-safe covariate-shift importance weights.

    Within the training window only, learn p(late | features) where "late" = the newest
    `late_days` of training ads; weight each train ad by the density ratio p/(1-p),
    median-normalized and clipped. Up-weights training ads that look like the recent edge
    of the distribution (the closest analogue to the forward predict set) using the FEATURE
    distribution, generalizing the hand-tuned date recency-decay.
    """
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
    w = p / (1 - p)
    w = w / np.median(w)
    return np.clip(w, 1.0 / clip, clip)
