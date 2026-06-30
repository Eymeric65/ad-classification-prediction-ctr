"""Diagnose WHERE the forward-in-time drift comes from.

Two analyses, both model-light (LightGBM only):

(A) Time-discriminator. Train a classifier to predict whether an ad is "recent"
    (min_date >= cutoff) vs "old". AUC >> 0.5 means the feature distribution itself
    shifts over time (covariate shift). Per-feature importance ranks WHICH features
    drift. Features that strongly predict time are the non-stationary ones whose
    learned feature->curve mapping is least trustworthy forward.

(B) Target drift. Compare each PCA score's distribution old vs recent (mean shift in
    units of pooled std), and the decoded curve-level. This separates covariate shift
    (X moves) from concept/label shift (Y | nothing, i.e. the base rates move).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/workspace/experiments")
from harness import load_everything
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

D = load_everything()
CUT = pd.Timestamp("2022-11-01")
recent = (D["date"] >= CUT).to_numpy().astype(int)
print(f"old={int((recent==0).sum()):,}  recent={int((recent==1).sum()):,}  "
      f"(cutoff {CUT.date()})\n")

cat, num = D["cat"], D["num"]
X = D["X"].copy()

# ---- (A) time-discriminator -------------------------------------------------
pre = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ("num", SimpleImputer(strategy="median"), num),
])
Xt = pre.fit_transform(X)
clf = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1)
proba = cross_val_predict(clf, Xt, recent, cv=3, method="predict_proba")[:, 1]
auc = roc_auc_score(recent, proba)
print(f"(A) TIME-DISCRIMINATOR AUC = {auc:.3f}")
print("    0.5 = no covariate shift; ->1.0 = the feature distribution moves over time.\n")

# per-RAW-feature drift: single-feature AUC (how well each one alone predicts recency)
print("    per-feature drift (single-feature time-AUC, top 20):")
rows = []
for c in num:
    v = pd.to_numeric(X[c], errors="coerce").to_numpy()
    m = ~np.isnan(v)
    if m.sum() < 100 or len(np.unique(v[m])) < 2:
        continue
    a = roc_auc_score(recent[m], v[m])
    rows.append((c, max(a, 1 - a)))
for c in cat:
    # categorical: AUC of P(recent|category) mapped per row
    rate = pd.Series(recent).groupby(X[c].to_numpy()).transform("mean").to_numpy()
    try:
        a = roc_auc_score(recent, rate)
        rows.append((c + " [cat]", max(a, 1 - a)))
    except ValueError:
        pass
for c, a in sorted(rows, key=lambda r: -r[1])[:20]:
    print(f"      {a:.3f}  {c}")

# ---- (B) target drift -------------------------------------------------------
print("\n(B) TARGET DRIFT (PCA scores + curve level, old vs recent):")
Y = D["Y"]
from compress_curves import decode
level = decode(Y, D["ccfg"], D["cb"]).mean(axis=1)
o, r = recent == 0, recent == 1
print(f"    curve level (mean CTR):  old={level[o].mean():.4%}  recent={level[r].mean():.4%}  "
      f"shift={ (level[r].mean()-level[o].mean()) / level.std():+.2f} sd")
for i in range(Y.shape[1]):
    s = Y[:, i]
    d = (s[r].mean() - s[o].mean()) / (s.std() + 1e-12)
    print(f"    pc{i}: old={s[o].mean():+.3f} recent={s[r].mean():+.3f}  shift={d:+.2f} sd")

# platform mix drift (a likely confounder)
print("\n    platform mix:  old yda%={:.1%}  recent yda%={:.1%}".format(
    (D["platform"][o] == "yda").mean(), (D["platform"][r] == "yda").mean()))
