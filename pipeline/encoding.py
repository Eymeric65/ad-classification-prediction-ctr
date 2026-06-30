"""Leakage-safe target encoding of advertiser_id by historical CTR level.

90.7% of prediction-set rows come from an advertiser that also appears in training, and an
advertiser's CTR level is a relatively stationary, much finer signal than the coarse industry
categorical the model otherwise keys on. Encoding it forward-in-time lifts grader-faithful
(day 1..30) RMSE ~1.8% and skill ~1.5pp, robustly across temporal cutoffs (see TUNING.md §advertiser).

Leakage discipline — the whole point:
  - TRAIN rows  -> leave-one-out smoothed mean (an ad never sees its own target).
  - TEST/PREDICT rows -> full-TRAIN smoothed mean (the future is encoded only from the past).
  - Unseen advertiser -> the global training mean (smoothing prior handles small advertisers too).
Smoothing: enc = (sum + alpha*global) / (n + alpha); larger alpha = trust the advertiser less.
The encoded target is the ad's mean cumulative-CTR level (decoded curve level), the quantity PC0
governs — encoding the raw level beat encoding the PC0 score or campaign_id in testing.
"""
import numpy as np
import pandas as pd

DEFAULT_ALPHA = 20.0


def fit_advertiser_encoder(adv_train, level_train, alpha=DEFAULT_ALPHA):
    """Build the encoder from training advertiser ids + their curve levels."""
    adv = pd.Series(np.asarray(adv_train).astype(str))
    lvl = pd.Series(np.asarray(level_train, float)).to_numpy()
    s = pd.Series(lvl).groupby(adv.to_numpy()).sum()
    c = pd.Series(lvl).groupby(adv.to_numpy()).count()
    return {"sum": s, "cnt": c, "global": float(lvl.mean()), "alpha": float(alpha)}


def encode_loo(enc, adv, level):
    """Leave-one-out encoding for the SAME rows the encoder was fit on (train)."""
    adv = np.asarray(adv).astype(str); level = np.asarray(level, float)
    s = enc["sum"].reindex(adv).to_numpy(); c = enc["cnt"].reindex(adv).to_numpy()
    a, g = enc["alpha"], enc["global"]
    return (s - level + a * g) / (c - 1 + a)


def encode_new(enc, adv):
    """Full-train-mean encoding for rows NOT in the fit set (test / prediction set)."""
    adv = np.asarray(adv).astype(str)
    s = np.nan_to_num(enc["sum"].reindex(adv).to_numpy())
    c = np.nan_to_num(enc["cnt"].reindex(adv).to_numpy())
    a, g = enc["alpha"], enc["global"]
    return (s + a * g) / (c + a)
