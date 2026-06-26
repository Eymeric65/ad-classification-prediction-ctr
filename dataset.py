"""Clean access to the dataset. Import these instead of ever reading the CSV again.

    from dataset import load_ads, load_laps, load_train, load_predict

Grain:
    ads   - one row per `ad_id` (static creative attrs + annotation features)
    laps  - one row per (ad_id, delivery_days): cumulative metrics over time
"""
import numpy as np
import pandas as pd

DATA = "data"

# annotation feature columns produced by build_dataset.py
AREA_COLS = [
    "area_person_photo", "area_person_illust", "area_animal_photo", "area_animal_illust",
    "area_product_photo", "area_product_illust", "area_logo", "area_text",
    "area_other_photo", "area_other_illust",
]
ANN_FLAG_COLS = ["ann_has_product", "ann_has_character", "ann_has_button",
                 "n_face_tags", "n_text_tags"]
CAT_COLS = ["digital_large", "digital_small", "campaign_goal",
            "creative_format", "creative_call_to_action_type", "creative_size"]

# Release-date seasonality features, derived from `min_date` (see add_date_features).
DATE_COLS = ["month_sin", "month_cos", "doy_sin", "doy_cos"]


def _cyc(values, period):
    """Map a periodic integer (month, day-of-year) to a (sin, cos) pair on the unit circle,
    so the encoding is continuous across the wrap-around (December is adjacent to January)."""
    rad = 2.0 * np.pi * np.asarray(values, float) / period
    return np.sin(rad), np.cos(rad)


def add_date_features(df, date_col="min_date"):
    """Add cyclical calendar features from the ad's release date (`min_date`).

    Encodes SEASON, not absolute time: month and day-of-year are sin/cos encoded.
    Year is deliberately omitted — the prediction set is a forward holdout (all 2023),
    where a raw year carries no transferable signal, whereas cyclical season ("August
    behaves like August") carries across years. `min_date` exists in both the train and
    predict tables, so these are leakage-free and available at predict time.
    """
    out = df.copy()
    if date_col in out.columns:
        d = pd.to_datetime(out[date_col])
        out["month_sin"], out["month_cos"] = _cyc(d.dt.month.to_numpy(), 12)
        out["doy_sin"], out["doy_cos"] = _cyc(d.dt.dayofyear.to_numpy(), 365.25)
    return out


def load_ads():
    """Static creative table, indexed by ad_id (training set)."""
    return add_date_features(pd.read_parquet(f"{DATA}/ads.parquet"))


def load_predict():
    """Prediction set — same columns as load_ads(), but no laps exist for these."""
    return add_date_features(pd.read_parquet(f"{DATA}/ads_predict.parquet"))


def load_laps():
    """Time series: (ad_id, delivery_days) -> cumulative metrics."""
    return pd.read_parquet(f"{DATA}/laps.parquet")


def load_train(at_day=None):
    """Join ads to their laps for modeling.

    at_day=None  -> every (ad, day) lap row joined to its static features.
    at_day=N     -> one row per ad: its cumulative state on exactly delivery_day N
                    (ads that never reached day N are dropped).
    """
    ads, laps = load_ads(), load_laps()
    if at_day is not None:
        laps = laps[laps["delivery_days"] == at_day]
    return laps.merge(ads, on="ad_id", how="inner")
