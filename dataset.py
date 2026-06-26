"""Clean access to the dataset. Import these instead of ever reading the CSV again.

    from dataset import load_ads, load_laps, load_train, load_predict

Grain:
    ads   - one row per `ad_id` (static creative attrs + annotation features)
    laps  - one row per (ad_id, delivery_days): cumulative metrics over time
"""
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


def load_ads():
    """Static creative table, indexed by ad_id (training set)."""
    return pd.read_parquet(f"{DATA}/ads.parquet")


def load_predict():
    """Prediction set — same columns as load_ads(), but no laps exist for these."""
    return pd.read_parquet(f"{DATA}/ads_predict.parquet")


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
