"""Turn the messy source CSVs into clean, typed Parquet — run once.

Outputs (in data/):
  ads.parquet          one row per (filename, min_date): static creative attrs + annotation features
  laps.parquet         (filename, min_date, delivery_days) -> cumulative metrics  [the time series]
  ads_predict.parquet  same schema as ads.parquet, for the prediction set (no laps)

Usage:  uv run python build_dataset.py
"""
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = "data"
TRAIN_CSV = f"{DATA}/2_2. training_data_lap.csv"
PRED_CSV = f"{DATA}/3_2. prediction_data_lap.csv"
CHUNK = 200_000

# ---- annotation parsing -----------------------------------------------------

# banner_tagging_area is a fixed set of 10 image-region buckets -> clean numeric vector
AREA_NAMES = {
    8000: "area_person_photo",  8001: "area_person_illust",
    8002: "area_animal_photo",  8003: "area_animal_illust",
    8004: "area_product_photo", 8005: "area_product_illust",
    8006: "area_logo",          8007: "area_text",
    8008: "area_other_photo",   8009: "area_other_illust",
}


def _names(v):
    """Normalize an annotation value (string OR list of {id,name}) to a list of names."""
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [it.get("name", "") for it in v if isinstance(it, dict)]
    return []


def parse_annotation(s):
    """JSON annotation string -> flat feature dict. Robust to missing keys / light schema."""
    out = {name: 0.0 for name in AREA_NAMES.values()}
    out.update(
        ann_schema="light", ann_has_product=0, ann_has_character=0,
        ann_has_button=0, n_face_tags=0, n_text_tags=0,
    )
    if not isinstance(s, str):
        return out
    try:
        d = json.loads(s)
    except (ValueError, TypeError):
        return out

    if "banner_tagging_object" in d:           # the full 8-key schema
        out["ann_schema"] = "full"
    for x in d.get("banner_tagging_area", []) or []:
        nm = AREA_NAMES.get(x.get("id"))
        if nm:
            out[nm] = float(x.get("area", 0) or 0)

    obj = _names(d.get("banner_tagging_object"))
    out["ann_has_product"] = int(any("商品あり" in n for n in obj))
    per = _names(d.get("banner_tagging_person"))
    out["ann_has_character"] = int("キャラクターあり" in per)
    band = _names(d.get("banner_tagging_button_band"))
    out["ann_has_button"] = int(any(("ボタン有" in n) or ("帯あり" in n) for n in band))
    out["n_face_tags"] = len(d.get("face_genderage", []) or [])
    out["n_text_tags"] = len(d.get("creative_text_tagging_detect", []) or [])
    return out


# ---- column groups ----------------------------------------------------------

# ad_id is the true unique key for a run: one creative (filename) can run as several ad_ids
KEY = ["ad_id"]
ID_COLS = ["ad_id", "advertiser_id", "campaign_id", "adgroup_id"]
STATIC = ID_COLS + [
    "filename", "min_date",
    "digital_large", "digital_small", "campaign_goal",
    "creative_format", "creative_call_to_action_type", "creative_size",
]
LAP_METRICS = ["delivery_days", "cumu_cost", "cumu_impressions", "cumu_clicks", "cumu_ctr"]


def clean_static(df):
    df = df.copy()
    df["min_date"] = pd.to_datetime(df["min_date"], errors="coerce")
    for c in ID_COLS:                       # ids are opaque (advertiser_id is a bytes-repr) -> string
        df[c] = df[c].astype("string")
    for c in ["digital_large", "digital_small", "campaign_goal",
              "creative_format", "creative_call_to_action_type", "creative_size"]:
        df[c] = df[c].astype("string")
    return df


def ads_rows_from_chunk(chunk, store):
    """Accumulate one richest record per (filename, min_date) into `store`."""
    chunk = clean_static(chunk)
    feats = chunk["annotation"].map(parse_annotation).apply(pd.Series)
    rows = pd.concat([chunk[STATIC], feats], axis=1)
    # keep the row with the richest annotation per ad (full > light, then most area mass)
    rows["_rich"] = (rows["ann_schema"] == "full").astype(int)
    for rec in rows.to_dict("records"):
        k = rec["ad_id"]
        prev = store.get(k)
        if prev is None or rec["_rich"] > prev["_rich"]:
            store[k] = rec


def build_ads(csv, out_path, with_laps):
    print(f"building ads from {csv} ...")
    store = {}
    lap_writer = None
    usecols = STATIC + ["annotation"] + (LAP_METRICS if with_laps else [])
    n = 0
    for chunk in pd.read_csv(csv, usecols=usecols, chunksize=CHUNK):
        n += len(chunk)
        ads_rows_from_chunk(chunk, store)
        if with_laps:
            laps = clean_static(chunk)[KEY + LAP_METRICS]
            laps["delivery_days"] = laps["delivery_days"].astype("int32")
            laps["cumu_impressions"] = laps["cumu_impressions"].astype("int64")
            laps["cumu_clicks"] = laps["cumu_clicks"].astype("int64")
            table = pa.Table.from_pandas(laps, preserve_index=False)
            if lap_writer is None:
                lap_writer = pq.ParquetWriter(f"{DATA}/laps.parquet", table.schema)
            lap_writer.write_table(table)
        print(f"  ...{n:,} rows", end="\r")
    print()
    if lap_writer is not None:
        lap_writer.close()
        print(f"  wrote {DATA}/laps.parquet")

    ads = pd.DataFrame(list(store.values())).drop(columns="_rich")
    ads.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}: {len(ads):,} ads, {len(ads.columns)} cols")
    return ads


if __name__ == "__main__":
    build_ads(TRAIN_CSV, f"{DATA}/ads.parquet", with_laps=True)
    build_ads(PRED_CSV, f"{DATA}/ads_predict.parquet", with_laps=False)
    print("done.")
