"""Turn the messy source CSVs into clean, typed Parquet — run once.

Training is built from TWO source files for two ad platforms whose schemas differ:
  2_1. training_data_yda.csv  (~11.1M laps)  device-targeted display; has `device` +
                              `purpose_of_conversion_measurement`; mostly FULL annotations.
  2_2. training_data_lap.csv  (~2.46M laps)  has `creative_format` +
                              `creative_call_to_action_type`; mostly LIGHT annotations.
The real prediction set is a MIX of both platforms, so we train ONE unified model on the
union of both schemas. Columns a platform lacks are filled with the sentinel "__NA__", which
doubles as a free platform indicator the model can key on.

Outputs (in data/):
  ads.parquet          one row per ad_id: static creative attrs + annotation features
  laps.parquet         (ad_id, delivery_days) -> cumulative metrics  [the time series]
  ads_predict.parquet  same schema as ads.parquet, for the prediction set (no laps)

Usage:  uv run python build_dataset.py
"""
import glob
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = "data"
TRAIN_CSVS = [f"{DATA}/2_1. training_data_yda.csv", f"{DATA}/2_2. training_data_lap.csv"]
# The prediction set is also a MIX of both platforms. Glob every `3_*. prediction_data_*.csv`
# so dropping in the yda file (e.g. `3_1. prediction_data_yda.csv`) is picked up automatically;
# build_ads unions them with the same __NA__ sentinel fill as the training side.
PRED_CSVS = sorted(glob.glob(f"{DATA}/3_*. prediction_data_*.csv"))
CHUNK = 200_000
NA = "__NA__"                       # sentinel for a column a platform doesn't have

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
# Union of static categorical features across both platforms. A row missing one of these
# (because its source CSV doesn't have the column) gets the "__NA__" sentinel.
CAT_FEATURES = [
    "digital_large", "digital_small", "campaign_goal", "creative_size",   # shared
    "creative_format", "creative_call_to_action_type",                    # lap platform only
    "device", "purpose_of_conversion_measurement",                        # yda platform only
]
STATIC = ID_COLS + ["filename", "min_date"] + CAT_FEATURES
LAP_METRICS = ["delivery_days", "cumu_cost", "cumu_impressions", "cumu_clicks", "cumu_ctr"]


def clean_static(df):
    df = df.copy()
    df["min_date"] = pd.to_datetime(df["min_date"], errors="coerce")
    for c in ID_COLS:                       # ids are opaque (advertiser_id is a bytes-repr) -> string
        df[c] = df[c].astype("string")
    for c in CAT_FEATURES:                  # categoricals -> string; empty/absent -> sentinel
        df[c] = df[c].astype("string")
        df[c] = df[c].where(df[c].notna() & (df[c].str.len() > 0), NA)
    return df


def ads_rows_from_chunk(chunk, store):
    """Keep, per ad_id, the static record + raw annotation of its RICHEST row.

    Richness is detected with a cheap substring test (full schema => has
    'banner_tagging_object'); the expensive JSON parse is deferred to once-per-ad at the end,
    so we don't json.loads ~13.5M rows to keep ~140k.
    """
    chunk = clean_static(chunk)
    ann = chunk["annotation"]
    rich = ann.fillna("").str.contains("banner_tagging_object", regex=False).to_numpy()
    static = chunk[STATIC].to_dict("records")
    ann_list = ann.tolist()
    for rec, a, r in zip(static, ann_list, rich):
        k = rec["ad_id"]
        prev = store.get(k)
        if prev is None or (r and not prev[2]):
            store[k] = (rec, a, bool(r))


def build_ads(csvs, out_path, with_laps):
    store = {}
    lap_writer = None
    for csv in csvs:
        print(f"building from {csv} ...")
        header = pd.read_csv(csv, nrows=0).columns.tolist()
        # each source only has its own columns; read what's present, fill the rest as NA later
        present_static = [c for c in STATIC if c in header]
        usecols = present_static + ["annotation"] + (LAP_METRICS if with_laps else [])
        n = 0
        for chunk in pd.read_csv(csv, usecols=usecols, chunksize=CHUNK):
            for c in STATIC:                       # add columns this platform lacks
                if c not in chunk.columns:
                    chunk[c] = pd.NA
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

    # one expensive annotation parse per kept ad, now that we know the richest row
    recs = [{**rec, **parse_annotation(a)} for rec, a, _ in store.values()]
    ads = pd.DataFrame(recs)
    ads.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}: {len(ads):,} ads, {len(ads.columns)} cols "
          f"({(ads['ann_schema'] == 'full').mean():.0%} full-schema)")
    return ads


if __name__ == "__main__":
    build_ads(TRAIN_CSVS, f"{DATA}/ads.parquet", with_laps=True)
    print(f"prediction sources: {PRED_CSVS}")
    build_ads(PRED_CSVS, f"{DATA}/ads_predict.parquet", with_laps=False)
    print("done.")
