# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An ML project to predict an advertising creative's cumulative-CTR trajectory from its
static features (industry, campaign goal, creative attributes, and computer-vision
annotations of the banner image), before the ad has run. The performance target is a
*curve* (cumulative CTR over the ad's lifetime), not a single number.

## Environment & commands

Uses [uv](https://docs.astral.sh/uv/). There is no test suite or linter configured.

```bash
uv sync                              # create venv, install deps
uv run python build_dataset.py       # raw CSV -> data/*.parquet  (~4 min, run ONCE)
uv run python compress_curves.py fit            # fit PCA, fill codebook, write scores.parquet
uv run python compress_curves.py encode         # re-encode curves from saved codebook
uv run python compress_curves.py decode <ad_id> # print a reconstructed curve
```

`main.py` is a placeholder stub.

## Data pipeline (the big picture)

Three stages, each writing into `data/` (gitignored, not in the repo):

1. **`build_dataset.py`** — parses the two messy source CSVs *once* into clean, typed
   Parquet. Never read the CSVs in downstream code. Outputs:
   - `ads.parquet` — one row per `ad_id`: static creative attrs + flattened annotation
     features (training set).
   - `laps.parquet` — one row per `(ad_id, delivery_days)`: the cumulative time series
     (`cumu_cost/impressions/clicks/ctr`). Written incrementally with a `ParquetWriter`
     because it's ~2.5M rows; CSV is read in 200k-row chunks.
   - `ads_predict.parquet` — same columns as `ads.parquet`, for the prediction set (no laps).

2. **`dataset.py`** — the *only* sanctioned way to load data. Use `load_ads`,
   `load_laps`, `load_predict`, and `load_train(at_day=N)` (joins each ad's static
   features to its lap state on a given delivery day; `at_day=None` joins all lap rows).
   Module-level lists (`AREA_COLS`, `ANN_FLAG_COLS`, `CAT_COLS`) name the feature groups.

3. **`compress_curves.py`** — compresses each ad's variable-length `cumu_ctr` curve into
   `n_components` PCA scores and reconstructs it back. Curves are resampled onto a common
   grid in log-CTR space, then PCA'd. The modeling idea: predict the few PCA scores from
   creative features, then `decode` them to the full curve. Outputs `data/scores.parquet`
   (one row per ad: `ad_id`, `pc0..pcN`).

## Key conventions & gotchas

- **`ad_id` is the primary key**, not `filename`. The same creative image can run as
  several `ad_id`s, so `filename` is not unique. `(ad_id, delivery_days)` is unique in laps.
- **ID columns are opaque bytes-repr strings** (e.g. `advertiser_id` = `b'\xd2\xa7…'`);
  `build_dataset.py` casts all IDs to pandas `string`.
- **Annotation train/predict schema mismatch** — only ~34% of training rows have the full
  8-key annotation schema; the other ~66% carry only a bare `banner_tagging_background`
  string. The prediction set is 100% full-schema. `parse_annotation` handles both and
  tags each row `ann_schema = "full" | "light"`. When using annotation features, prefer
  the full-schema rows (those match the prediction set). The 10-dim `area_*` vector is the
  cleanest annotation feature.
- **Leakage to watch:** `delivery_days` and the `cumu_*` columns exist only in training,
  not at predict time — they are the target side, never features.

## `compress_curves.py` config

`curve_config.yaml` is the single source of truth. The top params (`n_components`,
`grid_points`, `time_axis` = `normalized_lifetime` | `absolute_days`, `log_transform`)
are editable by hand; the `codebook` block (grid + PCA mean + components +
explained-variance) is *fitted output* — regenerate it by re-running `fit`, don't edit it.
`DEFAULT_PARAMS` in the script fills any params missing from the YAML, so old configs stay
forward-compatible. The current 8-component fit captures ~98.5% of variance (PC0 alone ~98.5%).

## Reference

`README.md` has the full source-CSV column dictionary and a detailed breakdown of the
`annotation` JSON structure (the 8 `banner_tagging_*` keys, their Japanese label values,
and area-bucket IDs). Consult it before working with raw annotations.
