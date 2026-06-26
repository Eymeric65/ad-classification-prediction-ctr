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
uv run python extract_image_features.py         # masks -> data/image_features.parquet (cache; incremental, --rebuild to redo)
uv run python compress_curves.py fit            # fit PCA, fill codebook, write scores.parquet
uv run python compress_curves.py encode         # re-encode curves from saved codebook
uv run python compress_curves.py decode <ad_id> # print a reconstructed curve
```

`main.py` is a placeholder stub.

## Data pipeline (the big picture)

Four stages, each writing into `data/` (gitignored, not in the repo):

1. **`build_dataset.py`** — parses the messy source CSVs *once* into clean, typed
   Parquet. Never read the CSVs in downstream code. Training is the **union of two ad
   platforms** whose schemas differ (the real prediction set is a mix of both):
   - `2_1. training_data_yda.csv` (~11.1M laps, ~85k ads) — device-targeted display; has
     `device` + `purpose_of_conversion_measurement`; ~67% full annotations.
   - `2_2. training_data_lap.csv` (~2.46M laps, ~49k ads) — has `creative_format` +
     `creative_call_to_action_type`; mostly light annotations.
   A column a platform lacks is filled with the sentinel `"__NA__"`, which doubles as a free
   **platform indicator** the model keys on. Outputs:
   - `ads.parquet` — one row per `ad_id` (~134k ads): static attrs (union schema) + flattened
     annotation features (training set).
   - `laps.parquet` — one row per `(ad_id, delivery_days)`: the cumulative time series
     (`cumu_cost/impressions/clicks/ctr`), ~13.5M rows. Written incrementally with a
     `ParquetWriter`; CSVs read in 200k-row chunks. Annotation JSON is parsed once-per-ad
     (richest row), not per lap row.
   - `ads_predict.parquet` — same columns as `ads.parquet`, for the prediction set (no laps).

2. **`extract_image_features.py`** — turns each creative's *segmentation mask* into spatial
   features and **caches** them in `data/image_features.parquet` (one row per `filename`).
   Masks live in `data/images_lap/` + `data/images_yda/` (both platforms ship them now, so
   **every ad gets image features** — not a one-platform signal). Each pixel is one of 10 flat
   palette colors = content categories (see the color-master slide / README). Per category we
   compute **coverage** (pixel fraction, 0..1) and the **normalized centroid** (`cx,cy` in
   0..1 — where on the banner that content sits); absent category → coverage 0, NaN centroid.
   30 columns: `img_cov_/img_cx_/img_cy_<cat>`. The run is **incremental** (skips filenames
   already cached); compute is decoupled from training so the ML side only reads the cache.

3. **`dataset.py`** — the *only* sanctioned way to load data. Use `load_ads`,
   `load_laps`, `load_predict`, and `load_train(at_day=N)` (joins each ad's static
   features to its lap state on a given delivery day; `at_day=None` joins all lap rows).
   `load_ads`/`load_predict` left-join the cached image features on `filename`. Module-level
   lists (`AREA_COLS`, `ANN_FLAG_COLS`, `CAT_COLS`, `DATE_COLS`, `IMG_COLS`) name the feature
   groups; `dataset_config.yaml`'s `features:` block toggles each group on/off.

4. **`compress_curves.py`** — compresses each ad's variable-length `cumu_ctr` curve into
   `n_components` PCA scores and reconstructs it back. Curves are resampled onto a common
   grid in log-CTR space, then PCA'd. The modeling idea: predict the few PCA scores from
   creative features, then `decode` them to the full curve. Outputs `data/scores.parquet`
   (one row per ad: `ad_id`, `pc0..pcN`).

## Key conventions & gotchas

- **`ad_id` is the primary key**, not `filename`. The same creative image can run as
  several `ad_id`s, so `filename` is not unique. `(ad_id, delivery_days)` is unique in laps.
- **ID columns are opaque bytes-repr strings** (e.g. `advertiser_id` = `b'\xd2\xa7…'`);
  `build_dataset.py` casts all IDs to pandas `string`.
- **Two-platform union & the `__NA__` sentinel** — `CAT_COLS` is the union of both platforms'
  categoricals. yda ads carry `__NA__` for `creative_format`/`creative_call_to_action_type`;
  lap ads carry `__NA__` for `device`/`purpose_of_conversion_measurement`. This is intentional:
  the sentinel pattern *is* a platform indicator. Beware Simpson's paradox in metrics — pooled
  PC0 R² (~0.72) is inflated by between-platform level separation; honest within-platform R² is
  ~0.53 (lap) / ~0.59 (yda). Always sanity-check per-platform, not just pooled.
- **Annotation schema mix** — ~67% of training ads now have the full 8-key annotation schema
  (yda is richer); the rest carry only a bare `banner_tagging_background` string. The prediction
  set is 100% full-schema. `parse_annotation` handles both and tags each row
  `ann_schema = "full" | "light"`. The 10-dim `area_*` vector is the cleanest annotation feature.
- **Leakage to watch:** `delivery_days` and the `cumu_*` columns exist only in training,
  not at predict time — they are the target side, never features.
- **CTR floor before log** — `compress_curves.CTR_EPS` (1e-6) floors `cumu_ctr` before the log
  transform, because yda has early zero-click laps (`cumu_ctr == 0`) that would otherwise give
  `log(0) = -inf` and break the PCA fit.
- **Mask features join on `filename`, not `ad_id`** — masks are per-creative, so the cache is
  keyed by `filename` (one row reused across an image's several `ad_id`s). Masks are clean
  indexed-color PNGs (exact palette, no anti-aliasing); `extract_image_features.py` packs RGB to
  an int24 and counts the 10 known colors. Adding the image group lifts held-out skill all
  +26.1%→+27.6%, tail +29.8%→+30.9%, and flips the middle −1.0%→+3.1% (the layout/centroid
  signal is what finally helps the everyday low-CTR ad the header categoricals couldn't separate).

## `compress_curves.py` config

`curve_config.yaml` is the single source of truth. The top params (`n_components`,
`grid_points`, `time_axis` = `normalized_lifetime` | `absolute_days`, `log_transform`)
are editable by hand; the `codebook` block (grid + PCA mean + components +
explained-variance) is *fitted output* — regenerate it by re-running `fit`, don't edit it.
`DEFAULT_PARAMS` in the script fills any params missing from the YAML, so old configs stay
forward-compatible. The current 8-component fit on the two-platform population captures ~97.5%
of variance (PC0 alone ~94.6%). PC0's share dropped from the old single-platform ~98.5% because
the two platforms have more diverse curve *shapes*, so the shape components (PC1+) now matter more.

## Reference

`README.md` has the full source-CSV column dictionary and a detailed breakdown of the
`annotation` JSON structure (the 8 `banner_tagging_*` keys, their Japanese label values,
and area-bucket IDs). Consult it before working with raw annotations.
