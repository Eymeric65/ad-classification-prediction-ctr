# petit-projet-philippe

A tiny machine learning project. Setup uses [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync          # create the venv and install dependencies
uv run python …  # run anything inside the project environment
```

Dependencies: `pandas`, `numpy`, `scikit-learn`.

## Data — use the Parquet, not the CSV

The raw CSVs are messy (bytes-string ids, a JSON `annotation` blob, ads and time series
glued together). [build_dataset.py](build_dataset.py) parses them **once** into clean,
typed Parquet. After that, never touch the CSV — load via [dataset.py](dataset.py).

```bash
uv run python build_dataset.py     # raw CSV -> data/*.parquet  (~4 min, run once)
```

| file | grain | rows | what |
|------|-------|------|------|
| `data/ads.parquet` | one row per `ad_id` | ~49,149 | static creative attrs + annotation features (train) |
| `data/laps.parquet` | `(ad_id, delivery_days)` | ~2,462,571 | the **time series**: cumulative cost/impr/clicks/ctr |
| `data/ads_predict.parquet` | one row per `ad_id` | 10,664 | prediction set — **same columns as ads.parquet** |

`ad_id` is the primary key (the same creative image can run as several `ad_id`s, so
`filename` is *not* unique). `ad_id + delivery_days` is unique in laps.

```python
from dataset import load_ads, load_laps, load_train, load_predict

ads  = load_ads()           # static features per ad
laps = load_laps()          # full time series
df   = load_train(at_day=7) # one row per ad at delivery_day 7, joined to features
pred = load_predict()       # what you score at inference time
```

The annotation features are already flattened into columns (`area_*`, `ann_has_*`,
`n_face_tags`, `n_text_tags`, `ann_schema`) — see the annotation section below for what
they mean and the train/predict schema caveat.

### Raw source files (reference only)

Two CSV files live in [data/](data/).

### Training — [data/2_2. training_data_lap.csv](data/2_2.%20training_data_lap.csv)

~2,462,571 rows. This is the file you fit on.

| # | column | notes |
|---|--------|-------|
| 1 | `filename` | creative image file (`.png`) |
| 2 | `min_date` | start date of the period |
| 3 | `delivery_days` | days the ad ran |
| 4 | `advertiser_id` | hashed/encoded id |
| 5 | `campaign_id` | |
| 6 | `adgroup_id` | |
| 7 | `ad_id` | |
| 8 | `cumu_cost` | cumulative cost |
| 9 | `cumu_impressions` | cumulative impressions |
| 10 | `cumu_clicks` | cumulative clicks |
| 11 | `cumu_ctr` | cumulative click-through rate (clicks / impressions) |
| 12 | `digital_large` | industry (large category) |
| 13 | `digital_small` | industry (small category) |
| 14 | `campaign_goal` | e.g. `APP_INSTALL`, `WEBSITE_CONVERSION` |
| 15 | `creative_format` | e.g. `IMAGE` |
| 16 | `creative_call_to_action_type` | e.g. `LEARN_MORE` |
| 17 | `creative_size` | e.g. `1080x1080` |
| 18 | `annotation` | JSON string of creative tags |

### Industry columns (`digital_large` / `digital_small`)

Japanese industry-classification labels: `digital_large` is the broad category (**31
values**), `digital_small` the finer sub-category (**51 values**). Both are
high-cardinality categoricals dominated by a single "missing" bucket.

> ⚠️ **~70% of training ads are `業種不明` ("industry unknown")** in both columns (34,361 of
> ~49k). A second missing flavor, `未設定` ("not set"), also appears in `digital_small`. The
> real signal lives in the ~30% that are actually labeled — encode accordingly.

Most common labeled values (training, `ads.parquet`):

| `digital_large` | meaning | count | | `digital_small` | meaning | count |
|---|---|---|---|---|---|---|
| ファッション・アクセサリー | Fashion & accessories | 7,029 | | カジュアルウェア | Casual wear | 3,978 |
| カメラ・時計・精密機器・事務用品 | Cameras/watches/precision/office | 2,050 | | ハイブランド | Luxury brands | 2,734 |
| 自動車・輸送用機器・用品 | Automotive & transport | 1,708 | | 時計 | Watches | 1,973 |
| 官公庁・各種団体 | Government / public bodies | 707 | | 自動車メーカー | Auto manufacturers | 1,708 |
| 化粧品・医薬部外品 | Cosmetics / quasi-drugs | 456 | | 官公庁・各種団体 | Government bodies | 698 |
| 通信 | Telecom | 448 | | 化粧品 | Cosmetics | 453 |
| 製薬・医薬品 | Pharma | 318 | | 通信 | Telecom | 448 |
| 住宅・不動産・建設 | Housing / real estate / constr. | 312 | | 住宅・不動産総合サービス | Real estate services | 281 |

…then a long tail down to single-digit counts. Worth checking the same distribution in
`ads_predict.parquet` for train/predict skew before relying on these features.

### Prediction — [data/3_2. prediction_data_lap.csv](data/3_2.%20prediction_data_lap.csv)

~10,664 rows. Same schema as training **minus the performance columns** — these are the
fields you have at predict time, before the ad has run:

`filename`, `min_date`, `advertiser_id`, `campaign_id`, `adgroup_id`, `ad_id`,
`digital_large`, `digital_small`, `campaign_goal`, `creative_format`,
`creative_call_to_action_type`, `creative_size`, `annotation`

**Columns present in training but absent here** (i.e. the candidate targets / leakage to
watch for): `delivery_days`, `cumu_cost`, `cumu_impressions`, `cumu_clicks`, `cumu_ctr`.

> Note: `advertiser_id` (and other id columns) are stored as Python `bytes` repr strings
> like `b'\xd2\xa7…'`.

## The `annotation` column

`annotation` is a **JSON string** holding computer-vision tags of the banner image
(`filename`). It is a single JSON object with up to **8 keys**. Every key's value is a
**list of `{"id": …, "name": …}`** objects (multi-label), except as noted.

| key | meaning | shape | example names |
|-----|---------|-------|---------------|
| `banner_tagging_area` | how the image area is split across object types | **10 fixed buckets**, each `{id, name, area}` with `area` ∈ [0,1] | 人物/動物/商品 (photo & illustration), ロゴ, テキスト, その他 |
| `banner_tagging_background` | background style | list (or bare string, see below) | 背景(フォト)=photo bg, 背景パターン有り=patterned, ベタ一色/色 白=solid white |
| `banner_tagging_dominant` | dominant color | usually 1 item | 最頻色 白/青/グレー… (white/blue/grey…) |
| `banner_tagging_button_band` | CTA button & band | multi-label | ボタン無し=no button, 帯あり=band, ボタン有 位置（右下）=button bottom-right, color tags |
| `banner_tagging_object` | product presence | 1 item, 3 values | 商品無し / 商品あり 写真 / 商品あり イラスト |
| `banner_tagging_person` | person/character presence | multi-label, 3 values | 人物無し, キャラクター無し, キャラクターあり |
| `face_genderage` | faces detected (gender + age) | multi-label, **can be `[]`** | 人物あり（女性）, 人物あり（20～34歳）… |
| `creative_text_tagging_detect` | OCR'd marketing keywords | multi-label, **can be `[]`**, open vocab (~30+) | 価格=price, 無料=free, 限定=limited, プレゼント=gift |

### ⚠️ Train/predict schema mismatch — read before fitting

The annotation schema is **not** consistent between the two files:

| | training | prediction |
|---|---|---|
| rows with the **full 8-key** schema | **~34%** | **100%** |
| rows with **only** `banner_tagging_background` (the rest missing) | **~66%** | 0% |
| type of `banner_tagging_background` | **bare string** on the 66%, **list** on the 34% | always a **list** |

So ~2/3 of training rows have *no* annotation features except a single background string.
Practical options:

- **Safest:** restrict annotation features to the **~34% full-schema** training rows
  (those that contain `banner_tagging_object`), which match the prediction file exactly.
- Use `banner_tagging_area` as the workhorse feature: it's a clean, **fixed-length
  10-dim numeric vector** (the `area` fractions) whenever the full schema is present.
- Treat the categorical keys as **multi-hot** (lists can hold several tags).
- Normalize `banner_tagging_background` to handle both the string and the list form.

> Quick recipe: `json.loads(row.annotation)`, then
> `{f"area_{x['id']}": x['area'] for x in d.get('banner_tagging_area', [])}` gives the
> 10 numeric area features; multi-hot the `name`s of the other keys.
