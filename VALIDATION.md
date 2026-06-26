# Validation — expected prediction-set performance

How reliably the curve model (features → 8 PCA scores → `decode` → cumulative-CTR
curve) performs on unseen ads, estimated by refitting on multiple random
train/test splits. Config is the live one: `dataset_config.yaml` (features,
`threshold` sample-weighting) + `pipeline_config.yaml` (LightGBM) + the fitted
`curve_codebook.yaml`. 20% of ads held out per split, seeds 42–46.

> Numbers below reflect the **tuned config** (LightGBM 800 trees / 95 leaves,
> sample-weighting `threshold 0.0115 / boost 8`) — see `TUNING.md` for the sweep that
> set it. The previous 400/31 model RMSE'd ~0.00265; tuning brought it to ~0.00258.

## Grader metric — plain RMSE on raw `cumu_ctr`

The grader compares predicted `cumu_ctr` vs **actual** `cumu_ctr` per
`(ad, delivery_day)` row and takes `sqrt(mean(squared error))`. The number below
reproduces that on held-out **training** ads (the only ads with known actuals):
train on the rest, predict each held-out ad's curve, interpolate it to that ad's
real delivery-days, RMSE against its raw laps. It therefore includes **both** model
error **and** PCA reconstruction error — exactly what the grader sees.

| seed | RMSE | yda | lap | baseline (mean curve) |
|------|----------|----------|----------|-----------------------|
| 42 | 0.002581 | 0.000938 | 0.005834 | 0.003791 |
| 43 | 0.002468 | 0.000924 | 0.005419 | 0.003728 |
| 44 | 0.002590 | 0.000978 | 0.005629 | 0.003825 |
| 45 | 0.002541 | 0.000949 | 0.005656 | 0.003750 |
| 46 | 0.002696 | 0.000955 | 0.005969 | 0.003839 |
| **mean** | **0.002575** | **0.000949** | **0.005701** | 0.003787 |
| **std**  | **±0.000074** | ±0.000018 | ±0.000188 | ±0.000042 |

- **Expected grader RMSE ≈ 0.00258, std ≈ 0.00007** (~2.9% relative). Seed-to-seed
  range is a tight **0.00247–0.00270**.
- Beats the mean-curve baseline (**0.00379**) by ~32% RMSE, consistently.
- **yda RMSE (0.00095) ≪ lap RMSE (0.00570)** — yda CTRs are smaller numbers, so
  squared errors are smaller. The prediction set is ~75% yda rows (vs 63% in
  training), so the real submission RMSE is likely **slightly below 0.00258**.

## Skill metric (context) — % RMSE reduction vs mean-curve baseline

`skill % = 1 − model_RMSE / mean-curve_RMSE` in decoded-CTR space. Reported overall,
for the high-CTR **tail** (top 10%) vs the **middle**, and **within each platform**
(pooled skill is inflated by between-platform level separation — Simpson's paradox —
so the within-platform numbers are the honest read).

| metric | mean | std |
|--------|------|-----|
| weighted score-space RMSE | 4.771 | ±0.036 |
| all | 31.4% | ±0.86 |
| tail (top 10%) | 35.5% | ±1.12 |
| middle | 3.8% | ±1.76 |
| yda (within-platform) | 40.2% | ±0.58 |
| lap (within-platform) | 30.7% | ±0.93 |

The modest **middle** skill (+3.8%) is by design: the `threshold` sample-weighting
trades middle accuracy for the high-CTR tail (the documented Pareto choice). The tuned
model lifts every bucket over the old 400/31 config (all 28.4→31.4%, tail 32.2→35.5%,
both within-platform skills up ~3–6pp) — see `TUNING.md`.

## Caveats

1. **Random split, not temporal.** These splits measure *sampling* stability (now
   confirmed tight). The real prediction set is a **forward-in-time holdout** (all
   2023). If CTR/creative trends drifted, the real RMSE could sit slightly **above**
   this band. A time-based split (train early, validate latest) would estimate that.
2. The std is the spread of the *estimate* across held-out samples; the actual
   submission is a single draw, but given the tight band, expect to land within
   ~0.0001 of 0.00258.

## Reproduce

Both harnesses live in `pipeline/`:

```bash
uv run python pipeline/rmse_eval.py 5    # grader RMSE table above
uv run python pipeline/seed_eval.py 5    # skill table above
```

Each refits the full model per seed (~40 s/fit); `rmse_eval.py` also loads the
13.5M-row `laps.parquet` for the raw actuals.
