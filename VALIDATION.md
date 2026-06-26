# Validation — expected prediction-set performance

How reliably the curve model (features → 8 PCA scores → `decode` → cumulative-CTR
curve) performs on unseen ads, estimated by refitting on multiple random
train/test splits. Config is the live one: `dataset_config.yaml` (features,
`threshold` sample-weighting) + `pipeline_config.yaml` (LightGBM) + the fitted
`curve_codebook.yaml`. 20% of ads held out per split, seeds 42–46.

## Grader metric — plain RMSE on raw `cumu_ctr`

The grader compares predicted `cumu_ctr` vs **actual** `cumu_ctr` per
`(ad, delivery_day)` row and takes `sqrt(mean(squared error))`. The number below
reproduces that on held-out **training** ads (the only ads with known actuals):
train on the rest, predict each held-out ad's curve, interpolate it to that ad's
real delivery-days, RMSE against its raw laps. It therefore includes **both** model
error **and** PCA reconstruction error — exactly what the grader sees.

| seed | RMSE | yda | lap | baseline (mean curve) |
|------|----------|----------|----------|-----------------------|
| 42 | 0.002690 | 0.001033 | 0.006037 | 0.003791 |
| 43 | 0.002563 | 0.001009 | 0.005586 | 0.003728 |
| 44 | 0.002658 | 0.001063 | 0.005731 | 0.003825 |
| 45 | 0.002580 | 0.001029 | 0.005690 | 0.003750 |
| 46 | 0.002751 | 0.001035 | 0.006046 | 0.003839 |
| **mean** | **0.002648** | **0.001034** | **0.005818** | 0.003787 |
| **std**  | **±0.000070** | ±0.000017 | ±0.000188 | ±0.000042 |

- **Expected grader RMSE ≈ 0.00265, std ≈ 0.00007** (~2.6% relative). Seed-to-seed
  range is a tight **0.00258–0.00275**.
- Beats the mean-curve baseline (**0.00379**) by ~30% RMSE, consistently.
- **yda RMSE (0.00103) ≪ lap RMSE (0.00582)** — yda CTRs are smaller numbers, so
  squared errors are smaller. The prediction set is ~75% yda rows (vs 63% in
  training), so the real submission RMSE is likely **slightly below 0.00265**.

## Skill metric (context) — % RMSE reduction vs mean-curve baseline

`skill % = 1 − model_RMSE / mean-curve_RMSE` in decoded-CTR space. Reported overall,
for the high-CTR **tail** (top 10%) vs the **middle**, and **within each platform**
(pooled skill is inflated by between-platform level separation — Simpson's paradox —
so the within-platform numbers are the honest read).

| metric | mean | std |
|--------|------|-----|
| weighted score-space RMSE | 5.175 | ±0.031 |
| all | 28.4% | ±0.68 |
| tail (top 10%) | 32.2% | ±0.85 |
| middle | 2.0% | ±1.02 |
| yda (within-platform) | 34.3% | ±0.95 |
| lap (within-platform) | 27.9% | ±0.75 |

The near-zero **middle** skill is by design: the `threshold` sample-weighting trades
middle accuracy for the high-CTR tail (the documented Pareto choice).

## Caveats

1. **Random split, not temporal.** These splits measure *sampling* stability (now
   confirmed tight). The real prediction set is a **forward-in-time holdout** (all
   2023). If CTR/creative trends drifted, the real RMSE could sit slightly **above**
   this band. A time-based split (train early, validate latest) would estimate that.
2. The std is the spread of the *estimate* across held-out samples; the actual
   submission is a single draw, but given the tight band, expect to land within
   ~0.0001 of 0.00265.

## Reproduce

Both harnesses live in `pipeline/`:

```bash
uv run python pipeline/rmse_eval.py 5    # grader RMSE table above
uv run python pipeline/seed_eval.py 5    # skill table above
```

Each refits the full model per seed (~40 s/fit); `rmse_eval.py` also loads the
13.5M-row `laps.parquet` for the raw actuals.
