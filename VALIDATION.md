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

## Temporal (forward-in-time) validation — the HONEST number

The tables above use **random** splits, which let the model train on ads interleaved in
time with the test ads. But the real prediction set is a strict **forward holdout**:
training ends `min_date` 2023-04-30, predict is 2023-05-01 → 2023-11-07.
`pipeline/temporal_eval.py` measures the right geometry — split the training ads on
`min_date` at a cutoff, train only on what precedes it, evaluate on the future slice —
with a **matched random control** (same test fraction) so the only difference is
time-ordering. The gap *is* the cost of forward-in-time prediction.

**The drift (config that was tuned on the random split — 800 trees / 95 leaves / boost 8 /
date features on):** temporal vs its matched random control, per cutoff —

| cutoff (horizon) | split | grader RMSE | all | tail | mid | yda | lap |
|---|---|---|---|---|---|---|---|
| **2022-11-01** (6mo) | **temporal** | **0.00414** | **+16.8%** | +18.0 | −0.1 | +23.1 | +16.4 |
| | random (control) | 0.00264 | +30.2% | +33.6 | +5.2 | +39.5 | +29.4 |
| 2022-08-01 (9mo) | temporal | 0.00374 | +16.4% | +18.3 | −6.0 | +18.4 | +16.2 |
| 2023-01-01 (4mo) | temporal | 0.00475 | +16.9% | +17.9 | +2.4 | +24.2 | +16.6 |

- **Forward-in-time roughly HALVES skill: ~31% → ~16–17% all, ~35% → ~18% tail.** The
  random-split numbers in the sections above are optimistic by ~2×.
- **It's drift, not just harder ads.** The mean-curve *baseline* also degrades forward,
  but the model degrades *more relative to it* — the learned feature→curve mapping is
  non-stationary. yda + tail transfer best; **lap + middle worst (middle ≈0)**.
- **The ~16.5% floor is stable** across 4/6/9-month horizons and 80k–110k training sizes
  — genuine distribution shift, **not** a sample-size problem (more rows won't close it).

### Mitigation — the forward-tuned config (LIVE)

Re-tuning against the temporal objective (see `TUNING.md` §temporal) changed four things:
**num_leaves 95→47** (the 95-leaf model overfits time-specific structure), **boost 8→6**
(8 leaves the everyday ad underwater forward-in-time), **recency weighting half_life=120d**
(lean the fit toward the newest ads — the closest analogue to the forward predict set), and
**date features OFF** (cyclical month/day-of-year are mildly time-leaking forward). Result,
temporal split, before→after, all three horizons:

| cutoff (horizon) | grader RMSE | all | tail | mid | yda | lap |
|---|---|---|---|---|---|---|
| 2022-08-01 (9mo) | 0.00374→— | 16.4→**16.9** | 18.3→18.9 | −6.0→−5.9 | 18.4→**21.8** | 16.2→16.6 |
| **2022-11-01 (6mo)** | 0.00414→**0.00407** | 16.8→**17.7** | 18.0→18.5 | −0.1→**+7.0** | 23.1→**27.5** | 16.4→17.2 |
| 2023-01-01 (4mo) | 0.00475→— | 16.9→**18.2** | 17.9→18.8 | +2.4→**+8.4** | 24.2→**28.5** | 16.6→17.7 |

- **Every bucket improves at every horizon.** At the operative ~6-month horizon (closest
  to the real May–Nov 2023 set): all **+0.9pp**, tail +0.5pp, yda **+4.4pp**, and the
  middle goes from underwater to **+7.0%**. Honest submission estimate is now **~17.5%
  all-skill / ~18.5% tail / yda ~27%**, grader RMSE ~0.00407.
- **The gains are small relative to the drift.** Re-pointing the model at the right
  objective recovers ~1pp of the ~14pp drift gap — the drift is mostly *irreducible*; the
  bigger win is no longer optimizing for a metric the real task doesn't reward.
- **Tradeoff (by design):** the live config scores *worse on the random split* (all
  31.4→27.0%, middle +3.8→−14.5% — see `seed_eval.py`). That's expected — the random
  split is the wrong geometry; the live config is specialized to the forward holdout. Do
  **not** judge it by the random tables above.
- A rejected lever: temporal-watch-fold **early stopping made things worse** (the watch
  fold already shows drift, so per-PC trees stop at 3–30 iters and underfit). See TUNING.md.
- Reproduce: `uv run python pipeline/temporal_eval.py [cutoff ...]` (default 2022-11-01).

## Caveats

1. **Tune on the temporal split, not the random one.** The live config is now tuned
   against `temporal_eval.py`; the random-split skill tables earlier in this file reflect
   the *old* config and the *wrong* objective — kept only for the drift contrast. Validate
   any future change on the temporal split.
2. The honest single-draw submission estimate is **~17.5% all-skill** (6-month horizon,
   forward-tuned config), grader RMSE ~0.00407 — not the ~31% / 0.00258 the random tables
   report.

## Reproduce

Both harnesses live in `pipeline/`:

```bash
uv run python pipeline/rmse_eval.py 5    # grader RMSE table above
uv run python pipeline/seed_eval.py 5    # skill table above
```

Each refits the full model per seed (~40 s/fit); `rmse_eval.py` also loads the
13.5M-row `laps.parquet` for the raw actuals.
