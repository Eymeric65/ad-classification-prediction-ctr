# Hyperparameter tuning log — 2026-06-26

Tuning of the score-predictor (features → 8 PCA scores → `decode` → cumulative-CTR curve).
All sweeps share **one** held-out split (`test_size=0.2`, `random_state=42`; train=107,140 /
test=26,785) so configs are directly comparable; the winner is re-checked with 4-fold CV.

**Metrics.** `wRMSE` = explained-variance-weighted score-space RMSE (PC0 ≈ 95% of the weight).
`all / tail / mid` = curve-space **skill** = `1 − model_RMSE / mean-curve_RMSE`, in decoded-CTR
space, over all test ads / the high-CTR top-10% tail / the rest. **Objective:** maximize `tail`
subject to `mid ≥ 0` (the everyday ad must still beat the mean-curve baseline), with `all` /
`wRMSE` as tie-breakers. Tune on **curve-space skill**, not wRMSE — they diverge at high model
capacity (wRMSE keeps dropping while real tail-skill turns over from overfit).

## Starting point

| config | all% | tail% | mid% | wRMSE |
|---|---|---|---|---|
| model 400/31, sample_weighting threshold 0.012 / boost 5 / clip 20 | 27.6 | 30.9 | +3.1 | 5.28 |

## Stage 1 — sample_weighting `threshold` (model fixed at 400/31)

Train mean-CTR percentiles (context): p50 0.0011, p90 0.0070, p95 0.0103, p98 0.0160, p99 0.0222.
Coarse then fine sweep:

| threshold | n_boost | all% | tail% | mid% | wRMSE |
|---|---|---|---|---|---|
| 0.0030 | 31,118 | 25.31 | 25.67 | 22.16 | 5.2425 |
| 0.0050 | 17,474 | 26.36 | 29.48 | +3.28 | 5.2625 |
| 0.0080 | 8,603 | 27.08 | 31.43 | −3.55 | 5.2189 |
| 0.0100 | 5,681 | 27.32 | 31.31 | −1.17 | 5.1755 |
| 0.0105 | 5,164 | 27.42 | 31.27 | −0.21 | 5.1607 |
| 0.0110 | 4,711 | 27.36 | 31.04 | +0.75 | 5.1541 |
| **0.0115** | 4,292 | 27.69 | **31.25** | **+1.81** | 5.1483 |
| 0.0120 (old) | 3,936 | 27.60 | 30.94 | +3.13 | 5.1378 |
| 0.0150 | 2,446 | 27.80 | 30.13 | +9.93 | 5.0910 |
| 0.0200 | 1,342 | 27.53 | 28.65 | +18.32 | 5.0516 |

**Pick: threshold 0.0115** — the knee. Highest tail skill of any threshold that keeps mid ≥ 0;
below ~0.011 the everyday ad drops under baseline, above it the tail erodes. The unconstrained
tail peak (31.43 @ 0.008) costs −3.6% on the middle — rejected.

## Stage 2 — LightGBM capacity (the big win; sample_weighting fixed at 0.0115 / boost 5)

The default `400 trees / 31 leaves` was **badly underfit**.

| n_estimators | learning_rate | num_leaves | wRMSE | all% | tail% | mid% |
|---|---|---|---|---|---|---|
| 400 | 0.05 | 31 | 5.1483 | 27.69 | 31.25 | +1.81 |
| 400 | 0.05 | 63 | 4.9500 | 29.71 | 32.69 | +7.58 |
| 800 | 0.05 | 31 | 4.9766 | 29.50 | 32.60 | +6.67 |
| 800 | 0.05 | 63 | 4.7833 | 30.95 | 33.40 | +12.34 |
| 800 | 0.03 | 63 | 4.8922 | 30.17 | 32.92 | +9.59 |
| 1200 | 0.03 | 63 | 4.7944 | 30.86 | 33.25 | +12.65 |
| 1500 | 0.02 | 63 | 4.8334 | 30.68 | 33.18 | +11.77 |

Higher-capacity round (lr 0.05):

| n_estimators | num_leaves | wRMSE | all% | tail% | mid% |
|---|---|---|---|---|---|
| 800 | 63 | 4.7833 | 30.95 | 33.40 | +12.34 |
| 1200 | 63 | 4.7003 | 31.28 | 33.45 | +14.54 |
| **800** | **95** | 4.6826 | 31.38 | **33.50** | +15.04 |
| 1200 | 95 | 4.6195 | **31.44** | 33.33 | +16.72 |
| 800 | 127 | 4.6328 | 31.44 | 33.34 | +16.65 |
| 1200 | 127 | 4.5822 | 31.35 | 33.06 | +17.89 |
| 2000 | 127 | 4.5526 | 30.96 | 32.46 | +19.00 |

**Pick: 800 trees / 95 leaves / lr 0.05.** Curve-space tail skill peaks ~95 leaves; past 127/2000
trees `wRMSE` keeps falling but real tail skill turns over (overfit) — the divergence that says
optimize on skill, not wRMSE.

## Stage 3 — joint `threshold × boost` (re-tuned at model 800/95)

The better-fit model leaves the middle strongly positive, so the tail boost can go harder. `clip` 30.

| threshold | boost | all% | tail% | mid% | wRMSE |
|---|---|---|---|---|---|
| 0.0115 | 5.0 | 31.38 | 33.50 | +15.04 | 4.6826 |
| **0.0115** | **8.0** | **31.44** | 34.90 | **+6.48** | 4.7378 |
| 0.0115 | 12.0 | 30.68 | 35.22 | −0.73 | 4.7953 |
| 0.0100 | 8.0 | 31.29 | 35.12 | +4.04 | 4.7641 |
| 0.0080 | 8.0 | 31.24 | 35.56 | +1.15 | 4.7183 |
| 0.0080 | 12.0 | 30.88 | 36.69 | −7.34 | 4.8941 |
| 0.0060 | 20.0 | 29.52 | 36.85 | −16.41 | 5.1196 |

**Pick: threshold 0.0115 / boost 8.0 / clip 30.** Best `all`, tail up to 34.9% (+1.4pp over
boost 5), middle a robust +6.5%. boost > 8 buys ~1pp more tail but drives the middle negative —
rejected. The aggressive alternative (0.008 / boost 8 → tail 35.6%) leaves a fragile +1.1% middle.

## Final config (committed)

- `pipeline_config.yaml`: `n_estimators 800`, `num_leaves 95` (lr 0.05, others unchanged).
- `dataset_config.yaml`: `threshold 0.0115`, `boost 8.0`, `clip 30`.

Production held-out report:

| bucket | model RMSE | baseline | skill |
|---|---|---|---|
| all | 0.003726 | 0.005435 | **+31.4%** |
| tail (top 10%) | 0.010601 | 0.016283 | **+34.9%** |
| middle | 0.001714 | 0.001833 | **+6.5%** |

wRMSE 5.28 → 4.74; PC0 score-RMSE 5.36 → 4.81.

## 4-fold CV confirmation (KFold shuffle, random_state 42)

Single-split gains re-checked across 4 folds — the lift holds and exceeds the fold-to-fold noise.

| metric | original (400/31, 0.012/5) | tuned (800/95, 0.0115/8) | Δ |
|---|---|---|---|
| wRMSE | 5.1871 ± 0.0399 | **4.8051 ± 0.0370** | −0.38 |
| all | 27.98% ± 0.81 | **30.88% ± 0.78** | +2.9pp |
| tail | 31.70% ± 1.20 | **34.90% ± 1.13** | +3.2pp |
| mid | 1.71% ± 0.65 | **3.04% ± 1.03** | +1.3pp |

The +2.9pp all-skill gap is ~3.6× the fold std, and **every** bucket improves (incl. the middle).

## Levers not pulled

- `min_child_samples`, `colsample_bytree` — fine regularization, held fixed; could trim the
  95-leaf overfit risk for a fraction of a point.
- `learning_rate` < 0.05 with proportionally more trees — 0.05 already won at these tree counts.
- Aggressive tail mode (`threshold 0.008 / boost 8`) → tail +35.6% but middle a fragile +1.1%.

## Caveats

- Single random 80/20 split for the sweeps (CV-confirmed for the winner). The real prediction
  set is a **forward-in-time holdout** (all 2023) — see VALIDATION.md; a temporal split would
  estimate drift the random split can't see.
- VALIDATION.md's skill table predates this tuning (reflects the old 400/31 config) — refresh it
  by re-running `pipeline/seed_eval.py` before trusting those exact numbers.

## Reproduce

Harnesses (kept in the session scratchpad, not committed): `tune.py` (stages `model` / `model2` /
`sw`), `sweep_threshold.py`, `cv_confirm.py`. Each assembles+splits once and refits per config.
