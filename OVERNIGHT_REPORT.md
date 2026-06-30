# Overnight investigation — neural nets & the forward-in-time (temporal) problem

**Date:** 2026-06-28 (overnight run) · **Branch:** `temporal-forward-tuning`
**Goal you set:** the LightGBM model is "only a classifier" and we want to "actually
predict the future" — explore moving to neural nets, and above all *try new methods to
attack the temporal (forward-in-time) issue* we've established is the real bottleneck.

> TL;DR — Three honest headlines, all validated on the forward holdout:
> 1. **A plain MLP loses to LightGBM overall, exactly as the tabular-ML literature predicts** —
>    but it is *much* stronger on the everyday "middle" ad and on yda, the two buckets temporal
>    drift hurts most. The two model families are **complementary**, not substitutes.
> 2. **A LightGBM+MLP ensemble is the night's best result**: it improves the real grader RMSE
>    and lifts yda (≈75% of the true predict set) and the middle bucket meaningfully, at a small
>    cost to pooled tail-skill. This is a *defensible, shippable* win. *(see validated numbers below)*
> 3. **The drift is, as VALIDATION.md already suspected, mostly irreducible.** I diagnosed it
>    precisely (large covariate shift, stable marginal target) and tried the textbook fixes.
>    Covariate-shift importance weighting gives a small consistent gain; dropping the most-drifting
>    feature *hurts*. No method "solves" the future — the gap is structural.

---

## 1. Setup & what I had to rebuild

- The container's `uv` and the project `.venv` were **macOS leftovers** (the venv's `python`
  symlinked into `/Library/Developer/...`). I built a clean Linux env (Python 3.10) with
  numpy/pandas/pyarrow/scikit-learn/lightgbm and **PyTorch 2.6 + CUDA 12.4**, verified on the
  **RTX 3070** GPU. All experiment code is self-contained under `experiments/`.
- `pipeline/temporal_eval.py` has a **hardcoded Mac path** (`_ROOT = "/Users/tokoemy/..."`) and
  can't run here. I re-implemented its geometry in `experiments/harness.py` as a *model-agnostic*
  harness so MLP / ensemble / LightGBM are all scored on the **identical** forward holdout.

### The number every method is judged against
Reproducing the live config (LightGBM 800 trees / 47 leaves, tail boost, recency decay,
advertiser encoding) on the **2022-11-01 forward holdout** (train ads `< cutoff`, test ads `≥`):

```
baseline (lightgbm)   grader RMSE=0.003953   all=+19.6%  tail=+20.3  mid=+9.6  yda=+26.3  lap=+19.2
```

This is ~2pp above VALIDATION.md's documented +17.7% because the doc predates the
latest advertiser-encoding commit (`6a55114`) — consistent with its noted +1.5pp. The harness
geometry is therefore confirmed correct.

---

## 2. Diagnosis — *what kind* of temporal problem is this?

Before trying fixes, I measured the drift's nature (`experiments/drift_diagnosis.py`).

**(A) Covariate shift is large.** A LightGBM time-discriminator trained to tell "recent"
(min_date ≥ 2022-11-01) from "old" ads scores **AUC = 0.755** (0.5 = no shift). The *feature
distribution itself moves substantially over time.* The biggest drifters:

| feature | single-feature time-AUC |
|---|---|
| `campaign_goal` (categorical) | **0.787** |
| `area_text` | 0.730 |
| `area_logo` | 0.728 |
| `area_other_illust` | 0.716 |
| `n_text_tags` | 0.659 |
| `ann_has_button` | 0.649 |
| platform mix (yda share) | 61.4% → 68.6% |

**(B) The marginal target barely moves.** Every PCA score shifts < 0.06 sd old→recent; the
decoded curve level shifts only +0.06 sd. So this is **not** marginal label shift.

**Conclusion:** the drift is **covariate shift** (X moves into new regions) **plus concept
shift** (the feature→curve *mapping* changes — the model leans on features like `campaign_goal`
whose relationship to CTR is non-stationary), **not** a shift in the overall CTR base rate.
This is the regime where (a) importance weighting and (b) leaning on stationary signals
(advertiser history — already in the model) are the textbook responses, and where simply
deleting the drifting feature is *not* expected to help. All three predictions held (§4).

---

## 3. Neural nets — do they "predict the future" better? (the framing question)

Two factual corrections that shaped the experiments, then the result.

- The current LightGBM is **not a classifier** — it's a multi-output *regressor* (one GBDT per
  PCA component) predicting a continuous curve. "Predict the future" is a property of the
  **train/test split** (the forward holdout), **not** of trees-vs-nets. Swapping in a neural net
  does not by itself unlock future prediction; both are supervised regressors on the same data.
- A useful math fact: PCA components are **orthonormal**, so squared error in the 8-score space
  equals squared error of the reconstructed log-curve. That makes *unweighted raw-score MSE* the
  principled NN training loss — which is what `experiments/mlp.py` minimizes (with the same
  tail+recency sample weights LightGBM uses).

**Result (2022-11-01 forward holdout):**

```
lightgbm (baseline)   RMSE=0.003953  all=+19.6  tail=+20.3  mid=+9.6   yda=+26.3  lap=+19.2
mlp (3-seed avg)      RMSE=0.003963  all=+18.5  tail=+18.5  mid=+17.9  yda=+27.0  lap=+18.0
```

The MLP **loses overall (+18.5 vs +19.6)** — precisely the tabular-ML consensus (tree
ensembles still beat plain deep nets on medium heterogeneous tables). **But look at the buckets:**
the MLP is **+8pp better in the middle** (the everyday low-CTR ad) and slightly better on yda,
while giving up ~2pp on the high-CTR tail. The tree nails the tail; the net is smoother in the
middle. That is textbook GBDT/NN complementarity — and the middle + yda are exactly the buckets
VALIDATION.md flags as worst-hit by forward drift. So the right use of the net is **not to replace
the tree but to ensemble with it.**

### The ensemble — validated across 3 cutoffs (the credible result)

Blending the two families' predicted scores (`0.5·LightGBM + 0.5·MLP`), refit at the three
cutoffs VALIDATION.md uses (2022-08-01 / 2022-11-01 / 2023-01-01), **mean across cutoffs**:

| method (mean of 3 cutoffs) | grader RMSE | all | tail | mid | yda | lap |
|---|---|---|---|---|---|---|
| **baseline (LightGBM, live config)** | 0.004062 | +18.9 | +19.8 | +8.3 | +26.5 | +18.5 |
| mlp (3-seed avg) | 0.004051 | +18.4 | +18.8 | +13.3 | +26.0 | +18.0 |
| **ensemble 0.5/0.5** | **0.004046** | +18.9 | +19.3 | **+15.0** | **+28.2** | +18.4 |

- **Grader RMSE improves at every cutoff** (mean −0.4%). The grader metric *is* RMSE, so this
  is the number that counts.
- **The middle bucket nearly doubles (+8.3 → +15.0)** and **yda gains +1.7pp (+26.5 → +28.2)**.
  These are the two buckets temporal drift hits hardest, and yda is **~75% of the real predict
  set** — so the ensemble's gains land disproportionately where the real submission is scored.
- The cost is ~0.5pp of pooled *tail* skill (the tree alone is the best pure-tail model). The
  pooled "all" skill is flat because it is tail-dominated; every *disaggregated* bucket that
  matters for the real set is up or flat.
- **At the longest horizon (9-month, 2022-08-01) the MLP even beats LightGBM outright** on both
  RMSE and all-skill — evidence that the smoother neural function **extrapolates further forward**
  better than trees, which overfit time-specific structure (recall leaves had to be cut 95→47).
  The further into the future you predict, the more the net earns its place.

---

## 4. Attacking the drift directly (`experiments/drift_attack.py`)

All on the 2022-11-01 forward holdout, vs the +19.6% baseline:

| method | all-skill | note |
|---|---|---|
| baseline | +19.6 | live config |
| **no recency-decay** | +15.9 | **recency weighting is worth ~3.7pp — the biggest lever already in place** |
| feature-importance-weighting (alone) | +15.7 | ≈ recency alone (no tail boost) |
| **feat-IW + recency (on top of config)** | **+19.8** | small, consistent gain (RMSE 0.003953→0.003936) |
| drop `campaign_goal` (biggest drifter) | +18.8 | **−0.8pp — deleting the drifting feature HURTS** |
| drop `campaign_goal` + IW | +19.1 | still below baseline |

**Covariate-shift importance weighting** (`methods.feature_iw`) is a leakage-safe, feature-based
generalization of the hand-tuned date recency-decay: within the training window it learns
`p(late | features)` and up-weights training ads that *look like* the recent edge of the
distribution. It gives a **small but consistent** gain stacked on the existing recency decay.
Dropping the most non-stationary feature **hurts** — confirming the diagnosis that `campaign_goal`
is non-stationary *but still net-informative forward*. The drift is mostly irreducible, as the
docs already concluded; these methods trim the edges, they don't close the ~9pp temporal-vs-random gap.

**Validated across 3 cutoffs**, the importance-weighting gain is small but real on the
tail/pooled side (mean: RMSE 0.004062→**0.004049**, all +18.9→**+19.1**, tail +19.8→**+20.1**),
while costing a little in the middle. It is genuinely *complementary* to the ensemble, which
wins the middle and yda — so the strongest recipe stacks them (see §5).

---

## 5. What I'd ship / recommend

**Ship the ensemble.** The night's best, validated across all three forward cutoffs, is

> **`cumu_ctr = decode( 0.6 · iw-LightGBM_scores + 0.4 · MLP_scores )`**

i.e. blend the live LightGBM (with covariate-shift importance weighting added to its sample
weights) with the torch MLP, 0.6/0.4 in PCA-score space. Mean over the 3 cutoffs:

| recipe | grader RMSE | all | tail | mid | yda | lap |
|---|---|---|---|---|---|---|
| baseline (LightGBM, live) | 0.004062 | +18.9 | **+19.8** | +8.3 | +26.5 | +18.5 |
| **0.6·iw-LightGBM + 0.4·MLP** | **0.004041** | **+19.1** | +19.5 | **+13.8** | **+27.9** | **+18.6** |
| 0.5·LightGBM + 0.5·MLP (no iw, simpler) | 0.004046 | +18.9 | +19.3 | +15.0 | +28.2 | +18.4 |

- **It beats the baseline on the real grader metric (−0.5% RMSE) and on all-skill, middle, yda,
  and lap** — regressing only ~0.3pp on the pooled tail. Because the true predict set is ~75% yda
  and graded on RMSE, the real-submission expectation is a *net improvement*.
- It is produced by `experiments/predict_ensemble.py`, which writes
  `data/7. result_group_X_filled_ensemble.csv` **without touching your live filled file** — so you
  can diff/submit it side-by-side. The plain (no-iw) 0.5 blend is a fine simpler fallback that
  captures most of the gain.
- **The submission file was produced and sanity-checked tonight** (trained on all 133,925 ads,
  filled all 1,289,400 result rows, 0 NaNs). It correlates 0.83 with the live LightGBM file and
  has a slightly **fatter upper tail** (p99 0.0173 vs 0.0108; max 0.063 vs 0.029) — the MLP is
  less tail-conservative than the heavily tail-tuned tree. That is benign given the holdout RMSE
  win, but if you want to be conservative, clip predicted CTR at a sane cap (e.g. the training
  max) before submitting.

**Do not replace LightGBM with a neural net.** Alone the MLP is worse on the tail and pooled
skill. Its value is purely as an ensemble partner / middle-and-far-horizon specialist.

**Honest expectation management.** None of this "solves" forward prediction. The temporal-vs-random
gap is ~9pp and is mostly **structural** (covariate + concept shift, §2). Recency weighting (~3.7pp,
already in the model) remains the single biggest lever; the ensemble adds a real but modest few
tenths of a pp of grader RMSE on top, concentrated in the buckets that matter for the real set.

### Ideas worth a follow-up night (ranked)
1. **Per-bucket / level-conditioned blend weight** — the optimal LGBM:MLP ratio differs by curve
   level (tree better on the tail, net better in the middle). A small gate that sets the blend
   weight from predicted level could capture both without the tail cost. (Risk: overfitting; validate
   per-cutoff.)
2. **A proper tabular-transformer (FT-Transformer / TabR)** instead of a plain MLP — the literature
   says these are the architectures that actually close the gap to GBDTs; the MLP is the weakest
   neural baseline and it already ensembles well, so a stronger net is the obvious next lever.
3. **Time-slice GroupDRO** — train to minimize the *worst* time-slice loss rather than the average;
   a principled attack on concept drift I scoped but did not run tonight.
4. **Re-fit the curve PCA on recent-only curves** — the codebook itself is fit on all-time data; a
   recency-weighted PCA basis might represent the forward curve shapes better (the shape components
   PC1+ are where the two platforms diverge).

## 6. Reproduce

```bash
PY=<the night venv python>   # /home/claudeuser/.claude/jobs/.../venv/bin/python (rebuild if gone)
$PY experiments/run_baseline.py      # reproduce the LightGBM forward-holdout baseline
$PY experiments/drift_diagnosis.py   # covariate/label shift diagnosis
$PY experiments/drift_attack.py      # importance weighting / feature ablation
$PY experiments/run_nn.py            # MLP vs LightGBM + ensemble (single cutoff)
$PY experiments/multi_cutoff.py      # validate winners across 3 cutoffs
$PY experiments/final_recipe.py      # stack importance-weighting + ensemble (the winner)
$PY experiments/predict_ensemble.py  # write the production ensemble submission CSV
```

All experiment code is under `experiments/`: `harness.py` (model-agnostic forward-holdout eval),
`mlp.py` (torch MLP), `methods.py` (covariate-shift importance weighting), and the run scripts
above. Nothing in the original repo was modified — the work is additive and side-by-side.
