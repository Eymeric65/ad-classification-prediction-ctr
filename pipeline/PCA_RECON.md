# PCA compression: is it biased toward outlier curves?

**Question.** PCA in log-CTR space minimises *total* reconstruction variance, so a few
extreme curves can pull the mean/components toward themselves. Does the compression
reconstruct high-CTR / outlier ads worse than the bulk — i.e. is it "biased in the
outliers"?

**How to reproduce.** [`pipeline/pca_recon.py`](pca_recon.py) — encodes every ad's real
curve to PCA scores and decodes it straight back through the saved codebook. No model,
no train/test split: this is the *pure compression floor*, the best any downstream
predictor could ever hit.

```bash
uv run python -m pipeline.pca_recon
```

Outputs two PNGs in [`pipeline/plots/`](plots/):
- `recon_curves.png` — 9 curves sampled best→worst round-trip error (actual vs PCA reconstruction).
- `recon_diagnostics.png` — error histogram, error-vs-magnitude scatter, mean error per final-CTR decile.

## Result: yes, biased toward the high-CTR tail — but negligibly

Measured over 49,149 curves, PCA(8), log-CTR space, normalized lifetime:

| signal | value |
|---|---|
| mean recon RMSE | 0.00007 CTR |
| median recon RMSE | 0.00003 CTR |
| p99 recon RMSE | 0.00070 CTR |
| worst 1% of curves carry | **18.3%** of all reconstruction error |
| top final-CTR decile vs bottom decile | **28.6×** worse (0.00028 vs 0.00001 CTR) |

The error-vs-magnitude scatter and the per-decile bars both rise monotonically with the
curve's final cumulative CTR. So the compression **is** sacrificing the high-CTR outliers
to fit the low-CTR bulk — classic variance-dominated PCA behaviour, and the log-transform
softens but doesn't remove it.

## Decision: PCA is fine for now — don't fix it yet

The bias is real and structural, but tiny in absolute terms: even the *worst* magnitude
decile averages 0.00028 CTR error (~0.03 percentage points, on curves whose level is
~1.7%). More importantly it sits **orders of magnitude below** the feature model's
curve-space RMSE — the predictor is nowhere near limited by the compression floor.

The outlier bias only becomes worth chasing once the predictor is good enough to be
bottlenecked by the PCA itself. If/when that day comes, the levers are:
- weight the PCA fit toward the extremes,
- fit in a different space (e.g. a different transform than log),
- or first check whether the top-decile curves are genuine or data artifacts
  (eyeball the `recon_curves.png` p98 panel).
