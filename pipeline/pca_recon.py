"""Inspect the PCA compression itself — is it biased toward outlier curves?

This is NOT the prediction model (see pipeline.visualize for that). Here there is
no train/test split and no feature model: we just encode each ad's real curve to
its PCA scores and decode it straight back, and ask how faithful that round-trip
is across the dataset — especially on the extremes.

Why this matters: PCA minimises *total* reconstruction variance in log-CTR space,
so a few extreme curves can dominate the mean/components. The symptom is a
reconstruction error that grows with curve magnitude (high-CTR / oddly-shaped ads
reconstructed worst) — i.e. the compression is "biased in the outliers".

Writes two PNGs into pipeline/plots/ :
  recon_curves.png       sample curves across the recon-error range: actual vs PCA round-trip
  recon_diagnostics.png  error distribution + error-vs-magnitude scatter + per-decile error

Run from the repo root:
  uv run python -m pipeline.pca_recon
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Make repo-root modules importable whether run as `-m pipeline.pca_recon` or directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from compress_curves import build_matrix, decode, load_codebook, load_settings

OUT = os.path.join(os.path.dirname(__file__), "plots")


def roundtrip(laps, ccfg, cb):
    """Encode every ad's real curve to PCA scores and decode straight back.

    Returns (true, rec, scores, ad_ids) — true/rec are curves in CTR space on the
    common grid, scores are the per-ad PCA coordinates. No model, no split: this is
    the pure compression floor (the best any downstream predictor could ever do).
    """
    mean, comps = np.asarray(cb["mean"]), np.asarray(cb["components"])
    X, ad_ids = build_matrix(laps, ccfg)          # (n_ads, grid_points), in log space if configured
    scores = (X - mean) @ comps.T                 # encode
    rec = decode(scores, ccfg, cb)                # decode back to CTR space
    true = np.exp(X) if ccfg["log_transform"] else X
    return true, rec, scores, ad_ids


def main():
    ccfg = load_settings()
    cb = load_codebook(ccfg)
    grid = np.asarray(cb["grid"])
    laps = pd.read_parquet(ccfg["paths"]["laps"])

    print(f"round-tripping {laps[ccfg['columns']['id']].nunique():,} curves "
          f"through PCA({ccfg['n_components']}) ...")
    true, rec, scores, ad_ids = roundtrip(laps, ccfg, cb)

    # Per-curve reconstruction error (CTR space) — the compression floor.
    recon_rmse = np.sqrt(((true - rec) ** 2).mean(axis=1))
    # Magnitude proxy for "outlier-ness": the curve's final cumulative CTR level.
    final_ctr = true[:, -1]
    xlabel = "normalized lifetime" if ccfg["time_axis"] == "normalized_lifetime" else "days (capped, normalized)"
    os.makedirs(OUT, exist_ok=True)

    # ---- figure 1: 9 curves sampled across the recon-error range (best -> worst) ----
    # y auto-scales per panel (sharey=False) ON PURPOSE: outliers have huge CTR and a
    # shared axis would flatten the typical curves into invisibility.
    order = np.argsort(recon_rmse)
    pcts = np.linspace(2, 98, 9)
    idx = order[(pcts / 100 * (len(order) - 1)).astype(int)]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for ax, i, p in zip(axes.ravel(), idx, pcts):
        ax.plot(grid, true[i], color="#1f77b4", lw=2, label="actual")
        ax.plot(grid, rec[i], color="#d62728", lw=2, ls="--", label="PCA round-trip")
        ax.set_title(f"p{int(p)} recon error   rmse={recon_rmse[i]:.4f}\nfinal CTR={final_ctr[i]:.3%}",
                     fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("PCA round-trip fidelity: actual vs reconstructed   —   left=best, right=worst",
                 fontsize=12)
    fig.supxlabel(xlabel)
    fig.supylabel("cumulative CTR")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recon_curves.png"), dpi=120)

    # ---- figure 2: is the error biased toward big / outlier curves? ----
    fig2, (axa, axb, axc) = plt.subplots(1, 3, figsize=(18, 5))

    # (a) distribution of the reconstruction floor.
    axa.hist(recon_rmse, bins=80, color="#1f77b4", alpha=0.7)
    axa.axvline(recon_rmse.mean(), color="k", ls="--", lw=1,
                label=f"mean={recon_rmse.mean():.4f}")
    axa.axvline(np.percentile(recon_rmse, 99), color="#d62728", ls=":", lw=1,
                label=f"p99={np.percentile(recon_rmse, 99):.4f}")
    axa.set_title("PCA reconstruction RMSE per curve")
    axa.set_xlabel("recon RMSE (CTR)")
    axa.set_ylabel("# ads")
    axa.legend(fontsize=8)
    axa.grid(alpha=0.3)

    # (b) the smoking gun: does error grow with curve magnitude?
    #     A rising cloud = PCA is sacrificing the high-CTR outliers (biased fit).
    axb.scatter(final_ctr, recon_rmse, s=5, alpha=0.15, color="#1f77b4")
    axb.set_xscale("log")
    axb.set_title("recon error vs curve magnitude")
    axb.set_xlabel("final cumulative CTR (log scale)")
    axb.set_ylabel("recon RMSE (CTR)")
    axb.grid(alpha=0.3, which="both")

    # (c) same signal, de-noised: mean error per magnitude decile.
    deciles = pd.qcut(final_ctr, 10, labels=False, duplicates="drop")
    ndec = deciles.max() + 1
    by_dec = [recon_rmse[deciles == d].mean() for d in range(ndec)]
    med_ctr = [np.median(final_ctr[deciles == d]) for d in range(ndec)]
    axc.bar(range(ndec), by_dec, color="#d62728", alpha=0.7)
    axc.set_xticks(range(ndec))
    axc.set_xticklabels([f"{c:.2%}" for c in med_ctr], rotation=45, ha="right", fontsize=7)
    axc.set_title("mean recon error by final-CTR decile")
    axc.set_xlabel("magnitude decile (median final CTR)")
    axc.set_ylabel("mean recon RMSE (CTR)")
    axc.grid(alpha=0.3, axis="y")

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "recon_diagnostics.png"), dpi=120)

    # ---- console summary: quantify the bias ----
    top1 = recon_rmse >= np.percentile(recon_rmse, 99)
    print(f"\nrecon RMSE (CTR):  mean={recon_rmse.mean():.5f}  "
          f"median={np.median(recon_rmse):.5f}  p99={np.percentile(recon_rmse, 99):.5f}  "
          f"max={recon_rmse.max():.5f}")
    print(f"worst 1% of curves carry {recon_rmse[top1].sum() / recon_rmse.sum() * 100:.1f}% "
          f"of all reconstruction error")
    print(f"mean recon error  — bottom magnitude decile: {by_dec[0]:.5f}   "
          f"top decile: {by_dec[-1]:.5f}   (ratio {by_dec[-1] / by_dec[0]:.1f}x)")
    print(f"\nwrote {OUT}/recon_curves.png and {OUT}/recon_diagnostics.png  ({len(recon_rmse):,} ads)")


if __name__ == "__main__":
    main()
