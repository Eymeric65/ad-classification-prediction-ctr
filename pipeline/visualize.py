"""Visualise held-out test predictions for the curve-score model.

Reuses pipeline.run.run() so the fit + split match `python -m pipeline.run` exactly.
Writes two PNGs into pipeline/plots/ :
  curves.png  - sample test ads spanning the error range: actual vs predicted curve
  scores.png  - PC0 predicted-vs-actual scatter + per-curve RMSE distribution

Run from the repo root:
  uv run python -m pipeline.visualize
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from pipeline.run import run
from compress_curves import decode

OUT = os.path.join(os.path.dirname(__file__), "plots")

# Shared y-axis for the curve panels, so shapes are comparable instead of each
# subplot auto-zooming into its own noise. CTR is a fraction (0.04 == 4%).
CTR_YLIM = (0.0, 0.04)


def main():
    r = run()
    ccfg, cb = r["ccfg"], r["cb"]
    grid = np.asarray(cb["grid"])

    # Decode test scores back to real cumulative-CTR curves to judge in curve space.
    true = decode(r["Yte"], ccfg, cb)                               # (n_test, grid_points)
    pred = decode(r["Ypred"], ccfg, cb)
    mean_curve = decode(np.zeros((1, len(r["pc_cols"]))), ccfg, cb)[0]   # the "predict the average" floor

    per_curve_rmse = np.sqrt(((true - pred) ** 2).mean(axis=1))
    base_rmse = np.sqrt(((true - mean_curve) ** 2).mean(axis=1))
    xlabel = "normalized lifetime" if ccfg["time_axis"] == "normalized_lifetime" else "days (capped, normalized)"
    os.makedirs(OUT, exist_ok=True)

    # ---- figure 1: 9 test ads sampled across the error range (best -> worst) ----
    order = np.argsort(per_curve_rmse)
    pcts = np.linspace(2, 98, 9)
    idx = order[(pcts / 100 * (len(order) - 1)).astype(int)]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for ax, i, p in zip(axes.ravel(), idx, pcts):
        ax.plot(grid, true[i], color="#1f77b4", lw=2, label="actual")
        ax.plot(grid, pred[i], color="#d62728", lw=2, ls="--", label="predicted")
        ax.plot(grid, mean_curve, color="gray", lw=1, ls=":", label="mean curve")
        ax.set_title(f"p{int(p)} error   rmse={per_curve_rmse[i]:.4f}", fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylim(*CTR_YLIM)         # shared y -> applies to every panel
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Held-out curves: actual vs predicted  ({r['pcfg']['model']})   —   left=best, right=worst",
                 fontsize=12)
    fig.supxlabel(xlabel)
    fig.supylabel("cumulative CTR")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curves.png"), dpi=120)

    # ---- figure 2: PC0 scatter (it IS ~98% of the curve) + error distribution ----
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))

    t0, p0 = r["Yte"][:, 0], r["Ypred"][:, 0]
    axa.scatter(t0, p0, s=6, alpha=0.25, color="#1f77b4")
    lim = [min(t0.min(), p0.min()), max(t0.max(), p0.max())]
    axa.plot(lim, lim, "k--", lw=1)
    r2 = 1 - ((t0 - p0) ** 2).sum() / ((t0 - t0.mean()) ** 2).sum()
    axa.set_title(f"PC0 predicted vs actual   (R² = {r2:.3f})")
    axa.set_xlabel("actual PC0")
    axa.set_ylabel("predicted PC0")
    axa.grid(alpha=0.3)

    axb.hist(base_rmse, bins=60, alpha=0.5, color="gray",
             label=f"mean-curve baseline (mean={base_rmse.mean():.4f})")
    axb.hist(per_curve_rmse, bins=60, alpha=0.6, color="#d62728",
             label=f"{r['pcfg']['model']} (mean={per_curve_rmse.mean():.4f})")
    axb.set_title("Per-curve RMSE distribution (test set)")
    axb.set_xlabel("curve RMSE (CTR)")
    axb.set_ylabel("# test ads")
    axb.legend()
    axb.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "scores.png"), dpi=120)

    print(f"wrote {OUT}/curves.png and {OUT}/scores.png  ({len(per_curve_rmse):,} test ads)")


if __name__ == "__main__":
    main()
