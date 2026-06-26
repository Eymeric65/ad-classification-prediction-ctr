"""Quick feature ablation: how much does the area / annotation signal actually buy?

Re-fits the configured model with feature groups toggled on/off, on the SAME
train/test split, and compares curve-space RMSE + PC0 R² against each other and
the mean-curve baseline. Nothing here mutates the config files.

  uv run python -m pipeline.ablate
"""
import copy

import numpy as np

from pipeline.run import run, load_yaml, DATASET_CONFIG
from compress_curves import decode


def evaluate(dcfg):
    r = run(dcfg=dcfg)
    ccfg, cb = r["ccfg"], r["cb"]
    true = decode(r["Yte"], ccfg, cb)
    pred = decode(r["Ypred"], ccfg, cb)
    mean_curve = decode(np.zeros((1, len(r["pc_cols"]))), ccfg, cb)
    rmse = np.sqrt(((true - pred) ** 2).mean())
    base = np.sqrt(((true - mean_curve) ** 2).mean())
    t0, p0 = r["Yte"][:, 0], r["Ypred"][:, 0]
    r2 = 1 - ((t0 - p0) ** 2).sum() / ((t0 - t0.mean()) ** 2).sum()
    return rmse, base, r2


def main():
    base_dcfg = load_yaml(DATASET_CONFIG)
    # (label, feature overrides) — split is identical across all, so this is a clean A/B.
    variants = [
        ("area ON  (full)", {"area": True,  "ann_flags": True}),
        ("area OFF",        {"area": False, "ann_flags": True}),
        ("annotation OFF",  {"area": False, "ann_flags": False}),
    ]
    print(f"{'config':18}  {'curve RMSE':>11}  {'vs base':>8}  {'PC0 R2':>7}")
    print("-" * 50)
    for label, overrides in variants:
        dcfg = copy.deepcopy(base_dcfg)
        dcfg["features"].update(overrides)
        rmse, base, r2 = evaluate(dcfg)
        print(f"{label:18}  {rmse:11.6f}  {(1 - rmse / base) * 100:+7.1f}%  {r2:7.3f}")
    print("-" * 50)
    print(f"baseline (mean curve) curve RMSE = {base:.6f}")


if __name__ == "__main__":
    main()
