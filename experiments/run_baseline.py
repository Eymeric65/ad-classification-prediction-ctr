"""Reproduce the LightGBM forward-holdout baseline through the harness."""
import sys
sys.path.insert(0, "/workspace/experiments")
from harness import (load_everything, eval_split, temporal_indices,
                     random_indices, lgbm_factory, show)

D = load_everything()
print(f"assembled {len(D['df']):,} ads  ({(D['platform']=='yda').sum():,} yda / "
      f"{(D['platform']=='lap').sum():,} lap)  feat={len(D['cat'])}cat+{len(D['num'])}num")
print(f"min_date spans {D['date'].min().date()} -> {D['date'].max().date()}\n")

fac = lgbm_factory(D["pcfg"])
for cutoff in ["2022-11-01"]:
    tr, te = temporal_indices(D, cutoff)
    frac = len(te) / len(D["df"])
    print(f"### cutoff {cutoff}  ->  train {len(tr):,} / test {len(te):,} ({frac:.0%})")
    show("TEMPORAL", eval_split(D, tr, te, fac))
    rtr, rte = random_indices(D, frac)
    show("RANDOM (control)", eval_split(D, rtr, rte, fac))
    print()
