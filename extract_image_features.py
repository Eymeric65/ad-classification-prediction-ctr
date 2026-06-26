"""Turn each creative's segmentation mask into spatial features.

The provided images are *resolved* masks: every pixel is painted one of 10 flat
palette colors, each color a content category (see the "Image color master" slide
and README). From each mask we derive, per category:
  - coverage           fraction of the image that is this category   (0..1)
  - center of mass x,y  normalized centroid of the category's pixels  (0..1, NaN if absent)

Coverage overlaps the annotation `area_*` buckets, but the centroid (where on the
banner the person / text / product sits) is new signal the annotations don't carry.

Both platforms ship masks (`data/images_lap/` + `data/images_yda/`), so every ad
gets image features — these are universal, not a one-platform signal.

Output: `data/image_features.parquet`, one row per `filename` (the creative is the
image; the same file can back several ad_ids), columns `img_cov_/img_cx_/img_cy_<cat>`.
The run is INCREMENTAL: filenames already in the cache are skipped, so re-running after
new masks arrive only processes the new ones. Pass --rebuild to recompute everything.

Usage:
  uv run python extract_image_features.py            # process new masks, update cache
  uv run python extract_image_features.py --rebuild  # recompute all masks from scratch
  uv run python extract_image_features.py <file.png> # debug one mask, print features
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

IMG_DIRS = ["data/images_lap", "data/images_yda"]
OUT_PATH = "data/image_features.parquet"

# (R, G, B) -> category short name. Names mirror the annotation `area_*` columns so the
# two feature families read consistently. Order matches the color-master slide.
PALETTE = [
    ((255, 0, 0),     "person_photo"),    # Person (Photo)
    ((128, 0, 0),     "person_illust"),   # Person (Drawing)
    ((0, 255, 0),     "animal_photo"),     # Animals (Photo)
    ((0, 128, 0),     "animal_illust"),    # Animals (Drawing)
    ((0, 0, 255),     "product_photo"),    # Object (Photo)
    ((0, 0, 128),     "product_illust"),   # Object (Drawing)
    ((255, 255, 0),   "logo"),             # Symbol
    ((255, 255, 255), "text"),             # Text
    ((128, 128, 128), "other_photo"),      # Other_Photo
    ((0, 0, 0),       "other_illust"),     # Other (Drawing)
]
CATS = [name for _, name in PALETTE]
# Pack each palette RGB into one int24 so a mask becomes a single 2-D int array.
CODES = np.array([r * 65536 + g * 256 + b for (r, g, b), _ in PALETTE], dtype=np.int64)

FEATURE_COLS = ([f"img_cov_{c}" for c in CATS]
                + [f"img_cx_{c}" for c in CATS]
                + [f"img_cy_{c}" for c in CATS])


def features_for_mask(arr):
    """RGB uint8 array (H, W, 3) -> dict of the 30 feature values."""
    h, w = arr.shape[:2]
    code = (arr[:, :, 0].astype(np.int64) * 65536
            + arr[:, :, 1].astype(np.int64) * 256
            + arr[:, :, 2].astype(np.int64))
    n = float(h * w)
    cov, cx, cy = {}, {}, {}
    for cc, name in zip(CODES, CATS):
        ys, xs = np.nonzero(code == cc)
        k = xs.size
        cov[name] = k / n
        if k:
            cx[name] = (xs.mean() + 0.5) / w   # pixel-center fraction, in (0, 1)
            cy[name] = (ys.mean() + 0.5) / h
        else:
            cx[name] = np.nan
            cy[name] = np.nan
    out = {}
    out.update({f"img_cov_{c}": cov[c] for c in CATS})
    out.update({f"img_cx_{c}": cx[c] for c in CATS})
    out.update({f"img_cy_{c}": cy[c] for c in CATS})
    return out


def _process(item):
    path, fname = item
    try:
        arr = np.asarray(Image.open(path).convert("RGB"))
        row = features_for_mask(arr)
        row["filename"] = fname
        return row
    except Exception as e:                       # don't let one bad file kill the run
        print(f"  skip {fname}: {e}", file=sys.stderr)
        return None


def _all_masks():
    """[(full_path, filename)] across every image dir. filename is the join key."""
    items = []
    for d in IMG_DIRS:
        if os.path.isdir(d):
            items += [(os.path.join(d, f), f) for f in os.listdir(d) if f.endswith(".png")]
    return items


def main(rebuild=False):
    items = _all_masks()
    cached = pd.DataFrame()
    if os.path.exists(OUT_PATH) and not rebuild:
        cached = pd.read_parquet(OUT_PATH)
        done = set(cached["filename"])
        items = [it for it in items if it[1] not in done]
        print(f"cache has {len(done):,} masks; {len(items):,} new to process ...")
    else:
        print(f"extracting image features from {len(items):,} masks ...")

    rows = []
    if items:
        with Pool() as pool:
            for i, row in enumerate(pool.imap_unordered(_process, items, chunksize=64)):
                if row is not None:
                    rows.append(row)
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1:,}/{len(items):,}")

    df = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True) if rows else cached
    df = df[["filename"] + FEATURE_COLS]
    df.to_parquet(OUT_PATH, index=False)
    matched = df[FEATURE_COLS[:len(CATS)]].gt(0).any(axis=1).sum()
    print(f"  wrote {OUT_PATH}: {len(df):,} masks x {len(FEATURE_COLS)} features "
          f"({matched:,} non-empty)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg.endswith(".png"):               # debug one file
        path = next((os.path.join(d, arg) for d in IMG_DIRS
                     if os.path.exists(os.path.join(d, arg))), None)
        arr = np.asarray(Image.open(path).convert("RGB"))
        for k, v in features_for_mask(arr).items():
            print(f"  {k:22s} {v:.4f}")
    else:
        main(rebuild=(arg == "--rebuild"))
