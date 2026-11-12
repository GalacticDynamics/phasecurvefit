"""Summarize coverage, band width and cost from out/ds*.npz."""

import glob

import numpy as np
from common import S_GRID, SIGMA, coverage

files = sorted(glob.glob("out/ds*.npz"))
ds = [np.load(f) for f in files]
print(f"{len(ds)} datasets, true width sigma={SIGMA}")
# where along the track: ends (first/last 15% of the grid) vs middle
ends = (S_GRID < 0.2) | (S_GRID > 0.8)
for m, label in [
    ("A", "bootstrap whole pipeline"),
    ("B", "seed ensemble"),
    ("C", "last-layer Laplace"),
]:
    offs = np.stack([d[m] for d in ds])  # (R, K, S)
    c = coverage(offs)
    t = np.mean([float(d["t" + m]) for d in ds])
    rms = np.sqrt(np.nanmean(np.nanmean(offs, 1) ** 2))  # error of the band centre
    print(
        f"\n{label} (K={offs.shape[1]} per dataset, {t:.1f} s "
        f"{'per member' if m != 'C' else 'per dataset on top of one fit'})"
    )
    for lev in ("68", "95"):
        print(
            f"  {lev}% coverage: all {c[lev].mean():.2f}  middle {c[lev][~ends].mean():.2f}"
            f"  ends {c[lev][ends].mean():.2f}   median half-width {np.nanmedian(c['w' + lev]):.3f}"
        )
    print(f"  rms offset of band centre from truth {rms:.3f}")
    print("  68% coverage along s:", " ".join(f"{v:.1f}" for v in c["68"][::3]))
# true error of a single fit, for scale
offsB = np.stack([d["B"][0] for d in ds])
print(
    f"\nsingle-fit rms error {np.sqrt(np.nanmean(offsB**2)):.3f}, "
    f"middle {np.sqrt(np.nanmean(offsB[:, ~ends] ** 2)):.3f}, ends {np.sqrt(np.nanmean(offsB[:, ends] ** 2)):.3f}"
)
