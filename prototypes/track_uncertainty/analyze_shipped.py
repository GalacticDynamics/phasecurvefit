"""Summarize out_shipped/ds*.npz: interval and tube coverage of the shipped bootstrap."""

import glob

import numpy as np
from common import S_GRID, coverage

ds = [np.load(f) for f in sorted(glob.glob("out_shipped/ds*.npz"))]
A = np.stack([d["A"] for d in ds])  # (R, K, S)
ref = np.stack([d["ref"] for d in ds])  # (R, S)
c = coverage(A)
print(
    f"{len(ds)} datasets, {A.shape[1]} members, {np.mean([d['t'] for d in ds]):.1f} s per fit"
)
for lev in ("68", "95"):
    r = np.stack([d["r" + lev] for d in ds])
    tube = np.abs(ref) <= r  # NaN radius or NaN offset counts as a miss
    print(
        f"{lev}%: interval coverage {c[lev].mean():.2f} (half-width {np.nanmedian(c['w' + lev]):.3f}),"
        f" tube coverage {tube.mean():.2f} (median radius {np.nanmedian(r):.3f},"
        f" NaN radius at {np.isnan(r).mean():.2f} of points)"
    )
print(f"reference rms error {np.sqrt(np.nanmean(ref**2)):.3f}")
print(
    f"member spread {np.nanmean(np.nanstd(A, axis=1)):.3f}, scatter of the reference across datasets {np.nanmean(np.nanstd(ref, axis=0)):.3f}"
)
bias = np.nanmean(ref, axis=0)
print(
    f"bias rms {np.sqrt(np.nanmean(bias**2)):.3f}; largest at s =",
    ", ".join(
        f"{S_GRID[i]:.2f} ({bias[i]:+.3f})" for i in np.argsort(-np.abs(bias))[:3]
    ),
)
print("68% interval coverage along s:", " ".join(f"{v:.1f}" for v in c["68"][::3]))
