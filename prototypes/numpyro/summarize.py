"""Summarize coverage, width, cost and diagnostics: python summarize.py [suffix]."""

import glob
import sys

import numpy as np
from common import S_GRID, coverage

suf = sys.argv[1] if len(sys.argv) > 1 else ""
files = sorted(f for f in glob.glob(f"out/ds*{suf}.npz") if suf or "_bad" not in f)
ds = [np.load(f) for f in files]
print(len(ds), "datasets:", [f.split("/")[-1] for f in files])
ends = (S_GRID < 0.2) | (S_GRID > 0.8)
for m in ["fixed", "marginal", "svi_normal", "svi_mvn", "svi_laplace"]:
    have = [d for d in ds if m in d]
    if not have:
        print(f"{m:12s} none")
        continue
    offs = np.stack([d[m] for d in have])
    c = coverage(offs)
    t = np.mean([float(d["t_" + m]) for d in have])
    rms = np.sqrt(np.nanmean(np.nanmean(offs, 1) ** 2))
    extra = ""
    if "ess_" + m in have[0]:
        g = lambda k: np.array([float(d[k + m]) for d in have])
        extra = (
            f" ess_min med {np.median(g('ess_')):.0f} (min {g('ess_').min():.0f})"
            f" track-ESS min med {np.median(g('essx_')):.0f} (min {g('essx_').min():.0f})"
            f" rhat_max {g('rhat_').max():.2f} div/1000 med {np.median(g('div_')):.0f}"
            f" sigma {np.median(g('sigma_')):.3f}"
        )
    print(
        f"{m:12s} n={len(have):2d} 68%: {c['68'].mean():.2f} (ends {c['68'][ends].mean():.2f})"
        f"  95%: {c['95'].mean():.2f} (ends {c['95'][ends].mean():.2f})"
        f"  hw68 {np.nanmedian(c['w68']):.3f} hw95 {np.nanmedian(c['w95']):.3f}"
        f"  centre rms {rms:.3f}  t {t:.1f}s{extra}"
    )
single = np.stack([d["single"][0] for d in ds])
print(
    f"single fit rms error {np.sqrt(np.nanmean(single**2)):.3f}, t {np.mean([float(d['t_single']) for d in ds]):.1f}s"
)
