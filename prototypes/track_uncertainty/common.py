"""Shared mock data and coverage measurement for the track-uncertainty prototypes."""

import jax.numpy as jnp
import numpy as np

SIGMA = 0.2  # true half-width of the stream
S_GRID = np.linspace(0.05, 0.95, 46)  # positions along the true track, as arc fraction


def true_curve(s):
    """Sine-shaped arc, s in [0, 1]. Returns (x, y)."""
    x = 10.0 * s
    y = 2.0 * np.sin(1.5 * np.pi * s)
    return x, y


def _tangent_normal(s):
    eps = 1e-4
    x0, y0 = true_curve(s - eps)
    x1, y1 = true_curve(s + eps)
    t = np.stack([x1 - x0, y1 - y0], -1)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    n = np.stack([-t[..., 1], t[..., 0]], -1)
    return t, n


def make_data(seed, n=400):
    """N samples about the true curve with Gaussian scatter across it, velocity along it."""
    rng = np.random.default_rng(seed)
    # uniform in arc length: dense table inverse
    ss = np.linspace(0, 1, 4001)
    x, y = true_curve(ss)
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s = np.interp(rng.uniform(0, arc[-1], n), arc, ss)
    t, nrm = _tangent_normal(s)
    x, y = true_curve(s)
    off = rng.normal(0, SIGMA, (n, 1)) * nrm
    q = np.stack([x, y], -1) + off
    p = t + rng.normal(0, 0.1, (n, 2))
    pos = {"x": jnp.asarray(q[:, 0]), "y": jnp.asarray(q[:, 1])}
    vel = {"x": jnp.asarray(p[:, 0]), "y": jnp.asarray(p[:, 1])}
    return pos, vel


def normal_offsets(curve_xy):
    """Signed offset of a sampled curve from the truth, along the true normal, at S_GRID.

    For each true point c(s), find where the sampled curve crosses the true normal
    line through c(s) (nearest crossing). NaN where it doesn't (track too short).
    curve_xy: (M, 2) densely sampled polyline.
    """
    xt, yt = true_curve(S_GRID)
    t, nrm = _tangent_normal(S_GRID)
    out = np.full(S_GRID.shape, np.nan)
    a, b = curve_xy[:-1], curve_xy[1:]
    for i in range(len(S_GRID)):
        c = np.array([xt[i], yt[i]])
        # along-tangent coordinate of each vertex relative to c: crossing where it changes sign
        u0 = (a - c) @ t[i]
        u1 = (b - c) @ t[i]
        cross = np.nonzero((u0 <= 0) & (u1 > 0) | (u0 > 0) & (u1 <= 0))[0]
        if cross.size == 0:
            continue
        f = u0[cross] / (u0[cross] - u1[cross])
        pts = a[cross] + f[:, None] * (b[cross] - a[cross])
        d = (pts - c) @ nrm[i]
        out[i] = d[np.argmin(np.abs(d))]
    return out


def coverage(offsets):
    """offsets: (R datasets, K samples, S). Returns dict of coverage and widths per s.

    A band misses when the true curve (offset 0) lies outside the central 68/95%
    of the samples. A sample that doesn't reach s (NaN) counts as not covering.
    """
    res = {}
    for lev, (lo, hi) in {"68": (16, 84), "95": (2.5, 97.5)}.items():
        qlo = np.nanpercentile(offsets, lo, axis=1)
        qhi = np.nanpercentile(offsets, hi, axis=1)
        reach = np.mean(np.isfinite(offsets), axis=1) > 0.5
        cov = (qlo <= 0) & (qhi >= 0) & reach
        res[lev] = cov.mean(0)
        res["w" + lev] = np.nanmedian((qhi - qlo) / 2, axis=0)
    return res
