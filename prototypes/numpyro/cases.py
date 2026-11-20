"""Validation cases with a known true track, and a truth metric for any dimension.

Each case returns (q, p, w, truth): positions (N, Dq), velocities (N, Dq), per-star
weights or None, and a dense true polyline (M, Dq).
"""

import numpy as np

S_GRID = np.linspace(0.05, 0.95, 46)


def density(s):
    """Line density of the ``vary`` case (per unit s, unnormalised): a dense clump
    near s = 0.35 and sparse tails, like a stream near its progenitor."""
    return 0.15 + np.exp(-0.5 * ((s - 0.35) / 0.08) ** 2)


def width(s):
    """Scatter of the ``vary`` case: narrow at the clump, fanning out at the far end."""
    return 0.06 + 0.3 * s**2


def _sample_curve(f, rng, n, s_keep=None, pdf=None):
    """n points uniform in arc length along f(s), s in [0, 1]; s_keep masks out s.
    ``pdf(s)`` (max <= 1.15 here) thins them by rejection to that line density."""
    ss = np.linspace(0, 1, 8001)
    c = f(ss)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    s = np.interp(rng.uniform(0, arc[-1], 4 * n), arc, ss)
    if s_keep is not None:
        s = s[s_keep(s)]
    if pdf is not None:
        s = np.interp(rng.uniform(0, arc[-1], 40 * n), arc, ss)
        s = s[rng.uniform(0, 1.15, s.size) < pdf(s)]
    s = s[:n]
    eps = 1e-4
    t = f(s + eps) - f(s - eps)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return s, f(s), t


def _scatter_across(q, t, sigma, rng):
    """Isotropic Gaussian scatter projected off the tangent (scatter across the track)."""
    e = rng.normal(0, sigma, q.shape)
    e -= np.sum(e * t, 1, keepdims=True) * t
    return q + e


def sine(s):
    return np.stack([10 * s, 2 * np.sin(1.5 * np.pi * s)], -1)


def helix(s):
    return np.stack([10 * s, 2 * np.sin(1.5 * np.pi * s), 1.5 * np.cos(np.pi * s)], -1)


def loop(s):
    # prolate trochoid: one loop that crosses itself near s = 0.5
    t = (0.15 + 0.7 * s) * 4 * np.pi
    return np.stack([1.0 * t - 2.5 * np.sin(t), 2.5 * np.cos(t)], -1)


def make(case, seed):
    rng = np.random.default_rng(seed)
    w = None
    if case in ("helix", "sine"):
        f, n, sig = (helix if case == "helix" else sine), 400, 0.2
        s, q, t = _sample_curve(f, rng, n)
    elif case == "loop":
        f, n, sig = loop, 600, 0.15
        s, q, t = _sample_curve(f, rng, n)
    elif case == "gap":
        f, n, sig = sine, 400, 0.2
        s, q, t = _sample_curve(f, rng, n, s_keep=lambda s: (s < 0.45) | (s > 0.55))
    elif case.startswith("image"):
        f, n, sig = sine, 4000, 0.2
        s, q, t = _sample_curve(f, rng, n)
    elif case == "vary":
        f, n = sine, 800
        s, q, t = _sample_curve(f, rng, n, pdf=density)
        sig = width(s)[:, None]
    q = _scatter_across(q, t, sig, rng)
    p = t + rng.normal(0, 0.1, t.shape)
    if case.startswith("image"):
        # bin into 0.1-wide pixels; observations are pixel centres, weights the counts
        px = 0.1
        ij = np.floor(q / px).astype(int)
        uniq, cnt = np.unique(ij, axis=0, return_counts=True)
        q = (uniq + 0.5) * px
        p = np.zeros_like(q)
        w = cnt.astype(float)
        if case == "image_norm":  # weights as a normalised density, peak 1
            w = w / w.max()
        elif case == "image_x5":  # weights five times the counts
            w = 5 * w
    truth = f(np.linspace(0, 1, 4001))
    return q, p, w, truth


def _arc_frac(c):
    a = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    return a / a[-1]


def _track_s(curve, truth, st, step=0.05):
    """True-curve parameter of each vertex of ``curve``, found by walking along it.

    Each vertex is projected onto the truth only within ``step`` of the previous
    vertex's projection, so at a self-crossing the walk stays on its own branch.
    """
    j = np.argmin(np.linalg.norm(truth - curve[0], axis=1))
    out = np.empty(len(curve))
    for i, x in enumerate(curve):
        near = np.nonzero(np.abs(st - st[j]) < step)[0]
        j = near[np.argmin(np.linalg.norm(truth[near] - x, axis=1))]
        out[i] = st[j]
    return out


def offsets(curve, truth, window=0.03):
    """Offsets of a sampled curve from the truth in the truth's normal plane, at S_GRID.

    For each true point c(s) with tangent t, find where the sampled curve crosses the
    hyperplane through c normal to t. Only crossings that project onto the truth within
    ``window`` of s count (see ``_track_s``), so a self-crossing curve's other branch is
    ignored. Returns (S, D-1): the offset in an orthonormal basis of the normal plane,
    NaN where there is no crossing.
    """
    D = truth.shape[1]
    # orient the curve like the truth (a ridge ordering has no preferred direction)
    if np.linalg.norm(curve[0] - truth[0]) > np.linalg.norm(curve[-1] - truth[0]):
        curve = curve[::-1]
    st = _arc_frac(truth)
    u = _track_s(curve, truth, st)
    out = np.full((len(S_GRID), D - 1), np.nan)
    a, b = curve[:-1], curve[1:]
    ua = u[:-1]
    for i, s in enumerate(S_GRID):
        j = np.searchsorted(st, s)
        c = truth[j]
        t = truth[min(j + 1, len(truth) - 1)] - truth[max(j - 1, 0)]
        t /= np.linalg.norm(t)
        # orthonormal basis of the normal plane
        basis = np.linalg.svd(np.eye(D) - np.outer(t, t))[0][:, : D - 1].T
        u0 = (a - c) @ t
        u1 = (b - c) @ t
        cross = np.nonzero(((u0 <= 0) & (u1 > 0)) | ((u0 > 0) & (u1 <= 0)))[0]
        cross = cross[np.abs(ua[cross] - s) < window]
        if cross.size == 0:
            continue
        fr = u0[cross] / (u0[cross] - u1[cross])
        pts = a[cross] + fr[:, None] * (b[cross] - a[cross])
        d = (pts - c) @ basis.T
        out[i] = d[np.argmin(np.linalg.norm(d, axis=1))]
    return out


def coverage(offs):
    """offs: (R datasets, K draws, S, C). Coverage and half-widths, averaged over C."""
    res = {}
    for lev, (lo, hi) in {"68": (16, 84), "95": (2.5, 97.5)}.items():
        qlo = np.nanpercentile(offs, lo, axis=1)
        qhi = np.nanpercentile(offs, hi, axis=1)
        reach = np.mean(np.isfinite(offs), axis=1) > 0.5
        cov = (qlo <= 0) & (qhi >= 0) & reach  # (R, S, C)
        res[lev] = cov.mean((0, 2))
        res["w" + lev] = np.nanmedian((qhi - qlo) / 2, axis=(0, 2))
    return res


if __name__ == "__main__":  # self-check: the truth has zero offset from itself
    for case in ("sine", "helix", "loop", "gap"):
        q, p, w, tr = make(case, 0)
        o = offsets(tr[::3], tr)
        assert np.nanmax(np.abs(o)) < 1e-2, (case, np.nanmax(np.abs(o)))
        assert np.isfinite(o).all(), case
        if tr.shape[1] == 2:  # a copy shifted 0.1 along the normal reads 0.1 everywhere
            t = np.gradient(tr, axis=0)
            t /= np.linalg.norm(t, axis=1, keepdims=True)
            o = offsets((tr + 0.1 * np.stack([-t[:, 1], t[:, 0]], -1))[::3], tr)
            assert np.allclose(np.abs(o), 0.1, atol=5e-3), (case, o.min(), o.max())
        print(case, q.shape, "ok")
    q, p, w, tr = make("image", 0)
    print("image pixels", q.shape, "total count", w.sum())
    q, p, w, tr = make("vary", 0)
    s = _arc_frac(tr)[
        np.argmin(np.linalg.norm(q[:, None] - tr[None, ::4], axis=2), 1) * 4
    ]
    h = np.histogram(s, bins=10, range=(0, 1))[0]
    print("vary", q.shape, "counts per tenth", h)
    assert h[3] > 5 * h[9], h  # the clump is much denser than the tail
