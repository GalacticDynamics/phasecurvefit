"""The ridge orderer: a backbone along the ridge of the weighted sample density.

Built for density fields -- an image whose pixels enter as samples weighted by
the density there -- but it works on any weighted or unweighted point cloud.

1. *Density*: a Gaussian kernel density estimate of the weighted samples at
   width ``bandwidth``. Smoothing a uniform tube or a blurred line puts the
   ridge of this density on the centre line. By default the bandwidth is
   estimated from the data (see ``RidgeOrderer``).
2. *Ends*: the two ends of the longest shortest path, by length, through the
   dense core (density at least ``core`` times its peak). Ranking the ends by
   the ridge cost below instead would pick a faint sample on the *side* of the
   stream, since reaching one costs about ``1 / density**2``.
3. *Ridge path*: the shortest path between the ends on the ``k``-nearest-
   neighbour graph, with each edge costing its length over the squared density
   averaged over its two ends, so the path follows the ridge.
4. *Trim*: the path runs out through the faint ends of the stream; it is cut
   where the density along it first and last reaches ``trim`` times its median.
   For a blurred line the density at the true end is half the interior value,
   so ``trim=0.5`` is the half-maximum end.
5. *Backbone*: the path, smoothed over ``bandwidth / 2`` of arc length. Every
   sample is ordered by its arc-length position along it (``chord``).

Velocities are not used, so ``velocity_aware`` is ``False`` and the direction
along the curve is arbitrary. The backbone seeds a SOM well:
``RidgeOrderer() | SOMOrderer()``.

The neighbour searches and graph run on the host with SciPy, like
``MSTOrderer(neighbors=SciPy())``, so ``order()`` cannot be traced.
"""

__all__: tuple[str, ...] = ("RidgeOrderer",)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import plum
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from .base import AbstractOrderer, _check_component_keys
from .result import OrderingResult
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.checks import check_weights
from phasecurvefit._src.custom_types import FSz0, FSzN, VectorComponents

# ponytail: the bandwidth is estimated on a strided subsample of at most this
# many samples, so a wide first guess can't pair every pixel with every other.
_N_ESTIMATE = 2000


class RidgeOrderer(AbstractOrderer):
    """Order samples along the ridge of their weighted density.

    Parameters
    ----------
    bandwidth
        Width of the Gaussian kernel for the density estimate, in position
        units: roughly the half-width of the stream. Too small and the ridge
        follows noise or pixel structure; too large and it cuts bends.
        ``None`` (default) estimates it from the data. It starts at a tenth of
        the data's extent, finds the backbone, sets the bandwidth to the
        weighted RMS distance of the samples from the backbone, and repeats
        until that changes by less than 2%. The estimate runs on a strided
        subsample of at most 2000 samples, and is never below their median
        nearest-neighbour spacing.
    k
        Neighbours each sample links to in the graph the ridge path runs on.
    core
        Density, as a fraction of its peak, above which a sample can be an end.
    trim
        The backbone is cut where the density along it falls below this
        fraction of its median.

    Notes
    -----
    The ends are found to within about ``bandwidth``: the longest path through
    the core starts at its far corner, off the ridge, and a uniform image's
    round caps hold more density than a blurred line's ends, so ``trim`` stops
    a little past them. A higher ``core`` narrows the core around the ridge,
    but too high a value splits it at any gap in the stream, and then the
    backbone covers only one side of the gap.

    Examples
    --------
    An image of a blurred arc: each pixel is a sample weighted by its density.

    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> g = jnp.linspace(-1.5, 1.5, 61)
    >>> X, Y = jnp.meshgrid(g, g)
    >>> R = jnp.hypot(X, Y)
    >>> image = jnp.exp(-((R - 1.0) ** 2) / (2 * 0.1**2)) * (Y > 0)
    >>> keep = image.ravel() > 0.05
    >>> pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
    >>> vel = {"x": jnp.zeros_like(pos["x"]), "y": jnp.zeros_like(pos["y"])}
    >>> result = pcf.order(
    ...     pos,
    ...     vel,
    ...     pcf.orderers.RidgeOrderer(),
    ...     weights=image.ravel()[keep],
    ... )
    >>> radius = jnp.hypot(result.backbone["x"], result.backbone["y"])
    >>> bool(jnp.median(jnp.abs(radius - 1.0)) < 0.03)
    True

    """

    bandwidth: float | FSz0 | None = None
    k: int = eqx.field(static=True, default=8)
    core: float = eqx.field(static=True, default=0.2)
    trim: float = eqx.field(static=True, default=0.5)

    def __check_init__(self) -> None:
        """Reject invalid configuration at construction."""
        # A Quantity is checked once its unit is stripped (the unxt dispatch
        # rebuilds the orderer with the bare number); anything else must be a
        # positive scalar, including a 0-d array.
        bw = self.bandwidth
        if bw is not None and not hasattr(bw, "unit") and np.ndim(bw) != 0:
            msg = f"bandwidth must be a scalar, got shape {np.shape(bw)}."
            raise ValueError(msg)
        if bw is not None and not hasattr(bw, "unit") and not float(bw) > 0:
            msg = f"bandwidth must be positive, got {self.bandwidth}."
            raise ValueError(msg)
        if self.k < 1:
            msg = f"k must be at least 1, got {self.k}."
            raise ValueError(msg)
        for name in ("core", "trim"):
            if not 0 < getattr(self, name) <= 1:
                msg = f"{name} must be in (0, 1], got {getattr(self, name)}."
                raise ValueError(msg)

    @plum.dispatch
    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,  # noqa: ARG002
        init: AbstractResult | None = None,  # noqa: ARG002
        weights: FSzN | None = None,
    ) -> OrderingResult:
        """Order samples along the ridge of their weighted density."""
        _check_component_keys(positions, velocities)
        arrays = [*positions.values(), *([] if weights is None else [weights])]
        if not all(jax.core.is_concrete(jnp.asarray(a)) for a in arrays):
            msg = (
                "RidgeOrderer runs on the host with SciPy and cannot be traced "
                "under jit, vmap or grad. Call it outside the transform, or use "
                "MSTOrderer, which traces."
            )
            raise TypeError(msg)
        comps = sorted(positions)
        pts = np.stack([np.asarray(positions[c], dtype=float) for c in comps], axis=1)
        n = pts.shape[0]
        if weights is not None:
            weights = check_weights(weights, n)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        opts = {"k": self.k, "core": self.core, "trim": self.trim}

        if n < 2:  # nothing to connect: identity ordering, as MSTOrderer
            backbone, chord = pts, np.zeros(n)
        else:
            h = (
                _estimate_bandwidth(pts, w, **opts)
                if self.bandwidth is None
                else float(self.bandwidth)
            )
            backbone, chord = _backbone(pts, w, h, **opts)
        indices = np.argsort(chord, kind="stable").astype(np.int32)
        # Back in the positions' float dtype: the host pipeline runs in float64,
        # which must not upcast a float32 caller's arrays under 64-bit JAX, and
        # integer positions (pixel indices) must not truncate the chord. The
        # Python float is weakly typed, so float32 stays float32.
        dtype = jnp.result_type(*(jnp.asarray(v) for v in positions.values()), float)
        backbone, chord = backbone.astype(dtype), chord.astype(dtype)
        return OrderingResult(
            positions={c: jnp.asarray(v) for c, v in positions.items()},
            velocities={c: jnp.asarray(v) for c, v in velocities.items()},
            indices=jnp.asarray(indices),
            gamma_range=(-1.0, 1.0),
            backbone={c: jnp.asarray(backbone[:, i]) for i, c in enumerate(comps)},
            chord=jnp.asarray(chord),
            weights=weights,
            velocity_aware=False,
        )


def _backbone(
    pts: np.ndarray, w: np.ndarray, h: float, /, *, k: int, core: float, trim: float
) -> tuple[np.ndarray, np.ndarray]:
    """Density, ridge path, trim and smoothing: steps 1-5 of the module docstring."""
    depth = _density(pts, w, h)
    path = _ridge_path(pts, depth, k=min(k, len(pts) - 1), core=core)
    d_path = depth[path]
    inside = np.flatnonzero(d_path >= trim * np.median(d_path))
    return _smooth_and_project(pts, path[inside[0] : inside[-1] + 1], h)


def _estimate_bandwidth(
    pts: np.ndarray,
    w: np.ndarray,
    /,
    *,
    k: int,
    core: float,
    trim: float,
    max_iter: int = 10,
    rtol: float = 0.02,
) -> float:
    """Self-consistent bandwidth: the weighted RMS distance from the backbone.

    Starts wide, at a tenth of the data's extent, where the ridge is smooth but
    cuts bends, so the scatter about it overestimates the width; each round
    then shrinks towards the scatter about the ridge it defines. Starting
    narrow instead could lock onto noise or pixel structure.
    """
    step = max(1, len(pts) // _N_ESTIMATE)
    sub, ws = pts[::step], w[::step]
    if not ws.any():  # the stride missed every weighted sample
        sub, ws = pts, w
    extent = float(np.ptp(pts, axis=0).max())
    if extent == 0:  # every sample at one point: any positive width orders them
        return 1.0
    # Floor at the typical gap between distinct samples; duplicates (gap 0)
    # would otherwise allow a zero bandwidth, which the density divides by.
    nn = cKDTree(sub).query(sub, 2)[0][:, 1]
    spacing = float(np.median(nn[nn > 0])) if (nn > 0).any() else 1e-3 * extent
    h = 0.1 * extent
    for _ in range(max_iter):
        backbone, _ = _backbone(sub, ws, h, k=k, core=core, trim=trim)
        dist, _ = cKDTree(backbone).query(sub)
        h_new = max(float(np.sqrt(np.sum(ws * dist**2) / np.sum(ws))), spacing)
        if abs(h_new - h) <= rtol * h:
            return h_new
        h = h_new
    return h


def _density(pts: np.ndarray, w: np.ndarray, h: float, /) -> np.ndarray:
    """Weighted Gaussian KDE at every sample, from neighbours within ``3 h``.

    Normalized to a peak of 1. Floored above 0 so a zero-weight sample far from
    the rest gives a finite, if huge, ridge cost rather than a division by 0.
    """
    n = len(pts)
    # ponytail: all pairs within 3h are held at once, O(N * (3h/spacing)^D)
    # memory; chunk the query if images get much larger than ~1e5 pixels.
    # query_pairs keeps coincident samples (distance 0), each pair once (i < j).
    i, j = cKDTree(pts).query_pairs(3 * h, output_type="ndarray").T
    kern = np.exp(-0.5 * (np.linalg.norm(pts[i] - pts[j], axis=1) / h) ** 2)
    depth = (
        w  # each sample's own term
        + np.bincount(i, weights=w[j] * kern, minlength=n)
        + np.bincount(j, weights=w[i] * kern, minlength=n)
    )
    depth = depth / depth.max()
    return np.maximum(depth, np.finfo(float).tiny ** 0.25)


def _ridge_path(
    pts: np.ndarray, depth: np.ndarray, /, *, k: int, core: float
) -> np.ndarray:
    """Sample indices of the ridge path between the two ends of the core."""
    n = len(pts)
    dist, nbr = cKDTree(pts).query(pts, k + 1)
    # Drop each sample's link to itself by index, not by position: with a
    # coincident duplicate, the sample need not come first in its own list.
    rows = np.repeat(np.arange(n), k + 1)
    not_self = nbr.ravel() != rows
    rows, cols = rows[not_self], nbr.ravel()[not_self]
    # Coincident samples would give zero-length edges, which a sparse graph
    # reads as no edge at all.
    ell = np.maximum(dist.ravel()[not_self], np.finfo(float).eps)

    def graph(data: np.ndarray, keep: np.ndarray) -> sparse.csr_matrix:
        return sparse.coo_matrix((data[keep], (rows[keep], cols[keep])), (n, n)).tocsr()

    # Ends: two-pass longest path, by length, within the core around the peak.
    in_core = depth >= core
    g_len = graph(ell, in_core[rows] & in_core[cols])
    d0 = csgraph.dijkstra(g_len, directed=False, indices=int(np.argmax(depth)))
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da = csgraph.dijkstra(g_len, directed=False, indices=a)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))

    # Ridge path between them: edge cost = length / (mean of the end densities)^2.
    cost = ell / (0.5 * (depth[rows] + depth[cols])) ** 2
    _, pred = csgraph.dijkstra(
        graph(cost, np.ones_like(rows, dtype=bool)),
        directed=False,
        indices=a,
        return_predecessors=True,
    )
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    return np.asarray(path[::-1])


def _smooth_and_project(
    pts: np.ndarray, path: np.ndarray, h: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the path into a backbone, and project every sample onto it.

    Returns the backbone vertices and each sample's arc length along it: the
    arc length to its nearest vertex plus its offset along the tangent there.
    """
    raw = pts[path]
    if len(raw) < 2:  # a single vertex: no length to smooth over or project onto
        return raw, np.zeros(len(pts))
    step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    # Capped at the path's own length: wider means nothing, and coincident
    # vertices (step 0) would otherwise ask for an astronomically wide kernel.
    sigma = min(0.5 * h / max(float(np.median(step)), np.finfo(float).eps), len(raw))
    backbone = ndimage.gaussian_filter1d(raw, sigma, axis=0, mode="nearest")
    tangent = np.gradient(backbone, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-300)
    s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(backbone, axis=0), axis=1))]
    )
    _, nearest = cKDTree(backbone).query(pts)
    along = np.einsum("ij,ij->i", pts - backbone[nearest], tangent[nearest])
    return backbone, s[nearest] + along
