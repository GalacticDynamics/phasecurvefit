r"""Self-Organizing Map core, in pure JAX.

A 1-D SOM learns ``K`` prototype vectors in phase space whose lattice indices
carry the ordering: prototypes ``i`` and ``j`` are neighbours iff ``|i - j| = 1``
(Starkman et al. 2023, Appendix A). Training moves the prototypes onto the data;
projecting the data back onto the prototype polyline yields an arc-length
*chord* parameter that orders the observations.

Everything in this module is a pure function of explicit arrays with static
shapes, so it is jit- and vmap-traceable. No host callbacks, no Python
branching on traced values, no dynamic shapes.

References
----------
Starkman, N., Bovy, J., Webb, J. J., Calvetti, D., & Somersalo, E. (2023).
*On the Fast Track: Rapid construction of stellar stream paths.*
MNRAS 522(4), 5022-5036. https://arxiv.org/abs/2212.00949

If you use this SOM stage in published work, please cite that paper. It
deviates from the paper's method in nine places, listed in
:doc:`/guides/som`.

"""

__all__: tuple[str, ...] = (
    "SOM1D",
    "FitResult",
    "bmu_distance",
    "chord",
    "densify",
    "fit",
    "init_prototypes",
)

from typing import ClassVar, Final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxmore import bounded_while_loop
from zeroth import zeroth

from phasecurvefit._src.custom_types import FSz0, ISzN, VectorComponents
from phasecurvefit._src.metrics import (
    AbstractDistanceMetric,
    SpatialDistanceMetric,
)

_TINY: Final = 1e-12
# Mirrors `MSTOrderer`'s `_EDGE_CLIP_MIN_RATIO`: a point's distance to its
# best-matching unit must be at least this multiple of the (kept) median before
# it is a clip candidate, floors the (multiplicative) threshold so a well-fit
# lattice -- where the robust spread collapses to ~0 -- is not shredded by
# microscopic quantization-error variation.
_OUTLIER_CLIP_MIN_RATIO: Final = 2.0


def _check_matching_keys(**components: VectorComponents) -> None:
    """Reject component dicts that do not all carry the same keys.

    Every dict here is stacked against one shared key tuple, so a mismatch is
    either a `KeyError` from deep inside a jitted region or -- when the odd dict
    has *extra* keys -- a silent drop of a component. Both are worse than
    failing here with the names.

    Empty dicts are rejected for the same reason: with no components there is
    nothing to stack, and the failure surfaces as a bare `StopIteration` from
    the first `zeroth` call with no message at all.
    """
    (ref_name, ref), *rest = components.items()
    for name, comps in rest:
        if set(comps) != set(ref):
            msg = (
                f"{name} and {ref_name} must have the same component keys; "
                f"{name}={sorted(comps)}, {ref_name}={sorted(ref)}."
            )
            raise ValueError(msg)
    # After the loop, so a mismatch is reported as a mismatch rather than as
    # emptiness; the keys are equal by here, so one empty means all empty.
    if not ref:
        msg = f"{ref_name} has no components; expected at least one, e.g. {{'x': ...}}."
        raise ValueError(msg)


def _check_symmetric(metric: AbstractDistanceMetric, /) -> None:
    """Reject a metric whose distance depends on the order of its arguments.

    The SOM assigns each datum to its *nearest prototype*, which is only
    meaningful when ``d(a, b) == d(b, a)``. An asymmetric metric scores
    "forward from the query point", so every datum prefers whatever lies ahead
    of it and the lattice collapses toward the curve's head.
    """
    if not metric.is_symmetric:
        msg = (
            f"{type(metric).__name__} is not symmetric, so nearest-prototype "
            f"assignment is not well defined; the lattice collapses toward the "
            f"curve's head. Use a symmetric metric (e.g. SpatialDistanceMetric "
            f"or FullPhaseSpaceDistanceMetric)."
        )
        raise ValueError(msg)


def _safe_norm(diffs: Array, /) -> Array:
    """Euclidean norm over the last axis whose gradient is finite at zero.

    ``jnp.linalg.norm`` has a 0/0 VJP at the origin, so a single coincident pair
    of vertices NaNs every gradient flowing through it -- and coincident
    vertices are not exotic: duplicated observations produce them directly from
    ``init_prototypes``. The doubled ``where`` keeps the forward value exact
    (the zero case really does return 0.0) while handing the reverse pass a
    constant to differentiate instead of a singularity.
    """
    sq = jnp.sum(diffs**2, axis=-1)
    # ``!= 0`` rather than ``> 0``: a NaN compares False to *both*, so ``> 0``
    # routed NaN into the zero branch and returned 0.0, silently swallowing an
    # invalid input that ``jnp.linalg.norm`` would have propagated. ``!= 0`` is
    # True for NaN, so it takes the sqrt branch and stays NaN, while an exact
    # zero still gets the constant that keeps the reverse pass finite.
    nonzero = sq != 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, sq, 1.0)), 0.0)


def _stack(
    components: VectorComponents, keys: tuple[str, ...], /
) -> Float[Array, "N D"]:
    """Stack selected dict components into a dense ``(N, D)`` array."""
    return jnp.stack([jnp.asarray(components[k]) for k in keys], axis=-1)


def init_prototypes(
    positions: VectorComponents,
    velocities: VectorComponents,
    /,
    *,
    n_prototypes: int,
    ordering: ISzN | None = None,
) -> tuple[VectorComponents, VectorComponents]:
    """Initialize prototypes by equi-frequency binning.

    This is the paper's default initialization: place a prototype at the average
    location of every ``n``-th point along an ordering.

    Parameters
    ----------
    positions, velocities
        Phase-space components, 1-D arrays of shape ``(N,)``.
    n_prototypes
        Number of prototypes, ``K``. Must be at least 2 and at most the number
        of points being binned.
    ordering
        Indices in order, as from a previous stage. When ``None``, bin along the
        first principal axis of the positions -- the N-D replacement for the
        paper's "bin in an observational longitude". Requires that the curve
        not double back along that axis: past about one turn the initial lattice
        is tangled and training cannot repair it. Curves that wind further must
        pass an ``ordering`` from a prior stage.

    Returns
    -------
    tuple[dict, dict]
        Prototype positions and velocities, each of shape ``(K,)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit import som

    >>> pos = {"x": jnp.linspace(0.0, 9.0, 10), "y": jnp.zeros(10)}
    >>> vel = {"x": jnp.ones(10), "y": jnp.zeros(10)}
    >>> pq, pp = som.init_prototypes(pos, vel, n_prototypes=5)
    >>> pq["x"].shape
    (5,)

    """
    _check_matching_keys(positions=positions, velocities=velocities)
    if n_prototypes < 2:
        msg = f"n_prototypes must be >= 2, got {n_prototypes}."
        raise ValueError(msg)

    keys = tuple(sorted(positions))

    if ordering is None:
        q = _stack(positions, keys)
        centered = q - jnp.mean(q, axis=0)
        # SVD on the centred matrix directly, not eigh on its Gram
        # (``centered.T @ centered``): the Gram squares the position
        # magnitudes, which overflows float32 well within ordinary unit
        # systems (1 kpc in SI is ~3e19 m; squared, that is past float32's
        # ~3.4e38 ceiling) with no exception and no NaN in the result --
        # just a wrong axis and a silently corrupted ordering. The first
        # principal axis is the leading right-singular vector.
        axis = jnp.linalg.svd(centered, full_matrices=False)[2][0]
        ordering = jnp.argsort(centered @ axis)
    ordering = jnp.asarray(ordering)
    if not jnp.issubdtype(ordering.dtype, jnp.integer):
        # Otherwise this surfaces from inside the repeat guard as "x argument
        # to bincount must have an integer type", naming a function the caller
        # never invoked. A boolean mask is the likely mistake: `visited` rather
        # than `ordering`.
        msg = (
            f"ordering must be an integer index array; got dtype "
            f"{ordering.dtype}. It holds observation indices in visit order, "
            f"not a mask -- pass an OrderingResult's `ordering`, not `visited`."
        )
        raise TypeError(msg)
    n_obs = len(zeroth(positions.values()))
    ordering = eqx.error_if(
        ordering,
        jnp.any(ordering < 0) | jnp.any(ordering >= n_obs),
        "ordering must contain only valid indices into the data, i.e. in "
        "[0, n_obs); out-of-range entries silently mis-bin the prototypes "
        "rather than raising. An OrderingResult's `indices` field is -1-padded "
        "for unvisited observations -- pass its `ordering` property instead. "
        "Dropping the padding gives a variable-length result, so `ordering` "
        "(like `indices[indices >= 0]`) cannot be evaluated under `jit`; "
        "filter outside the traced region and pass the result in.",
    )
    # The range check above does not catch a non-permutation. A repeated index
    # bins the same observation twice and leaves the lattice non-monotone,
    # silently breaking the premise that lattice position carries order --
    # which `fit` then cannot repair. A strict *subset* is legitimate (a prior
    # stage's visited set), so this tests for repeats, not for covering
    # [0, n_obs).
    ordering = eqx.error_if(
        ordering,
        jnp.max(jnp.bincount(jnp.clip(ordering, 0, n_obs - 1), length=n_obs)) > 1,
        "ordering must not repeat an index: a repeat bins one observation into "
        "two prototypes and leaves the lattice non-monotone, which training "
        "cannot undo. Pass each visited index at most once.",
    )

    m = int(ordering.shape[0])
    if m < n_prototypes:
        msg = (
            "binning needs at least n_prototypes points; got "
            f"{m} points for n_prototypes={n_prototypes}."
        )
        raise ValueError(msg)

    # Contiguous, equal-count bins along the ordering. ``m`` and
    # ``n_prototypes`` are both static, so which form is safe is known at trace
    # time. The direct product overflows int32 once ``m * K > 2**31`` (N=2.2e6
    # with K=1000 suffices, and x64 is off by default); the wrapped negative ids
    # are then silently dropped by ``segment_sum`` and rescued by
    # ``maximum(counts, 1)`` into prototypes stacked at the origin -- a wrong
    # lattice, with no error raised.
    if m * n_prototypes < 2**31:
        segment = jnp.minimum(jnp.arange(m) * n_prototypes // m, n_prototypes - 1)
    else:
        # Bin ``j`` starts at ``ceil(j * m / K)``, formed host-side in exact
        # int64 so nothing on device can overflow. Only taken above the
        # threshold: the binary search costs ~1.7x the direct product, and the
        # host-side ceil is O(K), which is not free at K = 1e5.
        starts = -(-np.arange(n_prototypes, dtype=np.int64) * m // n_prototypes)
        segment = jnp.searchsorted(jnp.asarray(starts), jnp.arange(m), side="right") - 1
    counts = jnp.maximum(jnp.bincount(segment, length=n_prototypes), 1)

    def bin_mean(values: Array) -> Array:
        ordered = jnp.asarray(values)[ordering]
        total = jax.ops.segment_sum(ordered, segment, num_segments=n_prototypes)
        return total / counts

    def finite_bin_mean(values: Array) -> Array:
        # Velocities only: skip non-finite entries, as `fit` does. A NaN
        # prototype would otherwise win every velocity-aware `argmin` (which
        # returns the first NaN) on the first epoch. A bin with no finite entry
        # takes the mean of all finite entries (0 if there are none).
        ordered = jnp.asarray(values)[ordering]
        ok = jnp.isfinite(ordered)
        clean = jnp.where(ok, ordered, 0.0)
        total = jax.ops.segment_sum(clean, segment, num_segments=n_prototypes)
        n_ok = jax.ops.segment_sum(ok.astype(clean.dtype), segment, n_prototypes)
        fallback = jnp.sum(clean) / jnp.maximum(jnp.sum(ok), 1)
        return jnp.where(n_ok > 0, total / jnp.maximum(n_ok, 1), fallback)

    return jt.map(bin_mean, positions), jt.map(finite_bin_mean, velocities)


def _distance_matrix(
    metric: AbstractDistanceMetric,
    metric_scale: float | FSz0,
    positions: VectorComponents,
    velocities: VectorComponents,
    proto_positions: VectorComponents,
    proto_velocities: VectorComponents,
    /,
) -> Float[Array, "N K"]:
    """Phase-space distance from every datum to every prototype.

    A datum whose distances are not all finite -- under a velocity-aware
    metric, one with a non-finite velocity -- falls back to position distance.
    Left non-finite, ``argmin`` sends it to prototype 0 (it returns the first
    NaN), dragging that prototype across the curve: on two close,
    anti-parallel arms one NaN velocity took the arm switches from 1 to ~25,
    where the fallback keeps 1. Whether missing velocities may reach this point
    at all is ``SOMOrderer.nan_policy``'s decision.

    The fallback costs ~10% of a velocity-aware ``fit`` on clean data (XLA
    shares the position term), and nothing for a position-only metric, which
    skips it. ``lax.cond`` per row would not save it: under ``vmap`` it lowers
    to a ``select`` that evaluates both branches.
    """

    def one(pos_n: VectorComponents, vel_n: VectorComponents) -> Array:
        d = metric(pos_n, vel_n, proto_positions, proto_velocities, metric_scale)
        if not metric.uses_velocity:
            return d
        spatial = SpatialDistanceMetric()(
            pos_n, vel_n, proto_positions, proto_velocities, 0.0
        )
        return jnp.where(jnp.all(jnp.isfinite(d)), d, spatial)

    return jax.vmap(one)(positions, velocities)


class FitResult(NamedTuple):
    """Trained prototypes, and which data survived outlier clipping.

    A named result rather than a bare tuple so ``kept`` -- absent unless
    ``outlier_clip_sigma`` is set -- never changes how many values ``fit``
    returns.

    Attributes
    ----------
    prototype_positions, prototype_velocities : VectorComponents
        Trained prototypes, shape ``(K,)`` per component.
    kept : Bool[Array, " N"]
        ``True`` for data that survived every outlier-clipping round.
        All-``True`` when ``outlier_clip_sigma`` was ``None``.

    """

    prototype_positions: VectorComponents
    prototype_velocities: VectorComponents
    kept: Bool[Array, " N"]


def _fit_core(
    proto_positions: VectorComponents,
    proto_velocities: VectorComponents,
    positions: VectorComponents,
    velocities: VectorComponents,
    /,
    *,
    metric: AbstractDistanceMetric,
    metric_scale: float | FSz0,
    n_epochs: int,
    sigma_start: float | None,
    sigma_end: float,
    weights: Float[Array, " N"] | None,
) -> tuple[VectorComponents, VectorComponents]:
    r"""Run the batch-Kohonen scan itself, without the outlier-clip wrapper.

    Split out of :func:`fit` so the clip-and-refit loop there can call this
    directly instead of recursing into :func:`fit`.
    """
    _check_matching_keys(
        proto_positions=proto_positions,
        proto_velocities=proto_velocities,
        positions=positions,
        velocities=velocities,
    )
    _check_symmetric(metric)
    if n_epochs < 1:
        msg = f"n_epochs must be >= 1, got {n_epochs}."
        raise ValueError(msg)
    if sigma_end <= 0:
        msg = f"sigma_end must be positive, got {sigma_end}."
        raise ValueError(msg)

    keys = tuple(sorted(proto_positions))
    n_prototypes = len(zeroth(proto_positions.values()))
    if n_prototypes < 2:
        # K=0 reaches `argmin` on an empty (N, 0) distance matrix; K=1 trains a
        # lattice that `densify` and `chord` then reject. Same floor as
        # `init_prototypes`, `densify` and `chord`.
        msg = (
            f"fit needs at least 2 prototypes, got {n_prototypes}. A lattice "
            f"that small has no neighbourhood to smooth over and cannot be "
            f"densified into a backbone."
        )
        raise ValueError(msg)
    start = max(n_prototypes / 4.0, sigma_end) if sigma_start is None else sigma_start
    if start <= 0:
        msg = f"sigma_start must be positive, got {start}."
        raise ValueError(msg)

    lattice = jnp.arange(n_prototypes)
    data_q = jt.map(jnp.asarray, positions)
    data_p = jt.map(jnp.asarray, velocities)
    ratio = sigma_end / start
    span = max(n_epochs - 1, 1)

    # The neighbourhood is evaluated on the (K, K) lattice, not an (N, K)
    # array, using
    #     h[n, k] = hkk[bmu[n], k]   =>   sum_n h[n,k] w_n
    #                                   = sum_c hkk[c,k] (sum_{bmu(n)=c} w_n)
    # Exact, not an approximation, and combines K partial sums rather than N.
    vel_keys = tuple(sorted(proto_velocities))
    n_pos = len(keys)
    stacked = jnp.stack(
        [data_q[k] for k in keys] + [data_p[k] for k in vel_keys], axis=-1
    )
    # Cast before squaring, not after: the integer square would overflow int32
    # past K = 46342, and casting the (K,) lattice is cheaper than casting the
    # (K, K) result. Exact integers stop at 2**24, but those entries are far
    # off-diagonal and their Gaussian underflows to zero regardless.
    lat = lattice.astype(stacked.dtype)
    lattice_sq = (lat[:, None] - lat[None, :]) ** 2
    omega = (
        jnp.ones(stacked.shape[0], dtype=stacked.dtype)
        if weights is None
        else jnp.asarray(weights, dtype=stacked.dtype)
    )
    # A non-finite velocity (common in catalogues: no radial velocity) is left
    # out of that component's averages rather than summed in: one NaN would
    # otherwise reach every prototype through the neighbourhood and make every
    # prototype velocity NaN. Per column, so the datum still pulls on positions
    # and its finite velocity components. Positions are not masked: a datum
    # without a position has nowhere to be placed.
    present = jnp.isfinite(stacked).at[:, :n_pos].set(True)
    contrib = omega[:, None] * present
    weighted_stacked = jnp.where(present, stacked, 0.0) * omega[:, None]

    def epoch(
        carry: tuple[VectorComponents, VectorComponents], step: Array
    ) -> tuple[tuple[VectorComponents, VectorComponents], None]:
        pq, pp = carry
        bmu = jnp.argmin(
            _distance_matrix(metric, metric_scale, data_q, data_p, pq, pp), axis=1
        )
        # `step` comes from `arange(n_epochs)`, which is int64 under x64; left
        # alone the division promotes sigma -- and the whole update with it --
        # to float64, and the scan carry no longer matches its float32 input.
        sigma = start * ratio ** (step.astype(stacked.dtype) / span)
        # Segment reductions rather than an (N, K) one-hot. The one-hot fed two
        # consumers -- a column sum and a matmul -- which blocked XLA from
        # fusing the whole (N, K) chain, so it materialized: `fit` peaked at
        # 416 MB for N=1e6, K=100 against 16 MB this way.
        counts = jax.ops.segment_sum(contrib, bmu, num_segments=n_prototypes)
        totals = jax.ops.segment_sum(weighted_stacked, bmu, num_segments=n_prototypes)
        neighbourhood = jnp.exp(-lattice_sq / (2.0 * sigma**2))
        weight = neighbourhood.T @ counts
        # A lattice unit no datum reaches has zero weight *and* zero numerator
        # (the Gaussian underflows in float32 ~11 units from the nearest
        # best-matching unit); likewise a velocity component every nearby datum
        # lacks. Leave it where it is rather than dividing 0/0.
        live = weight > jnp.finfo(stacked.dtype).tiny
        means = (neighbourhood.T @ totals) / jnp.where(live, weight, 1.0)
        current = jnp.stack([pq[k] for k in keys] + [pp[k] for k in vel_keys], axis=-1)
        means = jnp.where(live, means, current)
        return (
            {k: means[:, i] for i, k in enumerate(keys)},
            {k: means[:, n_pos + i] for i, k in enumerate(vel_keys)},
        ), None

    init = (jt.map(jnp.asarray, proto_positions), jt.map(jnp.asarray, proto_velocities))
    (trained_q, trained_p), _ = jax.lax.scan(epoch, init, jnp.arange(n_epochs))
    return trained_q, trained_p


def bmu_distance(
    metric: AbstractDistanceMetric,
    metric_scale: float | FSz0,
    positions: VectorComponents,
    velocities: VectorComponents,
    proto_positions: VectorComponents,
    proto_velocities: VectorComponents,
    /,
) -> Float[Array, " N"]:
    """Each datum's phase-space distance to its own best-matching prototype.

    The quantization error `fit` minimizes but never returns: a large value
    means a datum sits far from every prototype, whether because it is an
    outlier or because the lattice is too coarse to reach it. Used internally
    by :func:`fit` to decide which is which when ``outlier_clip_sigma`` is set.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> from phasecurvefit import som

    >>> pos = {"x": jnp.linspace(0.0, 9.0, 50), "y": jnp.zeros(50)}
    >>> vel = {"x": jnp.ones(50), "y": jnp.zeros(50)}
    >>> pq, pp = som.init_prototypes(pos, vel, n_prototypes=6)
    >>> result = som.fit(
    ...     pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric(), n_epochs=20
    ... )
    >>> d = som.bmu_distance(
    ...     pcf.metrics.SpatialDistanceMetric(),
    ...     0.0,
    ...     pos,
    ...     vel,
    ...     result.prototype_positions,
    ...     result.prototype_velocities,
    ... )
    >>> d.shape
    (50,)
    >>> bool(jnp.all(d >= 0))
    True

    """
    d = _distance_matrix(
        metric, metric_scale, positions, velocities, proto_positions, proto_velocities
    )
    return jnp.min(d, axis=1)


def fit(
    proto_positions: VectorComponents,
    proto_velocities: VectorComponents,
    positions: VectorComponents,
    velocities: VectorComponents,
    /,
    *,
    metric: AbstractDistanceMetric,
    metric_scale: float | FSz0 = 0.0,
    n_epochs: int = 10,
    sigma_start: float | None = None,
    sigma_end: float = 0.7,
    weights: Float[Array, " N"] | None = None,
    outlier_clip_sigma: float | None = None,
    outlier_clip_max_iters: int = 5,
) -> FitResult:
    r"""Train a 1-D SOM by batch Kohonen updates.

    Each epoch assigns every datum to its best-matching unit, then replaces
    every prototype by the neighbourhood-weighted mean of all the data:

    .. math::

        p_k \leftarrow \frac{\sum_n h_{c(n),k}\, \omega_n\, w_n}
                             {\sum_n h_{c(n),k}\, \omega_n},
        \qquad h_{ij} = \exp\!\left(-\frac{(i-j)^2}{2\sigma^2}\right)

    The neighbourhood is equation (A8) of the paper on the paper's linear
    lattice. The update replaces the paper's online form (A9)/(A10): there is no
    learning rate. This batch form is the fixed point of the conventional
    online Kohonen update, whose increment is proportional to ``w - p^(k)``
    (:doc:`/guides/som` notes how that differs from (A9) as printed). ``omega_n``
    is ``weights``, defaulting to ``1`` for every datum -- the paper has no such
    term, so this is an addition, not a further deviation from (A9)/(A10): with
    the default, the sum above is exactly the un-weighted one. A non-finite
    velocity component is left out of both sums for that component, so a
    missing velocity cannot poison every prototype through the neighbourhood.

    ``sigma`` anneals geometrically from ``sigma_start`` to ``sigma_end``, so the
    global ordering forms first and local detail is refined afterwards.
    ``sigma_end`` is reached on the *last* epoch, so ``n_epochs=1`` runs its one
    epoch at ``sigma_start`` and never narrows; pass ``sigma_start=sigma_end``
    to train a single epoch at the final width.

    Setting ``outlier_clip_sigma`` layers on iterated, robust rejection of
    quantization-error outliers, mirroring
    :attr:`~phasecurvefit.orderers.MSTOrderer.edge_clip_sigma` /
    ``edge_clip_max_iters`` in both mechanism and iteration, applied to each
    datum's distance to its best-matching unit (:func:`bmu_distance`) rather
    than MST edge length. Each iteration:

    1. refits with the current per-datum weight (``1`` for a kept datum, ``0``
       for a rejected one -- so rejected data stop pulling on any prototype
       without changing any array's shape);
    2. among the still-kept data, in log space (distances are positive and
       heavy-tailed): computes the median and a robust spread
       (``1.4826 * MAD``), then rejects everything beyond
       ``median + max(outlier_clip_sigma * spread, log(2))`` -- the floor
       keeps a well-fit lattice, where the spread collapses to ~0, from being
       shredded by microscopic quantization-error variation;
    3. stops when a round rejects nothing new, or after
       ``outlier_clip_max_iters`` rounds, whichever first -- a rejection is
       never undone once made.

    There is no graph here, so there is no analogue of MST's component-size
    veto (a long edge that would fragment the graph is kept); a quantization
    error is a per-datum quantity, not shared between data, so nothing here
    can "reconnect" the way a spared MST edge can.

    Parameters
    ----------
    proto_positions, proto_velocities
        Initial prototypes, shape ``(K,)`` per component.
    positions, velocities
        The data, shape ``(N,)`` per component.
    metric
        Any *symmetric* :class:`~phasecurvefit.metrics.AbstractDistanceMetric`;
        see :class:`~phasecurvefit.orderers.SOMOrderer` for which, and why.
    metric_scale
        Scale parameter handed to ``metric``; see
        :class:`~phasecurvefit.orderers.SOMOrderer`.
    n_epochs
        Number of batch updates. Static.
    sigma_start
        Initial neighbourhood width in lattice units. ``None`` uses
        ``max(K / 4, sigma_end)``, i.e. ``K / 4`` floored at ``sigma_end`` so
        the anneal is never inverted on a very small lattice.
    sigma_end
        Final neighbourhood width in lattice units.
    weights
        Per-datum weight ``omega_n`` in the update above, shape ``(N,)``.
        ``None`` (default) weighs every datum equally. A datum weighted ``0``
        contributes to no prototype's update without changing any array's
        shape. Mutually exclusive with ``outlier_clip_sigma``, which computes
        its own weights round by round.
    outlier_clip_sigma
        Robust-sigma threshold in log space for outlier rejection. ``None``
        (default) disables it -- ``fit`` then runs exactly one batch-Kohonen
        pass, as if ``outlier_clip_sigma`` never existed. Otherwise must be
        positive.
    outlier_clip_max_iters
        Maximum number of refit-and-reclip rounds when ``outlier_clip_sigma``
        is set. Static; must be >= 1. Unused otherwise.

    Returns
    -------
    FitResult
        Trained prototype positions and velocities, and a boolean ``(N,)``
        ``kept`` mask -- all-``True`` unless ``outlier_clip_sigma`` rejected
        something. A named result rather than a bare tuple so adding
        ``outlier_clip_sigma`` never changed how many values this function
        returns.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> from phasecurvefit import som

    >>> pos = {"x": jnp.linspace(0.0, 9.0, 50), "y": jnp.zeros(50)}
    >>> vel = {"x": jnp.ones(50), "y": jnp.zeros(50)}
    >>> pq, pp = som.init_prototypes(pos, vel, n_prototypes=6)
    >>> result = som.fit(
    ...     pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric(), n_epochs=20
    ... )
    >>> bool(jnp.all(jnp.diff(result.prototype_positions["x"]) > 0))
    True
    >>> bool(jnp.all(result.kept))
    True

    With one wild outlier, ``outlier_clip_sigma`` finds it:

    >>> pos_c = dict(pos)
    >>> pos_c["y"] = pos_c["y"].at[25].set(50.0)
    >>> pq, pp = som.init_prototypes(pos_c, vel, n_prototypes=6)
    >>> result = som.fit(
    ...     pq,
    ...     pp,
    ...     pos_c,
    ...     vel,
    ...     metric=pcf.metrics.SpatialDistanceMetric(),
    ...     n_epochs=20,
    ...     outlier_clip_sigma=3.0,
    ... )
    >>> bool(result.kept[25])
    False
    >>> int(result.kept.sum())
    49

    """
    if outlier_clip_sigma is None:
        trained_q, trained_p = _fit_core(
            proto_positions,
            proto_velocities,
            positions,
            velocities,
            metric=metric,
            metric_scale=metric_scale,
            n_epochs=n_epochs,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            weights=weights,
        )
        n = len(zeroth(positions.values()))
        return FitResult(trained_q, trained_p, jnp.ones(n, dtype=bool))

    if weights is not None:
        msg = "weights and outlier_clip_sigma are mutually exclusive."
        raise ValueError(msg)
    if outlier_clip_sigma <= 0:
        msg = f"outlier_clip_sigma must be positive, got {outlier_clip_sigma}."
        raise ValueError(msg)
    if outlier_clip_max_iters < 1:
        msg = f"outlier_clip_max_iters must be >= 1, got {outlier_clip_max_iters}."
        raise ValueError(msg)

    n = len(zeroth(positions.values()))
    dtype = zeroth(positions.values()).dtype
    log_floor = jnp.log(_OUTLIER_CLIP_MIN_RATIO)

    def refit(kept: Bool[Array, " N"]) -> tuple[VectorComponents, VectorComponents]:
        return _fit_core(
            proto_positions,
            proto_velocities,
            positions,
            velocities,
            metric=metric,
            metric_scale=metric_scale,
            n_epochs=n_epochs,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            weights=kept.astype(dtype),
        )

    State = tuple[Bool[Array, " N"], VectorComponents, VectorComponents, Array]

    def cond_fn(state: State) -> Array:
        return state[-1]

    def body_fn(state: State) -> State:
        kept, _pq, _pp, _cut_any = state
        pq, pp = refit(kept)
        dist = bmu_distance(metric, metric_scale, positions, velocities, pq, pp)
        candidate = kept & (dist > 0)
        loglen = jnp.where(candidate, jnp.log(jnp.where(candidate, dist, 1.0)), jnp.nan)
        med = jnp.nanmedian(loglen)
        mad = jnp.nanmedian(jnp.abs(loglen - med))
        threshold = med + jnp.maximum(outlier_clip_sigma * 1.4826 * mad, log_floor)
        cut = candidate & (loglen > threshold)
        return kept & ~cut, pq, pp, jnp.any(cut)

    run_first_iteration = True
    init_state: State = (
        jnp.ones(n, dtype=bool),
        jt.map(jnp.asarray, proto_positions),
        jt.map(jnp.asarray, proto_velocities),
        jnp.asarray(run_first_iteration),
    )
    kept, trained_q, trained_p, _ = bounded_while_loop(
        cond_fn,
        body_fn,
        init_state,
        max_steps=outlier_clip_max_iters,
        check_termination=False,
    )
    return FitResult(trained_q, trained_p, kept)


def bootstrap_weights(key: PRNGKeyArray, n_obs: int, /) -> Float[Array, " N"]:
    """Multinomial counts for one bootstrap resample, as ``weights`` for :func:`fit`.

    Draws ``n_obs`` indices with replacement and returns how often each datum
    was drawn. Handed to ``fit`` as ``weights`` this *is* a bootstrap resample:
    a datum drawn twice counts twice in the neighbourhood-weighted mean, and
    one never drawn counts not at all. Roughly ``1/e`` of the data is left out
    of any member, which is where an ensemble's spread comes from.

    This exists because ``vmap`` alone does not give you an ensemble. Batch
    Kohonen replaces each prototype outright with a mean of the data, so a
    prototype survives only through which datum it wins; members that agree on
    their assignments stay merged for every later epoch, and perturbing a
    shared initialization converges to bit-identical members. The diversity has
    to come from the data.

    Counts rather than gathered indices, because counts keep every array at
    shape ``(N,)``: an ensemble is a ``vmap`` over an ``(M, N)`` weight matrix
    with the data passed once, not ``M`` copies of the data.

    Parameters
    ----------
    key
        A :func:`jax.random.key`. Split it per ensemble member.
    n_obs
        Number of observations, ``N``. Static.

    Returns
    -------
    Array, shape (N,)
        Counts summing to ``n_obs``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> from phasecurvefit import som

    >>> pos = {"x": jnp.linspace(0.0, 9.0, 50), "y": jnp.zeros(50)}
    >>> vel = {"x": jnp.ones(50), "y": jnp.zeros(50)}
    >>> w = som.bootstrap_weights(jax.random.key(0), 50)
    >>> w.shape, float(w.sum())
    ((50,), 50.0)

    An ensemble maps over the weights, not over copies of the data:

    >>> keys = jax.random.split(jax.random.key(0), 4)
    >>> ws = jax.vmap(lambda k: som.bootstrap_weights(k, 50))(keys)
    >>> pq, pp = som.init_prototypes(pos, vel, n_prototypes=6)
    >>> metric = pcf.metrics.SpatialDistanceMetric()
    >>> fq = jax.vmap(lambda w: som.fit(pq, pp, pos, vel, metric=metric, weights=w))(
    ...     ws
    ... ).prototype_positions
    >>> fq["x"].shape
    (4, 6)

    """
    if n_obs < 1:
        msg = f"n_obs must be >= 1, got {n_obs}."
        raise ValueError(msg)
    drawn = jax.random.randint(key, (n_obs,), 0, n_obs)
    return jnp.bincount(drawn, length=n_obs).astype(float)


def _catmull_rom(
    P: Float[Array, "K D"], /, *, factor: int, knot_cols: int
) -> Float[Array, "M D"]:
    """Centripetal Catmull-Rom through ``P``, ``factor`` samples per segment.

    Centripetal (alpha = 1/2) knot spacing is used rather than uniform because
    it cannot produce cusps or self-intersections when the prototypes are
    unevenly spaced. Knot distances use only the first ``knot_cols`` columns
    (the positions), so that mixing position and velocity units cannot distort
    the parameterization. Evaluated by the Barry-Goldman pyramid, which is a
    chain of six lerps and vectorizes over all segments at once.
    """
    n_proto, n_dim = P.shape
    if n_proto < 2:
        return P
    # Duplicate the end points to give the first and last segments a full
    # four-point stencil.
    padded = jnp.concatenate([P[:1], P, P[-1:]], axis=0)
    p0, p1, p2, p3 = padded[:-3], padded[1:-2], padded[2:-1], padded[3:]

    def next_knot(a: Array, b: Array, t: Array) -> Array:
        # Floored *before* the root, not after. The stencil is padded with
        # duplicated endpoints, so the first and last pairs are exactly
        # coincident by construction; `jnp.linalg.norm` has a 0/0 VJP there,
        # which made every gradient through `densify` NaN at both ends even
        # though the forward value was guarded. Squaring first keeps the same
        # floor (|d| >= _TINY <=> d^2 >= _TINY^2) with a finite derivative.
        d_sq = jnp.sum(((b - a)[:, :knot_cols]) ** 2, axis=-1)
        return t + jnp.sqrt(jnp.sqrt(jnp.maximum(d_sq, _TINY**2)))

    # Every array below is pinned to ``P.dtype``. Left at JAX's default a
    # float32 input silently returns a float64 backbone under x64.
    t0 = jnp.zeros(n_proto - 1, dtype=P.dtype)
    t1 = next_knot(p0, p1, t0)
    t2 = next_knot(p1, p2, t1)
    t3 = next_knot(p2, p3, t2)

    u = jnp.linspace(0.0, 1.0, factor, endpoint=False, dtype=P.dtype)[None, :, None]
    T0, T1, T2, T3 = (x[:, None, None] for x in (t0, t1, t2, t3))
    P0, P1, P2, P3 = (x[:, None, :] for x in (p0, p1, p2, p3))
    t = T1 + u * (T2 - T1)

    def lerp(pa: Array, pb: Array, ta: Array, tb: Array) -> Array:
        # Must stay in this form, not ((tb-t)/span)*pa + ((t-ta)/span)*pb: on
        # the duplicated end stencil those two coefficients are individually
        # ~1e6 and cancel to O(1), which float32 cannot do. Here the huge
        # coefficient multiplies (pb - pa), exactly zero on that stencil.
        span = jnp.maximum(tb - ta, _TINY)
        frac = (t - ta) / span
        return pa + frac * (pb - pa)

    a1 = lerp(P0, P1, T0, T1)
    a2 = lerp(P1, P2, T1, T2)
    a3 = lerp(P2, P3, T2, T3)
    b1 = lerp(a1, a2, T0, T2)
    b2 = lerp(a2, a3, T1, T3)
    curve = lerp(b1, b2, T1, T2)

    return jnp.concatenate([curve.reshape(-1, n_dim), P[-1:]], axis=0)


def _resample_uniform(
    C: Float[Array, "M D"], /, *, knot_cols: int
) -> Float[Array, "M D"]:
    """Resample ``C`` to vertices equally spaced in *position* arc length.

    Needed by ``OrderingResult._interp_backbone``, which interpolates by vertex
    index. :func:`chord` does *not* depend on it: it computes true cumulative
    arc length and works on any backbone.
    """
    step = _safe_norm(jnp.diff(C[:, :knot_cols], axis=0))
    s = jnp.concatenate([jnp.zeros(1, dtype=step.dtype), jnp.cumsum(step)])
    # jnp.interp requires strictly increasing xp, so nudge coincident vertices
    # apart by one ULP at the arc length's own magnitude. The nudge must stay
    # relative: an absolute floor is a sizeable fraction of s[-1] on a track
    # shorter than one unit, destroying the uniformity produced here.
    # Applied only when actually needed: the ramp shifts every entry, and the
    # shift accumulates with M -- 6e-4 of the total arc length at M=5001 --
    # so nudging an already strictly increasing ``s`` would perturb the common
    # case for nothing. Selected branchlessly to stay traceable.
    eps = jnp.finfo(s.dtype).eps * jnp.where(s[-1] > 0, s[-1], 1.0)
    ramp = s + jnp.arange(s.shape[0]) * eps
    s = jnp.where(jnp.all(jnp.diff(s) > 0), s, ramp)
    target = jnp.linspace(0.0, s[-1], C.shape[0], dtype=s.dtype)
    # vmapped rather than a Python loop over columns: one batched `interp`
    # instead of D copies of it, which compiles about a third faster.
    return jax.vmap(lambda col: jnp.interp(target, s, col), in_axes=1, out_axes=1)(C)


def densify(
    proto_positions: VectorComponents,
    proto_velocities: VectorComponents,
    /,
    *,
    factor: int = 5,
) -> tuple[VectorComponents, VectorComponents]:
    """Turn prototypes into a smooth, arc-length-uniform backbone polyline.

    Densifying to a C1 curve is what removes the paper's 2-D "convexity" case.
    On a piecewise-linear polyline every point whose nearest polyline point is a
    vertex receives the same arc length -- a tie the paper broke with an
    ``arctan2`` angle sweep that does not generalize past 2-D. On a smooth curve
    those wedges have angular extent of order (curvature x spacing) and vanish
    as the sampling densifies, so no angle machinery is needed at all.

    Parameters
    ----------
    proto_positions, proto_velocities
        Prototypes, shape ``(K,)`` per component.
    factor
        Samples per prototype segment; the result has ``factor * (K - 1) + 1``
        vertices. ``factor=1`` still evaluates the spline, but at one sample
        per segment, which reproduces the prototypes to float32 rounding -- so
        it smooths nothing. It does **not** skip the arc-length
        resampling, so the result is a polyline whose vertices are redistributed
        to equal spacing rather than the prototypes themselves -- on unevenly
        spaced prototypes it cuts corners, and a corner prototype can end up off
        the emitted polyline entirely.
        Raising it buys down the interior geometric error but not the
        error in the first and last segments: the duplicated end stencil gives
        the phantom knot a near-zero spacing, so the parameterization stalls at
        the tips and those two segments keep a fixed error floor. The tips are
        where :func:`chord` extrapolates past the ends, so that floor bounds
        end-cap accuracy.

    Returns
    -------
    tuple[dict, dict]
        Backbone positions and velocities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit import som

    >>> pq = {"x": jnp.linspace(0.0, 3.0, 4), "y": jnp.zeros(4)}
    >>> pp = {"x": jnp.ones(4), "y": jnp.zeros(4)}
    >>> bq, bp = som.densify(pq, pp, factor=5)
    >>> bq["x"].shape
    (16,)

    """
    if factor < 1:
        msg = f"factor must be >= 1, got {factor}."
        raise ValueError(msg)

    _check_matching_keys(
        proto_positions=proto_positions, proto_velocities=proto_velocities
    )
    n_prototypes = len(zeroth(proto_positions.values()))
    if n_prototypes < 2:
        msg = (
            f"densify needs at least 2 prototypes to form a backbone, got "
            f"{n_prototypes}. A single vertex has no segment to measure arc "
            f"length along, and `chord` cannot project onto it."
        )
        raise ValueError(msg)

    q_keys = tuple(sorted(proto_positions))
    p_keys = tuple(sorted(proto_velocities))
    n_q = len(q_keys)

    P = jnp.concatenate(
        [_stack(proto_positions, q_keys), _stack(proto_velocities, p_keys)], axis=-1
    )
    curve = _resample_uniform(
        _catmull_rom(P, factor=factor, knot_cols=n_q), knot_cols=n_q
    )
    backbone_q = {k: curve[:, i] for i, k in enumerate(q_keys)}
    backbone_p = {k: curve[:, n_q + i] for i, k in enumerate(p_keys)}
    return backbone_q, backbone_p


def _segment_projection(
    q: Float[Array, " D"], B: Float[Array, "M D"], k: Array, n_seg: int, /
) -> tuple[Array, Array]:
    """Clamped projection of ``q`` onto backbone segment ``k``.

    Returns ``(t, distance)``. The clamp is relaxed at the two *global* ends so
    that data beyond the tips extrapolate (the paper's "end-cap" case) instead
    of piling up at the endpoints.
    """
    p = B[k]
    s = B[k + 1] - B[k]
    # Guard on exact zero rather than an absolute floor. ``_TINY`` is compared
    # against |s|^2, which carries (length)^2 units, so a fixed 1e-12 took over
    # the denominator for any segment shorter than 1e-6 in the caller's units
    # and scaled ``t`` down toward zero -- correct in kpc, silently wrong in
    # normalised or radian coordinates. A zero-length segment has a numerator
    # of exactly zero, so substituting 1.0 there yields t=0 exactly.
    ss = jnp.sum(s * s)
    t = jnp.sum((q - p) * s) / jnp.where(ss > 0, ss, 1.0)
    # Bounds pinned to ``t``'s dtype. The literals are weakly typed, so under
    # x64 ``jnp.where`` resolves them to float64; ``jnp.clip`` happens to keep
    # the operand's dtype, so nothing leaks today -- but that is the clip's
    # doing, not ours, and the rest of this module pins its constants rather
    # than relying on a downstream op to absorb them.
    lo = jnp.where(k == 0, -jnp.inf, 0.0).astype(t.dtype)
    hi = jnp.where(k == n_seg - 1, jnp.inf, 1.0).astype(t.dtype)
    t = jnp.clip(t, lo, hi)
    # _safe_norm rather than jnp.linalg.norm: this distance currently
    # feeds only the d_prev <= d_next comparison, whose cotangent JAX
    # prunes, so the 0/0 VJP at a zero residual is not reachable today. It is a
    # trap for whoever next uses the returned distance in a value, and the same
    # 0/0 has already produced NaN gradients twice in this module.
    return t, _safe_norm(q - (p + t * s))


def _chord_one(
    q: Float[Array, " D"],
    j: Array,
    B: Float[Array, "M D"],
    L: Array,
    seglen: Array,
    n_seg: int,
    /,
) -> Array:
    """Arc length of ``q``, refined over the two segments adjacent to vertex ``j``.

    ``L`` is the cumulative arc length at each backbone vertex and ``seglen``
    the length of each segment; both are exact regardless of whether the
    backbone is arc-length-uniform.
    """
    k_prev = jnp.clip(j - 1, 0, n_seg - 1)
    k_next = jnp.clip(j, 0, n_seg - 1)
    t_prev, d_prev = _segment_projection(q, B, k_prev, n_seg)
    t_next, d_next = _segment_projection(q, B, k_next, n_seg)
    use_prev = d_prev <= d_next
    k = jnp.where(use_prev, k_prev, k_next)
    t = jnp.where(use_prev, t_prev, t_next)
    # Cumulative arc length at vertex k, plus the fractional distance into
    # segment k. t may fall outside [0, 1] on the two global end segments
    # (see _segment_projection), which extrapolates correctly here too.
    return L[k] + t * seglen[k]


def chord(
    backbone_positions: VectorComponents,
    backbone_velocities: VectorComponents,
    positions: VectorComponents,
    velocities: VectorComponents,
    /,
    *,
    metric: AbstractDistanceMetric,
    metric_scale: float | FSz0 = 0.0,
) -> Float[Array, " N"]:
    """Project data onto the backbone and return the arc-length chord parameter.

    This is the N-dimensional generalization of the paper's Section 2.2.2. Each
    datum is assigned to its nearest backbone vertex **under** ``metric``, and
    then refined to sub-vertex resolution **in position space**, where
    projection onto a segment is actually defined.

    Whether that assignment is velocity-aware depends on ``metric`` and
    ``metric_scale``. At the default ``metric_scale=0.0`` the phase-space
    metric reduces to pure position distance, so anti-parallel arms of a
    near-closed loop can capture each other; :doc:`/guides/som` gives the
    contamination rates and :class:`~phasecurvefit.orderers.SOMOrderer` covers
    choosing a scale.

    The split is forced: a metric such as
    :class:`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` is not induced
    by an inner product, so "project onto a line segment" has no meaning under
    it. The metric therefore chooses *which* segment; Euclidean position
    geometry chooses *where* within it. The returned chord is consequently a
    genuine physical arc length along the track, computed as true cumulative arc
    length over the backbone -- which need not be arc-length-uniform
    (:func:`densify` produces one that is, but this function does not assume
    it).

    Returns
    -------
    Float[Array, " N"]
        Arc length per observation, in input order. Values outside
        ``[0, L]`` are data beyond the tips (see ``_segment_projection``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> from phasecurvefit import som

    >>> bq = {"x": jnp.linspace(0.0, 10.0, 11), "y": jnp.zeros(11)}
    >>> bp = {"x": jnp.ones(11), "y": jnp.zeros(11)}
    >>> pos = {"x": jnp.array([0.0, 5.0, 10.0]), "y": jnp.zeros(3)}
    >>> vel = {"x": jnp.ones(3), "y": jnp.zeros(3)}
    >>> lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    >>> jnp.round(lam, 3)
    Array([ 0.,  5., 10.], dtype=float32)

    """
    _check_matching_keys(
        positions=positions,
        velocities=velocities,
        backbone_positions=backbone_positions,
        backbone_velocities=backbone_velocities,
    )
    _check_symmetric(metric)
    n_vertices = len(zeroth(backbone_positions.values()))
    if n_vertices < 2:
        msg = (
            f"backbone must have at least 2 vertices to project onto, got "
            f"{n_vertices}. With one vertex there is no segment, and the "
            f"cumulative arc length is empty."
        )
        raise ValueError(msg)

    q_keys = tuple(sorted(positions))
    B = _stack(backbone_positions, q_keys)
    data = _stack(positions, q_keys)

    nearest = jnp.argmin(
        _distance_matrix(
            metric,
            metric_scale,
            positions,
            velocities,
            backbone_positions,
            backbone_velocities,
        ),
        axis=1,
    )

    n_seg = int(B.shape[0]) - 1
    seglen = _safe_norm(jnp.diff(B, axis=0))
    L = jnp.concatenate([jnp.zeros(1, dtype=seglen.dtype), jnp.cumsum(seglen)])
    return jax.vmap(_chord_one, in_axes=(0, 0, None, None, None, None))(
        data, nearest, B, L, seglen, n_seg
    )


class SOM1D(eqx.Module):
    """A trained or untrained 1-D Self-Organizing Map.

    A thin, composable object over the functional core: it carries the
    prototypes and the hyperparameters, and its methods forward to
    :func:`fit`, :func:`densify` and :func:`chord`. The functions remain the
    primary interface -- an ensemble maps ``vmap`` over those, not this class.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> from phasecurvefit import som

    >>> pos = {"x": jnp.linspace(0.0, 9.0, 60), "y": jnp.zeros(60)}
    >>> vel = {"x": jnp.ones(60), "y": jnp.zeros(60)}

    >>> model = som.SOM1D.make(
    ...     pos, vel, n_prototypes=8, metric=pcf.metrics.SpatialDistanceMetric()
    ... )
    >>> model.n_prototypes
    8
    >>> trained = model.fit(pos, vel)
    >>> trained.chord(pos, vel).shape
    (60,)

    """

    prototype_positions: VectorComponents
    prototype_velocities: VectorComponents
    metric: AbstractDistanceMetric
    metric_scale: float | FSz0 = 0.0
    n_epochs: int = eqx.field(static=True, default=10)
    sigma_start: float | None = eqx.field(static=True, default=None)
    sigma_end: float = eqx.field(static=True, default=0.7)
    densify_factor: int = eqx.field(static=True, default=5)

    __citation__: ClassVar[str] = "https://arxiv.org/abs/2212.00949"

    @classmethod
    def make(
        cls,
        positions: VectorComponents,
        velocities: VectorComponents,
        /,
        *,
        n_prototypes: int,
        ordering: ISzN | None = None,
        metric: AbstractDistanceMetric | None = None,
        **kwargs: object,
    ) -> "SOM1D":
        """Build an untrained SOM with prototypes initialized from the data."""
        pq, pp = init_prototypes(
            positions, velocities, n_prototypes=n_prototypes, ordering=ordering
        )
        chosen = SpatialDistanceMetric() if metric is None else metric
        return cls(pq, pp, chosen, **kwargs)  # type: ignore[arg-type]

    @property
    def n_prototypes(self) -> int:
        """Number of prototypes, ``K``."""
        return len(zeroth(self.prototype_positions.values()))

    def fit(
        self, positions: VectorComponents, velocities: VectorComponents, /
    ) -> "SOM1D":
        """Train on the data, returning a new SOM with updated prototypes."""
        result = fit(
            self.prototype_positions,
            self.prototype_velocities,
            positions,
            velocities,
            metric=self.metric,
            metric_scale=self.metric_scale,
            n_epochs=self.n_epochs,
            sigma_start=self.sigma_start,
            sigma_end=self.sigma_end,
        )
        return eqx.tree_at(
            lambda m: (m.prototype_positions, m.prototype_velocities),
            self,
            (result.prototype_positions, result.prototype_velocities),
        )

    def backbone(self) -> tuple[VectorComponents, VectorComponents]:
        """Return the densified, arc-length-uniform backbone polyline."""
        return densify(
            self.prototype_positions,
            self.prototype_velocities,
            factor=self.densify_factor,
        )

    def chord(
        self, positions: VectorComponents, velocities: VectorComponents, /
    ) -> Float[Array, " N"]:
        """Arc-length chord parameter of the data on this SOM's backbone."""
        bq, bp = self.backbone()
        return chord(
            bq,
            bp,
            positions,
            velocities,
            metric=self.metric,
            metric_scale=self.metric_scale,
        )
