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
:doc:`/explanation/som`.

"""

__all__: tuple[str, ...] = (
    "SOM1D",
    "FitResult",
    "SOMMixture",
    "bmu_distance",
    "bootstrap_weights",
    "chord",
    "densify",
    "fit",
    "init_prototypes",
)

from dataclasses import dataclass
from typing import ClassVar, Final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from zeroth import zeroth

from phasecurvefit._src._repr import ReprMixin
from phasecurvefit._src.checks import value_error_if
from phasecurvefit._src.custom_types import (
    ArrayComponents,
    FSz0,
    ISzN,
    VectorComponents,
)
from phasecurvefit._src.metrics import (
    AbstractDistanceMetric,
    SpatialDistanceMetric,
)
from phasecurvefit._src.nn.membership import uniform_background_density

_TINY: Final = 1e-12
# Early in training the outlier mixture widens the curve by this fraction of a
# neighbourhood's physical length, annealed down to ``sigma_end``; see `fit`.
# Measured on a contaminated arc: 0.1 kept every member however long the
# training ran, where no widening lost 256 of 300 at 40 epochs; 0.17 and up let
# a background of 240 in 540 drag the backbone off at 20 epochs.
_MIXTURE_ANNEAL: Final = 0.1
# A mixture fit whose member fraction ends below this keeps every datum; see
# `fit`. The SOM's input is a sample of a curve, so a fit that calls most of
# it background has fitted the bounding box instead.
_MIN_MEMBER_FRACTION: Final = 0.5


def _n_rows(components: VectorComponents, /) -> int:
    """Return the number of points: the length of the first component."""
    return len(zeroth(components.values()))  # type: ignore[arg-type]  # ArrayLike includes scalars; components are 1-D


def _check_matching_keys(**components: VectorComponents) -> None:
    """Reject component dicts that do not all carry the same keys.

    Every dict here is stacked against one shared key tuple, so a mismatch is
    either a `KeyError` from deep inside a jitted region or -- when the odd dict
    has *extra* keys -- a silent drop of a component. Both are worse than
    failing here with the names.

    Empty dicts are rejected too, with a `ValueError` that names the dict.
    Without this check there would be nothing to stack, and the failure would
    surface later as a bare `StopIteration` from the first `zeroth` call, with
    no message at all.
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


def _pack(
    q: VectorComponents, p: VectorComponents, keys: tuple[str, ...], /
) -> Float[Array, "N TwoD"]:
    """Stack positions, then velocities, each in ``keys`` order: ``(N, 2D)``."""
    return jnp.concatenate([_stack(q, keys), _stack(p, keys)], axis=-1)


def _unpack(
    arr: Float[Array, "N TwoD"], keys: tuple[str, ...], /
) -> tuple[ArrayComponents, ArrayComponents]:
    """Split a `_pack`-ed array back into position and velocity dicts."""
    d = len(keys)
    q = {k: arr[:, i] for i, k in enumerate(keys)}
    return q, {k: arr[:, d + i] for i, k in enumerate(keys)}


def init_prototypes(
    positions: VectorComponents,
    velocities: VectorComponents,
    /,
    *,
    n_prototypes: int,
    ordering: ISzN | None = None,
) -> tuple[ArrayComponents, ArrayComponents]:
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
    n_obs = _n_rows(positions)
    ordering = value_error_if(
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
    ordering = value_error_if(
        ordering,
        jnp.max(jnp.bincount(jnp.clip(ordering, 0, n_obs - 1), length=n_obs)) > 1,
        "ordering must not repeat an index: a repeat bins one observation into "
        "two prototypes and leaves the lattice non-monotone, which training "
        "cannot undo. Pass each visited index at most once.",
    )

    m = int(ordering.shape[0])
    return _bin_prototypes(
        jt.map(lambda v: jnp.asarray(v)[ordering], positions),
        jt.map(lambda v: jnp.asarray(v)[ordering], velocities),
        m,
        n_prototypes,
    )


def _bin_prototypes(
    positions: VectorComponents,
    velocities: VectorComponents,
    n_valid: int | Array,
    n_prototypes: int,
    /,
) -> tuple[ArrayComponents, ArrayComponents]:
    """Equi-frequency bin the first ``n_valid`` points, already in order.

    ``positions`` and ``velocities`` are in the order to bin along. Only the
    first ``n_valid`` entries are binned; the rest are padding and reach no
    prototype. ``n_valid`` may be traced, so a stage chained after another can
    bin exactly what that stage visited while every array keeps the static
    length ``M``. With ``n_valid == M`` this is plain equi-frequency binning.
    """
    big_m = _n_rows(positions)
    if _is_concrete(n_valid):
        if int(n_valid) < n_prototypes:
            msg = (
                "binning needs at least n_prototypes points; got "
                f"{int(n_valid)} points for n_prototypes={n_prototypes}."
            )
            raise ValueError(msg)
    else:
        n_valid = value_error_if(
            n_valid,
            n_valid < n_prototypes,
            "binning needs at least n_prototypes points; got fewer visited "
            f"points than n_prototypes={n_prototypes}.",
        )

    # Contiguous, equal-count bins along the ordering: rank ``r`` goes to bin
    # ``floor(r * K / m)``. ``big_m`` and ``n_prototypes`` are both static, and
    # ``r < m <= big_m``, so which form is safe is known at trace time. The
    # direct product overflows int32 once ``m * K > 2**31`` (N=2.2e6 with
    # K=1000 suffices, and x64 is off by default); the wrapped negative ids
    # are then silently dropped by ``segment_sum`` and rescued by
    # ``maximum(counts, 1)`` into prototypes stacked at the origin -- a wrong
    # lattice, with no error raised.
    rank = jnp.arange(big_m)
    valid = rank < n_valid
    if big_m * n_prototypes < 2**31:
        segment = jnp.minimum(rank * n_prototypes // n_valid, n_prototypes - 1)
    else:
        # Bin ``j`` starts at ``ceil(j * m / K)``. With ``m = a K + b`` that is
        # ``j a + ceil(j b / K)``, and ``j b < K**2`` fits int32 for every
        # lattice ``_fit_core`` can square without overflow. Exact, and needs
        # no host-side ``m``, so it traces. Only taken above the threshold: the
        # binary search costs ~1.7x the direct product.
        if n_prototypes > 46340:
            msg = (
                f"n_prototypes={n_prototypes} is too large to bin "
                f"{big_m} points without int32 overflow; use at most 46340."
            )
            raise ValueError(msg)
        a, b = n_valid // n_prototypes, n_valid % n_prototypes
        j = jnp.arange(n_prototypes)
        starts = j * a + (j * b + n_prototypes - 1) // n_prototypes
        segment = jnp.searchsorted(starts, rank, side="right") - 1
    # Padding goes to bin ``K``, which ``bincount`` and ``segment_sum`` drop.
    segment = jnp.where(valid, segment, n_prototypes)
    counts = jnp.maximum(jnp.bincount(segment, length=n_prototypes), 1)

    def bin_mean(values: Array) -> Array:
        total = jax.ops.segment_sum(values, segment, num_segments=n_prototypes)
        return total / counts

    def finite_bin_mean(values: Array) -> Array:
        # Velocities only: skip non-finite entries, as `fit` does. A NaN
        # prototype would otherwise win every velocity-aware `argmin` (which
        # returns the first NaN) on the first epoch. A bin with no finite entry
        # takes the mean of all finite entries (0 if there are none).
        ok = jnp.isfinite(values) & valid
        clean = jnp.where(ok, values, 0.0)
        total = jax.ops.segment_sum(clean, segment, num_segments=n_prototypes)
        n_ok = jax.ops.segment_sum(ok.astype(clean.dtype), segment, n_prototypes)
        fallback = jnp.sum(clean) / jnp.maximum(jnp.sum(ok), 1)
        return jnp.where(n_ok > 0, total / jnp.maximum(n_ok, 1), fallback)

    return jt.map(bin_mean, positions), jt.map(finite_bin_mean, velocities)


def _is_concrete(x: object, /) -> bool:
    """Whether ``x`` has a value here, i.e. it is not an abstract tracer.

    The same test as ``jax.core.is_concrete``, which jax 0.12 removed. A
    ``grad`` tracer carries its primal value and counts as concrete; a ``jit``
    tracer has none.
    """
    return not isinstance(x, jax.core.Tracer) or x.to_concrete_value() is not None  # type: ignore[no-untyped-call]  # untyped jax internal


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
    """Trained prototypes, and each datum's membership of the curve.

    Attributes
    ----------
    prototype_positions, prototype_velocities : VectorComponents
        Trained prototypes, shape ``(K,)`` per component.
    kept : Bool[Array, " N"]
        ``True`` for data the outlier mixture counts as members of the curve
        (``responsibility >= 0.5``). All-``True`` when ``outlier_mixture`` was
        ``None``.
    responsibility : Float[Array, " N"]
        Posterior probability that each datum came from the curve rather than
        the background, under the final prototypes. All ones when
        ``outlier_mixture`` was ``None``.

    """

    prototype_positions: ArrayComponents
    prototype_velocities: ArrayComponents
    kept: Bool[Array, " N"]
    responsibility: Float[Array, " N"]


@dataclass(frozen=True, kw_only=True)
class SOMMixture:
    r"""Mixture-model outlier rejection for :func:`fit` and ``SOMOrderer``.

    Each datum's position comes either from the curve or from a flat
    background,

    .. math::

        \mathcal{L}_n = \pi\, f(q_n) + (1 - \pi)\, \rho_{\mathrm{bg}} ,

    where :math:`f` is a Gaussian tube of width :math:`s` about the polyline
    through the prototype positions, open at its two ends. Every epoch computes
    the posterior membership :math:`r_n` of each datum (the E-step), fits
    :math:`\pi` and :math:`s` by their closed-form M-steps, and counts each
    datum by :math:`\omega_n r_n` in the prototype update. A weight
    :math:`\omega_n` multiplies every sum and never enters :math:`r_n`.
    :doc:`/explanation/som` explains the model.

    The background needs room outside the curve. Samples that fill their own
    bounding box, as the pixels of a mask or an image do, are explained better
    by the background than by the curve, and EM ends with most of them called
    background. A fit whose member fraction :math:`\pi` ends below 0.5 has
    fitted the box rather than the curve, so it keeps every datum, as without
    the mixture.

    Attributes
    ----------
    background_density : float | None
        :math:`\rho_{\mathrm{bg}}`, in inverse position units to the power
        ``D``. ``None`` (default) uses the reciprocal volume of the positions'
        bounding box (:func:`~phasecurvefit.nn.uniform_background_density`).
    background_inflate : float
        Scales each axis of that bounding box when ``background_density`` is
        ``None``. Raise it when the data do not fill the field.

    Examples
    --------
    >>> from phasecurvefit import som
    >>> som.SOMMixture().background_density is None
    True

    """

    background_density: float | None = None
    background_inflate: float = 1.0

    __citation__: ClassVar[str] = "https://arxiv.org/abs/1008.4686"

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.background_density is not None and self.background_density <= 0:
            msg = f"background_density must be positive, got {self.background_density}."
            raise ValueError(msg)
        if self.background_inflate <= 0:
            msg = f"background_inflate must be positive, got {self.background_inflate}."
            raise ValueError(msg)


def _log_ndtr_diff(lo: Array, hi: Array, /) -> Array:
    """``log(Phi(hi) - Phi(lo))`` for ``lo <= hi``, accurate in both tails.

    Both arguments far above zero would round ``Phi`` to 1 for each and
    the difference to 0, so that side is reflected (``Phi(hi) - Phi(lo)
    = Phi(-lo) - Phi(-hi)``) and every evaluation happens where ``Phi`` is
    small and ``log_ndtr`` is exact.
    """
    flip = lo > 0
    a = jnp.where(flip, -hi, lo)
    b = jnp.where(flip, -lo, hi)
    log_b = jax.scipy.special.log_ndtr(b)
    # ``a < b``, so the exponent is negative; ``-expm1`` keeps the small-
    # interval end accurate.
    return log_b + jnp.log(-jnp.expm1(jax.scipy.special.log_ndtr(a) - log_b))


def _tube_e_step(
    q: Float[Array, "N D"],
    proto_q: Float[Array, "K D"],
    omega: Float[Array, " N"],
    pi: FSz0,
    s2: FSz0,
    log_bg: FSz0,
    /,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Membership of the curve, and the squared distance the width is fitted to.

    The curve is the polyline through the prototypes. Each of its ``K - 1``
    segments carries a share of the members, spread uniformly along it and
    scattered about it by an isotropic Gaussian of width ``s``. A segment from
    ``a`` with unit direction ``u`` and length ``l`` then has the density

    .. math::

        \frac{1}{l}\, \mathcal{N}_{D-1}(d_\perp; s)\,
        \left[\Phi\!\left(\frac{l - t}{s}\right)
              - \Phi\!\left(\frac{-t}{s}\right)\right],
        \qquad t = (q - a) \cdot u ,

    a tube about the segment with Gaussian caps at its ends. The first and
    last segments have no outer cap: the curve's ends are open, as in
    :func:`chord`, so the bracket is 1 on that side. A segment's share is the
    weighted fraction of the data it explains, counting each datum by its
    share-free density under each segment. A Gaussian per prototype
    instead is lumpy along the track wherever ``s`` is below the prototype
    spacing, and members between prototypes look like background (on a clean
    arc, 45 of 300 members were rejected).

    Returns ``(r, d2)``: the posterior membership,
    ``sigmoid(log_fg - log_bg)`` as in
    :func:`~phasecurvefit.nn.membership_responsibility`, and each datum's
    squared distance to the polyline, which the width's M-step uses.
    """
    n_dims = q.shape[1]
    a = proto_q[:-1]
    seg = proto_q[1:] - a
    length = _safe_norm(seg)
    s = jnp.sqrt(s2)
    u = seg / jnp.where(length > 0, length, 1.0)[:, None]
    # A zero-length segment (coincident prototypes) has no direction; floored
    # at a sliver of the width, it is a point Gaussian.
    length = jnp.maximum(length, 1e-3 * s)
    first = jnp.arange(a.shape[0]) == 0
    last = jnp.arange(a.shape[0]) == a.shape[0] - 1
    norm = 0.5 * (n_dims - 1) * jnp.log(2.0 * jnp.pi * s2) + jnp.log(length)

    def one(q_n: Float[Array, " D"]) -> tuple[Array, Array]:
        # Per datum, as in `_distance_matrix`, so the (K - 1, D) residuals are
        # never an (N, K - 1, D) array.
        rel = q_n - a
        t = jnp.sum(rel * u, axis=-1)
        perp2 = jnp.maximum(jnp.sum(rel**2, axis=-1) - t**2, 0.0)
        # The open ends. With Gaussian caps there, the data past a lattice that
        # the wide early neighbourhood has drawn in look like background, stop
        # pulling its ends back out, and the curve stays short: on a clean
        # arc, half the members were rejected.
        lo = jnp.where(first, -jnp.inf, -t / s)
        hi = jnp.where(last, jnp.inf, (length - t) / s)
        log_seg = -0.5 * perp2 / s2 + _log_ndtr_diff(lo, hi) - norm
        # Squared distance to the polyline, from the nearest point of the
        # nearest segment. Not an average over the segments by their share of
        # the density: at a large width that share reaches distant stretches
        # of the same curve, and the width fed on itself and ran away.
        lo_t = jnp.where(first, -jnp.inf, 0.0)
        hi_t = jnp.where(last, jnp.inf, length)
        d2 = perp2 + (t - jnp.clip(t, lo_t, hi_t)) ** 2
        return log_seg, jnp.min(d2)

    log_seg, d2 = jax.vmap(one)(q)
    # Each segment's share of the members is the weighted fraction of all the
    # data it explains, re-estimated every epoch from the data alone. Equal
    # shares failed when one outlier captured an end prototype: the segment
    # out to it, holding one datum, still claimed 1/(K - 1) of the members, so
    # the outlier stayed a member. Weighting the shares by membership instead
    # fed back on itself, and segments that lost members lost their share.
    # Each datum is split over the segments by its density under each, not
    # given to the nearest: a datum outside a bend is equally near both
    # segments meeting there, and a hard count flipped on rounding, which made
    # the fit jump (by 0.03 in float64 between weights and the copies they
    # stand for).
    count = omega @ jax.nn.softmax(log_seg, axis=1)
    tiny = jnp.finfo(count.dtype).tiny  # type: ignore[no-untyped-call]  # jnp.finfo is untyped
    log_share = jnp.log(jnp.maximum(count, tiny)) - jnp.log(
        jnp.maximum(jnp.sum(count), tiny)
    )
    log_fg = jax.nn.logsumexp(log_seg + log_share, axis=1)
    r = jax.nn.sigmoid(jnp.log(pi) - jnp.log1p(-pi) + log_fg - log_bg)
    return r, d2


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
    outlier or because the lattice is too coarse to reach it.

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


def _check_schedule(
    n_prototypes: int,
    n_epochs: int,
    sigma_start: float | None,
    sigma_end: float,
    /,
) -> float:
    """Validate `fit`'s lattice and schedule, and return the starting width."""
    if n_epochs < 1:
        msg = f"n_epochs must be >= 1, got {n_epochs}."
        raise ValueError(msg)
    if sigma_end <= 0:
        msg = f"sigma_end must be positive, got {sigma_end}."
        raise ValueError(msg)
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
    return start


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
    outlier_mixture: SOMMixture | None = None,
    mask: Bool[Array, " N"] | None = None,
) -> FitResult:
    r"""Train a 1-D SOM by batch Kohonen updates.

    Each epoch assigns every datum to its best-matching unit, then replaces
    every prototype by the neighbourhood-weighted mean of all the data:

    .. math::

        p_k \leftarrow \frac{\sum_n h_{c(n),k}\, \omega_n r_n\, w_n}
                             {\sum_n h_{c(n),k}\, \omega_n r_n},
        \qquad h_{ij} = \exp\!\left(-\frac{(i-j)^2}{2\sigma^2}\right)

    The neighbourhood is equation (A8) of the paper on the paper's linear
    lattice. The update replaces the paper's online form (A9)/(A10): there is no
    learning rate. This batch form is the fixed point of the conventional
    online Kohonen update, whose increment is proportional to ``w - p^(k)``
    (:doc:`/explanation/som` notes how that differs from (A9) as printed). ``omega_n``
    is ``weights``, defaulting to ``1`` for every datum. ``r_n`` is the datum's
    membership of the curve under ``outlier_mixture``, and ``1`` without it.
    The paper has neither term, so with the defaults the sum above is exactly
    the un-weighted one. A non-finite velocity component is left out of both
    sums for that component, so a missing velocity cannot poison every
    prototype through the neighbourhood.

    ``sigma`` anneals geometrically from ``sigma_start`` to ``sigma_end``, so the
    global ordering forms first and local detail is refined afterwards.
    ``sigma_end`` is reached on the *last* epoch, so ``n_epochs=1`` runs its one
    epoch at ``sigma_start`` and never narrows; pass ``sigma_start=sigma_end``
    to train a single epoch at the final width.

    Setting ``outlier_mixture`` models each datum's position as drawn either
    from the curve or from a flat background (:class:`SOMMixture`). Each epoch
    computes every datum's membership ``r_n`` from the current prototypes, fits
    the mixing fraction and the curve's width by their closed-form M-steps, and
    then updates the prototypes as above. Early on the lattice is a rough copy
    of the curve, so the memberships are computed with the curve widened by a
    tenth of ``sigma_t`` times the mean prototype spacing, where ``sigma_t``
    anneals geometrically from ``max(sigma_start, K / 4)`` to ``sigma_end``
    over the epochs. The widening is zero at the last epoch. It anneals from
    at least ``K / 4`` even when ``sigma_start`` is narrow, as it is chained
    after another orderer, where the lattice starts from bin means that cut
    inside the curve's bends. A weight
    multiplies every sum, the M-steps included, so a datum of weight ``w``
    counts like ``w`` coincident copies; it never enters ``r_n``. A fit whose
    mixing fraction ends below 0.5 keeps every datum, with ``responsibility``
    1: it called most of the data background, which is what happens when the
    data fill their own bounding box.

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
        shape.
    outlier_mixture
        A :class:`SOMMixture` to reject outliers, or ``None`` (default) for none.
    mask
        Which data take part, shape ``(N,)``. ``None`` (default) means all of
        them. A ``False`` datum is treated as absent: its weight is ``0`` in
        every sum, the mixture's included, and it is ``False`` in ``kept`` with
        ``responsibility`` 0. Unlike dropping it, this keeps every shape static,
        so a mask known only at run time still traces. A masked datum must
        still hold finite positions, since a weight of ``0`` times a non-finite
        value is not ``0``, and it still counts towards the bounding box of the
        default ``background_density``.

    Returns
    -------
    FitResult
        Trained prototype positions and velocities, each datum's
        ``responsibility``, and the boolean ``kept`` mask
        ``responsibility >= 0.5``.

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

    With one wild outlier, the mixture finds it:

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
    ...     outlier_mixture=som.SOMMixture(),
    ... )
    >>> bool(result.kept[25])
    False
    >>> int(result.kept.sum())
    49

    """
    _check_matching_keys(
        proto_positions=proto_positions,
        proto_velocities=proto_velocities,
        positions=positions,
        velocities=velocities,
    )
    _check_symmetric(metric)
    keys = tuple(sorted(proto_positions))
    n_prototypes = _n_rows(proto_positions)
    start = _check_schedule(n_prototypes, n_epochs, sigma_start, sigma_end)

    data_q = jt.map(jnp.asarray, positions)
    data_p = jt.map(jnp.asarray, velocities)
    ratio = sigma_end / start
    # The outlier mixture's widening anneals from at least the standalone
    # starting width, even when the lattice itself starts narrow, as it does
    # chained. Tied to the lattice's own width instead, a chained SOM got no
    # widening at all, and on a near-closed three-lobed loop with 15
    # prototypes the bin-mean lattice it starts from cut inside the lobes:
    # 306 of 450 members were rejected.
    start_t = max(start, n_prototypes / 4.0)
    ratio_t = sigma_end / start_t
    span = max(n_epochs - 1, 1)

    # The neighbourhood is evaluated on the (K, K) lattice, not an (N, K)
    # array, using
    #     h[n, k] = hkk[bmu[n], k]   =>   sum_n h[n,k] w_n
    #                                   = sum_c hkk[c,k] (sum_{bmu(n)=c} w_n)
    # Exact, not an approximation, and combines K partial sums rather than N.
    # `_check_matching_keys` made both dicts share ``keys``.
    n_pos = len(keys)
    stacked = _pack(data_q, data_p, keys)
    dtype = stacked.dtype
    # Cast before squaring, not after: the integer square would overflow int32
    # past K = 46342, and casting the (K,) lattice is cheaper than casting the
    # (K, K) result. Exact integers stop at 2**24, but those entries are far
    # off-diagonal and their Gaussian underflows to zero regardless.
    lat = jnp.arange(n_prototypes).astype(dtype)
    lattice_sq = (lat[:, None] - lat[None, :]) ** 2
    omega = (
        jnp.ones(stacked.shape[0], dtype=dtype)
        if weights is None
        else jnp.asarray(weights, dtype=dtype)
    )
    if mask is not None:
        omega = jnp.where(mask, omega, 0.0)
    # A non-finite velocity (common in catalogues: no radial velocity) is left
    # out of that component's averages rather than summed in: one NaN would
    # otherwise reach every prototype through the neighbourhood and make every
    # prototype velocity NaN. Per column, so the datum still pulls on positions
    # and its finite velocity components. Positions are not masked: a datum
    # without a position has nowhere to be placed.
    present = jnp.isfinite(stacked).at[:, :n_pos].set(True)
    clean = jnp.where(present, stacked, 0.0)
    data_qs = stacked[:, :n_pos]

    # The mixture lives in position space, like the autoencoder's. Its setup is
    # a few reductions; the scan carries its two scalars either way, and
    # ``use_mix`` (static) keeps them out of the update when it is off.
    use_mix = outlier_mixture is not None
    mix = SOMMixture() if outlier_mixture is None else outlier_mixture
    rho_bg: float | Array
    if mix.background_density is None:
        rho_bg = uniform_background_density(data_qs, inflate=mix.background_inflate)
    else:
        rho_bg = mix.background_density
    log_bg = jnp.log(jnp.asarray(rho_bg, dtype=dtype))
    # A floor on the width's square, relative to the field so it carries the
    # data's units: all-coincident data would otherwise fit s = 0 and send
    # every log density to -inf.
    extent = jnp.max(data_qs, axis=0) - jnp.min(data_qs, axis=0)
    s2_floor = 1e-12 * jnp.sum(jnp.where(extent > 0, extent, 1.0) ** 2)
    total_w = jnp.maximum(jnp.sum(omega), jnp.finfo(dtype).tiny)  # type: ignore[no-untyped-call]  # jnp.finfo is untyped
    pi_bounds = (jnp.asarray(1e-6, dtype), jnp.asarray(1.0 - 1e-6, dtype))

    # The tube scatters members across the curve, in D - 1 dimensions; a 1-D
    # field has none, and its width only shapes the curve's two ends.
    n_cross = max(n_pos - 1, 1)

    def width_m_step(wr: Array, msd: Array, s2_old: Array) -> Array:
        # Weighted by omega * r, like every other sum: a weight counts copies.
        mass = jnp.sum(wr)
        s2 = jnp.sum(wr * msd) / (n_cross * jnp.where(mass > 0, mass, 1.0))
        return jnp.maximum(jnp.where(mass > 0, s2, s2_old), s2_floor)

    def epoch(
        carry: tuple[VectorComponents, VectorComponents, Array, Array], step: Array
    ) -> tuple[tuple[VectorComponents, VectorComponents, Array, Array], None]:
        pq, pp, pi, s2 = carry
        bmu = jnp.argmin(
            _distance_matrix(metric, metric_scale, data_q, data_p, pq, pp), axis=1
        )
        # `step` comes from `arange(n_epochs)`, which is int64 under x64; left
        # alone the division promotes sigma -- and the whole update with it --
        # to float64, and the scan carry no longer matches its float32 input.
        sigma = start * ratio ** (step.astype(dtype) / span)
        if use_mix:
            # Early on, the lattice is a rough copy of the curve: smoothed on
            # the scale of sigma * spacing while the neighbourhood is wide, or
            # the bin means it was seeded with. The data scatter about it more
            # widely than about the final lattice. Without widening the curve
            # for it, members the early lattice misses are called background,
            # stop pulling it, and stay lost: on a clean arc trained 40 epochs,
            # 256 of 300. The widening falls to zero by the last epoch.
            proto_qs = _stack(pq, keys)
            spacing = jnp.sum(_safe_norm(jnp.diff(proto_qs, axis=0))) / (
                n_prototypes - 1
            )
            sigma_t = start_t * ratio_t ** (step.astype(dtype) / span)
            anneal = (_MIXTURE_ANNEAL * spacing) ** 2 * jnp.maximum(
                sigma_t**2 - sigma_end**2, 0.0
            )
            r, msd = _tube_e_step(data_qs, proto_qs, omega, pi, s2 + anneal, log_bg)
            pi = jnp.clip(jnp.sum(omega * r) / total_w, *pi_bounds)
            s2 = width_m_step(omega * r, msd, s2)
            wr = omega * r
        else:
            wr = omega
        # Segment reductions rather than an (N, K) one-hot. The one-hot fed two
        # consumers -- a column sum and a matmul -- which blocked XLA from
        # fusing the whole (N, K) chain, so it materialized: `fit` peaked at
        # 416 MB for N=1e6, K=100 against 16 MB this way.
        counts = jax.ops.segment_sum(
            wr[:, None] * present, bmu, num_segments=n_prototypes
        )
        totals = jax.ops.segment_sum(
            clean * wr[:, None], bmu, num_segments=n_prototypes
        )
        neighbourhood = jnp.exp(-lattice_sq / (2.0 * sigma**2))
        weight = neighbourhood.T @ counts
        # A lattice unit no datum reaches has zero weight *and* zero numerator
        # (the Gaussian underflows in float32 ~11 units from the nearest
        # best-matching unit); likewise a velocity component every nearby datum
        # lacks. Leave it where it is rather than dividing 0/0.
        live = weight > jnp.finfo(dtype).tiny  # type: ignore[no-untyped-call]  # jnp.finfo is untyped
        means = (neighbourhood.T @ totals) / jnp.where(live, weight, 1.0)
        means = jnp.where(live, means, _pack(pq, pp, keys))
        return (*_unpack(means, keys), pi, s2), None

    init_q = jt.map(jnp.asarray, proto_positions)
    init_p = jt.map(jnp.asarray, proto_velocities)
    # Width seeded from each datum's nearest initial prototype, and an even
    # mixing fraction; EM refines both from the first epoch. Skipped when the
    # mixture is off, which would otherwise pay for a distance matrix.
    s2_init = jnp.asarray(1.0, dtype)
    if use_mix:
        nearest = bmu_distance(
            SpatialDistanceMetric(), 0.0, data_q, data_p, init_q, init_p
        )
        s2_init = width_m_step(omega, nearest**2, s2_init)
    init = (init_q, init_p, jnp.asarray(0.5, dtype), s2_init)
    (trained_q, trained_p, pi, s2), _ = jax.lax.scan(epoch, init, jnp.arange(n_epochs))

    n = stacked.shape[0]
    member = jnp.ones(n, dtype=bool) if mask is None else jnp.asarray(mask, bool)
    if not use_mix:
        return FitResult(trained_q, trained_p, member, member.astype(dtype))
    resp, _ = _tube_e_step(data_qs, _stack(trained_q, keys), omega, pi, s2, log_bg)
    # A fit that calls most of the data background has fitted the box, not
    # the curve: samples that fill their own bounding box (a mask's pixels)
    # are explained better by the flat background than by any tube, so EM
    # drifts there (pi 0.33 on a straight mask of half-width 0.3). Neither a
    # larger field nor a floor on pi in the M-step fixed that without losing
    # heavily contaminated arcs, whose pi stays above 0.76. Such a fit keeps
    # every datum, as without the mixture.
    resp = jnp.where(pi < _MIN_MEMBER_FRACTION, jnp.ones_like(resp), resp)
    # Masked data stay out either way.
    resp = jnp.where(member, resp, 0.0)
    return FitResult(trained_q, trained_p, member & (resp >= 0.5), resp)


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
    eps = jnp.finfo(s.dtype).eps * jnp.where(s[-1] > 0, s[-1], 1.0)  # type: ignore[no-untyped-call]  # jnp.finfo is untyped
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
) -> tuple[ArrayComponents, ArrayComponents]:
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
    n_prototypes = _n_rows(proto_positions)
    if n_prototypes < 2:
        msg = (
            f"densify needs at least 2 prototypes to form a backbone, got "
            f"{n_prototypes}. A single vertex has no segment to measure arc "
            f"length along, and `chord` cannot project onto it."
        )
        raise ValueError(msg)

    keys = tuple(sorted(proto_positions))  # shared, per `_check_matching_keys`
    n_q = len(keys)
    P = _pack(proto_positions, proto_velocities, keys)
    curve = _resample_uniform(
        _catmull_rom(P, factor=factor, knot_cols=n_q), knot_cols=n_q
    )
    return _unpack(curve, keys)


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
    near-closed loop can capture each other; :doc:`/explanation/som` gives the
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
    n_vertices = _n_rows(backbone_positions)
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


class SOM1D(ReprMixin, eqx.Module):
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
        return _n_rows(self.prototype_positions)

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

    def backbone(self) -> tuple[ArrayComponents, ArrayComponents]:
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
