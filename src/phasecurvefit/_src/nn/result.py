"""Autoencoder result types with interpolation capability."""

__all__: tuple[str, ...] = ("AutoencoderResult", "fill_ordering_gaps")

from dataclasses import KW_ONLY
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from scipy.spatial import cKDTree

from .abstractautoencoder import AbstractAutoencoder
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.custom_types import BSzN, FSzN, ISzN, VectorComponents


class AutoencoderResult(AbstractResult):
    r"""Result of autoencoder training and prediction with interpolation.

    Extends AbstractResult with training outputs (gamma, membership_prob) and
    provides spatial interpolation via the trained decoder network.

    Attributes
    ----------
    positions : VectorComponents
        Original position data (dict with 1D arrays).
    velocities : VectorComponents
        Original velocity data (dict with 1D arrays).
    indices : ISzN
        Indices sorted by gamma value.
    gamma_range : tuple[float, float]
        Valid range for the ordering parameter: calling the result accepts any
        $\gamma$ in it, including every tracer's own.
    track_range : tuple[float, float] | None
        The range of $\gamma$ the track covers, for drawing it:
        ``result(jnp.linspace(*result.track_range, n))``. Each end is where
        the weighted number of member tracers per unit length along the track
        falls below half its value just inside, so faint tracers past the end
        of a stream -- the blurred tail of a density field -- don't carry the
        track past the end, while a stream that narrows or fades keeps its
        ends. Within ``gamma_range``. ``None`` for a result built by hand;
        training sets it.
    gamma : FSzN
        Ordering parameters for all tracers.
    membership_prob : Array
        Membership probabilities for all tracers.
    model : AbstractAutoencoder
        Trained autoencoder model with decoder for spatial interpolation.

    """

    positions: VectorComponents
    velocities: VectorComponents
    indices: ISzN
    gamma: FSzN
    membership_prob: FSzN
    model: AbstractAutoencoder
    _: KW_ONLY
    gamma_range: tuple[float, float]
    track_range: tuple[float, float] | None = None

    def __call__(
        self, gamma: Array, /, *, key: PRNGKeyArray | None = None
    ) -> VectorComponents:
        r"""Interpolate spatial positions using the decoder network.

        Parameters
        ----------
        gamma : Array
            Ordering parameter in [min_gamma, max_gamma], shape (...).
            Will be clipped to valid range and normalized for interpolation.
        key : PRNGKeyArray, optional
            JAX random key passed to the decoder for stochastic evaluation.
            If None, uses deterministic evaluation with a zero key. Default is None.

        Returns
        -------
        positions : VectorComponents
            Interpolated spatial positions from the decoder with same shape
            (...) as gamma.

        """
        # Error if gamma is out of range
        min_gamma, max_gamma = self.gamma_range
        gamma = eqx.error_if(
            gamma,
            (gamma < min_gamma) | (gamma > max_gamma),
            "gamma must be within the valid gamma_range",
        )

        return self.model.decode(gamma, key=key)


def fill_ordering_gaps(
    model: AbstractAutoencoder,
    result: AbstractResult,
    /,
    prob_threshold: float = 0.5,
) -> AutoencoderResult:
    r"""Use trained autoencoder to fill gaps in phase-flow walk ordering.

    This function predicts $\gamma$ values for all tracers (including those
    skipped by phase-flow walk) and returns a complete ordering.

    Parameters
    ----------
    model : PathAutoencoder
        Trained autoencoder model.
    result : AbstractResult
        Result from an orderer.
    prob_threshold : float, optional
        Minimum membership probability to include. Default: 0.5.

    Returns
    -------
    result : AutoencoderResult
        Complete ordering including previously skipped tracers.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = pcf.order(pos, vel)
    >>> keys = jax.random.split(jax.random.key(0), 2)
    >>> normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
    >>> model = pcf.nn.PathAutoencoder.make(
    ...     normalizer, gamma_range=result.gamma_range, key=keys[0]
    ... )
    >>> cfg = pcf.nn.TrainingConfig(show_pbar=False)
    >>> result, *_ = pcf.nn.train_autoencoder(model, result, config=cfg, key=keys[1])

    """
    return _encode_to_result(
        model,
        result.positions,
        result.velocities,
        prob_threshold=prob_threshold,
        gamma_range=result.gamma_range,
        weights=getattr(result, "weights", None),
    )


def _encode_to_result(
    model: AbstractAutoencoder,
    q: VectorComponents,
    p: VectorComponents,
    /,
    *,
    prob_threshold: float,
    gamma_range: tuple[float, float],
    weights: FSzN | None = None,
    track_ends: Literal["half_max", "mask"] = "half_max",
) -> AutoencoderResult:
    """Encode every tracer; order by gamma, keeping those with p >= threshold."""
    gamma, prob = model.encode(q, p)
    order = jnp.argsort(gamma)
    return AutoencoderResult(
        positions=q,
        velocities=p,
        indices=order[prob[order] >= prob_threshold],
        gamma_range=gamma_range,
        track_range=_track_range(
            model, q, gamma, prob >= prob_threshold, weights, ends=track_ends
        ),
        gamma=gamma,
        membership_prob=prob,
        model=model,
    )


def _track_range(
    model: AbstractAutoencoder,
    q: VectorComponents,
    gamma: FSzN,
    member: BSzN,
    weights: FSzN | None,
    /,
    *,
    ends: Literal["half_max", "mask"] = "half_max",
) -> tuple[float, float]:
    """Find the span of gamma between the ends of the curve the samples trace.

    ``ends="half_max"`` (default) cuts where the line density falls to half,
    as below. ``ends="mask"`` instead reads the ends from the shape of a mask's
    caps (`_mask_ends`), for images made by thickening a curve.

    Works in physical length: decode the track over the members' span of
    gamma, project every member onto it
    (extending past each end along the tangent there), and cut each end where
    the weighted number of members per unit length falls below half its local
    value (`_trim_ends`). Physical length, because near an end the encoder
    squeezes many samples into little gamma, so counting per unit gamma makes a
    faint tail look dense. Local, because a non-uniform stream's line density
    varies along it: a narrow or faint stretch near an end is not an end.

    For a curve blurred by any symmetric kernel the line density at the true end
    is half its value just inside, so this is the half-maximum end. A mask
    built by thickening a curve has a cap past each end that holds more weight,
    so the cut falls partway into the cap.

    Eager: it runs in NumPy when a trained result is built, which happens
    outside ``jit``. With no members it falls back to the span of all the
    gamma; with members of total weight 0, or a track that can't be decoded,
    to the members' span.
    """
    if ends not in ("half_max", "mask"):
        msg = f"track_ends must be 'half_max' or 'mask', got {ends!r}."
        raise ValueError(msg)
    g = np.asarray(gamma, dtype=float)
    keep = np.asarray(member, dtype=bool)
    if not keep.any():  # nothing to trim against
        return float(g.min()), float(g.max())
    w = np.ones_like(g) if weights is None else np.asarray(weights, dtype=float)
    g, w = g[keep], w[keep]
    lo, hi = float(g.min()), float(g.max())
    if not (w > 0).any() or hi <= lo:  # nothing to trim by
        return lo, hi

    grid = np.linspace(lo, hi, 400)
    keys = list(q)
    track = _magnitudes(model.decode(jnp.asarray(grid)), keys)
    finite = np.isfinite(track).all(axis=1)  # a running mean can leave gaps
    grid, track = grid[finite], track[finite]
    if len(grid) < 2:
        return lo, hi
    s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(track, axis=0), axis=1))]
    )
    tangent = np.gradient(track, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-300)
    pts = _magnitudes(q, keys)[keep]
    _, near = cKDTree(track).query(pts)
    offset = pts - track[near]
    step = np.einsum("ij,ij->i", offset, tangent[near])
    along = s[near] + step

    if ends == "mask":
        across = np.linalg.norm(offset - step[:, None] * tangent[near], axis=1)
        s_lo, s_hi = _mask_ends(along, across, s[-1])
    else:
        s_lo, s_hi = _trim_ends(along, w)
    s_lo, s_hi = np.clip([s_lo, s_hi], 0.0, s[-1])
    # np.interp needs strictly increasing lengths; repeated vertices (a running
    # mean's plateaus) give equal ones, and all their gamma decode to one point.
    s, first = np.unique(s, return_index=True)
    grid = grid[first]
    return float(np.interp(s_lo, s, grid)), float(np.interp(s_hi, s, grid))


def _magnitudes(comps: VectorComponents, keys: list[str], /) -> np.ndarray:
    """Stack position components as float64 magnitudes, shape ``(N, D)``.

    Quantities are taken in the first component's unit, so lengths along the
    track are physical whatever units the components carry. Float64 because the
    tangent's 1e-300 guard underflows to 0 in float32.
    """
    unit = getattr(comps[keys[0]], "unit", None)
    return np.stack(
        [
            np.asarray(comps[k] if unit is None else comps[k].ustrip(unit), dtype=float)
            for k in keys
        ],
        axis=1,
    )


def _mask_ends(along: FSzN, across: FSzN, length: float, /) -> tuple[float, float]:
    """Read each end of a thickened curve from the shape of its cap.

    A mask made by thickening a curve by a half-width ``w`` has a cap past each
    end that reaches exactly ``w`` beyond it. So the end is the tip, the
    farthest any sample reaches along the track, minus the half-width just
    inside it: the widest any sample reaches across the track within two
    widths of that end (first guessed from the outer 15% of the track). This
    uses where samples are, not their weights, so the end shifts linearly with
    the cap rather than through a slow fall in density.
    """
    a = np.asarray(along, dtype=float)
    r = np.asarray(across, dtype=float)

    def half_width(inner: np.ndarray, /) -> float:
        # Widest reach across the track among samples that project onto the
        # stretch `inner` of the track measured from this end.
        def widest(d: float, /) -> float:
            near = (inner >= 0) & (inner <= d)
            return float(r[near].max()) if near.any() else 0.0

        first = widest(0.15 * length)
        return widest(2 * first) if first > 0 else first

    s_lo = a.min() + half_width(a)
    s_hi = a.max() - half_width(length - a)
    if s_hi <= s_lo:  # too short to have caps: keep everything
        return float(a.min()), float(a.max())
    return float(s_lo), float(s_hi)


def _trim_ends(
    along: FSzN, weights: FSzN, /, *, frac: float = 0.5, window: int = 3
) -> tuple[float, float]:
    """Cut each end where the line density falls below ``frac`` of its local value.

    Bins ``along`` (distance along the track), one bin per 10 samples up to 40.
    From each end, the end is the first bin holding at least ``frac`` times the
    median weight of the next ``window`` bins inward. A cap falls off within a
    bin or two, while widening, brightening and fading vary over many bins, so
    only the cap is cut. A faint stretch longer than ``window`` bins counts as
    part of the stream: it is too long to be a cap.

    An end bin also needs the ``window`` bins inward of it to be non-empty. A few
    interlopers that the track bends out to reach fill scattered bins with empty
    ones between them, and would otherwise pass as a faint tail. If no bin
    passes that test on both sides, as for sparse samples, the test is dropped.
    """
    a = np.asarray(along, dtype=float)
    w = np.asarray(weights, dtype=float)
    n_bins = int(np.clip(a.size // 10, 1, 40))
    edges = np.linspace(a.min(), a.max(), n_bins + 1)
    count, _ = np.histogram(a, edges, weights=w)

    def first(c: np.ndarray, /, *, contiguous: bool) -> int:
        for j in range(len(c) - 1):
            inward = c[j + 1 : j + 1 + window]
            if contiguous and not (inward > 0).all():
                continue
            if c[j] > 0 and c[j] >= frac * np.median(inward):
                return j
        # No bin passed: with the contiguity test, past the far end, so that
        # lo > hi below and the scan reruns without it.
        return len(c) if contiguous else len(c) - 1

    def scan(*, contiguous: bool) -> tuple[int, int]:
        lo = first(count, contiguous=contiguous)
        return lo, len(count) - 1 - first(count[::-1], contiguous=contiguous)

    # Without the contiguity test the fullest bin passes from both sides (its
    # inward median is at most itself), so the scan from the left stops at or
    # before it, the scan from the right at or after it, and lo <= hi.
    lo, hi = scan(contiguous=True)
    if lo > hi:
        lo, hi = scan(contiguous=False)
    return float(edges[lo]), float(edges[hi + 1])
