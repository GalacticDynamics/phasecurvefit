"""One-call, end-to-end path fitting: order, refine, and train.

Collapses the multi-step recipe in the quickstart and autoencoder guides
(:func:`~phasecurvefit.orderers.default_pipeline` feeding a
:class:`~phasecurvefit.nn.PathAutoencoder`) into a single function, for
callers who want a fitted track without assembling the steps themselves.
"""

__all__: tuple[str, ...] = ("TrackEnsemble", "bootstrap_track", "fit_track")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from phasecurvefit._src._repr import ReprMixin
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.nn import (
    AutoencoderResult,
    PathAutoencoder,
    StandardScalerNormalizer,
    TrainingConfig,
    train_autoencoder,
)
from phasecurvefit._src.orderers.default_pipeline import default_pipeline
from phasecurvefit._src.som import SOMMixture

_DEFAULT_MIXTURE = SOMMixture()


def fit_track(
    positions: VectorComponents,
    velocities: VectorComponents,
    *,
    key: PRNGKeyArray,
    metadata: StateMetadata | None = None,
    weights: Float[Array, " N"] | None = None,
    n_prototypes: int = 15,
    outlier_mixture: SOMMixture | None = _DEFAULT_MIXTURE,
    training_config: TrainingConfig | None = None,
) -> tuple[AbstractResult, AutoencoderResult, Float[Array, " n_epochs"]]:
    """Order, refine, and fit a smooth track through phase-space tracers.

    Runs :func:`~phasecurvefit.orderers.default_pipeline` (the MST backbone
    refined by a SOM stage), builds a
    :class:`~phasecurvefit.nn.PathAutoencoder` sized to match, and trains it
    with :func:`~phasecurvefit.nn.train_autoencoder` -- the same three steps
    as the quickstart and autoencoder guides, in one call.

    For control over any individual step (a different orderer, a
    pre-built model, a decoder swap, ...), call the pieces directly instead;
    this is the convenience path, not a replacement for them.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers.
    key
        Split once for model initialization and once for training.
    metadata
        Passed through to :func:`~phasecurvefit.orderers.default_pipeline`.
    weights
        Per-tracer weights, e.g. the density at each pixel of a density field.
        Passed to :func:`~phasecurvefit.orderers.default_pipeline`, whose result
        carries them on to :func:`~phasecurvefit.nn.train_autoencoder`.
    n_prototypes
        Passed to :func:`~phasecurvefit.orderers.default_pipeline`.
    outlier_mixture
        A :class:`~phasecurvefit.som.SOMMixture` for the SOM stage of
        :func:`~phasecurvefit.orderers.default_pipeline`, which then leaves out
        of the ordering any tracer the curve-or-background mixture calls
        background. The encoder learns those tracers as non-members, so the
        track does not bend out to reach interlopers at the stream's tips.
        Defaults to ``SOMMixture()``. A fit that calls most tracers background,
        as it does when they fill their own bounding box like a mask's pixels,
        keeps them all. ``None`` keeps every tracer in the ordering. With fewer
        than ``n_prototypes`` tracers the SOM stage, and so the rejection, is
        skipped. It works with ``weights``.
    training_config
        Passed to :func:`~phasecurvefit.nn.train_autoencoder`. ``None`` uses
        its own defaults.

    Returns
    -------
    ordering_result : AbstractResult
        What :func:`~phasecurvefit.orderers.default_pipeline` produced --
        useful on its own for diagnosing the ordering stage independently of
        the fit.
    autoencoder_result : AutoencoderResult
        The trained result: call it with a ``gamma`` value to interpolate
        along the fitted track, or read ``.model`` for the network itself.
    losses : Array
        Training losses, for diagnosing convergence.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> ang = jnp.linspace(0.0, jnp.pi, 60)
    >>> pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    >>> vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    >>> cfg = pcf.nn.TrainingConfig(
    ...     n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
    ... )
    >>> ordering, fitted, losses = pcf.fit_track(
    ...     pos, vel, key=jax.random.key(0), n_prototypes=12, training_config=cfg
    ... )
    >>> int(ordering.n_visited)
    60
    >>> fitted.model.normalizer.n_spatial_dims
    2

    """
    key_model, key_train = jr.split(key)
    ordering_result = default_pipeline(
        positions,
        velocities,
        metadata=metadata,
        weights=weights,
        n_prototypes=n_prototypes,
        outlier_mixture=outlier_mixture,
    )
    normalizer = StandardScalerNormalizer(positions, velocities)
    model = PathAutoencoder.make(
        normalizer=normalizer, gamma_range=ordering_result.gamma_range, key=key_model
    )
    autoencoder_result, _opt_state, losses = train_autoencoder(
        model, ordering_result, config=training_config, key=key_train
    )
    return ordering_result, autoencoder_result, losses


class TrackEnsemble(ReprMixin, eqx.Module):
    r"""A fit to the data and fits to bootstrap resamples of it.

    Made by :func:`bootstrap_track`. ``reference`` is the track fitted to all
    the data, and ``members`` are the tracks fitted to resamples. The spread of
    the members about the reference is the track's uncertainty.

    Each fit has its own $\gamma$, so members are compared with the reference
    in space: at a point on the reference, a member's offset is the vector
    from that point to the member's nearest point, minus its component along
    the reference's tangent.

    Attributes
    ----------
    reference : AutoencoderResult
        The track fitted to all the data.
    members : tuple[AutoencoderResult, ...]
        The tracks fitted to the resamples.

    """

    reference: AutoencoderResult
    members: tuple[AutoencoderResult, ...]

    def offsets(
        self, gamma: Float[Array, " G"], /, *, n_dense: int = 1000
    ) -> Float[Array, "M G D"]:
        r"""Offset of each member from the reference, across the reference.

        Parameters
        ----------
        gamma
            Points on the reference, as its own $\gamma$ (within
            ``reference.gamma_range``).
        n_dense
            Points at which each member's track is evaluated, over its
            ``track_range``, to find the one nearest each reference point.

        Returns
        -------
        Array, shape (n_members, len(gamma), n_spatial_dims)
            The perpendicular offsets, in the order of the position
            components. NaN where a member's nearest point is one of its
            ends, which means its track stops short of that reference point.

        """
        ref = self.reference
        gamma = jnp.atleast_1d(jnp.asarray(gamma))
        keys = list(ref.positions)

        def stack(q: dict[str, Array], /) -> Float[Array, "N D"]:
            return jnp.stack([q[k] for k in keys], axis=-1)

        c = stack(ref.model.decode(gamma))
        dc = jax.vmap(jax.jacfwd(lambda g: stack(ref.model.decode(g))))(gamma)
        t = dc / jnp.linalg.norm(dc, axis=-1, keepdims=True)

        def one(member: AutoencoderResult, /) -> Float[Array, "G D"]:
            if member.track_range is None:  # pragma: no cover - training sets it
                msg = "every member needs a track_range."
                raise ValueError(msg)
            v = stack(member(jnp.linspace(*member.track_range, n_dense)))
            i = jnp.argmin(jnp.sum((v[None] - c[:, None]) ** 2, axis=-1), axis=1)
            d = v[i] - c
            d = d - jnp.sum(d * t, axis=-1, keepdims=True) * t
            stops_short = (i == 0) | (i == n_dense - 1)
            return jnp.where(stops_short[:, None], jnp.nan, d)

        return jnp.stack([one(m) for m in self.members])

    def radius(
        self, gamma: Float[Array, " G"], /, level: float = 0.68, *, n_dense: int = 1000
    ) -> Float[Array, " G"]:
        r"""Radius of the tube about the reference that holds ``level`` of the members.

        At each point it is the ``level`` quantile of the members' distances
        from the reference (the norms of :meth:`offsets`), leaving out
        members whose track stops short of that point. Plot the reference
        with this radius on either side of it.

        It measures the variance of the fit and not its bias, so where the
        track bends the true curve can lie outside it more often than
        ``level`` implies; see the explanation page on track uncertainty.

        """
        dist = jnp.linalg.norm(self.offsets(gamma, n_dense=n_dense), axis=-1)
        return jnp.nanquantile(dist, level, axis=0)


def bootstrap_track(
    positions: VectorComponents,
    velocities: VectorComponents,
    *,
    key: PRNGKeyArray,
    n_samples: int = 20,
    metadata: StateMetadata | None = None,
    weights: Float[Array, " N"] | None = None,
    n_prototypes: int = 15,
    outlier_clip_sigma: float | None = 3.0,
    training_config: TrainingConfig | None = None,
) -> TrackEnsemble:
    """Fit a track and its uncertainty, by refitting bootstrap resamples.

    Runs :func:`fit_track` once on all the data, then ``n_samples`` more times,
    each on ``N`` tracers drawn from the ``N`` with replacement and with its own
    training key. Every stage is refitted, so the spread of the members covers
    the ordering, the outlier clipping, the membership, the training and the
    noise in the data.

    A resample repeats some tracers and leaves out others. Repeats are passed as
    repeated rows rather than as weights, so :func:`fit_track`'s outlier
    clipping stays on.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers.
    key
        Split once for the reference fit and once for each member.
    n_samples
        Number of resamples. Each costs one :func:`fit_track`.
    weights
        Per-tracer weights, as in :func:`fit_track`. A resampled tracer keeps
        its weight. A bootstrap treats each tracer as one independent draw, so
        it gives an honest uncertainty only when the tracers are independent,
        for example stars, and not the pixels of an image.
    metadata, n_prototypes, outlier_clip_sigma, training_config
        Passed to every :func:`fit_track` call.

    Returns
    -------
    TrackEnsemble
        The full-data fit and the fits to the resamples.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> ang = jnp.linspace(0.0, jnp.pi, 60)
    >>> pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    >>> vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    >>> cfg = pcf.nn.TrainingConfig(
    ...     n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
    ... )
    >>> ens = pcf.bootstrap_track(
    ...     pos,
    ...     vel,
    ...     key=jax.random.key(0),
    ...     n_samples=2,
    ...     n_prototypes=12,
    ...     training_config=cfg,
    ... )
    >>> len(ens.members)
    2
    >>> gamma = jnp.linspace(*ens.reference.track_range, 5)
    >>> ens.offsets(gamma).shape
    (2, 5, 2)

    """
    if n_samples < 1:
        msg = f"n_samples must be >= 1, got {n_samples}."
        raise ValueError(msg)

    def fit(
        pos: VectorComponents,
        vel: VectorComponents,
        w: Float[Array, " N"] | None,
        k: PRNGKeyArray,
        /,
    ) -> AutoencoderResult:
        return fit_track(
            pos,
            vel,
            key=k,
            metadata=metadata,
            weights=w,
            n_prototypes=n_prototypes,
            outlier_clip_sigma=outlier_clip_sigma,
            training_config=training_config,
        )[1]

    n = next(iter(positions.values())).shape[0]
    key_ref, *keys = jr.split(key, n_samples + 1)
    reference = fit(positions, velocities, weights, key_ref)
    members = []
    # ponytail: a Python loop, since fit_track isn't vmappable (host-side track_range).
    for k in keys:
        key_draw, key_fit = jr.split(k)
        idx = jr.randint(key_draw, (n,), 0, n)
        members.append(
            fit(
                {c: v[idx] for c, v in positions.items()},
                {c: v[idx] for c, v in velocities.items()},
                None if weights is None else jnp.asarray(weights)[idx],
                key_fit,
            )
        )
    return TrackEnsemble(reference=reference, members=tuple(members))
