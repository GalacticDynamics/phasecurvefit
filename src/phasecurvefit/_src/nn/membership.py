r"""Mixture-model membership: outlier rejection via a generative likelihood.

Motivation
----------
The default membership head is trained as a *classifier*: phase-1
(`phasecurvefit._src.nn.order_net.encoder_loss`) pushes $p \to 1$ on every
tracer the orderer visited, and $p \to 0$ on samples drawn uniformly from the
phase-space bounding box. Two things go wrong.

1. **The positive labels come from the orderer.** An MST (or any spanning
   structure) has no reject option -- it threads outliers into the ordering
   along their cheapest edge. Those outliers then land in the positive set, and
   the network is explicitly supervised to call them members. The membership
   head becomes a *student of the orderer*, and faithfully reproduces its blind
   spot.

2. **The reconstruction residual is never used.** The quantity that actually
   *defines* an outlier -- the distance $r_n = \lVert q_n - x_\theta(\gamma_n)
   \rVert$ from the fitted track -- is computed by the decoder and then
   discarded as far as $p$ is concerned. No gradient flows from "this star is
   far from the track" to "lower its membership probability".

The result is a membership probability that is badly *calibrated*: it saturates
near 1 for anything remotely near the data manifold, so thresholding it at
``member_threshold`` separates almost nothing.

The model
---------
This module replaces the classifier with a **generative mixture model**, in the
sense of Hogg, Bovy & Lang (2010), §3 ("Pruning outliers"). Each star is drawn
either from the stream or from a smooth background:

.. math::

    \mathcal{L}_n = \pi_n \, \mathcal{N}\!\left(q_n \,;\, x_\theta(\gamma_n),\,
                     \sigma^2(\gamma_n) \mathbb{I}\right)
                  + (1 - \pi_n) \, \rho_{\mathrm{bg}}

and we minimise the negative log-likelihood $-\sum_n \log \mathcal{L}_n$.

- $\pi_n$ is the encoder's membership output. It plays the role of $(1 - P_b)$
  in Hogg et al., except that instead of a single global bad-data fraction it is
  **amortised**: a neural network predicts a per-star mixture weight from the
  star's own phase-space coordinates.
- $\mathcal{N}(\cdot)$ is the foreground (stream) density: a Gaussian of width
  $\sigma(\gamma)$ about the decoded track.
- $\rho_{\mathrm{bg}}$ is the background density -- a flat density over the
  field (Hogg et al. use a broad Gaussian $(Y_b, V_b)$; a uniform density is the
  natural analogue when the "field" is a bounded survey footprint).

Now the gradient with respect to $\pi_n$ is automatic and *correct*: for a star
close to the track the Gaussian term dominates and $\pi_n \to 1$; for a star far
from the track the background term dominates and $\pi_n \to 0$. Outlier
rejection falls out of the likelihood rather than being imposed by a
self-referential label.

It also replaces the arbitrary ``member_threshold`` with $\rho_{\mathrm{bg}}$,
which is *physically interpretable* (a background surface density). Hogg et al.
are blunt about why this matters: sigma-clipping and hand-tuned cuts are "a
procedure and not the outcome of justifiable modeling".

Why $\sigma$ varies with $\gamma$
---------------------------------
A single global $\sigma$ has a breakdown point. If a handful of outliers sit
only a few stream-widths off the track, the likelihood is happier inflating
$\sigma$ to swallow them than lowering their $\pi_n$ -- and you detect nothing.
Two defences, both implemented here:

- **$\sigma$ is a function of $\gamma$** (`WidthNet`), so a stream that
  fans out towards its ends does not force a single compromise width that is too
  wide everywhere else.
- **$\sigma$ is annealed from above** (`sigma_ceiling`). Training starts with a
  generous ceiling -- while the track is still bad, everything should look like a
  member -- and the ceiling is squeezed down over epochs. The width may go
  *below* the ceiling if the data want it to; it may not go above. This directly
  prevents the inflation failure mode.

References
----------
Hogg, D. W., Bovy, J., & Lang, D. (2010). "Data analysis recipes: Fitting a
model to data." arXiv:1008.4686. See §3, "Pruning outliers", for the mixture
model this module implements.

Nibauer, J., et al. (2022). "Charting Galactic Accelerations with Stellar
Streams and Machine Learning." ApJ 940, 22.

"""

__all__: tuple[str, ...] = (
    "MixtureMembershipConfig",
    "WidthNet",
    "membership_rampup",
    "membership_responsibility",
    "mixture_membership_loss",
    "sigma_ceiling",
    "uniform_background_density",
)

import functools as ft
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Real

from phasecurvefit._src.custom_types import FSz0, FSzN

# Membership is clamped into [_EPS, 1 - _EPS] before any log is taken.
#
# This is not (only) about avoiding log(0). The derivative of `log(pi)` is
# `1 / pi`, so at pi = 0 the gradient is 1 / _EPS: with _EPS = 1e-12 that is
# 1e12, which in float32 is close enough to the edge that a subsequent
# multiply-by-zero in a fused kernel produces NaN rather than 0. 1e-6 bounds the
# gradient at a harmless 1e6, and the difference between a membership of 1e-6 and
# one of 0 is not a distinction anyone can act on.
_EPS: float = 1e-6

# `exp` overflows to `inf` above ~88 in float32. The width network is free to
# predict any log-width; clamp it so a badly-initialised or diverging net returns
# a large-but-finite width rather than `inf` (which would poison every downstream
# residual with NaN).
_LOG_SIGMA_MAX: float = 20.0
_LOG_SIGMA_MIN: float = -20.0


class WidthNet(eqx.Module):
    r"""Width $\sigma$ of the foreground component along the ordering parameter.

    The foreground density need not have a uniform width. A stream, for example,
    is typically narrow near the progenitor and fans out towards the tidal
    tails; the same is true of many curved distributions. A single scalar $\sigma$
    forces one compromise, which is both a worse fit and -- more importantly --
    a worse *outlier detector*, because the compromise width is too generous
    wherever the stream is genuinely thin.

    This is a small MLP $\gamma \mapsto \log \sigma_{\mathrm{raw}}(\gamma)$,
    exponentiated to guarantee positivity and then capped from above by an
    annealing ceiling (see `sigma_ceiling`):

    .. math::

        \sigma(\gamma, t) = \min\!\big(
            \sigma_{\mathrm{raw}}(\gamma),\; \sigma_{\mathrm{ceil}}(t)
        \big)

    The floor is enforced by construction (``exp`` is positive) plus
    ``sigma_min``, which keeps the Gaussian from collapsing onto individual
    stars -- the classic degenerate maximum of any mixture likelihood, where one
    component shrinks to zero width around a single point and the likelihood
    diverges.

    Parameters
    ----------
    width_size, depth : int
        Hidden-layer size and number of hidden layers of the MLP.
    sigma_init : float
        Width the network is initialised to predict, everywhere. Choose
        something comparable to the expected stream width; it only sets the
        starting point.
    sigma_min : float
        Hard floor on the returned width. Guards the collapse-onto-a-star
        degeneracy. Should be well below the true stream width (e.g. the
        typical positional uncertainty).
    key : PRNGKeyArray
        JAX random key for initialisation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> import phasecurvefit as pcf

    >>> width = pcf.nn.WidthNet(sigma_init=0.2, key=jr.key(0))

    At initialisation the predicted width is close to ``sigma_init`` everywhere:

    >>> sigma = width(jnp.array(0.3))
    >>> bool(0.1 < sigma < 0.4)
    True

    It is a scalar-in, scalar-out function, so ``vmap`` it over a batch:

    >>> import jax
    >>> gammas = jnp.linspace(-1.0, 1.0, 5)
    >>> jax.vmap(width)(gammas).shape
    (5,)

    Widths are strictly positive, always:

    >>> bool(jnp.all(jax.vmap(width)(gammas) > 0))
    True

    """

    mlp: eqx.nn.MLP

    sigma_min: float = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    __citation__: ClassVar[str] = "https://arxiv.org/abs/1008.4686"

    def __init__(
        self,
        width_size: int = 32,
        depth: int = 2,
        *,
        sigma_init: float = 0.1,
        sigma_min: float = 1e-3,
        key: PRNGKeyArray,
    ) -> None:
        if sigma_init <= 0:
            msg = f"sigma_init must be positive, got {sigma_init}."
            raise ValueError(msg)
        if sigma_min <= 0:
            msg = f"sigma_min must be positive, got {sigma_min}."
            raise ValueError(msg)

        self.sigma_min = sigma_min
        self.width_size = width_size
        self.depth = depth

        mlp = eqx.nn.MLP(
            in_size="scalar",
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        # Initialise the network to predict log(sigma_init) everywhere, by
        # zeroing the final layer's weights and setting its bias. Without this
        # the initial width is an arbitrary O(1) number, and a mixture model
        # started at the wrong width is very good at finding the wrong optimum.
        final = mlp.layers[-1]
        mlp = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            mlp,
            (
                jnp.zeros_like(final.weight),
                jnp.full_like(final.bias, jnp.log(sigma_init)),
            ),
        )
        self.mlp = mlp

    @ft.partial(eqx.filter_jit)
    def __call__(self, gamma: FSz0, /, key: PRNGKeyArray | None = None) -> FSz0:
        """Return the stream half-width at ordering parameter ``gamma``.

        Parameters
        ----------
        gamma : Array, shape ()
            Ordering parameter.
        key : PRNGKeyArray | None
            Unused; accepted for signature consistency with the other networks.

        Returns
        -------
        sigma : Array, shape ()
            Stream half-width, strictly positive and at least ``sigma_min``.

        """
        del key
        # Clamp before exponentiating: an unclamped `exp` overflows to `inf` for
        # log-widths above ~88 in float32, and an infinite width turns every
        # residual downstream into NaN.
        log_sigma = jnp.clip(self.mlp(gamma), _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        return self.sigma_min + jnp.exp(log_sigma)


def sigma_ceiling(
    epoch_idx: Int[Array, ""] | int,
    num_epochs: int,
    /,
    *,
    start: float,
    stop: float,
) -> FSz0:
    r"""Annealing ceiling on the stream width, geometric from `start` to `stop`.

    The mixture likelihood has a well-known failure mode: rather than lowering
    the membership of a few nearby outliers, it can simply *widen* the stream
    component until they are explained. The likelihood genuinely prefers this,
    so no amount of training fixes it -- it is not a convergence problem.

    Annealing removes the option. Early on the ceiling is generous, which is
    what we want: the decoded track is still garbage, so every star is far from
    it, and an aggressive width would reject the entire stream. As the track
    sharpens, the ceiling is squeezed and stars that *stay* far from the track
    are progressively forced to explain themselves as background.

    The schedule is geometric (linear in $\log \sigma$), which is the natural
    choice for a scale parameter.

    Parameters
    ----------
    epoch_idx : Array, shape () | int
        Current epoch, in ``[0, num_epochs)``. May be a tracer.
    num_epochs : int
        Total number of epochs. Must be static.
    start, stop : float
        Ceiling at the first and last epoch. ``start`` should be comfortably
        larger than the expected stream width (a few times is fine); ``stop``
        should be of order the stream width.

    Returns
    -------
    ceiling : Array, shape ()
        The width ceiling for this epoch.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit.nn import sigma_ceiling

    The ceiling starts at ``start`` and ends at ``stop``:

    >>> float(sigma_ceiling(0, 5, start=1.0, stop=0.1))
    1.0
    >>> round(float(sigma_ceiling(4, 5, start=1.0, stop=0.1)), 6)
    0.1

    ...and is geometric in between, so the midpoint is the *geometric* mean:

    >>> round(float(sigma_ceiling(2, 5, start=1.0, stop=0.01)), 6)
    0.1

    A single-epoch run just uses ``start``:

    >>> float(sigma_ceiling(0, 1, start=1.0, stop=0.1))
    1.0

    """
    if start <= 0 or stop <= 0:
        msg = f"start and stop must be positive, got start={start}, stop={stop}."
        raise ValueError(msg)

    if num_epochs <= 1:
        return jnp.asarray(start, dtype=float)

    frac = jnp.asarray(epoch_idx, dtype=float) / (num_epochs - 1)
    log_start, log_stop = jnp.log(start), jnp.log(stop)
    return jnp.exp(log_start + (log_stop - log_start) * frac)


def membership_rampup(
    epoch_idx: Int[Array, ""] | int,
    num_epochs: int,
    /,
    *,
    warmup_frac: float,
) -> FSz0:
    r"""Ramp the membership term in from zero: 0 during warm-up, 1 afterwards.

    Why this is not optional
    ------------------------
    The mixture likelihood has a second degenerate optimum, and it is *worse*
    than the width-inflation one because it is self-reinforcing.

    At initialisation the decoded track is meaningless, so **every** star has a
    huge residual. The likelihood's cheapest move is to declare the whole
    dataset background: push $\pi_n \to 0$ for all $n$. But once $\pi_n = 0$ the
    stream component carries no weight, so *no gradient reaches the decoder* --
    the track is free to drift anywhere, residuals grow further, and $\pi$ is
    pinned at zero forever. Training collapses to "there is no stream". We have
    measured this: with outliers at 20 stream-widths, an un-ramped mixture drove
    the median inlier residual to ~8 (stream width 0.15) and rejected all 400
    genuine members.

    The cure is the standard EM initialisation: fit the track *first*, with
    membership effectively pinned at 1, and only then let the model start
    disowning stars. Concretely we use

    .. math::

        \pi^{\mathrm{eff}}_n = 1 - w(t)\,(1 - \pi_n)

    so that at $w = 0$ the loss degenerates to the plain Gaussian reconstruction
    NLL (the background term vanishes), and at $w = 1$ it is the full mixture.
    Gradients still flow into $\pi$ during warm-up (through the $w \pi_n$ term),
    they simply cannot yet *act* on the reconstruction.

    Together with `sigma_ceiling` this brackets the problem from both sides:
    the ramp stops membership collapsing before the track exists, and the
    ceiling stops the width inflating once it does.

    Parameters
    ----------
    epoch_idx : Array, shape () | int
        Current epoch, in ``[0, num_epochs)``. May be a tracer.
    num_epochs : int
        Total number of epochs. Must be static.
    warmup_frac : float
        Fraction of training spent ramping, in ``[0, 1)``. ``w`` rises linearly
        from 0 to 1 across the first ``warmup_frac * (num_epochs - 1)`` epochs
        and stays at 1 thereafter. ``0`` disables the ramp (full mixture from
        epoch zero) -- don't, unless you have a good reason.

    Returns
    -------
    w : Array, shape ()
        Ramp value in [0, 1].

    Examples
    --------
    >>> from phasecurvefit.nn import membership_rampup

    With a 50% warm-up the mixture is fully on by the halfway point:

    >>> float(membership_rampup(0, 10, warmup_frac=0.5))
    0.0
    >>> float(membership_rampup(5, 10, warmup_frac=0.5))
    1.0
    >>> float(membership_rampup(9, 10, warmup_frac=0.5))
    1.0

    ...and rises linearly in between:

    >>> round(float(membership_rampup(2, 10, warmup_frac=0.5)), 3)
    0.444

    ``warmup_frac=0`` means no warm-up at all:

    >>> float(membership_rampup(0, 10, warmup_frac=0.0))
    1.0

    """
    if not 0.0 <= warmup_frac < 1.0:
        msg = f"warmup_frac must be in [0, 1), got {warmup_frac}."
        raise ValueError(msg)

    if warmup_frac == 0.0 or num_epochs <= 1:
        return jnp.asarray(1.0, dtype=float)

    n_warm = warmup_frac * (num_epochs - 1)
    frac = jnp.asarray(epoch_idx, dtype=float) / n_warm
    return jnp.clip(frac, 0.0, 1.0)


def uniform_background_density(
    qs: Float[Array, "N D"], /, *, inflate: float = 1.0
) -> FSz0:
    r"""Flat background density $\rho_{\mathrm{bg}}$ from the field's extent.

    The mixture model needs a density for the "not a stream member" component.
    Hogg, Bovy & Lang (2010) §3 use a broad Gaussian $(Y_b, V_b)$ and marginalise
    over it; when the field is a bounded footprint -- as it is here -- the natural
    analogue is a uniform density over that footprint,

    .. math::

        \rho_{\mathrm{bg}} = \frac{1}{\prod_d \left(\max_d q - \min_d q\right)} .

    This is *not* a free knob to tune: it is the reciprocal of the field volume,
    and it is what makes membership a calibrated posterior rather than an
    arbitrary cut. It is, however, the quantity that sets where the
    stream/background crossover sits, so it is worth being deliberate about the
    footprint you hand it.

    Parameters
    ----------
    qs : Array, shape (N, D)
        Positions defining the field.
    inflate : float, optional
        Multiply the extent of each axis by this factor before taking the
        volume. Use ``> 1`` if the observed points do not fill the true survey
        footprint. Default 1.

    Returns
    -------
    rho_bg : Array, scalar
        Background density, in units of (length)^-D. Returned as a 0-d array so
        that it stays trace-transparent under ``jit`` (e.g. inside
        `posterior_membership`); use ``float(...)`` if you need a Python scalar.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit.nn import uniform_background_density

    A unit square has unit background density:

    >>> qs = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    >>> float(uniform_background_density(qs))
    1.0

    A 2x2 square has a quarter the density:

    >>> qs = jnp.array([[0.0, 0.0], [2.0, 2.0]])
    >>> float(uniform_background_density(qs))
    0.25

    """
    if inflate <= 0:
        msg = f"inflate must be positive, got {inflate}."
        raise ValueError(msg)

    extent = inflate * (jnp.max(qs, axis=0) - jnp.min(qs, axis=0))
    # A degenerate axis (all points share a coordinate) would give zero volume
    # and an infinite density. Treat it as unit extent -- the axis carries no
    # information either way.
    extent = jnp.where(extent > 0, extent, 1.0)
    return 1.0 / jnp.prod(extent)


def _log_foreground_density(r2: FSzN, sigma: FSzN, n_dims: int) -> FSzN:
    r"""Log isotropic Gaussian stream density, $\log \mathcal{N}(r; 0, \sigma^2 I)$."""
    return (
        -0.5 * r2 / jnp.square(sigma)
        - n_dims * jnp.log(sigma)
        - 0.5 * n_dims * jnp.log(2 * jnp.pi)
    )


def membership_responsibility(
    prob: FSzN,
    r2: FSzN,
    sigma: FSzN,
    /,
    *,
    log_bg_density: FSz0 | float,
    n_dims: int,
) -> FSzN:
    r"""Posterior probability that each star is a stream member.

    This is the E-step of the mixture model: given the current track, width, and
    prior membership $\pi_n$, the *posterior* membership of star $n$ is

    .. math::

        \hat{q}_n = \frac{\pi_n \, \mathcal{N}(r_n; 0, \sigma_n^2)}
                         {\pi_n \, \mathcal{N}(r_n; 0, \sigma_n^2)
                          + (1 - \pi_n)\, \rho_{\mathrm{bg}}}

    which is Hogg, Bovy & Lang (2010)'s $q_i$ after marginalisation -- the
    quantity they recommend reporting instead of a hard cut.

    Note the distinction from ``prob``: $\pi_n$ is what the *encoder believes*
    from the star's phase-space coordinates alone, before seeing how far it
    actually landed from the track. $\hat{q}_n$ folds in the residual. **The
    responsibility is the number you want** when identifying members; ``prob``
    is only its prior.

    Parameters
    ----------
    prob : Array, shape (N,)
        Encoder membership output $\pi_n$, in [0, 1].
    r2 : Array, shape (N,)
        Squared residual $\lVert q_n - x_\theta(\gamma_n) \rVert^2$.
    sigma : Array, shape (N,)
        Stream half-width at each star's $\gamma$.
    log_bg_density : float
        $\log \rho_{\mathrm{bg}}$.
    n_dims : int
        Spatial dimensionality $D$.

    Returns
    -------
    responsibility : Array, shape (N,)
        Posterior membership in [0, 1].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit.nn import membership_responsibility

    A star sitting exactly on the track, with a flat prior, is confidently a
    member; one far away is confidently not:

    >>> prob = jnp.array([0.5, 0.5])
    >>> r2 = jnp.array([0.0, 100.0])  # on the track / far off it
    >>> sigma = jnp.array([0.1, 0.1])
    >>> q = membership_responsibility(
    ...     prob, r2, sigma, log_bg_density=float(jnp.log(0.01)), n_dims=2
    ... )
    >>> bool(q[0] > 0.99), bool(q[1] < 0.01)
    (True, True)

    """
    prob = jnp.clip(prob, _EPS, 1.0 - _EPS)
    log_fg = jnp.log(prob) + _log_foreground_density(r2, sigma, n_dims)
    log_bg = jnp.log1p(-prob) + log_bg_density
    # exp(log_fg - logaddexp(log_fg, log_bg)) == sigmoid(log_fg - log_bg), but
    # the sigmoid form is the numerically stable one.
    return jax.nn.sigmoid(log_fg - log_bg)


@eqx.filter_jit
def mixture_membership_loss(
    qs_meas: Float[Array, "N D"],
    qs_pred: Float[Array, "N D"],
    prob: FSzN,
    sigma: FSzN,
    mask: Bool[Array, " N"],
    /,
    *,
    log_bg_density: float,
    rampup: FSz0 | float = 1.0,
) -> tuple[FSz0, FSzN]:
    r"""Negative log-likelihood of the stream/background mixture.

    Implements the "mixture" model of Hogg, Bovy & Lang (2010), §3, with the
    bad-data fraction amortised into a per-star, network-predicted $\pi_n$:

    .. math::

        -\log \mathcal{L} = -\frac{1}{N} \sum_n \log\!\left[
            \pi_n \, \mathcal{N}\!\left(q_n; x_\theta(\gamma_n),
                                        \sigma_n^2 \mathbb{I}\right)
            + (1 - \pi_n) \, \rho_{\mathrm{bg}}
        \right]

    Minimising this simultaneously fits the track ($x_\theta$), the width
    ($\sigma$), and the membership ($\pi$) -- and the last of these is now driven
    by the residual, which is the entire point.

    Evaluated with `jax.nn.logsumexp` semantics (via `jnp.logaddexp`) so that a
    star which is *implausible under both components* -- very far from the track
    and in a low-density corner -- does not underflow to ``-inf``.

    Parameters
    ----------
    qs_meas : Array, shape (N, D)
        Observed positions.
    qs_pred : Array, shape (N, D)
        Decoded track positions, $x_\theta(\gamma_n)$.
    prob : Array, shape (N,)
        Encoder membership output $\pi_n$, in [0, 1].
    sigma : Array, shape (N,)
        Stream half-width at each star's $\gamma$. Must be positive.
    mask : Array, shape (N,)
        True for real, usable stars; False for padding or ignorable rows. Only
        masked-in stars contribute.
    log_bg_density : float
        $\log \rho_{\mathrm{bg}}$; see `uniform_background_density`.
    rampup : Array, shape () | float, optional
        Membership ramp $w \in [0, 1]$ from `membership_rampup`. The effective
        membership is $\pi^{\mathrm{eff}} = 1 - w (1 - \pi)$, so ``w=0`` gives a
        pure reconstruction NLL and ``w=1`` the full mixture. Defaults to 1
        (no warm-up); see `membership_rampup` for why you almost certainly want
        to ramp.

    Returns
    -------
    loss : Array, shape ()
        Mean negative log-likelihood over the masked-in stars.
    responsibility : Array, shape (N,)
        Posterior membership $\hat{q}_n$ for every star (including masked-out
        ones, whose values are meaningless). Returned because the caller almost
        always wants it -- to weight the velocity-alignment term, and to report
        membership -- and recomputing it would duplicate the whole forward pass.

        Computed from the *effective* membership, so during warm-up it is
        (correctly) close to 1 everywhere.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit.nn import mixture_membership_loss

    Two stars on the track and one far off it. With a confident prior, the
    on-track stars get responsibility ~1 and the outlier ~0:

    >>> qs_meas = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 9.0]])
    >>> qs_pred = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    >>> prob = jnp.array([0.9, 0.9, 0.9])
    >>> sigma = jnp.array([0.1, 0.1, 0.1])
    >>> mask = jnp.ones(3, dtype=bool)
    >>> loss, resp = mixture_membership_loss(
    ...     qs_meas, qs_pred, prob, sigma, mask, log_bg_density=float(jnp.log(0.01))
    ... )
    >>> bool(resp[0] > 0.99), bool(resp[1] > 0.99), bool(resp[2] < 0.01)
    (True, True, True)

    The loss is finite even for the wildly-discrepant star:

    >>> bool(jnp.isfinite(loss))
    True

    """
    n_dims = qs_meas.shape[1]

    r2 = jnp.sum(jnp.square(qs_meas - qs_pred), axis=-1)

    # Warm-up: pull the membership towards 1 early on, so the track can fit
    # before the model is allowed to disown anything. See `membership_rampup`.
    prob_eff = 1.0 - jnp.asarray(rampup) * (1.0 - prob)
    prob_eff = jnp.clip(prob_eff, _EPS, 1.0 - _EPS)

    log_fg = jnp.log(prob_eff) + _log_foreground_density(r2, sigma, n_dims)
    log_bg = jnp.log1p(-prob_eff) + log_bg_density

    log_like = jnp.logaddexp(log_fg, log_bg)

    # The responsibility is the E-step: in EM it is held *fixed* while the
    # parameters are updated, so it must not carry gradient. Cutting it here is
    # both semantically right and numerically necessary -- under `jit`, XLA fuses
    # the dead VJP of this (unused-for-the-loss) output and the resulting
    # `0 * inf` yields NaN gradients at saturated membership.
    responsibility = jax.lax.stop_gradient(jax.nn.sigmoid(log_fg - log_bg))

    # `masked_mean` from jaxmore would do, but we want the mask applied to the
    # *log-likelihood* rather than to a residual, and a NaN-safe denominator.
    count = jnp.sum(mask)
    safe_count = jnp.where(count > 0, count, 1)
    loss = -jnp.where(count > 0, jnp.sum(log_like * mask) / safe_count, 0.0)

    return loss, responsibility


@dataclass(frozen=True)
class MixtureMembershipConfig:
    r"""Configuration for mixture-model membership (outlier rejection).

    Pass an instance of this to
    `phasecurvefit.nn.EncoderDecoderTrainingConfig.membership` (or
    `TrainingConfig.membership`) to swap the classifier-style membership loss for
    the generative mixture model of Hogg, Bovy & Lang (2010), §3.

    Leaving ``membership=None`` (the default) preserves the existing behaviour
    exactly.

    Attributes
    ----------
    sigma_init : float
        Stream half-width the `WidthNet` is initialised to predict. Set it
        to your best guess at the stream width, in the *normalised* coordinates
        the networks see (`StandardScalerNormalizer` makes this roughly "in units
        of the field's standard deviation").
    sigma_min : float
        Hard floor on the width, guarding the collapse-onto-a-single-star
        degeneracy that afflicts every mixture likelihood.
    sigma_ceiling : tuple[float, float]
        ``(start, stop)`` for the annealing ceiling on the width, applied
        geometrically across epochs; see `sigma_ceiling`. ``start`` should be a
        few times the expected width, ``stop`` of order the width. **This is one
        of the two knobs that matter**: with no ceiling, a free width will
        inflate to swallow nearby outliers rather than reject them.
    warmup_frac : float
        Fraction of training spent ramping the membership term in; see
        `membership_rampup`. **This is the other knob that matters**: with no
        warm-up, the model can collapse to "everything is background" before the
        track has fitted, and never recover. Set to 0 only if you are supplying
        an already-good track.
    background_density : float | None
        $\rho_{\mathrm{bg}}$. If None (default), computed from the extent of the
        training positions via `uniform_background_density`.
    background_inflate : float
        Passed to `uniform_background_density` when ``background_density`` is
        None. Increase if the observed stars do not fill the survey footprint.
    width_size, depth : int
        Architecture of the `WidthNet`.
    lambda_velocity : float
        Weight on the velocity-alignment term, which is reweighted by the
        posterior responsibility so that outliers cannot drag the track's
        tangent. Set to 0 to disable the velocity term entirely.

    Examples
    --------
    >>> import phasecurvefit as pcf

    Enable outlier rejection with a stream you expect to be ~0.1 wide:

    >>> membership = pcf.nn.MixtureMembershipConfig(
    ...     sigma_init=0.1, sigma_ceiling=(0.5, 0.1)
    ... )
    >>> cfg = pcf.nn.TrainingConfig(membership=membership, show_pbar=False)
    >>> cfg.membership.sigma_init
    0.1

    The default is None, i.e. the legacy classifier membership:

    >>> pcf.nn.TrainingConfig().membership is None
    True

    """

    _: KW_ONLY

    sigma_init: float = 0.1
    sigma_min: float = 1e-3
    sigma_ceiling: tuple[float, float] = (0.5, 0.1)

    warmup_frac: float = 0.3

    background_density: float | None = None
    background_inflate: float = 1.0

    width_size: int = 32
    depth: int = 2

    lambda_velocity: float = 1.0

    __citation__: ClassVar[str] = "https://arxiv.org/abs/1008.4686"

    def __post_init__(self) -> None:
        """Validate the schedule."""
        start, stop = self.sigma_ceiling
        if start <= 0 or stop <= 0:
            msg = f"sigma_ceiling entries must be positive, got {self.sigma_ceiling}."
            raise ValueError(msg)
        if stop > start:
            msg = (
                "sigma_ceiling must anneal downward (start >= stop), got "
                f"{self.sigma_ceiling}. A ceiling that *grows* would let the "
                "stream width inflate to absorb outliers, which is the failure "
                "mode this schedule exists to prevent."
            )
            raise ValueError(msg)
        if self.sigma_init <= 0:
            msg = f"sigma_init must be positive, got {self.sigma_init}."
            raise ValueError(msg)
        if not 0.0 <= self.warmup_frac < 1.0:
            msg = f"warmup_frac must be in [0, 1), got {self.warmup_frac}."
            raise ValueError(msg)

    def make_width_net(self, *, key: PRNGKeyArray) -> WidthNet:
        """Build the `WidthNet` this config describes."""
        return WidthNet(
            width_size=self.width_size,
            depth=self.depth,
            sigma_init=self.sigma_init,
            sigma_min=self.sigma_min,
            key=key,
        )

    def resolve_background_density(self, qs: Real[Array, "N D"], /) -> FSz0 | float:
        """Return the configured background density, or derive it from `qs`."""
        if self.background_density is not None:
            if self.background_density <= 0:
                msg = (
                    "background_density must be positive, got "
                    f"{self.background_density}."
                )
                raise ValueError(msg)
            return self.background_density
        return uniform_background_density(qs, inflate=self.background_inflate)
