"""NumPyro track models shared by the prototypes.

The track is x_d(gamma) = c0_d + c1_d gamma + f_d(gamma) for each spatial dim d, with
f_d a Hilbert-space GP (squared-exponential kernel, numpyro.contrib.hsgp) whose
amplitude and length scale are latent. Stars scatter about the track with an isotropic
Gaussian of latent width sigma.

Two likelihoods:

- ``fixed``: each star has a known gamma_n (from phasecurvefit's encoder), so this is
  Bayesian regression of q_n on gamma_n.
- ``marginal``: gamma_n is latent with a uniform prior on [-1, 1] and is summed out on a
  grid, p(q_n) = (1/K) sum_k N(q_n; x(gamma_k), sigma^2 I). No per-star gamma.

Either can add a Hogg, Bovy & Lang (2010) sec. 3 background: a flat density rho_bg with
latent fraction f_bg.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions

L_BOX = 1.3  # HSGP box half-width; gamma lives in [-1, 1]
M_BASIS = 16
CENTERED = False  # centred mixed worse here (min ESS 1 vs 35); see SURVEY
LINEAR_MEAN = False  # c0 + c1 gamma is near-degenerate with the lowest basis functions
K_GRID = 256
GAMMA_GRID = jnp.linspace(-1.0, 1.0, K_GRID)
OMEGA = jnp.arange(1, M_BASIS + 1) * jnp.pi / (2 * L_BOX)  # sqrt Laplacian eigenvalues


def sqrt_spd(amp, length):
    """Sqrt of the squared-exponential spectral density at OMEGA, amp = kernel std.

    Written in closed form because sqrt(numpyro's diag_spectral_density_...) underflows
    to 0 at short length scales and its gradient there is NaN. NumPyro's ``alpha`` is the
    kernel variance (it enters the density linearly); here ``amp**2`` plays that role.
    """
    return (
        amp
        * (2 * jnp.pi) ** 0.25
        * jnp.sqrt(length)
        * jnp.exp(-((length * OMEGA) ** 2) / 4)
    )


class Scaler:
    """Centre and scale positions by one scalar, so an isotropic width stays isotropic."""

    def __init__(self, q):
        self.mu = q.mean(0)
        self.s = float(q.std())

    def fwd(self, q):
        return (q - self.mu) / self.s

    def inv(self, z):
        return z * self.s + self.mu


def track(gam, D):
    """Sample the track's latents and return x(gam), shape (len(gam), D)."""
    phi = eigenfunctions(gam, L_BOX, M_BASIS)  # (G, M)
    with numpyro.plate("dim", D, dim=-2):
        alpha = numpyro.sample("alpha", dist.HalfNormal(1.0))
        length = numpyro.sample("length", dist.LogNormal(jnp.log(0.3), 0.7))
        if LINEAR_MEAN:
            c0 = numpyro.sample("c0", dist.Normal(0.0, 2.0))
            c1 = numpyro.sample("c1", dist.Normal(0.0, 2.0))
    spd = sqrt_spd(alpha, length)  # (D, M) by broadcasting (D, 1) with (M,)
    with numpyro.plate("dim", D, dim=-2), numpyro.plate("basis", M_BASIS, dim=-1):
        if CENTERED:
            spd = jnp.clip(spd, 1e-8)
            beta = numpyro.sample("beta", dist.Normal(0.0, spd))
        else:
            beta = numpyro.sample("beta", dist.Normal(0.0, 1.0)) * spd
    f = phi @ beta.T  # (G, D)
    if LINEAR_MEAN:
        f = f + c0[:, 0] + c1[:, 0] * gam[:, None]
    return f


def _log_normal_iso(r2, sigma, D):
    return -0.5 * r2 / sigma**2 - D * jnp.log(sigma) - 0.5 * D * jnp.log(2 * jnp.pi)


def configure(m_basis=16, k_grid=256):
    """Set the number of basis functions and the gamma grid size."""
    global M_BASIS, OMEGA, K_GRID, GAMMA_GRID
    M_BASIS, K_GRID = m_basis, k_grid
    OMEGA = jnp.arange(1, m_basis + 1) * jnp.pi / (2 * L_BOX)
    GAMMA_GRID = jnp.linspace(-1.0, 1.0, k_grid)


def model(
    z,
    gamma=None,
    *,
    n_q=None,
    log_rho_bg=None,
    weights=None,
    line_density=False,
    width_varies=False,
    vel_tangent=False,
    vel_slack=0.0,
):
    """z: (N, D) scaled phase-space coordinates, positions in the first ``n_q`` columns.

    gamma given -> ``fixed``; None -> ``marginal``. Columns after ``n_q`` (velocities)
    scatter about the track with their own width ``sigma_p``.

    In the marginal model, ``line_density`` replaces the uniform prior on gamma with a
    latent one, log pi(gamma) a GP normalised over the grid, so a sparse tail need not
    be crossed in a sliver of gamma. ``width_varies`` makes the position width
    sigma * exp(g(gamma)) with g a GP. Both are extra GP channels of ``track``.

    ``vel_tangent`` (marginal model only) ties the velocity mean to the track:
    exp(v(gamma)) times the unit tangent dx/dgamma, with v a GP log-speed channel. A
    track that runs the wrong way (globally, or round a loop) then misfits every
    velocity. Velocities must be scaled without centring for this.

    ``vel_slack > 0`` loosens the tie: the mean becomes exp(v) t + vel_slack * d(gamma),
    with d a GP per velocity component under the usual prior. Real stream stars move
    a few degrees off the track; with ``vel_slack`` around 0.05 of the speed that
    costs little, while running the wrong way still costs about twice the speed.
    """
    N, D = z.shape
    n_q = D if n_q is None else n_q
    n_extra = int(line_density) + int(width_varies) if gamma is None else 0
    tied = vel_tangent and gamma is None and n_q < D
    n_slack = n_q if tied and vel_slack > 0 else 0
    n_mean = n_q + 1 + n_slack if tied else D  # tied: positions, log-speed, offsets
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    x_all = track(GAMMA_GRID if gamma is None else gamma, n_mean + n_extra)
    x, col = x_all[:, :n_mean], n_mean
    if tied:
        t = jnp.gradient(x[:, :n_q], axis=0)
        t = t / jnp.linalg.norm(t, axis=-1, keepdims=True).clip(1e-12)
        v = jnp.exp(x[:, n_q : n_q + 1]) * t
        if n_slack:
            v = v + vel_slack * x[:, n_q + 1 :]
        x = jnp.concatenate([x[:, :n_q], v], -1)
    log_pi = -jnp.log(x_all.shape[0])
    if n_extra and line_density:
        log_pi, col = jax.nn.log_softmax(x_all[:, col]), col + 1
    if n_extra and width_varies:
        sigma = sigma * jnp.exp(x_all[:, col])  # (K,)
    if gamma is None:
        d2 = (z[:, None, :] - x[None]) ** 2  # (N, K, D)
    else:
        d2 = (z - x)[:, None, :] ** 2  # (N, 1, D)
    ll = _log_normal_iso(jnp.sum(d2[..., :n_q], -1), sigma, n_q)
    if n_q < D:
        sigma_p = numpyro.sample("sigma_p", dist.HalfNormal(0.5))
        ll = ll + _log_normal_iso(jnp.sum(d2[..., n_q:], -1), sigma_p, D - n_q)
    log_fg = jax.nn.logsumexp(ll + log_pi, -1)
    if log_rho_bg is not None:
        f_bg = numpyro.sample("f_bg", dist.Beta(1.0, 9.0))
        log_fg = jnp.logaddexp(jnp.log1p(-f_bg) + log_fg, jnp.log(f_bg) + log_rho_bg)
    if weights is not None:
        # weights act like counts. Multiplied in directly: handlers.scale refuses 0.
        log_fg = weights * log_fg
    with numpyro.plate("stars", N):
        numpyro.factor("loglik", log_fg)


def ls_init(z, gamma, D, n_q=None, n_extra=0):
    """Starting values: ridge least squares of z on the basis at fixed hyperparameters."""
    a0, l0 = 1.0, 0.3
    phi = np.asarray(eigenfunctions(jnp.asarray(gamma), L_BOX, M_BASIS))
    spd = np.asarray(sqrt_spd(a0, l0))
    X = phi * spd
    if LINEAR_MEAN:
        X = np.concatenate([np.ones((len(gamma), 1)), gamma[:, None], X], 1)
    A = X.T @ X + np.eye(X.shape[1]) * 1e-1
    coef = np.linalg.solve(A, X.T @ np.asarray(z))  # (2+M, D)
    resid = np.asarray(z) - X @ coef
    n0 = 2 if LINEAR_MEAN else 0
    beta = np.asarray(coef[n0:].T) * (spd if CENTERED else 1.0)
    beta = np.concatenate([beta, np.zeros((n_extra, beta.shape[1]))])  # flat extras
    init = {
        "alpha": jnp.full((D + n_extra, 1), a0),
        "length": jnp.full((D + n_extra, 1), l0),
        "beta": jnp.asarray(beta),
        "sigma": jnp.asarray(max(float(resid[:, :n_q].std()), 0.02)),
    }
    if n_q is not None and n_q < D:
        init["sigma_p"] = jnp.asarray(max(float(resid[:, n_q:].std()), 0.02))
    if LINEAR_MEAN:
        init["c0"], init["c1"] = (
            jnp.asarray(coef[0])[:, None],
            jnp.asarray(coef[1])[:, None],
        )
    return init


def basis_fit(gamma, y, a0=1.0, l0=0.3):
    """Ridge fit of y(gamma) onto the basis: the non-centred beta row for one channel."""
    X = np.asarray(eigenfunctions(jnp.asarray(gamma), L_BOX, M_BASIS)) * np.asarray(
        sqrt_spd(a0, l0)
    )
    return np.linalg.solve(X.T @ X + 1e-1 * np.eye(X.shape[1]), X.T @ np.asarray(y))


def track_draws(samples, gam, D=None):
    """Evaluate every GP channel at gam for every posterior draw: (S, len(gam), C)."""
    D = samples["beta"].shape[-2]
    from numpyro.infer import Predictive

    def f(**kw):
        numpyro.deterministic("x", track(gam, D))

    keep = {
        k: samples[k] for k in ("alpha", "length", "c0", "c1", "beta") if k in samples
    }
    return Predictive(f, posterior_samples=keep)(jax.random.key(0))["x"]
