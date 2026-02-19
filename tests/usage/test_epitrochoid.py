"""Matplotlib tests for epitrochoid autoencoder fitting."""

from collections.abc import Mapping

import jax
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

import quaxed.numpy as jnp
import unxt as u

import localflowwalk as lfw

plt = pytest.importorskip("matplotlib.pyplot")


def make_self_intersecting_stream(
    key: PRNGKeyArray,
    *,
    n: int,
    noise_sigma: float = 0.5,
    scale: float = 120.0,
    R: float = 5.0,
    r: float = 1.0,
    d: float = 4.5,
) -> tuple[jax.Array, Mapping[str, u.Quantity], Mapping[str, u.Quantity]]:
    """Return (t, pos, vel) for an OPEN epitrochoid curve with self-intersections.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation
    n : int
        Number of points along the curve
    noise_sigma : float, default=0.5
        Standard deviation of positional noise
    scale : float, default=120.0
        Overall scale factor
    R : float, default=5.0
        Outer circle radius parameter
    r : float, default=1.0
        Inner circle radius parameter
    d : float, default=4.5
        Distance parameter for epicycloid

    Returns
    -------
    t : Array
        Parameter values [0, 2π)
    pos : dict
        Position dictionary with keys "x", "y" as unitful Quantities
    vel : dict
        Velocity dictionary with keys "x", "y" as unitful Quantities

    """
    usys = u.unitsystems.si

    # Epitrochoid from 5° to 355° with 10° gap (open curve, doesn't close)
    t_start = 5.0 * jnp.pi / 180.0
    t_end = 355.0 * jnp.pi / 180.0
    t = jnp.linspace(t_start, t_end, n)

    ratio = (R + r) / r  # = 5 internal rotations per outer rotation
    x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
    y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0

    # Derivatives for velocity
    dx0 = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
    dy0 = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

    # Optional small positional noise
    kx, ky = jax.random.split(key)
    x = x0 + noise_sigma * jax.random.normal(kx, (n,))
    y = y0 + noise_sigma * jax.random.normal(ky, (n,))

    # Pack into unitful quantities
    pos = {"x": u.Q(x, usys["length"]), "y": u.Q(y, usys["length"])}
    vel = {"x": u.Q(dx0, usys["speed"]), "y": u.Q(dy0, usys["speed"])}
    return t, pos, vel


@pytest.mark.mpl_image_compare(deterministic=True)
def test_epitrochoid_autoencoder_fit() -> plt.Figure:
    """Test epitrochoid generation and autoencoder fitting with visualization.

    This test:
    1. Generates a self-intersecting epitrochoid curve with noise
    2. Shuffles the data and performs a local flow walk
    3. Trains an autoencoder on the walked path
    4. Creates a visualization with encoded points and predicted path
    5. Returns the figure for pytest-mpl comparison

    """
    usys = u.unitsystems.si

    # Generate epitrochoid
    key = jr.key(201030)
    key, subkey = jr.split(key)
    t, pos, vel = make_self_intersecting_stream(
        subkey,
        n=2048,
        noise_sigma=6,
    )

    # Shuffle the data
    key, subkey = jr.split(key)
    order = jr.permutation(subkey, jnp.arange(len(t)))

    qs = jax.tree.map(lambda x: x[order], pos)
    ps = jax.tree.map(lambda x: x[order], vel)

    # Determine the starting index as the point closest to the starting point
    start_idx = int(jnp.argsort(order)[0])

    # Walk configuration
    config = lfw.WalkConfig(
        strategy=lfw.strats.KDTree(k=60),
        metric=lfw.metrics.FullPhaseSpaceDistanceMetric(),
    )
    lam = u.Q(4, "s")
    max_dist = u.Q(40, "m")

    # Perform walk
    walkresult = lfw.walk_local_flow(
        qs,
        ps,
        start_idx=start_idx,
        lam=lam,
        max_dist=max_dist,
        config=config,
        direction="forward",
        metadata=lfw.StateMetadata(usys=usys),
    )

    # Train autoencoder
    key, model_key, train_key = jr.split(key, 3)
    normalizer = lfw.nn.StandardScalerNormalizer(qs, ps)
    model = lfw.nn.PathAutoencoder.make(normalizer, track_depth=4, key=model_key)
    train_config = lfw.nn.TrainingConfig(show_pbar=False)
    model, _, _ = lfw.nn.train_autoencoder(
        model, walkresult, key=train_key, config=train_config
    )

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Encode all points
    all_gamma, all_probs = model.encode(walkresult.positions, walkresult.velocities)
    rejected_membership = all_probs < 0.9

    # Decode predicted path
    qs_pred = model.decode(jnp.linspace(-1, 1, 1_000))

    # Extract values from Quantities for plotting
    qs_x_vals = jnp.asarray(u.ustrip(qs["x"]))
    qs_y_vals = jnp.asarray(u.ustrip(qs["y"]))
    qs_pred_x_vals = jnp.asarray(u.ustrip(qs_pred["x"]))
    qs_pred_y_vals = jnp.asarray(u.ustrip(qs_pred["y"]))

    # Plot all points with gradient coloring
    ax.scatter(
        qs_x_vals,
        qs_y_vals,
        s=50,
        c=jnp.asarray(all_gamma),
        cmap="RdYlBu",
        alpha=0.8,
        label="Stream members",
    )

    # Plot predicted mean path
    ax.plot(
        qs_pred_x_vals,
        qs_pred_y_vals,
        c="k",
        lw=3,
        label="Predicted mean path",
    )

    # Mark rejected samples in cyan
    ax.scatter(
        qs_x_vals[rejected_membership],
        qs_y_vals[rejected_membership],
        s=100,
        c="cyan",
        alpha=1.0,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label="Rejected samples",
    )

    # Mark start point
    ax.scatter(
        [qs_x_vals[start_idx]],
        [qs_y_vals[start_idx]],
        s=200,
        c="tab:green",
        marker="*",
        label="Start point",
        linewidths=2,
        zorder=5,
    )

    ax.set_xlabel(r"$x$ [m]", fontsize=14)
    ax.set_ylabel(r"$y$ [m]", fontsize=14)
    ax.set_title("ML Ordering of Stellar Stream with Predicted Mean Path")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)  # noqa: FBT003

    return fig
