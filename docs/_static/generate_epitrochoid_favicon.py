#!/usr/bin/env -S uv run python
"""Generate a favicon by fitting an autoencoder to an epitrochoid.

This script:
1. Creates a self-intersecting epitrochoid curve with noise
2. Shuffles the data and performs a local flow walk
3. Trains an autoencoder on the walked path
4. Generates a clean visualization (points colored by gamma, mean path as black line)
5. Saves as a PNG suitable for use as a favicon (no axes, labels, or legend)
"""

import sys
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import quaxed.numpy as jnp
import unxt as u

import phasecurvefit as pcf


def make_self_intersecting_stream(
    key: jax.Array,
    *,
    n: int,
    noise_sigma: float = 0.5,
    scale: float = 120.0,
    R: float = 5.0,
    r: float = 1.0,
    d: float = 4.5,
) -> tuple[jax.Array, Mapping[str, u.Quantity], Mapping[str, u.Quantity]]:
    """Return (pos, vel, t) for an OPEN epitrochoid curve with self-intersections.

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

    # Add positional noise
    kx, ky = jax.random.split(key)
    x = x0 + noise_sigma * jax.random.normal(kx, (n,))
    y = y0 + noise_sigma * jax.random.normal(ky, (n,))

    # Pack into unitful quantities
    pos = {"x": u.Q(x, usys["length"]), "y": u.Q(y, usys["length"])}
    vel = {"x": u.Q(dx0, usys["speed"]), "y": u.Q(dy0, usys["speed"])}
    return t, pos, vel


def main(
    output_path: str = "favicon.png",
    dpi: int = 100,
    size: int = 256,
) -> None:
    """Generate and save the epitrochoid favicon.

    Parameters
    ----------
    output_path : str, default="favicon.png"
        Path to save the output PNG file
    dpi : int, default=100
        DPI for the saved figure
    size : int, default=256
        Size of the square figure in pixels

    """
    print("Initializing random key...")
    key = jr.key(201030)

    usys = u.unitsystems.si

    # Generate epitrochoid data
    print("Generating epitrochoid curve...")
    key, subkey = jax.random.split(key)
    t, pos, vel = make_self_intersecting_stream(subkey, n=2048, noise_sigma=6)

    # Shuffle data
    print("Shuffling data...")
    key, subkey = jr.split(key)
    order = jr.permutation(subkey, jnp.arange(len(t)))

    qs = jax.tree.map(lambda x: x[order], pos)
    ps = jax.tree.map(lambda x: x[order], vel)

    # Find starting index
    start_idx = int(np.argsort(order)[0])

    # Perform local flow walk
    print("Performing local flow walk...")
    config = pcf.WalkConfig(
        strategy=pcf.strats.KDTree(k=60),
        metric=pcf.metrics.FullPhaseSpaceDistanceMetric(),
    )
    metric_scale = u.Q(4, "s")
    max_dist = u.Q(40, "m")

    walkresult = pcf.walk_local_flow(
        qs,
        ps,
        start_idx=start_idx,
        metric_scale=metric_scale,
        max_dist=max_dist,
        config=config,
        direction="forward",
        metadata=pcf.StateMetadata(usys=usys),
    )

    # Train autoencoder
    print("Training autoencoder...")
    key, model_key, train_key = jr.split(key, 3)
    normalizer = pcf.nn.StandardScalerNormalizer(qs, ps)
    model = pcf.nn.PathAutoencoder.make(normalizer, track_depth=4, key=model_key)
    model, _, losses = pcf.nn.train_autoencoder(model, walkresult, key=train_key)
    print(f"Final training loss: {losses[-1]:.6f}")

    # Encode all points and get mean path
    print("Encoding points and generating mean path...")
    all_gamma, _ = model.encode(walkresult.positions, walkresult.velocities)
    qs_pred = model.decode(jnp.linspace(*model.gamma_range, 1_000))

    # Create clean visualization for favicon
    print("Creating favicon visualization...")
    fig_size_inches = size / dpi  # Convert pixels to inches at given DPI
    fig, ax = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=dpi)

    # Plot colored points (epitrochoid)
    ax.scatter(
        np.asarray(qs["x"]),
        np.asarray(qs["y"]),
        s=3,
        c=np.asarray(all_gamma),
        cmap="RdYlBu",
        alpha=0.8,
    )

    # Plot predicted mean path as black line
    ax.plot(
        np.asarray(qs_pred["x"]),
        np.asarray(qs_pred["y"]),
        c="black",
        lw=1.5,
    )

    # Clean up: remove axes, labels, legend, etc.
    ax.axis("off")
    ax.set_aspect("equal")

    # Remove margins
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save figure
    print(f"Saving favicon to {output_path}...")
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
        facecolor="none",
        edgecolor="none",
        transparent=True,
    )
    plt.close(fig)

    print(f"✓ Favicon saved to {output_path}")


if __name__ == "__main__":
    # Get script directory
    script_dir = Path(__file__).parent

    # Determine output path (can be overridden via command line)
    output_path = str(script_dir / "favicon.png")
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    main(output_path=output_path)
