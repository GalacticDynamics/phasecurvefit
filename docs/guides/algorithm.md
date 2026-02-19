# Algorithm Details

This page explains the mathematical foundations and implementation details of
the phase flow walking algorithm.

## Mathematical Foundation

### Walking Algorithm (Metric-Agnostic)

The walking algorithm is **metric-agnostic**. It only requires a distance (or score)
function that compares the current point to candidate points. The algorithm then:

1. Computes distances to all unvisited points
2. Selects the minimum distance
3. Advances the walk
4. Stops early when termination criteria are met

The specific **distance metric is pluggable** and can be replaced without changing
the walking logic.

### Distance Metrics (Pluggable)

The default metric is **AlignedMomentumDistanceMetric**, which combines spatial
proximity with momentum alignment:

$$d = d_0 + \lambda \cdot (1 - \cos\theta)$$

where:

- $d_0 = \|\mathbf{p}_{\text{candidate}} - \mathbf{p}_{\text{current}}\|$ is the Euclidean distance in position space
- $\cos\theta$ is the cosine similarity between the velocity and the direction to the candidate
- $\lambda \geq 0$ is the momentum weight parameter

Other metrics are available (e.g., spatial-only), and custom metrics can be supplied.
See [Metrics](metrics.md) for the full list and extension guide.

### Breaking Down the Aligned-Momentum Components

#### Spatial Distance ($d_0$)

The Euclidean distance between two points in position space:

$$d_0 = \sqrt{\sum_i (p_i^{\text{candidate}} - p_i^{\text{current}})^2}$$

This measures how far apart the points are spatially.

#### Momentum Term ($\lambda \cdot (1 - \cos\theta)$)

The momentum term penalizes points that are not in the direction of the current velocity:

1. **Unit direction vector** from current to candidate:
   $$\hat{\mathbf{u}}_{\text{dir}} = \frac{\mathbf{p}_{\text{candidate}} - \mathbf{p}_{\text{current}}}{d_0}$$

2. **Unit velocity vector**:
   $$\hat{\mathbf{v}} = \frac{\mathbf{v}_{\text{current}}}{\|\mathbf{v}_{\text{current}}\|}$$

3. **Cosine similarity**:
   $$\cos\theta = \hat{\mathbf{u}}_{\text{dir}} \cdot \hat{\mathbf{v}}$$

4. **Momentum penalty**:
   - When $\cos\theta = 1$ (aligned): penalty = 0
   - When $\cos\theta = 0$ (perpendicular): penalty = $\lambda$
   - When $\cos\theta = -1$ (opposite): penalty = $2\lambda$

### Effect of $\lambda$ in the Default Metric

The momentum weight $\lambda$ controls the balance between spatial and momentum considerations:

- **$\lambda = 0$**: Pure nearest neighbor (spatial only)
  $$d = d_0$$
  Selects the closest point regardless of velocity direction.

- **$\lambda \to \infty$**: Pure momentum (directional only)
  $$d \approx \lambda \cdot (1 - \cos\theta)$$
  Strongly favors points in the velocity direction, even if far away.

- **$\lambda \approx 1$**: Balanced
  Both spatial proximity and momentum alignment matter equally.

### Physical Interpretation (Default Metric)

For stellar streams, the momentum term encodes the fact that stars in a stream follow
coherent trajectories. A star is more likely to be the next member of the stream if:

1. It's spatially close ($d_0$ is small)
2. It lies in the direction the stream is flowing ($\cos\theta \approx 1$)

The algorithm naturally traces the stream by following the "flow" of velocities
through phase-space.

## Algorithm Pseudocode

```text
Input:
    position: dict[str, Array]  # N points with shape (N,) per component
    velocity: dict[str, Array]  # N points with shape (N,) per component
    start_idx: int              # starting index
    metric_scale: float         # scale parameter for distance metric
    max_dist: float | None      # optional gap detection
    terminate_indices: Set[int] | None  # optional termination indices
    n_max: int | None           # optional iteration limit

Output:
    indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]

Procedure:
    Initialize:
        ordered_arr ← [-1, -1, ..., -1]  # Array to store ordered indices
        visited_mask ← [1.0, 1.0, ..., 1.0]  # 0 = visited, 1 = unvisited
        visited_mask[start_idx] ← 0.0
        ordered_arr[0] ← start_idx
        current_idx ← start_idx
        step ← 1

    While (step < n_max) AND (current_idx not in terminate_indices):
        # Get scalar phase-space data at current index
        current_pos ← {key: position[key][current_idx] for key in position}
        current_vel ← {key: velocity[key][current_idx] for key in velocity}

      # Vectorized computation to all points (vmap over array index)
      distances[i] ← metric(current_pos, current_vel, {key: position[key][i] for key}, {key: velocity[key][i] for key})

        # Mask visited points with infinity
        distances_masked[i] ← visited_mask[i] > 0.5 ? distances[i] : infinity

        # Find nearest unvisited neighbor
        min_dist ← min(distances_masked)
        best_idx ← argmin(distances_masked)

        # Check early termination
        if min_dist > max_dist:
            Break  # Gap detected, stop algorithm

        # Update state
        ordered_arr[step] ← best_idx
        visited_mask[best_idx] ← 0.0
        current_idx ← best_idx
        step ← step + 1

    # Extract valid indices
    skipped ← {i : visited_mask[i] > 0.5 for i in 0..N-1}
    Return (ordered_arr[:step], skipped)
```


## Gap Filling with Autoencoder

Due to the momentum condition, the walk algorithm inevitably skips some tracers.
To assign $\gamma$ values to these skipped particles, an **autoencoder neural
network** can interpolate based on phase-space location:

1. **Interpolation Network**: Learns $(x, v) \rightarrow (\gamma, p)$ from ordered tracers
2. **Param-Net**: Reconstructs positions from $\gamma$ values
3. **Momentum condition**: Ensures alignment with velocity field

See [Autoencoder for Gap Filling](autoencoder.md) for details.

## Extensions and Variants

The current implementation supports:

- **Arbitrary dimensions**: Works in 1D, 2D, 3D, or higher
- **Gap detection**: `max_dist` parameter
- **Conditional termination**: `terminate_indices` parameter
- **Limited search**: `n_max` parameter
- **Gap filling**: Autoencoder neural network for skipped tracers
- **Reverse walks**: `direction="backward"` parameter to trace streams backwards by negating velocities
- **Bidirectional walks**: `combine_flow_walks()` to trace streams in both directions simultaneously

### Reverse Walks

The `direction` parameter enables walking through phase-space in the opposite
direction by negating the velocity vectors. This is useful for:

- Tracing stellar streams backwards from the tidal tail towards the progenitor
- Exploring both branches of a bifurcated stream
- Testing bidirectional connectivity in phase-space

To use backward walks:

```python
import jax.numpy as jnp
import phasecurvefit as pcf

# Create sample data
pos = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
}
vel = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5, 0.5]),
}

# Forward walk (default)
result_forward = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

# Reverse walk from the same starting point
result_reverse = pcf.walk_local_flow(
    pos, vel, start_idx=0, metric_scale=1.0, direction="backward"
)
```

The negated velocities ensure that the algorithm follows the stream in the
opposite direction. This is mathematically equivalent to physically reversing
time in the dynamical system.

### Combining Forward and Reverse Walks

For stellar streams that extend in both directions from a starting point (e.g.,
from a progenitor or disruption point), the `combine_flow_walks()` function
combines the results of two separate walks into a single coherent ordering:

```python
# Run forward and reverse walks separately
result_forward = pcf.walk_local_flow(
    pos, vel, start_idx=2, metric_scale=1.0, direction="forward"
)
result_reverse = pcf.walk_local_flow(
    pos, vel, start_idx=2, metric_scale=1.0, direction="backward"
)

# Combine the results
result = pcf.combine_flow_walks(result_forward, result_reverse)

# Result indices are ordered: [reverse tail] → [start] → [forward tail]
```

This can be simplified to:

```python
result = pcf.walk_local_flow(pos, vel, start_idx=2, metric_scale=1.0, direction="both")
```

This is particularly useful for:

- **Complete stream tracing**: Get a full ordering from one tidal tail to the other
- **Progenitor analysis**: Start from the progenitor and trace both streams
- **Bifurcated streams**: Explore complex geometries extending in multiple directions
- **Stream validation**: Verify connectivity in both directions simultaneously

The combination strategy ensures spatial coherence by placing the starting point
near the center with the reverse tail on one end and the forward tail on the
other. Since you run the walks separately, you have full control over the
parameters for each direction (e.g., different `max_dist` thresholds).

Potential future extensions:
- Adaptive $\lambda$ based on local stream properties
- Multiple starting points with graph merging
- Probabilistic variants for noisy data
- GPU-specific optimizations for very large datasets

## References

Nibauer, J., et al. (2022). "Charting Galactic Accelerations with Stellar Streams and Machine Learning." arXiv:2201.12042.
