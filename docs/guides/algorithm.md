# Algorithm Details

This page explains the mathematical foundations and implementation details of the Nearest Neighbors with Momentum (NN+p) algorithm.

## Mathematical Foundation

### Distance Metric

The NN+p algorithm uses a composite distance metric that combines spatial proximity with momentum alignment:

$$d = d_0 + \lambda \cdot (1 - \cos\theta)$$

where:

- $d_0 = \|\mathbf{p}_{\text{candidate}} - \mathbf{p}_{\text{current}}\|$ is the Euclidean distance in position space
- $\cos\theta$ is the cosine similarity between the velocity and the direction to the candidate
- $\lambda \geq 0$ is the momentum weight parameter

### Breaking Down the Components

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

### Effect of Lambda (λ)

The momentum weight $\lambda$ controls the balance between spatial and momentum considerations:

- **$\lambda = 0$**: Pure nearest neighbor (spatial only)
  $$d = d_0$$
  Selects the closest point regardless of velocity direction.

- **$\lambda \to \infty$**: Pure momentum (directional only)
  $$d \approx \lambda \cdot (1 - \cos\theta)$$
  Strongly favors points in the velocity direction, even if far away.

- **$\lambda \approx 1$**: Balanced
  Both spatial proximity and momentum alignment matter equally.

### Physical Interpretation

For stellar streams, the momentum term encodes the fact that stars in a stream follow coherent trajectories. A star is more likely to be the next member of the stream if:

1. It's spatially close ($d_0$ is small)
2. It lies in the direction the stream is flowing ($\cos\theta \approx 1$)

The algorithm naturally traces the stream by following the "flow" of velocities through phase-space.

## Algorithm Pseudocode

```
Algorithm: Nearest Neighbors with Momentum

Input:
    position: dict[str, Array]  # N points with shape (N,) per component
    velocity: dict[str, Array]  # N points with shape (N,) per component
    start_idx: int              # starting index
    lam: float                  # momentum weight
    max_dist: float | None      # optional gap detection
    terminate_indices: Set[int] | None  # optional termination indices
    n_max: int | None           # optional iteration limit

Output:
    ordered_indices: tuple[int, ...]
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
        d0[i] ← euclidean_distance(current_pos, {key: position[key][i] for key})
        unit_dirs[i] ← unit_direction(current_pos, {key: position[key][i] for key})
        unit_vel ← unit_velocity(current_vel)  # Scalar dict
        cos_sim[i] ← cosine_similarity(unit_vel, unit_dirs[i])
        distances[i] ← d0[i] + lam * (1 - cos_sim[i])

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

## Implementation Details

### JAX Optimization

The implementation uses several JAX-specific optimizations:

1. **`jax.lax.while_loop`**: Compiles the main loop for maximum performance with JIT
2. **Scalar-first design**: Functions operate on scalar phase-space dicts, enabling JAX tracing
3. **vmap vectorization**: Uses `jax.vmap` to vectorize distance computation over candidate indices
4. **Masking with infinity**: Uses `jnp.where` to mask visited points instead of dynamic filtering
5. **Tuple state**: Immutable state tuple `(ordered_arr, visited_mask, current_idx, step, should_stop)` avoids PyTree overhead

### Scalar-First Design with vmap Vectorization

Distance computation follows a **scalar-first** design: core distance functions operate on
scalar phase-space dictionaries (single points), and vectorization is achieved through
`jax.vmap`. This ensures excellent JAX compatibility and JIT performance.

For the current point at index `current_idx`:

1. **Extract scalar phase-space dict**:
   <!-- skip: next -->
   ```python
   current_pos = {key: position[key][current_idx] for key in position}
   ```

2. **Define scalar distance function**:
   <!-- skip: next -->
   ```python
   def dist_to_candidate(idx):
       cand_pos = {key: position[key][idx] for key in position}
       d0 = euclidean_distance(current_pos, cand_pos)
       u_dir = unit_direction(current_pos, cand_pos)
       u_vel = unit_velocity(current_velocity)
       cos_sim = cosine_similarity(u_vel, u_dir)
       return d0 + lam * (1 - cos_sim)
   ```

3. **Vectorize over all point indices with vmap**:
   <!-- skip: next -->
   ```python
   distances = jax.vmap(dist_to_candidate)(jnp.arange(N))
   ```

4. **Mask visited points and find nearest**:
   <!-- skip: next -->
   ```python
   distances_masked = jnp.where(visited_mask < 0.5, distances, jnp.inf)
   best_idx = jnp.argmin(distances_masked)
   ```

This approach ensures:
- **JAX tracing**: Scalar operations compose naturally with JAX transformations
- **JIT compilation**: No dynamic shapes or fancy indexing
- **Flexibility**: Works across different coordinate systems (1D, 2D, 3D, etc.)
- **Modularity**: Distance computation is independent of array stacking

### Early Termination

The algorithm can terminate before visiting all points in several ways:

1. **max_dist**: Stops when the nearest unvisited point exceeds the distance threshold
2. **terminate_indices**: Stops when reaching specific indices
3. **n_max**: Stops after a maximum number of iterations

This is intentional! As noted in Nibauer et al. (2022):

> "Due to the momentum condition, the algorithm inevitably passes over some stream particles without incorporating them into the nearest neighbors graph."

Points that would require "going backwards" against the velocity direction receive high penalties and may be skipped.

## Edge Cases

### Stationary Points (Zero Velocity)

When the velocity is zero or very small:
- The unit velocity is undefined
- Implementation uses `safe_vel_norm = jnp.maximum(vel_norm, 1e-10)` to avoid division by zero
- Behavior degrades to pure nearest neighbor ($\lambda$ has no effect)

### Degenerate Positions (Zero Distance)

When two points occupy the same position:
- The direction is undefined
- Implementation uses `safe_d0 = jnp.maximum(d0, 1e-10)`
- The point with better momentum alignment is preferred

### Single Point

When there's only one point:
- Returns immediately with that point as the only element
- No distances are computed

## Computational Complexity

- **Time per iteration**: $O(N)$ for computing distances to all points
- **Total iterations**: Up to $N$ (but often terminates early)
- **Overall**: $O(N^2)$ worst case, but:
  - Can be much better with early termination
  - Highly optimized via JIT compilation
  - Vectorized operations enable GPU acceleration

## Gap Filling with Autoencoder

Due to the momentum condition, the NN+p algorithm inevitably skips some tracers. To assign $\gamma$ values to these skipped particles, an **autoencoder neural network** can interpolate based on phase-space location:

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

Potential future extensions:
- Adaptive $\lambda$ based on local stream properties
- Multiple starting points with graph merging
- Probabilistic variants for noisy data
- GPU-specific optimizations for very large datasets

## References

Nibauer, J., et al. (2022). "Charting Galactic Accelerations with Stellar Streams and Machine Learning." arXiv:2201.12042.
