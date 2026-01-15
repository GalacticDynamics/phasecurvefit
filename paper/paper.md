---
title:
  "localflowwalk: A High-Performance JAX Framework for Phase-Space Walks with
  Pluggable Metrics and Strategies"
tags:
  - Python
  - JAX
  - astronomy
  - stellar streams
  - galactic dynamics
  - phase-space analysis
authors:
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    corresponding: true
    affiliation: "1, 2"
  - name: Jacob Nibauer
    orcid: 0000-0002-5408-3992
    affiliation: "3"
  - name: Sirui Wu
    orcid: 0009-0003-4675-3622
    affiliation: "4"
affiliations:
  - name:
      MIT Kavli Institute for Astrophysics and Space Research, Massachusetts
      Institute of Technology, Cambridge, MA 02139, USA
    index: 1
  - name:
      Department of Astronomy, Case Western Reserve University, Cleveland, OH
      44106, USA
    index: 2
  - name: Department of Physics, Princeton University, Princeton, NJ 08544, USA
    index: 3
  - name:
      DARK, Niels Bohr Institute, University of Copenhagen, Jagtvej 155, DK-2200
      Copenhagen, Denmark
    index: 4
date: 14 January 2026
bibliography: paper.bib
---

# Summary

Stellar streams—elongated structures of gravitationally unbound stars stripped
from dwarf galaxies and globular clusters—trace the gravitational potential of
the Milky Way with remarkable fidelity. As sensitive probes of galactic
structure and dark matter distribution, streams have become essential tools in
near-field cosmology [@bonaca2020stellar]. A critical preprocessing step in
stream analysis is ordering the discrete, noisy observations of stream member
stars along the stream's one-dimensional trajectory through six-dimensional
phase-space.

`localflowwalk` is an open-source Python package for constructing such orderings
by _walking along the local phase-space flow_ using JAX [@jax2018github]. The
core routine, `walk_local_flow`, is factored into two independent choices:

1. a **distance metric** that scores candidate next steps in phase space, and
2. a **query strategy** that proposes which candidates to consider.

This design makes the Nearest Neighbors with Momentum (NN+p) method from
[@nibauer2022charting] one particular configuration (via
`AlignedMomentumDistanceMetric`), while enabling alternative metrics and search
strategies better matched to different data quality, dimensionality, and
performance constraints.

In addition to the walk itself, `localflowwalk` includes a neural-network
gap-filling component implementing the autoencoder described in
[@nibauer2022charting], which can assign a continuous ordering parameter to
tracers skipped during the walk and provide a membership probability estimate.

# Statement of Need

Ordering stream tracers is often treated as "just" a nearest-neighbor problem,
but there is no single universally appropriate notion of proximity in stellar
stream data: spatial coordinates may be precise while velocities are noisy (or
unavailable), different coordinate systems impose different scalings, and
outliers or contaminants can trigger spurious jumps that spoil an ordering. In
practice, analysts frequently need to experiment with _both_ (i) how distances
are measured in phase space and (ii) how aggressively the neighbor search is
restricted.

The original NN+p implementation accompanying [@nibauer2022charting]
demonstrated the value of including a momentum-alignment term, but was tightly
coupled to that specific metric and not built as a reusable, extensible library.

`localflowwalk` addresses this need, providing a JAX-based implementation that
is:

1. **A modular walk API** with user-selectable metrics and neighbor-query
   strategies
2. **Hardware acceleration and composability** through JAX transformations and
   XLA compilation, interoperating with libraries like Equinox
   [@kidger2021equinox] and Diffrax [@kidger2022diffrax]
3. **Differentiability** of metrics and neural components for gradient-based
   tuning and downstream integration
4. **Coordinate-agnostic inputs** through JAX PyTrees (e.g., dicts of
   components)
5. **Neural gap filling** to assign orderings to skipped tracers
6. **Rigorously tested** with a property-based test suite via Hypothesis
   [@maciver2019hypothesis]

# Package Design

`walk_local_flow` constructs an ordered index sequence by iterating from a
starting tracer and repeatedly selecting the next tracer that minimizes a
user-supplied metric. At step $i$ with current phase-space point
$(\mathbf{q}_i, \mathbf{p}_i)$, the next point is chosen as:

$$
j^\* = \arg\min_{j \in \mathcal{C}_i} \; M\!\left(\mathbf{q}_i, \mathbf{p}_i, \mathbf{q}_j, \mathbf{p}_j;\lambda\right),
$$

where $M$ is the selected **metric**, $\lambda$ is a metric-dependent tuning
parameter, and $\mathcal{C}_i$ is the **candidate set** provided by the chosen
**strategy**. The walk can terminate early when no candidate lies within a
user-specified `max_dist`, leaving the remaining points "skipped"—a behavior
that is useful for rejecting outliers and contaminants.

## Strategies: controlling the candidate set

Strategies implement how $\mathcal{C}_i$ is constructed at each step.

- **Brute-force (`BruteForceStrategy`)** considers all unvisited points. This is
  exact and simple, and is often competitive for moderate $N$ when vectorized
  and JIT-compiled.
- **KD-tree prefiltering (`KDTreeStrategy`)** uses a spatial KD-tree to propose
  the $k$ nearest candidates in position space before applying the metric. This
  is useful when long-range jumps are undesirable or when the metric is
  expensive and a local candidate set is sufficient.

Custom strategies can be implemented by subclassing `AbstractQueryStrategy`,
allowing integration with approximate-nearest-neighbor methods or domain-
specific candidate proposals.

## Metrics: defining "distance" in phase space

Metrics implement the scoring function $M$ and encode the physical assumptions
about what constitutes a good next step.

- **`SpatialDistanceMetric`** uses Euclidean distance in position space and
  ignores velocities. It is appropriate when velocities are missing or
  unreliable, or when a baseline spatial ordering is desired:

  $$
  M = d_0(\mathbf{q}_i, \mathbf{q}_j).
  $$

- **`FullPhaseSpaceDistanceMetric`** uses a true Euclidean distance in the full
  phase space by combining position and velocity differences. Here $\lambda$
  acts as a time-scale $\tau$ that converts velocity differences to position
  units:

  $$
  M = \sqrt{d_0(\mathbf{q}_i,\mathbf{q}_j)^2 + (\tau\, d_v(\mathbf{p}_i,\mathbf{p}_j))^2 }.
  $$

  This metric is useful when one wants 6D proximity without imposing a preferred
  flow direction.

- **`AlignedMomentumDistanceMetric`** reproduces the NN+p momentum-alignment
  idea from [@nibauer2022charting] by penalizing candidate steps that are not
  aligned with the current velocity direction:

  $$
  M = d_0(\mathbf{q}_i,\mathbf{q}_j) + \lambda\left(1-\cos\theta_{ij}\right).
  $$

  Here the _alignment_ angle $\theta_{ij}$ is defined between the current
  velocity direction and the direction to the candidate point,
  $\cos\theta_{ij}=\hat{\mathbf{p}}_i\cdot\widehat{(\mathbf{q}_j-\mathbf{q}_i)}$.
  Perfect alignment ($\cos\theta_{ij}=1$) incurs no penalty, while anti-
  alignment ($\cos\theta_{ij}=-1$) is strongly disfavored, discouraging
  "backtracking" along the stream.

Because metrics are first-class objects, users can implement additional
problem-specific metrics (e.g., non-Euclidean coordinate scalings or learned
embeddings) by subclassing `AbstractDistanceMetric`.

## Neural gap filling for skipped tracers

When the walk terminates early or skips tracers, `localflowwalk.nn` provides an
autoencoder-based interpolator from [@nibauer2022charting] that maps phase-space
coordinates $(\mathbf{q},\mathbf{p})$ to a continuous ordering parameter
$\gamma\in[-1,1]$ and a membership probability $p\in[0,1]$. After training on
the visited (high-confidence) portion of the walk, the network can predict
$\gamma$ for all tracers and thus produce a complete ordering and a
probability-based filter for contaminants.

```python
import jax
import localflowwalk as lfw

res = lfw.walk_local_flow(
    pos,
    vel,
    start_idx=0,
    metric=lfw.metrics.FullPhaseSpaceDistanceMetric(),
)

ae = lfw.nn.Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=3)
trained, _ = lfw.nn.train_autoencoder(ae, res)
filled = lfw.nn.fill_ordering_gaps(trained, res, prob_threshold=0.5)
```

## Implementation Details

`localflowwalk` represents phase-space data as pairs of Python dictionaries
mapping component names to JAX arrays:

```python
import jax.numpy as jnp
import localflowwalk as lfw

# Full 6D phase-space
position = {"x": x, "y": y, "z": z}
velocity = {"vx": vx, "vy": vy, "vz": vz}

# Order the observations
result = lfw.walk_local_flow(
    position,
    velocity,
    start_idx=0,  # Index of starting point
    lam=1.0,  # Metric parameter (interpretation depends on metric)
    max_dist=5.0,  # Optional early termination threshold
    metric=lfw.metrics.FullPhaseSpaceDistanceMetric(),  # Or AlignedMomentumDistanceMetric()
    strategy=lfw.BruteForceStrategy(),  # Or KDTreeStrategy(k=...)
)

# Access ordered indices and any skipped points
ordered = result.ordered_indices
skipped = result.skipped_indices
```

Using `AlignedMomentumDistanceMetric` reproduces the NN+p momentum-alignment
selection rule from [@nibauer2022charting].

# Performance Characteristics

The walk can require computing distances between a current point and many
candidate points across $N$ iterations for $N$ points. JAX's vectorization and
compilation provide substantial constant-factor improvements:

- Vectorized metric computations avoid Python loop overhead
- JIT compilation eliminates interpreter costs and enables XLA optimization
- GPU execution parallelizes the distance computations across thousands of cores

In practice, we observe high performance on datasets with tens of thousands of
points. The implementation also supports `jax.vmap` for parallel analysis of
multiple streams or parameter sweeps without code modification.

# Research Applications

`localflowwalk` enables easy data-driven reconstruction of stream paths in
simulations. Beyond stellar streams, the same walk abstraction applies to
ordering problems in other phase-space datasets where a coherent local flow is
expected.

# Acknowledgements

N.S. is supported by the Brinson Prize Fellowship at MIT. J.N. ... S.W. ... This
work made use of the JAX, NumPy, and Astropy software packages.

# References
