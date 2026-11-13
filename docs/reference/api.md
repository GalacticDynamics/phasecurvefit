# API Reference

Complete API documentation for phasecurvefit.

```{currentmodule} phasecurvefit
```

## Main Functions

`order` is the primary entry point. With no orderer it runs `default_pipeline`, an
MST backbone refined by a SOM; `fit_track` goes on to fit the autoencoder.

```{autofunction} order
```

```{autofunction} phasecurvefit.orderers.default_pipeline
```

```{autofunction} fit_track
```

```{autofunction} bootstrap_track
```

```{autoclass} TrackEnsemble
:members: offsets, radius
```

```{autofunction} sample_track
```

```{autoclass} TrackPosterior
```

```{autoclass} TrackDraw
:members: width
```

```{autofunction} track_model
```

```{autofunction} combine_results
```

## Orderers

Pluggable ordering stages, composable with `|`. See the
[Orderers guide](../how-to/choose-orderer.md).

```{currentmodule} phasecurvefit.orderers
```

```{autoclass} AbstractOrderer
:members: order
:show-inheritance:
```

```{autoclass} LocalFlowOrderer
:members: order
:show-inheritance:
```

```{autoclass} MSTOrderer
:members: order
:show-inheritance:
```

```{autoclass} SOMOrderer
:members: order
:show-inheritance:
```

```{autoclass} RidgeOrderer
:members: order
:show-inheritance:
```

```{autoclass} ChainOrderer
:members: order
:show-inheritance:
```

```{autoclass} OrderingResult
:members:
:special-members: __call__
:show-inheritance:
```

```{currentmodule} phasecurvefit
```

## Walk Configuration

```{autoclass} WalkConfig
:members:
```

```{autoclass} StateMetadata
:members:
:no-inherited-members:
```

## Result Accessor

Helper function to extract ordered data from results.

```{autofunction} order_w
```

## Distance Metrics

Pluggable distance metrics for controlling how the algorithm selects the next point.
See the [Metrics Guide](../how-to/tune-metric.md) for usage examples.

```{currentmodule} phasecurvefit.metrics
```

```{autoclass} AbstractDistanceMetric
:members:
:show-inheritance:
```

```{autoclass} AlignedMomentumDistanceMetric
:members:
:show-inheritance:
```

```{autoclass} SpatialDistanceMetric
:members:
:show-inheritance:
```

```{autoclass} FullPhaseSpaceDistanceMetric
:members:
:show-inheritance:
```

```{currentmodule} phasecurvefit
```

## Query Strategies

Neighbour-query strategies for the walk, set via `WalkConfig(strategy=...)`.

```{currentmodule} phasecurvefit.strats
```

```{autoclass} AbstractQueryStrategy
:members:
:show-inheritance:
```

```{autoclass} BruteForce
:members:
:show-inheritance:
```

```{autoclass} KDTree
:members:
:show-inheritance:
```

```{autoclass} QueryResult
:members:
```

```{currentmodule} phasecurvefit
```

## Phase-Space Utilities

Low-level functions for phase-space operations. Available in the `phasecurvefit.w` submodule.

```{currentmodule} phasecurvefit.w
```

```{autofunction} euclidean_distance
```

```{autofunction} unit_direction
```

```{autofunction} unit_velocity
```

```{autofunction} velocity_norm
```

```{autofunction} cosine_similarity
```

```{autofunction} get_w_at
```

```{currentmodule} phasecurvefit
```

## Types

```{autoclass} WalkLocalFlowResult
:members:
:show-inheritance:
```

````{py:data} ScalarComponents
:annotation: : TypeAlias = dict[str, RLikeSz0]

Type alias for dictionaries mapping component names to scalar JAX arrays.

Used for single phase-space points. Keys are coordinate/component names
(e.g., "x", "y", "z"), values are 0-dimensional JAX arrays.

Example:

```python
import jax.numpy as jnp

position = {
    "x": jnp.array(1.0),
    "y": jnp.array(2.0),
}
```
````

````{py:data} VectorComponents
:annotation: : TypeAlias = dict[str, RLikeSzN]

Type alias for dictionaries mapping component names to 1D JAX arrays.

Used for arrays of phase-space points. Keys are coordinate/component names
(e.g., "x", "y", "z"), values are 1-dimensional JAX arrays of shape (N,).

Example:

```python
import jax.numpy as jnp

position = {
    "x": jnp.array([0.0, 1.0, 2.0]),
    "y": jnp.array([0.0, 1.0, 2.0]),
}
```
````

## Autoencoder Module

Neural network for interpolating skipped tracers. See [Autoencoder Guide](../how-to/autoencoder.md) for details.

### Classes

```{autoclass} phasecurvefit.nn.PathAutoencoder
:members: encode, decode, decode_position, predict
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.OrderingNet
:members: __call__
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.TrackNet
:members: __call__
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.TrainingConfig
:members:
```

```{autoclass} phasecurvefit.nn.AbstractTrackNet
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.FourierTrackNet
:members: __call__
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.AbstractAutoencoder
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.AbstractExternalDecoder
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.EncoderExternalDecoder
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.RunningMeanDecoder
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.AutoencoderResult
:members:
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.OrderingTrainingConfig
:members:
```

```{autoclass} phasecurvefit.nn.AbstractNormalizer
:show-inheritance:
```

```{autoclass} phasecurvefit.nn.StandardScalerNormalizer
:show-inheritance:
```

### Training Functions

```{autofunction} phasecurvefit.nn.train_autoencoder
```

```{autofunction} phasecurvefit.nn.fill_ordering_gaps
```

```{autofunction} phasecurvefit.nn.train_ordering_net
```

```{autofunction} phasecurvefit.nn.encoder_loss
```

### Membership & Outlier Rejection

Mixture-model membership, after Hogg, Bovy & Lang (2010), §3. See
{doc}`/explanation/outliers`.

```{autoclass} phasecurvefit.nn.MixtureMembershipConfig
:members:
```

```{autoclass} phasecurvefit.nn.WidthNet
:members: __call__
:show-inheritance:
```

```{autofunction} phasecurvefit.nn.posterior_membership
```

```{autofunction} phasecurvefit.nn.mixture_membership_loss
```

```{autofunction} phasecurvefit.nn.membership_responsibility
```

```{autofunction} phasecurvefit.nn.sigma_ceiling
```

```{autofunction} phasecurvefit.nn.membership_rampup
```

```{autofunction} phasecurvefit.nn.uniform_background_density
```

## SOM Module

Self-Organizing Map ordering, after Starkman et al. (2023). See
{doc}`/explanation/som`.

{class}`~phasecurvefit.orderers.SOMOrderer` is listed under [Orderers](#orderers).

```{py:module} phasecurvefit.som
```

```{autoclass} phasecurvefit.som.SOM1D
:members: make, fit, backbone, chord, n_prototypes
:show-inheritance:
```

```{autofunction} phasecurvefit.som.init_prototypes
```

```{autofunction} phasecurvefit.som.fit
```

```{autofunction} phasecurvefit.som.densify
```

```{autofunction} phasecurvefit.som.chord
```

```{autofunction} phasecurvefit.som.bootstrap_weights
```

```{autofunction} phasecurvefit.som.bmu_distance
```

```{autoclass} phasecurvefit.som.FitResult
:members:
```

```{autoclass} phasecurvefit.som.SOMMixture
```

## Neighbour Backends

Exact k-nearest-neighbour backends for
{class}`~phasecurvefit.orderers.MSTOrderer` (`neighbors=`).

**Choosing a backend.** `SciPy` is **eager-only**: it raises `TypeError`
inside `jit`, `vmap` or `grad`, including on arrays a jitted function captures
rather than takes as arguments. Under those transforms use
`BucketKDTree` (or `BruteForce`/`JaxKD`). `MSTOrderer`'s default
(`neighbors=None`) makes that choice per call: `SciPy` for concrete inputs,
`BucketKDTree` when traced.

`BucketKDTree` and `BruteForce` wrap
[jaxknn](https://github.com/nstarman/jaxknn)'s `knn` and `brute_knn`. The
kd-tree buys traceability, not CPU speed. Compiled under `jit`, it is
slower than eager `SciPy`, and it compiles once per input shape. Measured on
an Apple M2 Pro (10 cores, under load, so treat these as rough):

| n | `jit` compile, first call | kNN (k=10): jit kd-tree / SciPy | `MSTOrderer` (k=10): jit kd-tree / SciPy |
|---|---|---|---|
| ~2,000 | ~8–9 s | ~3× slower | ~2× slower |
| 10,000–100,000 | ~11–13 s | ~6.5–8.5× slower | ~1.6–2.8× slower (20,000–100,000) |
| 1,000,000 | ~27 s | ~6× slower | not measured |

`SciPy` queries with every core (`workers=-1`); the kd-tree query is only
partly parallel on CPU. GPU was not measured.

```{py:module} phasecurvefit.neighbors
```

```{autoclass} phasecurvefit.neighbors.AbstractNeighborSearch
:members: knn
```

```{autoclass} phasecurvefit.neighbors.BucketKDTree
```

```{autoclass} phasecurvefit.neighbors.BruteForce
```

```{autoclass} phasecurvefit.neighbors.JaxKD
```

```{autoclass} phasecurvefit.neighbors.SciPy
```

## Index

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
