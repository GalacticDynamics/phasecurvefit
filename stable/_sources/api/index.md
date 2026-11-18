# API Reference

Complete API documentation for phasecurvefit.

```{eval-rst}
.. currentmodule:: phasecurvefit
```

## Main Functions

`order` is the primary entry point; `walk_local_flow` is its deprecated
predecessor.

```{eval-rst}
.. autofunction:: order

.. autofunction:: walk_local_flow

.. autofunction:: combine_results
```

## Orderers

Pluggable ordering algorithms. See the [Orderers guide](../guides/orderers.md).

```{eval-rst}
.. currentmodule:: phasecurvefit.orderers

.. autoclass:: AbstractOrderer
   :members: order
   :show-inheritance:

.. autoclass:: LocalFlowOrderer
   :members: order
   :show-inheritance:

.. autoclass:: MSTOrderer
   :members: order
   :show-inheritance:

.. autoclass:: OrderingResult
   :members:
   :special-members: __call__
   :show-inheritance:

.. currentmodule:: phasecurvefit
```

## Walk Configuration

```{eval-rst}
.. autoclass:: WalkConfig
   :members:

.. autoclass:: StateMetadata
   :members:
   :no-inherited-members:
```

## Result Accessor

Helper function to extract ordered data from results.

```{eval-rst}
.. autofunction:: order_w
```

## Distance Metrics

Pluggable distance metrics for controlling how the algorithm selects the next point.
See the [Metrics Guide](../guides/metrics.md) for usage examples.

```{eval-rst}
.. currentmodule:: phasecurvefit.metrics

.. autoclass:: AbstractDistanceMetric
   :members:
   :show-inheritance:

.. autoclass:: AlignedMomentumDistanceMetric
   :members:
   :show-inheritance:

.. autoclass:: SpatialDistanceMetric
   :members:
   :show-inheritance:

.. autoclass:: FullPhaseSpaceDistanceMetric
   :members:
   :show-inheritance:

.. currentmodule:: phasecurvefit
```

## Query Strategies

Neighbour-query strategies for the walk, set via `WalkConfig(strategy=...)`.

```{eval-rst}
.. currentmodule:: phasecurvefit.strats

.. autoclass:: AbstractQueryStrategy
   :members:
   :show-inheritance:

.. autoclass:: BruteForce
   :members:
   :show-inheritance:

.. autoclass:: KDTree
   :members:
   :show-inheritance:

.. autoclass:: QueryResult
   :members:

.. currentmodule:: phasecurvefit
```

## Phase-Space Utilities

Low-level functions for phase-space operations. Available in the `phasecurvefit.w` submodule.

```{eval-rst}
.. currentmodule:: phasecurvefit.w

.. autofunction:: euclidean_distance

.. autofunction:: unit_direction

.. autofunction:: unit_velocity

.. autofunction:: velocity_norm

.. autofunction:: cosine_similarity

.. autofunction:: get_w_at

.. currentmodule:: phasecurvefit
```

## Types

```{eval-rst}
.. autoclass:: WalkLocalFlowResult
   :members:
   :show-inheritance:

.. data:: ScalarComponents
   :annotation: : TypeAlias = Mapping[str, FLikeSz0]

   Type alias for dictionaries mapping component names to scalar JAX arrays.

   Used for single phase-space points. Keys are coordinate/component names
   (e.g., "x", "y", "z"), values are 0-dimensional JAX arrays.

   Example::

       position: ScalarComponents = {
           "x": jnp.array(1.0),
           "y": jnp.array(2.0),
       }

.. data:: VectorComponents
   :annotation: : TypeAlias = Mapping[str, FLikeSzN]

   Type alias for dictionaries mapping component names to 1D JAX arrays.

   Used for arrays of phase-space points. Keys are coordinate/component names
   (e.g., "x", "y", "z"), values are 1-dimensional JAX arrays of shape (N,).

   Example::

       position: VectorComponents = {
           "x": jnp.array([0.0, 1.0, 2.0]),
           "y": jnp.array([0.0, 1.0, 2.0]),
       }
```

## Autoencoder Module

Neural network for interpolating skipped tracers. See [Autoencoder Guide](../guides/nn.md) for details.

### Classes

```{eval-rst}
.. autoclass:: phasecurvefit.nn.PathAutoencoder
   :members: encode, decode, decode_position, predict
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.OrderingNet
   :members: __call__
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.TrackNet
   :members: __call__
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.TrainingConfig
   :members:

.. autoclass:: phasecurvefit.nn.AbstractTrackNet
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.FourierTrackNet
   :members: __call__
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.AbstractAutoencoder
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.AbstractExternalDecoder
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.EncoderExternalDecoder
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.RunningMeanDecoder
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.AutoencoderResult
   :members:
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.OrderingTrainingConfig
   :members:

.. autoclass:: phasecurvefit.nn.AbstractNormalizer
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.StandardScalerNormalizer
   :show-inheritance:
```

### Training Functions

```{eval-rst}
.. autofunction:: phasecurvefit.nn.train_autoencoder

.. autofunction:: phasecurvefit.nn.fill_ordering_gaps

.. autofunction:: phasecurvefit.nn.train_ordering_net

.. autofunction:: phasecurvefit.nn.encoder_loss
```

### Membership & Outlier Rejection

Mixture-model membership, after Hogg, Bovy & Lang (2010), §3. See
{doc}`/guides/outliers`.

```{eval-rst}
.. autoclass:: phasecurvefit.nn.MixtureMembershipConfig
   :members:

.. autoclass:: phasecurvefit.nn.WidthNet
   :members: __call__
   :show-inheritance:

.. autofunction:: phasecurvefit.nn.posterior_membership

.. autofunction:: phasecurvefit.nn.mixture_membership_loss

.. autofunction:: phasecurvefit.nn.membership_responsibility

.. autofunction:: phasecurvefit.nn.sigma_ceiling

.. autofunction:: phasecurvefit.nn.membership_rampup

.. autofunction:: phasecurvefit.nn.uniform_background_density
```

## Index

```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
