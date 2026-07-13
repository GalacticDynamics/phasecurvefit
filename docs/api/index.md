# API Reference

Complete API documentation for phasecurvefit.

```{eval-rst}
.. currentmodule:: phasecurvefit
```

## Main Function

```{eval-rst}
.. autofunction:: walk_local_flow
   :no-index:
```

## Result Accessor

Helper function to extract ordered data from results.

```{eval-rst}
.. autofunction:: order_w
   :no-index:
```

## Distance Metrics

Pluggable distance metrics for controlling how the algorithm selects the next point.
See the [Metrics Guide](../guides/metrics.md) for usage examples.

```{eval-rst}
.. currentmodule:: phasecurvefit.metrics

.. autoclass:: AbstractDistanceMetric
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: AlignedMomentumDistanceMetric
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: SpatialDistanceMetric
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: FullPhaseSpaceDistanceMetric
   :no-index:
   :members:
   :show-inheritance:

.. currentmodule:: phasecurvefit
```

## Phase-Space Utilities

Low-level functions for phase-space operations. Available in the `phasecurvefit.w` submodule.

```{eval-rst}
.. currentmodule:: phasecurvefit.w

.. autofunction:: euclidean_distance
   :no-index:

.. autofunction:: unit_direction
   :no-index:

.. autofunction:: unit_velocity
   :no-index:

.. autofunction:: cosine_similarity
   :no-index:

.. autofunction:: get_w_at
   :no-index:

.. currentmodule:: phasecurvefit
```

## Types

```{eval-rst}
.. autoclass:: WalkLocalFlowResult
   :no-index:
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

Neural network for interpolating skipped tracers. See [Autoencoder Guide](../guides/autoencoder.md) for details.

### Classes

```{eval-rst}
.. autoclass:: phasecurvefit.nn.PathAutoencoder
   :no-index:
   :members: encode, decode, decode_position, predict
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.OrderingNet
   :no-index:
   :members: __call__
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.TrackNet
   :no-index:
   :members: __call__
   :show-inheritance:

.. autoclass:: phasecurvefit.nn.TrainingConfig
   :no-index:
   :members:
```

### Training Functions

```{eval-rst}
.. autofunction:: phasecurvefit.nn.train_autoencoder
   :no-index:

.. autofunction:: phasecurvefit.nn.fill_ordering_gaps
   :no-index:
```

### Membership & Outlier Rejection

Mixture-model membership, after Hogg, Bovy & Lang (2010), §3. See
{doc}`/guides/outliers`.

```{eval-rst}
.. autoclass:: phasecurvefit.nn.MixtureMembershipConfig
   :no-index:
   :members:

.. autoclass:: phasecurvefit.nn.StreamWidthNet
   :no-index:
   :members: __call__
   :show-inheritance:

.. autofunction:: phasecurvefit.nn.stream_membership
   :no-index:

.. autofunction:: phasecurvefit.nn.mixture_membership_loss
   :no-index:

.. autofunction:: phasecurvefit.nn.membership_responsibility
   :no-index:

.. autofunction:: phasecurvefit.nn.sigma_ceiling
   :no-index:

.. autofunction:: phasecurvefit.nn.membership_rampup
   :no-index:

.. autofunction:: phasecurvefit.nn.uniform_background_density
   :no-index:
```

## Index

```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
