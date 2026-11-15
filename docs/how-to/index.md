# How-to guides

Each guide shows how to do one task. The [](../tutorials/index.md) show whole
analyses, [](../explanation/index.md) explains why the methods work, and the
[](../reference/index.md) lists every option.

An analysis has five steps, and each has its guides:

1. Prepare your data: put positions and velocities in the dictionary format
   ([](data.md)), with physical units if you have them ([](units.md)).
2. Order the samples: choose an orderer ([](choose-orderer.md)), chain stages
   ([](chain-orderers.md)), and tune the one you chose, either the local-flow walk
   ([](local-flow-walk.md), [](tune-metric.md), [](custom-metric.md), [](kdtree.md))
   or the self-organizing map ([](som.md)).
3. Fit a track: give every sample a value of $\gamma$ and fit a smooth track
   ([](autoencoder.md)), or interpolate along an ordering ([](interpolate.md)).
4. Reject outliers: give each sample a calibrated membership probability
   ([](outliers.md)), and put an uncertainty band on the track
   ([](track-uncertainty.md)).
5. Scale up: compile, batch and differentiate with JAX ([](jax.md)), and run on a
   GPU ([](gpu.md)).

```{toctree}
:maxdepth: 1
:hidden:

data
units
choose-orderer
chain-orderers
local-flow-walk
tune-metric
custom-metric
kdtree
som
autoencoder
interpolate
outliers
track-uncertainty
jax
gpu
```
