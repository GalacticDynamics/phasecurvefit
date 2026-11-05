# How-to guides

Each guide shows how to do one task. The {doc}`../tutorials/index` show whole
analyses, {doc}`../explanation/index` explains why the methods work, and the
{doc}`../reference/index` lists every option.

An analysis has five steps, and each has its guides:

1. **Prepare your data**: put positions and velocities in the dictionary format
   ({doc}`data`), with physical units if you have them ({doc}`units`).
2. **Order the samples**: choose an orderer ({doc}`choose-orderer`), chain stages
   ({doc}`chain-orderers`), and tune the one you chose: the local-flow walk
   ({doc}`local-flow-walk`, {doc}`tune-metric`, {doc}`custom-metric`, {doc}`kdtree`)
   or the self-organizing map ({doc}`som`).
3. **Fit a track**: give every sample a value of $\gamma$ and fit a smooth track
   ({doc}`autoencoder`), or interpolate along an ordering ({doc}`interpolate`).
4. **Reject outliers**: give each sample a calibrated membership probability
   ({doc}`outliers`).
5. **Scale up**: compile, batch and differentiate with JAX ({doc}`jax`), and run on
   a GPU ({doc}`gpu`).

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
jax
gpu
```
