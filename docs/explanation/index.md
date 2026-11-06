# How it works

These pages explain how each stage of phasecurvefit works and why it is built that
way. They contain no steps to follow; the {doc}`../how-to/index` do, and the
{doc}`../reference/index` lists every option.

An ordering and a track come from up to three stages. An orderer puts the samples in
order: by default an MST backbone ({doc}`mst`) refined by a self-organizing map
({doc}`som`), or the local-flow walk ({doc}`walk`). An autoencoder then gives every
sample an ordering coordinate and fits a smooth track ({doc}`autoencoder`). A
mixture model of the stream and a background can make that fit reject interlopers
({doc}`outliers`). Data given as an image rather than a list of samples, such as a
density map, goes through the same stages with each pixel weighted by its value
({doc}`density-fields`).

```{toctree}
:maxdepth: 1
:hidden:

walk
mst
som
autoencoder
outliers
density-fields
```
