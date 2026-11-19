# How it works

These pages explain how each stage of phasecurvefit works and why it is built that
way. They contain no steps to follow; the [](../how-to/index.md) do, and the
[](../reference/index.md) lists every option.

An ordering and a track come from up to three stages. An orderer puts the samples in
order: by default an MST backbone ([](mst.md)) refined by a self-organizing map
([](som.md)), or the local-flow walk ([](walk.md)). An autoencoder then gives every
sample an ordering coordinate and fits a smooth track ([](autoencoder.md)). A
mixture model of the stream and a background can make that fit reject interlopers
([](outliers.md)). Refitting bootstrap resamples gives the track's uncertainty
([](track-uncertainty.md)). Data given as an image rather than a list of samples, such as a
density map, goes through the same stages with each pixel weighted by its value
([](density-fields.md)).

```{toctree}
:maxdepth: 1
:hidden:

walk
mst
som
autoencoder
outliers
track-uncertainty
density-fields
```
