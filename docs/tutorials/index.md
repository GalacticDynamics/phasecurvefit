# Tutorials

Each tutorial is a complete analysis that runs from simulated data to a result.
The groups follow the kind of data you have; start with the group that matches
yours. New to phasecurvefit? Begin with the {doc}`../start/quickstart`, then the
stream autoencoder tutorial.

The notebooks need the `tutorials` extra, which adds `matplotlib` and `galax`
({doc}`../start/install`). Continuous integration runs every notebook on each
change, so each one is checked to run with the current code.

## A stream with velocities

- {doc}`stream_autoencoder`: order a simulated stellar stream from its progenitor,
  fit its mean track, and read the training loss. **Start here.**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/stream_autoencoder.ipynb)
- {doc}`stream_runningmean`: a faster, training-free mean track for the same
  stream, and when it is accurate enough.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/stream_runningmean.ipynb)

## No known start point

- {doc}`stream_mst`: order a stream with no progenitor or start index, and reject
  injected interlopers.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/stream_mst.ipynb)
- {doc}`epitrochoid_mst`: why the MST needs velocities on a self-crossing curve,
  and a Fourier-feature decoder.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_mst.ipynb)

## A curve that crosses itself

- {doc}`epitrochoid_autoencoder`: order a curve that crosses itself, using
  velocities to stay on the right strand.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_autoencoder.ipynb)
- {doc}`epitrochoid_runningmean`: why a running mean cuts inside tight bends, and
  how the window size controls it.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_runningmean.ipynb)

## Contaminated data

- {doc}`outlier_rejection`: reject interlopers with a stream-plus-background mixture
  model, and see why the orderers alone can't.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/outlier_rejection.ipynb)

## An image or density map

- {doc}`density_fields`: fit a track through an image of a stream, with each pixel a
  sample weighted by its value.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/density_fields.ipynb)

```{toctree}
:maxdepth: 1
:hidden:

stream_autoencoder
stream_runningmean
stream_mst
epitrochoid_mst
epitrochoid_autoencoder
epitrochoid_runningmean
outlier_rejection
density_fields
```
