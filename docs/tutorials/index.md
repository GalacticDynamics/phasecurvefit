# Tutorials

This section contains practical examples demonstrating how to use phasecurvefit.

Running these notebooks locally requires the `tutorials` extra, which adds
`matplotlib` and `galax` on top of `phasecurvefit[all]`:

```bash
pip install phasecurvefit[tutorials]
```

**Where to start:** new to `phasecurvefit`? Begin with the **Stream
Autoencoder** tutorial, which introduces ordering, the autoencoder and how to read
its training. The others build on it: the running-mean tutorials trade accuracy
for speed, the MST tutorials order without a starting point, the epitrochoid
tutorials handle curves that cross themselves, and Outlier Rejection covers
contaminated data.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Outlier Rejection
:link: outlier_rejection
:link-type: doc

Reject interlopers with a stream-plus-background mixture model, and see why the orderers alone can't.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/outlier_rejection.ipynb)
:::

:::{grid-item-card} Stream Autoencoder
:link: stream_autoencoder
:link-type: doc

**Start here.** Order a simulated stellar stream, fit its mean track, and read the training loss.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/stream_autoencoder.ipynb)
:::

:::{grid-item-card} Stream Running Mean Path
:link: stream_runningmean
:link-type: doc

A faster, training-free mean track for the same stream, and when it is accurate enough.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/stream_runningmean.ipynb)
:::

:::{grid-item-card} Epitrochoid Autoencoder
:link: epitrochoid_autoencoder
:link-type: doc

Order a curve that crosses itself, using velocities to stay on the right strand.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_autoencoder.ipynb)
:::

:::{grid-item-card} Epitrochoid Running Mean Path
:link: epitrochoid_runningmean
:link-type: doc

Why a running mean cuts inside tight bends, and how the window size controls it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_runningmean.ipynb)
:::

:::{grid-item-card} Stream MST Backbone
:link: stream_mst
:link-type: doc

Order a stream with no progenitor or start index, and reject injected interlopers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/stream_mst.ipynb)
:::

:::{grid-item-card} Epitrochoid MST Backbone
:link: epitrochoid_mst
:link-type: doc

Why the MST needs velocities on a self-crossing curve, plus a Fourier-feature decoder.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_mst.ipynb)
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

outlier_rejection
stream_autoencoder
stream_runningmean
epitrochoid_autoencoder
epitrochoid_runningmean
stream_mst
epitrochoid_mst
```
