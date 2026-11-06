# Density fields

A density field is data given as an image: a density map of a stream from a
simulation, or a probability map from a segmentation. phasecurvefit fits a track
through it by treating each pixel as a sample weighted by its value. This page
explains why that needs a different orderer, two training settings, and care at the
ends of the track. The [density-field tutorial](../tutorials/density_fields.ipynb)
runs the whole fit.

## Why the weights skip membership

A weight of 2 counts like two coincident samples, so the SOM fits the weighted
density and the autoencoder weights its fit to the track. The autoencoder's
membership probability is left unweighted, because a weight sets how much a sample
counts and says nothing about whether it belongs to the stream. Weighting it would
push low-weight samples below the membership threshold and out of the fit, and in an
image those are the faint pixels at the edges and tails.

## Why order along the ridge

The default pipeline's MST orders samples along the longest path of a tree linking
near neighbours. A thin stream of stars has one long path through it. A wide, evenly
filled region has many paths of about the same length, and the longest one zigzags
across the stream and ends in a corner. On a uniform mask of radius 0.5 and a line
blurred with $\sigma = 0.25$, its backbone cuts inside the bends and ends 0.3 to 1.1
from the true ends.

{class}`~phasecurvefit.orderers.RidgeOrderer` uses the shape of the density instead.
Smoothing a uniform tube, or a line blurred by any symmetric kernel, puts the highest
density on the centre line, so a path that prefers dense pixels follows it. On the
same two images, the ridge then a SOM ends within 0.04 of the true ends. The ridge
backbone only needs to be roughly right, because the SOM chained after it refines the
middle.

How wide to smooth is estimated from the data. Starting from a deliberately wide
width, each round sets it to the RMS distance of the samples from the ridge it just
found, and it settles on the stream's own scatter: 0.104 for a blur of
$\sigma = 0.1$. It starts wide because a wide kernel gives a smooth ridge that the
iteration then tightens, while a narrow one could lock onto noise.

## Why the SOM sits outside the bends

A SOM converges to a principal curve of the weighted density: each point is the
weighted mean of the samples nearest it. On a bend of radius $R$, a uniform tube of
half-width $r$ holds more area outside the bend than inside, and the mean radius
across it is

$$
\frac{\int_{R-r}^{R+r} \rho^2 \, d\rho}{\int_{R-r}^{R+r} \rho \, d\rho}
= R + \frac{r^2}{3R} .
$$

For $r = 0.5$ and $R = 1.85$ that offset is 0.045, and the SOM's largest error on such
a mask is 0.045.

## Why a wide stream trains differently

In a thin stream every sample sits close to the track, so there is little room for
$\gamma$ to vary across it. A wide field leaves that room, and two things go wrong:

- In the joint training phase, the reconstruction loss can lower itself by letting
  $\gamma$ encode position across the stream as well as along it, and the track bends
  into loops on the curves. Freezing the encoder in that phase keeps $\gamma$ an
  along-track coordinate.
- The samples' spread across the stream, with variance about $r^2/3$, is much larger
  than the error the decoder has left to fix, so small batches keep the optimizer
  jittering around the track. On a uniform tube of half-width 0.5, batches of 1000
  brought the track's largest error from 0.39 to 0.06.

{doc}`../how-to/autoencoder` shows the two settings.

## Where the track ends

The faint samples past each end of a stream get the most extreme $\gamma$, so the
whole of `gamma_range` runs past the ends. `track_range` stops each end where the
weighted number of samples per unit length along the track falls below half its
value just inside. It counts per unit length because near an end the encoder squeezes
many samples into little $\gamma$, which would make a faint tail look dense, and it
compares with the value just inside so that a stream that narrows or fades keeps its
ends.

For a curve blurred by a symmetric kernel, the line density at the true end is half
its value just inside, so this finds the end. A mask made by thickening a curve has a
cap past each end that holds real weight, and the half-density point falls partway
into it. In the tutorial the narrow end of the track is 0.05 from the true end, and
the wide end, where the cap reaches 0.7 past the curve, is 0.33 past it. The image
alone can't tell a short curve with a cap from a longer, blurrier one.
