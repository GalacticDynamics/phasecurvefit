# Density fields

A density field is data given as an image of a wide, diffuse curve, such as a
density map of a stellar stream from a simulation or a probability map from a
segmentation. phasecurvefit fits a track through it by treating each pixel as a
sample weighted by its value. This page
explains how the weights enter membership, and why a density field needs a different
orderer, two training settings, and care at the ends of the track. The [density-field tutorial](../tutorials/density_fields.md)
runs the whole fit.

## How the weights enter membership

A weight $\omega_n$ counts like $\omega_n$ coincident samples, so the SOM fits the
weighted density and the autoencoder weights its fit to the track. Taking that rule
literally also settles how the weights enter membership.

### In the mixture model

The mixture model of {doc}`outliers` gives each sample the likelihood

$$
\mathcal{L}_n = \pi_n \mathcal{N}_n + (1 - \pi_n)\,\rho_{\mathrm{bg}} ,
$$

where $\mathcal{N}_n$ is the Gaussian about the track evaluated at the sample. The
$\omega_n$ copies of sample $n$ sit at the same position, so they share
$\mathcal{L}_n$ and together add $\omega_n \log \mathcal{L}_n$ to the
log-likelihood. Training minimizes

$$
-\frac{1}{N} \sum_n \omega_n \log \mathcal{L}_n ,
$$

with the weights normalized to mean 1. Scaling every weight by the same factor scales
the loss and leaves its minimum where it was.

The posterior membership of a sample is

$$
\hat{q}_n = \frac{\pi_n \mathcal{N}_n}{\mathcal{L}_n} .
$$

The weight doesn't appear in it. The posterior depends on where the sample is and on
the model, and every copy of a sample has the same posterior however many copies
there are. The weight sets how hard the sample pulls on the model. For the membership,
the gradient is

$$
\frac{\partial}{\partial \pi_n} \, \omega_n \log \mathcal{L}_n
= \omega_n \, \frac{\mathcal{N}_n - \rho_{\mathrm{bg}}}{\mathcal{L}_n} .
$$

Its sign depends on whether the curve or the background explains the sample better,
and the weight only scales it. A faint pixel inside the curve pulls weakly towards
membership, and a faint pixel far outside it pulls weakly towards the background. In
the velocity-alignment term each sample counts by $\omega_n \hat{q}_n$, the expected
mass of members at that sample.

The SOM's outlier mixture follows the same rule: a weight multiplies a sample's terms
in the prototype update and in the fits of the member fraction and the curve's width,
and doesn't appear in the sample's membership, which depends on where the sample is
and on the model ({doc}`som`).

For an image the sum has a continuum limit. With each pixel weighted by the image
$I(q)$, the loss tends, as the pixels shrink, to

$$
-\frac{\int I(q) \log \mathcal{L}(q) \, dq}{\int I(q) \, dq} ,
$$

the cross-entropy between the normalized image and the model. Minimizing it is
maximum likelihood for an unlimited number of samples drawn from the image. Pixels of
value zero add nothing to it, so leaving them out of the samples changes nothing.

### Why the weight isn't a prior on membership

The weight could instead go into the prior, for example as $\omega_n \pi_n$ for
weights between 0 and 1, so that faint pixels are less likely to be members. That
reads the weight as evidence about which component a sample came from. In a density
map the weight says how much is there. Putting it in the prior sends the faint pixels
in the wings and tails of the curve to the background because they are faint, so
the model fits a narrower, shorter curve than the image shows.

The prior reading looks more natural for a segmentation map, whose values $p_n$ are
probabilities that each pixel belongs to the curve. Those values are already the
weights, though, so putting them in the prior as well would count them twice. Reading
$p_n$ as the probability that pixel $n$ is part of the data gives the expected
log-likelihood $\sum_n p_n \log \mathcal{L}_n$, the same loss as above.

### In the encoder

The encoder's first training phase learns its membership as a classifier, pushing
the ordered samples towards 1 and uniform random points towards 0
({doc}`outliers`). Its membership term is left unweighted. Weighting only the ordered
samples would let a faint pixel count less against the same random points around
it, and its membership would drop below the threshold. That comes from the random
points, which have no weights of their own; weighting consistently would mean
rescaling them as well, and that isn't implemented. The mixture model has no random
points, since the background enters as the density $\rho_{\mathrm{bg}}$, so its
weights need no such care.

### Uncertainties

The weighted sum gives the best-fitting track, which is all the training uses. Its
curvature gives the uncertainty of that track only when the weights are counts, such
as numbers of stars per pixel. For image intensities or segmentation probabilities,
an uncertainty would need a noise model for the pixel values.

## Why order along the ridge

The default pipeline's MST orders samples along the longest path of a tree linking
near neighbours. A thin curve of samples has one long path through it. A wide, evenly
filled region has many paths of about the same length, and the longest one zigzags
across the curve and ends in a corner. On a uniform mask of radius 0.5 and a line
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
found, and it settles on the curve's own scatter: 0.104 for a blur of
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

Where the curve is wide and noisy, the SOM's backbone can wiggle. With 30 prototypes
on the tutorial's image, the prototypes near the wide end sit closer together than the
curve is wide, so each follows the clumps of noise near it. The autoencoder fits one
smooth track through all the samples, so the wiggle doesn't reach the final track.

## Why a wide curve trains differently

Along a thin curve every sample sits close to the track, so there is little room for
$\gamma$ to vary across it. A wide field leaves that room, and two things go wrong:

- In the joint training phase, the reconstruction loss can lower itself by letting
  $\gamma$ encode position across the curve as well as along it, and the track bends
  into loops on the curves. Freezing the encoder in that phase keeps $\gamma$ an
  along-track coordinate.
- The samples' spread across the curve, with variance about $r^2/3$, is much larger
  than the error the decoder has left to fix, so small batches keep the optimizer
  jittering around the track. On a uniform tube of half-width 0.5, batches of 1000
  brought the track's largest error away from its ends from 0.11 to 0.04.

{doc}`../how-to/autoencoder` shows the two settings.

## Where the track ends

The faint samples past each end of a curve get the most extreme $\gamma$, so the
whole of `gamma_range` runs past the ends. `track_range` stops each end where the
weighted number of samples per unit length along the track falls below half its
value just inside. It counts per unit length because near an end the encoder squeezes
many samples into little $\gamma$, which would make a faint tail look dense, and it
compares with the value just inside so that a curve that narrows or fades keeps its
ends.

For a curve blurred by a symmetric kernel, the line density at the true end is half
its value just inside, so this finds the end. A mask made by thickening a curve has a
cap past each end that holds real weight, and the half-density point falls partway
into it. On the tutorial's image the wide end stops 0.33 past the true end, where the
cap reaches 0.7 past the curve. The image alone can't tell a short curve with a cap
from a longer, blurrier one, so the user has to say which it is.

`TrainingConfig(track_ends="mask")` says the image is a mask. A curve thickened to
half-width $w$ has a cap that reaches $w$ past its end, so each end is placed $w$ in
from the farthest sample along the track. $w$ is the widest any sample reaches across
the track near that end. This rule uses the samples' positions and ignores their
weights, so it suits segmentation maps and other images with a definite edge. On the
tutorial's image both ends land within 0.1 of the true ends. On a point cloud there is
no edge, and the half-density rule is the one to use.
