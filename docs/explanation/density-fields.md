# Density fields

A density field is data given as an image rather than a list of samples: a density
map of a stream from a simulation, or a probability map from a segmentation.
phasecurvefit fits a track through one by treating each pixel as a sample weighted by
its value. This page explains how each stage handles that, and why. The
[density-field tutorial](../tutorials/density_fields.ipynb) runs the whole fit.

## Pixels as weighted samples

Each pixel becomes one sample at its centre, with the pixel's value as its weight
({doc}`../how-to/data`). A weight of 2 counts like two coincident samples of weight 1.
The SOM fits the weighted density, and the autoencoder weights its ordering targets
and its fit to the track.

The autoencoder does not weight its membership probability. A weight sets how much a
sample counts in the track, and says nothing about whether it belongs to the stream. With
the membership term weighted, low-weight samples fell below the membership threshold
and dropped out of the fit: in a test with two parallel lines weighted 3 to 1, the
fainter line was lost. In a density field the low-weight samples are the faint pixels
at the edges and tails.

An image has no velocities. They are set to zero, the orderers used for images don't
read them, and the direction of the ordering, which end $\gamma$ starts from, is
arbitrary.

## Why the MST struggles with wide streams

The MST orders samples along the longest path of a tree linking near neighbours. A
thin stream of stars has one long path through it. A wide, evenly filled region has
many paths of about the same length, and the longest one zigzags across the stream
and ends in a corner of it. On the density-field tutorial's uniform mask and blurred
line, the default MST-then-SOM backbone cuts inside the bends, and its ends are 0.3 to
1.1 from the true ends.

## The ridge of the density

{class}`~phasecurvefit.orderers.RidgeOrderer` builds its backbone along the ridge of
the weighted density instead:

1. It smooths the weighted samples with a Gaussian kernel of width `bandwidth`.
   Smoothing a uniform tube, or a line already blurred by any symmetric kernel, puts
   the highest density on the centre line.
2. It takes the two ends of the longest path, by length, through the dense core: the
   samples with at least `core` times the peak density.
3. Between those ends it takes the shortest path on the nearest-neighbour graph, where
   each edge costs its length over the square of the mean density at its two ends. A
   step off the ridge costs more, so the path follows it.
4. It trims the path where the density along it falls below `trim` times its median,
   and smooths it over half a bandwidth.

The ends come from the core rather than from the ridge cost because reaching a faint
pixel on the side of the stream costs about one over the density squared, which would
make that pixel the farthest point. Past 10 000 samples, the backbone is found on a
random subsample of 10 000 and every sample is then projected onto it: pixels much
finer than the bandwidth add nothing to the density.

The ridge backbone is a starting point for a SOM. It finds the ends to within about
one bandwidth, and a SOM chained after it refines the middle.

## Choosing the bandwidth

With `bandwidth=None`, the default, the bandwidth is estimated from the data. It
starts at a tenth of the data's extent, finds the backbone, sets the bandwidth to the
weighted RMS distance of the samples from the backbone, and repeats until it changes
by less than 2%.

The estimate starts wide because a wide kernel gives a smooth ridge that cuts the
bends, so the scatter about it overestimates the width, and each round shrinks it
towards the scatter about the ridge it defines. A narrow first guess could instead
lock onto noise or pixel structure.

The estimate settles on the RMS scatter across the stream:

| Input | Expected | Estimated |
|---|---|---|
| line blurred with $\sigma = 0.1$ | 0.1 | 0.104 |
| uniform band of half-width 0.15 | $0.15 / \sqrt{3} = 0.087$ | 0.095 |
| samples scattered by 0.05 about a curve | 0.05 | 0.052 |

On the tutorial's soft mask, whose half-width grows from 0.3 to 0.7, it goes from 1.10
to 0.42, 0.254 and then 0.247, where it stays.

## What the SOM converges to

A SOM trained on weighted samples converges to a principal curve of the weighted
density: each point of the curve is the weighted mean of the samples nearest to it.
That is close to the centre line, with two known offsets.

On a bend of radius $R$, a uniform tube of half-width $r$ holds more area on the
outside of the bend than the inside. The mean radius across the tube is

$$
\frac{\int_{R-r}^{R+r} \rho^2 \, d\rho}{\int_{R-r}^{R+r} \rho \, d\rho}
= R + \frac{r^2}{3R} ,
$$

so the curve sits $r^2/(3R)$ outside the centre line. For the tutorial's uniform mask,
$r = 0.5$ and the tightest bend has $R = 1.85$, so the offset is 0.045. The SOM's
largest error there is 0.045.

At each end, the last prototype is a mean over the samples nearest it, which all lie
on the inner side, so the SOM can stop short of the true end. On the narrow end of the
tutorial's soft mask it stops 0.39 short. The autoencoder's track is evaluated over a
range found from the samples themselves, and it ends 0.05 from that end.

## Training the autoencoder on a wide stream

Two training settings matter more for a wide field than for a thin stream of stars:

- `freeze_encoder_final_training=True`. In the joint phase the reconstruction loss
  can lower itself by letting $\gamma$ encode position across the stream as well as
  along it, and the track then bends into loops on the curves. A thin stream gives
  that little room. Freezing the encoder in that phase keeps $\gamma$ an along-track
  coordinate.
- A large `batch_size`, such as 1000. Pixels spread across the stream with a variance
  of about $r^2/3$, which is much larger than the error the decoder has left to fix.
  With batches of 100, the optimizer keeps jittering around the mean position at each
  $\gamma$ instead of settling on it. On a uniform tube of half-width 0.5, batches of
  1000 brought the track's largest error from 0.39 to 0.06.

## Where the track ends

Decoding the whole of `gamma_range` carries the track past the ends of the stream,
because the faint samples beyond each end get the most extreme $\gamma$. Draw the
track over `result.track_range` instead.

`track_range` is found in physical length. The track is decoded, every member sample
is projected onto it, extending past each end along the tangent, and each end is cut
where the weighted number of samples per unit length falls below half its value just
inside. Counting per unit length rather than per unit $\gamma$ matters because near an
end the encoder squeezes many samples into little $\gamma$, which makes a faint tail
look dense. Comparing with the value just inside, rather than with the median of the
whole stream, keeps the ends of a stream that narrows or fades.

For a curve blurred by any symmetric kernel, the line density at the true end is half
its value just inside, so this rule finds the end. A mask made by thickening a curve
is different. Its cap past each end holds real weight, so the half-density point
falls partway into the cap. On the tutorial's soft mask, the narrow end of the track
is 0.05 from the true end and the wide end, where the cap reaches 0.7 past the curve,
is 0.33 past it. The image alone can't tell a short curve with a cap from a longer,
blurrier one.
