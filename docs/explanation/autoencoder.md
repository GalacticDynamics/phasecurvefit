# The autoencoder

An ordering is not yet a smooth track, and it may be incomplete. The local-flow walk
follows a single thread through the data, so it leaves out samples off to the side of
its path and everything beyond a gap larger than `max_dist`. On the simulated stream
in the [stream autoencoder tutorial](../tutorials/stream_autoencoder.md), the walk
orders 248 of 8000 stars. An outlier-rejecting chain also leaves samples out on
purpose. And any orderer gives only an order, not a curve you can evaluate. The
autoencoder, from Appendix A.2 of Nibauer et al. (2022), fills both gaps.
{doc}`../how-to/autoencoder` shows how to train it.

## Two networks

- The **encoder** maps a sample's phase-space coordinates to an ordering coordinate
  and a membership probability, $(x, v) \rightarrow (\gamma, p)$. It learns from the
  ordered samples and generalizes to the ones the orderer skipped.
- The **decoder** maps $\gamma$ back to a position, $\gamma \rightarrow x$. It is the
  smooth mean track $x(\gamma)$, which you can evaluate at any $\gamma$.

## Training

1. The orderer assigns $\gamma$ over its `gamma_range` ($[-1, 1]$ for the default
   pipeline) to the samples it ordered.
2. In the first training phase, the encoder learns to predict $\gamma$ from
   phase-space coordinates, and a membership probability $p$ that separates the
   stream from a background.
3. In the second, with the encoder frozen, the decoder is fitted to a running mean of
   the ordered member samples' positions, as a warm start for the track.
4. In the third, both networks train together on reconstructing positions plus
   aligning the track with the velocities. The alignment weight ramps linearly from
   `lambda_p[0]` to `lambda_p[1]` over this phase, which keeps the track's direction
   consistent with the samples' motion.

Each phase optimizes a different objective, so the training losses can be compared
only within a phase, and the returned model holds the final epoch's weights.

## A faster alternative

A running-mean decoder, {class}`~phasecurvefit.nn.RunningMeanDecoder`, can replace
the trained decoder. It averages the positions of member samples within a window in
$\gamma$, needs no training, and is fast enough to evaluate inside a likelihood at
every step of an MCMC. It cuts inside tight bends, by an amount set by the window size
([epitrochoid running-mean tutorial](../tutorials/epitrochoid_runningmean.md)).

## Membership

The encoder's membership probability is trained as a classifier against random
points, and on its own it cannot reject interlopers that the orderer threaded into the
ordering. {doc}`outliers` explains why, and the mixture model that fixes it.

## Interlopers at the tips

The default pipeline orders every sample it is given, interlopers included. An
interloper beyond the end of a stream lands at the end of the ordering, so the encoder
learns to map it to the most extreme $\gamma$, and the decoder bends the end of the
track out to reach it. On a gapped stream of width 0.2 with 4% interlopers (the
reproducer in issue #224), the two interlopers ordered last got $\gamma$ of 0.92 and
0.94, while no member went past 0.65, and the track ran 5.3 off the true curve.

{func}`~phasecurvefit.fit_track` guards against this in two places.

- It orders with the SOM's outlier clipping on, `outlier_clip_sigma=3`. A sample whose
  distance to its nearest prototype is an outlier among the rest is left out of the
  ordering, so the encoder learns it as a non-member and the track has no reason to
  reach for it. On the gapped stream this left out 9 of the 16 interlopers and none
  of the 404 members.
- `track_range` does not count scattered samples as the end of the stream. Each end
  needs the few bins of line density just inside it to be non-empty. A track that
  bends out to a handful of interlopers passes them one at a time, with empty bins
  in between, and that stretch is cut. This applies to every trained result, not just
  to `fit_track`.

The table gives the largest distance from the track over `track_range` to the true
curve, for two training seeds, with default training settings. The last three rows are
one sine-shaped stream of 420 samples with 4% interlopers, at three widths.

| stream | before | `track_range` change only | both changes |
|---|---|---|---|
| near-closed loop, no interlopers | 0.35, 0.40 | 0.35, 0.40 | 0.35, 0.40 |
| gapped, width 0.2 | 5.27, 5.27 | 0.60, 0.53 | 0.14, 0.21 |
| width 0.3 | 5.54, 5.96 | 0.48, 0.58 | 0.18, 0.06 |
| width 1.0 | 5.05, 2.95 | 1.60, 1.13 | 1.18, 2.51 |

On the loop every sample stayed in the ordering. On the widest stream the track still
strays by one to two and a half widths, with or without clipping.

The mixture model of {doc}`outliers` also removes the hook, but it is not on by
default. On the clean loop it rejected between a quarter and three fifths of the
genuine members on three of six seeds (issue #257).

## References

Nibauer, J., Belokurov, V., Cranmer, M., Goodman, J., & Ho, S. (2022). "Charting
Galactic Accelerations with Stellar Streams and Machine Learning." ApJ 940, 22.
[arXiv:2205.11767](https://arxiv.org/abs/2205.11767).
