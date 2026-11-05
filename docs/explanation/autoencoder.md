# The autoencoder

An ordering is not yet a smooth track, and it may be incomplete. The local-flow walk
follows a single thread through the data, so it leaves out samples off to the side of
its path and everything beyond a gap larger than `max_dist`. On the simulated stream
in the [stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb), the walk
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

1. **Initialization.** The orderer assigns $\gamma$ over its `gamma_range`
   ($[-1, 1]$ for the default pipeline) to the samples it ordered.
2. **Encoder.** The encoder learns to predict $\gamma$ from phase-space coordinates,
   and a membership probability $p$ that separates the stream from a background.
3. **Decoder.** With the encoder frozen, the decoder is fitted to a running mean of
   the ordered member samples' positions, as a warm start for the track.
4. **Both.** Both networks train together on reconstructing positions plus aligning
   the track with the velocities. The alignment weight ramps linearly from
   `lambda_p[0]` to `lambda_p[1]` over this phase, which keeps the track's direction
   consistent with the samples' motion.

Each phase optimizes a different objective, so the training losses can be compared
only within a phase, and the returned model holds the final epoch's weights.

## A faster alternative

A running-mean decoder, {class}`~phasecurvefit.nn.RunningMeanDecoder`, can replace
the trained decoder. It averages the positions of member samples within a window in
$\gamma$, needs no training, and is fast enough to evaluate inside a likelihood at
every step of an MCMC. It cuts inside tight bends, by an amount set by the window size
([epitrochoid running-mean tutorial](../tutorials/epitrochoid_runningmean.ipynb)).

## Membership

The encoder's membership probability is trained as a classifier against random
points, and on its own it cannot reject interlopers that the orderer threaded into the
ordering. {doc}`outliers` explains why, and the mixture model that fixes it.

## References

Nibauer, J., Belokurov, V., Cranmer, M., Goodman, J., & Ho, S. (2022). "Charting
Galactic Accelerations with Stellar Streams and Machine Learning." ApJ 940, 22.
[arXiv:2205.11767](https://arxiv.org/abs/2205.11767).
