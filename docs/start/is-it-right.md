# Is phasecurvefit right for my data?

phasecurvefit fits when three things hold. You have many samples. Each sample has a
position and a velocity, given as dictionaries of arrays with the same keys. And the
samples lie along one curve whose order you don't know. The tables below match what
you have and what you want to the call to use and the page to read.

## What you have

| You have | Use | Read more |
|---|---|---|
| positions and velocities along a curve, and no known start point | `pcf.order(pos, vel)`, the default pipeline | {doc}`quickstart`, {doc}`../tutorials/stream_mst` |
| a known start point, such as a stream's progenitor | {class}`~phasecurvefit.orderers.LocalFlowOrderer` with `start_idx`, and `direction="both"` if the curve flows away from it both ways | {doc}`../how-to/local-flow-walk`, {doc}`../tutorials/stream_autoencoder` |
| a curve that crosses itself | once: the default pipeline, whose MST weighs edges by velocity and crosses straight over. Many times: {class}`~phasecurvefit.orderers.LocalFlowOrderer`, or {class}`~phasecurvefit.orderers.MSTOrderer` with `sever_cos_threshold`. | {doc}`../how-to/choose-orderer`, {doc}`../tutorials/epitrochoid_autoencoder`, {doc}`../tutorials/epitrochoid_mst` |
| samples with some missing velocities (NaN) | the default pipeline, which joins a sample with a missing velocity by position alone and skips it when orienting the ordering | {doc}`../how-to/data` |
| `unxt` Quantities with physical units | any orderer, with `metadata=pcf.StateMetadata(usys=...)` | {doc}`../how-to/units` |
| a dataset where background points mix with the curve | an MST with `edge_clip_sigma`, then the mixture membership model | {doc}`../how-to/outliers`, {doc}`../tutorials/outlier_rejection` |
| an ordering to run inside `jit`, `vmap` or `grad` | any orderer, the default pipeline included, except the ones that run on the host: {class}`~phasecurvefit.orderers.RidgeOrderer` and `MSTOrderer(neighbors=SciPy())` | {doc}`../how-to/jax` |

## What you want

| You want | phasecurvefit gives | Read more |
|---|---|---|
| the order of the samples along the curve | `result.ordering`, or `pcf.order_w(result)` for the reordered arrays | {doc}`quickstart`, {doc}`../reference/result` |
| an ordering coordinate $\gamma$ for every sample | `fitted.gamma` from {func}`~phasecurvefit.fit_track` or {func}`~phasecurvefit.nn.train_autoencoder` | {doc}`../how-to/autoencoder` |
| a smooth mean track $x(\gamma)$ | calling the fitted result, `fitted(gamma)` | {doc}`../how-to/autoencoder`, {doc}`../how-to/interpolate` |
| a track fast enough for every step of an MCMC | {class}`~phasecurvefit.nn.RunningMeanDecoder`, which needs no training | {doc}`../tutorials/stream_runningmean` |
| a probability that each sample belongs to the curve | `fitted.membership_prob`, or {func}`~phasecurvefit.nn.posterior_membership` with the mixture model | {doc}`../how-to/outliers` |
| positions at given values of $\gamma$ from an ordering alone | calling the ordering result, `result(gamma)` | {doc}`../how-to/interpolate` |

## What phasecurvefit does not do

- Each call orders one curve. The default pipeline joins disconnected pieces into
  one ordering, so separate distinct streams in a catalogue before you order them.
- It does not model the dynamics. The track is a smooth curve through the data in
  phase space, and phasecurvefit fits no orbit, potential or progenitor.
- The local-flow walk does not choose its own start point. Chaining an
  {class}`~phasecurvefit.orderers.MSTOrderer` in front of it supplies an end of the
  curve ({doc}`../how-to/chain-orderers`).
