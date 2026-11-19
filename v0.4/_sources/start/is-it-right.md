# Is phasecurvefit right for my data?

phasecurvefit fits when three things hold. You have many samples. Each sample has a
position and a velocity, given as dictionaries of arrays with the same keys. And the
samples lie along one curve whose order you don't know. The tables below match what
you have and what you want to the call to use and the page to read.

## What you have

| You have | Use | Read more |
|---|---|---|
| positions and velocities along a curve, and no known start point | `pcf.order(pos, vel)`, the default pipeline | [](quickstart.md), [](../tutorials/stream_mst.md) |
| a known start point, such as a stream's progenitor | {class}`~phasecurvefit.orderers.LocalFlowOrderer` with `start_idx`, and `direction="both"` if the curve flows away from it both ways | [](../how-to/local-flow-walk.md), [](../tutorials/stream_autoencoder.md) |
| a curve that crosses itself | once: the default pipeline, whose MST weighs edges by velocity and crosses straight over. Many times: {class}`~phasecurvefit.orderers.LocalFlowOrderer`, or {class}`~phasecurvefit.orderers.MSTOrderer` with `sever_cos_threshold`. | [](../how-to/choose-orderer.md), [](../tutorials/epitrochoid_autoencoder.md), [](../tutorials/epitrochoid_mst.md) |
| samples with some missing velocities (NaN) | the default pipeline, which joins a sample with a missing velocity by position alone and skips it when orienting the ordering | [](../how-to/data.md) |
| `unxt` Quantities with physical units | any orderer, with `metadata=pcf.StateMetadata(usys=...)` | [](../how-to/units.md) |
| a dataset where background points mix with the curve | an MST with `edge_clip_sigma`, then the mixture membership model | [](../how-to/outliers.md), [](../tutorials/outlier_rejection.md) |
| an ordering to run inside `jit`, `vmap` or `grad` | any orderer, the default pipeline included, except the ones that run on the host: {class}`~phasecurvefit.orderers.RidgeOrderer` and `MSTOrderer(neighbors=SciPy())` | [](../how-to/jax.md) |

## What you want

| You want | phasecurvefit gives | Read more |
|---|---|---|
| the order of the samples along the curve | `result.ordering`, or `pcf.order_w(result)` for the reordered arrays | [](quickstart.md), [](../reference/result.md) |
| an ordering coordinate $\gamma$ for every sample | `fitted.gamma` from {func}`~phasecurvefit.fit_track` or {func}`~phasecurvefit.nn.train_autoencoder` | [](../how-to/autoencoder.md) |
| a smooth mean track $x(\gamma)$ | calling the fitted result, `fitted(gamma)` | [](../how-to/autoencoder.md), [](../how-to/interpolate.md) |
| a track fast enough for every step of an MCMC | {class}`~phasecurvefit.nn.RunningMeanDecoder`, which needs no training | [](../tutorials/stream_runningmean.md) |
| a probability that each sample belongs to the curve | `fitted.membership_prob`, or {func}`~phasecurvefit.nn.posterior_membership` with the mixture model | [](../how-to/outliers.md) |
| positions at given values of $\gamma$ from an ordering alone | calling the ordering result, `result(gamma)` | [](../how-to/interpolate.md) |

## What phasecurvefit does not do

- Each call orders one curve. The default pipeline joins disconnected pieces into
  one ordering, so separate distinct streams in a catalogue before you order them.
- It does not model the dynamics. The track is a smooth curve through the data in
  phase space, and phasecurvefit fits no orbit, potential or progenitor.
- The local-flow walk does not choose its own start point. Chaining an
  {class}`~phasecurvefit.orderers.MSTOrderer` in front of it supplies an end of the
  curve ([](../how-to/chain-orderers.md)).
