# What phasecurvefit does

Many datasets are samples along a curve in phase space whose order along the curve
is unknown. A stellar stream is the running example: you have each star's position
and velocity, but not where along the stream it sits. Before you can fit a model to
such a curve you need two things: an ordering coordinate for every sample, and a
smooth track through the samples.

phasecurvefit produces both. It takes positions and velocities as two dictionaries
with matching keys, one array per coordinate, in any number of dimensions:

- **An ordering.** {func}`~phasecurvefit.order` returns an
  {class}`~phasecurvefit.orderers.OrderingResult` holding the sample indices in order
  along the curve, and an ordering coordinate $\gamma$ that runs from one end of the
  curve to the other.
- **A smooth track.** An autoencoder learns $\gamma$ for every sample, including any
  the ordering left out, and a mean track $x(\gamma)$ that you can evaluate at any
  $\gamma$. {func}`~phasecurvefit.fit_track` runs the ordering and the fit in one call.
- **Membership.** A stream-plus-background mixture model gives each sample a
  probability of belonging to the curve, for data with interlopers.

## The default pipeline

With no other arguments, `pcf.order(pos, vel)` runs two stages. A minimum spanning
tree of the nearest-neighbour graph gives a backbone whose longest path runs from one
end of the curve to the other, so no start point is needed. A one-dimensional
self-organizing map then averages over many samples along that backbone, which
smooths the ordering against local noise. The velocities set the direction, so
$\gamma$ increases along the flow. {func}`~phasecurvefit.fit_track` adds the
autoencoder on top.

Other orderers can replace either stage. The local-flow walk follows the velocity
field from a start point, step by step, and is the better choice when a curve
crosses itself. {doc}`is-it-right` matches your data to an orderer.

## Built on JAX

The orderers, distance metrics and neural networks are written in JAX and are
[Equinox](https://docs.kidger.site/equinox/) modules. The local-flow walk and the
MST orderer work under `jit`, `vmap` and `grad`, and everything runs on CPU or GPU.
Inputs may carry physical units through [`unxt`](https://unxt.readthedocs.io).

## Read next

- {doc}`is-it-right`: which call suits your data.
- {doc}`quickstart`: a first ordering and track, step by step.
- {doc}`../explanation/index`: how each stage works.
