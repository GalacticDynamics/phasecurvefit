---
title:
  "phasecurvefit: A High-Performance JAX Framework for Phase-Space Walks with
  Pluggable Metrics and Strategies"
tags:
  - Python
  - JAX
  - astronomy
  - stellar streams
  - galactic dynamics
  - phase-space analysis
authors:
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    corresponding: true
    affiliation: "1, 2"
  - name: Jacob Nibauer
    orcid: 0000-0001-8042-5794
    affiliation: "3"
affiliations:
  - name:
      MIT Kavli Institute for Astrophysics and Space Research, Massachusetts
      Institute of Technology, Cambridge, MA 02139, USA
    index: 1
  - name:
      Department of Astronomy, Case Western Reserve University, Cleveland, OH
      44106, USA
    index: 2
  - name:
      Department of Astrophysical Sciences, Princeton University, 4 Ivy Ln,
      Princeton, NJ 08544, USA
    index: 3
date: 14 January 2026
bibliography: paper.bib
---

# Summary

Filamentary structures are ubiquitous in the physical sciences, ranging from
coherent streams of stars called stellar streams to elongated structures in
turbulent fluids, plasmas, and the interstellar medium. In the context of
stellar streams, a common preprocessing step is ordering observational or
simulation data to derive the mean path of the stream's trajectory through
phase-space.

`phasecurvefit` is an open-source Python package for constructing such orderings
and paths by _walking along the local phase-space flow_ using JAX [@Jax2018].
There are two core components: the first is `walk_local_flow` which builds an
approximate ordering and trajectory but which might miss some of the data; and
the second is a fast-to-train autoencoder that imputes the full ordering and
trajectory.

`walk_local_flow` is very modular and can be customized to use any:

1. **distance metric** that scores candidate next steps in phase space, and
2. **query strategy**, like brute-force or KD-trees, that proposes which
   candidates to consider.

This design makes the Nearest Neighbors with Momentum (NN+p) method from
[@Nibauer2022] one particular configuration (via
`phasecurvefit.metrics.AlignedMomentumDistanceMetric`), while enabling
alternative metrics and search strategies better matched to different data
scenarios and performance constraints.

In addition to the walk itself, `phasecurvefit` packages a neural-network
gap-filling component which assigns a continuous ordering parameter to data
skipped during the walk and reconstructs the spatial mean path of the structure.
This encoder develops upon the one in [@Nibauer2022]: speeding up different
components of training by between 1 and 3 orders of magnitude; adding an
intermediate decoder-only training that quarters the epochs necessary for
training the full autoencoder, halving the overall training time; and adding
stabilization of the loss function across training phases.

# Statement of Need

Ordering stream members is challenging, even in forward-model simulations where
all variables can be controlled. For most models, no property intrinsic to the
stream may be used to determine the path or path-order without prior knowledge
of the gravitional potential. Therefore, it is necessary to develop algorithms
which can infer the path-ordering of the stream. Moreover, it is necessary for
these methods to be performant to support the analysis of high performance
stream simulators, and auto-differentiable to support inference routines.

The original phase-flow walk implementation accompanying [@Nibauer2022]
demonstrated the efficacy of the core algorithms, but was not sufficiently
performant nor autodifferentiable, was inflexible regarding the distance metrics
and search strategies, and most importantly was not built as a reusable,
extensible library.

`phasecurvefit` addresses these needs, providing an implementation that has:

1. **A modular API** with user-selectable metrics and neighbor-query strategies;
2. **Hardware acceleration and composability** through JAX transformations and
   XLA compilation, interoperating with libraries like Equinox [@Kidger2021];
3. **Differentiability** of metrics and neural components for gradient-based
   tuning and downstream integration;
4. **Coordinate-agnostic inputs** through JAX PyTrees (e.g., `dict`s of
   components);
5. **Rigorous tests** with a test suite via `pytest` [@Pytest2004];
6. **Interoperability** with `unxt` [@Starkman2025] for unit support in JAX.

# Package Design

With `phasecurvefit`, fitting an affine parameter that orders a stream and
inferring the mean path requires 5 lines of code:

1. to construct an initial walk
2. to normalize the data
3. to define the autoencoder
4. to train the autoencoder
5. to get the final result

```python
import jax
import phasecurvefit as lfw

walkresult = lfw.walk_local_flow(pos, vel, ...)
normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
model = lfw.nn.PathAutoencoder.make(
    normalizer, gamma_range=walkresult, key=jax.random.key(0)
)
result, *_ = lfw.nn.train_autoencoder(model, walkresult, key=jax.random.key(1))
```

# Performance Characteristics

In practice, we observe high performance on datasets with tens of thousands of
points. The implementation also supports `jax.vmap` for parallel analysis of
multiple streams or parameter sweeps without code modification.

- `walk_local_flow` runs in under a second.
- Training the gap-filling path-ordering encoder takes under a second.
- Training the decoder to reconstruct a running mean takes under 4 seconds.
- Training the encoder and full path-inferring decoder takes a little under 16
  seconds.

The slowest step is the path-inferring decoder. Due to `phasecurvefit`'s modular
design this step is easily skipped and users can use the gap-filling encoder
with alternate or custom path-inferring functions. An included alternate
function is a path-ordered rolling mean, which consumes a small fraction of a
second after training the encoder.

# Research Applications

`phasecurvefit` enables easy data-driven reconstruction of stream paths in
simulations. Beyond stellar streams, the same walk abstraction applies to
ordering problems in other phase-space datasets where a coherent local flow is
expected.

# Acknowledgements

N.S. is supported by the Brinson Prize Fellowship at MIT. J.N. is supported by a
National Science Foundation Graduate Research Fellowship, Grant No. DGE-2039656.
This work made extensive use of the `JAX` [@Jax2018], and `unxt` [@Starkman2025]
software packages.

# References
