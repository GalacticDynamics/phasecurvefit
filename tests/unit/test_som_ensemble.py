"""The core must stay vmap-able, so a future SOM ensemble needs no changes.

If one fails, the core has acquired a host callback, a dynamic shape, or
Python branching on a traced value -- fix the core, not the test.
"""

import jax
import jax.numpy as jnp

import phasecurvefit as pcf
from phasecurvefit import som

N_ENSEMBLE = 4
N_PROTOTYPES = 10


def _ensemble_inits(pos, vel, key):
    """Genuinely distinct initializations, stacked on a leading axis.

    Batch Kohonen is strongly contractive: each epoch replaces every prototype
    with a neighbourhood-weighted mean of the *data*, so a previous prototype
    matters only through which datum it currently wins. Once two members' BMU
    assignments coincide every later epoch is byte-identical between them.

    So each member gets its own random *ordering* into ``init_prototypes`` --
    a different discrete lattice. Do not "tidy" this back down to a small
    jitter around one shared init, or the ensemble degenerates to one member
    broadcast over the batch axis.
    """
    n = next(iter(pos.values())).shape[0]
    keys = jax.random.split(key, N_ENSEMBLE)

    def one(k):
        ordering = jax.random.permutation(k, n)
        return som.init_prototypes(
            pos, vel, n_prototypes=N_PROTOTYPES, ordering=ordering
        )

    return jax.vmap(one)(keys)


def test_fit_vmaps_over_an_ensemble(helix):
    pos, vel, _ = helix()
    metric = pcf.metrics.SpatialDistanceMetric()
    eq, ep = _ensemble_inits(pos, vel, jax.random.key(0))

    fq, _ = jax.vmap(lambda a, b: som.fit(a, b, pos, vel, metric=metric, n_epochs=10))(
        eq, ep
    )

    # Indexed by component name, not `jt.leaves(...)[0]`: leaf order is an
    # implementation detail of the pytree, so a positional index silently
    # follows any change in structure to some other -- possibly constant -- leaf.
    assert fq["x"].shape == (N_ENSEMBLE, N_PROTOTYPES)
    # Members stay distinct ...
    assert not jnp.allclose(fq["x"][0], fq["x"][1])
    # ... and training actually moved them. Without this the assertion above is
    # satisfied by the distinct *initializations* alone, so the test passes
    # even if `fit` returns its input untouched.
    assert not jnp.allclose(fq["x"], eq["x"])


def test_chord_vmaps_to_a_posterior_of_orderings(helix):
    pos, vel, _ = helix()
    metric = pcf.metrics.SpatialDistanceMetric()
    eq, ep = _ensemble_inits(pos, vel, jax.random.key(0))

    def one(a, b):
        fq, fp = som.fit(a, b, pos, vel, metric=metric, n_epochs=10)
        bq, bp = som.densify(fq, fp, factor=5)
        return som.chord(bq, bp, pos, vel, metric=metric)

    def untrained(a, b):
        bq, bp = som.densify(a, b, factor=5)
        return som.chord(bq, bp, pos, vel, metric=metric)

    chords = jax.vmap(one)(eq, ep)
    assert chords.shape == (N_ENSEMBLE, 200)
    # Members stay distinct ...
    assert not jnp.allclose(chords[0], chords[1])
    # ... and the projection reflects training, not just the initial lattice:
    # member distinctness alone survives a `fit` that does nothing.
    assert not jnp.allclose(chords, jax.vmap(untrained)(eq, ep))

    orderings = jnp.argsort(chords, axis=1)
    assert orderings.shape == (N_ENSEMBLE, 200)


def test_whole_pipeline_jits_end_to_end(helix):
    pos, vel, _ = helix(n=100)
    metric = pcf.metrics.SpatialDistanceMetric()
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=N_PROTOTYPES)

    @jax.jit
    def pipeline(a, b):
        fq, fp = som.fit(a, b, pos, vel, metric=metric, n_epochs=5)
        bq, bp = som.densify(fq, fp, factor=5)
        return som.chord(bq, bp, pos, vel, metric=metric)

    assert pipeline(pq, pp).shape == (100,)
