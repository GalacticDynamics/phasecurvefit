"""The core must stay vmap-able, so a future SOM ensemble needs no changes.

If one fails, the core has acquired a host callback, a dynamic shape, or
Python branching on a traced value -- fix the core, not the test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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

    result = jax.vmap(lambda a, b: som.fit(a, b, pos, vel, metric=metric, n_epochs=10))(
        eq, ep
    )
    fq = result.prototype_positions

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
        result = som.fit(a, b, pos, vel, metric=metric, n_epochs=10)
        bq, bp = som.densify(
            result.prototype_positions, result.prototype_velocities, factor=5
        )
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
        result = som.fit(a, b, pos, vel, metric=metric, n_epochs=5)
        bq, bp = som.densify(
            result.prototype_positions, result.prototype_velocities, factor=5
        )
        return som.chord(bq, bp, pos, vel, metric=metric)

    assert pipeline(pq, pp).shape == (100,)


class TestBootstrapWeights:
    """``bootstrap_weights`` draws one resample, expressed as counts."""

    def test_counts_sum_to_n(self):
        """N draws with replacement, so the counts total N."""
        w = som.bootstrap_weights(jax.random.key(0), 300)
        assert w.shape == (300,)
        assert float(w.sum()) == pytest.approx(300.0)

    def test_leaves_out_about_one_in_e(self):
        """~1/e of the data missing per member is where the spread comes from.

        A sampler that quietly returned all-ones would satisfy the sum check
        above and produce a dead ensemble, so pin the omissions too.
        """
        w = som.bootstrap_weights(jax.random.key(0), 2000)
        left_out = float((w == 0).mean())
        assert left_out == pytest.approx(1 / np.e, abs=0.05)

    def test_vmaps_over_keys(self):
        """An ensemble maps over keys, so the draw must vmap and differ per key."""
        keys = jax.random.split(jax.random.key(0), 5)
        ws = jax.vmap(lambda k: som.bootstrap_weights(k, 120))(keys)
        assert ws.shape == (5, 120)
        assert not jnp.array_equal(ws[0], ws[1])


class TestEnsembleDiversityComesFromTheData:
    """The point of the weights: a posterior over *data*, not over inits.

    An ensemble whose only diversity is the starting lattice measures
    initialization sensitivity. These tests pin the contrast directly --
    the same shared init is degenerate without weights and alive with them --
    so a future change that re-merges the ensemble fails here rather than
    quietly returning a zero-width posterior.
    """

    @staticmethod
    def _setup(n=300, k=20):
        t = jnp.linspace(0.0, 2.0, n)
        pos = {"x": jnp.cos(t) * 3, "y": jnp.sin(t) * 3}
        vel = {"x": -jnp.sin(t), "y": jnp.cos(t)}
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=k)
        return pos, vel, pq, pp, pcf.metrics.SpatialDistanceMetric()

    def test_jittering_a_shared_init_collapses(self):
        """The degenerate construction, kept executable as the baseline.

        This is not a wish -- it is what batch Kohonen does. A prototype is
        replaced outright by a mean of the data, so it survives only through
        which datum it wins; once two members agree on their assignments they
        are merged for every later epoch. Measured here as *exactly* zero
        separation, not merely small.
        """
        pos, vel, pq, pp, metric = self._setup()

        def jittered(key):
            noisy = {
                k: v + jax.random.normal(key, v.shape) * 0.05 for k, v in pq.items()
            }
            return som.fit(
                noisy, pp, pos, vel, metric=metric, n_epochs=10
            ).prototype_positions

        fitted = jax.vmap(jittered)(jax.random.split(jax.random.key(0), 8))
        spread = jnp.abs(fitted["x"][:, None, :] - fitted["x"][None, :, :]).max()
        assert float(spread) == 0.0

    def test_bootstrap_weights_keep_members_apart(self):
        """Same shared init, same everything -- only the weights differ."""
        pos, vel, pq, pp, metric = self._setup()
        n = next(iter(pos.values())).shape[0]

        weights = jax.vmap(lambda k: som.bootstrap_weights(k, n))(
            jax.random.split(jax.random.key(0), 8)
        )
        fitted = jax.vmap(
            lambda w: som.fit(pq, pp, pos, vel, metric=metric, n_epochs=10, weights=w)
        )(weights).prototype_positions

        off_diagonal = jnp.abs(fitted["x"][:, None, :] - fitted["x"][None, :, :]).max(
            -1
        )[~jnp.eye(8, dtype=bool)]
        # Orders of magnitude above float32 noise, and above the exactly-zero
        # collapse the jittered construction produces.
        assert float(off_diagonal.min()) > 1e-3

    def test_the_chord_posterior_has_width(self):
        """What an ensemble is actually for: spread on the ordering itself."""
        pos, vel, pq, pp, metric = self._setup()
        n = next(iter(pos.values())).shape[0]

        def member(w):
            res = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=10, weights=w)
            bq, bp = som.densify(
                res.prototype_positions, res.prototype_velocities, factor=5
            )
            return som.chord(bq, bp, pos, vel, metric=metric)

        weights = jax.vmap(lambda k: som.bootstrap_weights(k, n))(
            jax.random.split(jax.random.key(0), 8)
        )
        chords = jax.vmap(member)(weights)

        assert chords.shape == (8, n)
        assert float(jnp.std(chords, axis=0).mean()) > 1e-3
