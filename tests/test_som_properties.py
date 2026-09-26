"""Property-based tests for the SOM core.

The example tests pin measured behaviour on particular curves. These assert the
invariants that must hold for *any* input, which is where hand-picked examples
are weakest.

Generation is deliberately constrained: JAX runs in float32 here, so strategies
produce finite, bounded values. Unbounded or `nan`-carrying inputs would fail
for reasons that say nothing about the code.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import phasecurvefit as pcf
from phasecurvefit import som
from phasecurvefit._src.orderers.base import chord_along_ordering

# JAX compiles per shape, so a handful of sizes keeps this from being a
# compilation benchmark. Deadlines are off for the same reason.
SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

coords = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def polylines(draw, min_points=3, max_points=40, dims=2):
    """Draw a polyline with no coincident consecutive vertices.

    Duplicate vertices make arc length degenerate, which the orderers guard
    against separately; this strategy is about the generic case.
    """
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    keys = ["x", "y", "z"][:dims]
    pts = np.array(
        [draw(st.lists(coords, min_size=n, max_size=n)) for _ in keys], dtype=np.float32
    )
    # nudge apart so consecutive points are distinct
    pts += np.arange(n, dtype=np.float32) * 1e-2
    return {k: jnp.asarray(pts[i]) for i, k in enumerate(keys)}


class TestChordAlongOrdering:
    """`chord_along_ordering` is arc length: its properties are geometric."""

    @SETTINGS
    @given(pos=polylines())
    def test_non_negative_and_monotone(self, pos):
        """Distance travelled never decreases as you walk the ordering."""
        n = next(iter(pos.values())).shape[0]
        chord = np.asarray(chord_along_ordering(pos, jnp.arange(n)))
        assert np.all(chord >= 0)
        assert np.all(np.diff(chord) >= -1e-4)

    @SETTINGS
    @given(pos=polylines(), shift=coords)
    def test_translation_invariant(self, pos, shift):
        """Arc length is a distance, so moving the whole curve cannot change it."""
        n = next(iter(pos.values())).shape[0]
        moved = {k: v + shift for k, v in pos.items()}
        a = np.asarray(chord_along_ordering(pos, jnp.arange(n)))
        b = np.asarray(chord_along_ordering(moved, jnp.arange(n)))
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)

    @SETTINGS
    @given(pos=polylines(), factor=st.floats(min_value=0.125, max_value=8.0, width=32))
    def test_scales_linearly(self, pos, factor):
        """Scaling the curve scales every arc length by the same factor."""
        n = next(iter(pos.values())).shape[0]
        scaled = {k: v * factor for k, v in pos.items()}
        a = np.asarray(chord_along_ordering(pos, jnp.arange(n))) * factor
        b = np.asarray(chord_along_ordering(scaled, jnp.arange(n)))
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-2)

    def test_walks_the_ordering_not_the_input_order(self):
        """The chord accumulates along the *visit* order, then scatters back.

        Every other test in this class passes ``arange``, i.e. the identity, so
        an implementation that gathered in input order would satisfy all of
        them. This pins a permutation whose answer is hand-computed and is
        neither the identity nor the x coordinate.
        """
        pos = {"x": jnp.asarray([0.0, 5.0, 1.0, 3.0]), "y": jnp.zeros(4)}
        ordering = jnp.asarray([2, 0, 3, 1])
        # walk: obs2 (x=1) -> obs0 (x=0) -> obs3 (x=3) -> obs1 (x=5)
        # arc:        0            1            4            6
        # scattered back to the observation's own slot:
        want = np.array([1.0, 6.0, 0.0, 4.0])
        got = np.asarray(chord_along_ordering(pos, ordering))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    @SETTINGS
    @given(pos=polylines(min_points=3), data=st.data())
    def test_matches_the_arc_length_of_the_permuted_walk(self, pos, data):
        """For *any* permutation, against an independent NumPy reference."""
        keys = sorted(pos)
        n = next(iter(pos.values())).shape[0]
        perm = np.asarray(data.draw(st.permutations(range(n))), dtype=np.int32)

        got = np.asarray(chord_along_ordering(pos, jnp.asarray(perm)))

        # Reference: walk the points in `perm` order, accumulate the segment
        # lengths, then put each running total in that observation's own slot.
        q = np.stack([np.asarray(pos[k]) for k in keys], axis=-1)[perm]
        walked = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=-1))]
        )
        want = np.empty(n, dtype=np.float64)
        want[perm] = walked
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)

    @SETTINGS
    @given(pos=polylines(min_points=4), n_hidden=st.integers(min_value=1, max_value=3))
    def test_unvisited_entries_are_nan(self, pos, n_hidden):
        """`nan` marks exactly the observations no stage visited."""
        n = next(iter(pos.values())).shape[0]
        indices = jnp.concatenate(
            [jnp.arange(n - n_hidden), jnp.full(n_hidden, -1, dtype=jnp.int32)]
        )
        chord = np.asarray(chord_along_ordering(pos, indices))
        assert np.isnan(chord).sum() == n_hidden
        assert not np.isnan(chord[: n - n_hidden]).any()


class TestDensify:
    """`densify` resamples a prototype set into a backbone."""

    @SETTINGS
    @given(pos=polylines(min_points=3, max_points=20), factor=st.integers(2, 8))
    def test_vertex_count_is_exact(self, pos, factor):
        """The emitted backbone has `factor * (K - 1) + 1` vertices."""
        k = next(iter(pos.values())).shape[0]
        vel = {key: jnp.zeros(k) for key in pos}
        bq, bp = som.densify(pos, vel, factor=factor)
        expected = factor * (k - 1) + 1
        assert next(iter(bq.values())).shape == (expected,)
        assert next(iter(bp.values())).shape == (expected,)

    @SETTINGS
    @given(pos=polylines(min_points=3, max_points=20), factor=st.integers(2, 8))
    def test_endpoints_are_preserved(self, pos, factor):
        """The backbone starts and ends on the first and last prototype.

        Arc-length *uniformity* is deliberately not asserted here. It holds
        tightly on a smooth curve -- 0.1%, pinned by
        ``test_densify_is_uniform_in_arc_length`` -- but this strategy also
        generates zig-zags and near-cusps, where a spline's arc-length
        parameterization varies fast inside one segment and
        ``_resample_uniform`` interpolates that linearly. The residual there is
        tens of percent and does not shrink with ``factor``, so a bound wide
        enough to pass would assert nothing.
        """
        k = next(iter(pos.values())).shape[0]
        vel = {key: jnp.zeros(k) for key in pos}
        bq, _ = som.densify(pos, vel, factor=factor)
        for key in pos:
            assert float(bq[key][0]) == pytest.approx(float(pos[key][0]), abs=1e-3)
            assert float(bq[key][-1]) == pytest.approx(float(pos[key][-1]), abs=1e-3)


class TestMetrics:
    """Distance metrics have contracts the orderers rely on."""

    @SETTINGS
    @given(
        pos=polylines(min_points=2, max_points=12), scale=st.floats(0.0, 5.0, width=32)
    )
    @pytest.mark.parametrize(
        "metric",
        [
            pcf.metrics.SpatialDistanceMetric(),
            pcf.metrics.FullPhaseSpaceDistanceMetric(),
        ],
        ids=["spatial", "full-phase-space"],
    )
    def test_non_negative_and_symmetric(self, metric, pos, scale):
        """These two are induced by an inner product, so they are symmetric."""
        keys = sorted(pos)
        n = pos[keys[0]].shape[0]
        vel = {k: jnp.zeros(n) for k in keys}
        one = lambda d, i: {k: d[k][i] for k in keys}  # a single point
        many = lambda d, i: {k: d[k][i][None] for k in keys}  # ...as a 1-array

        forward = np.asarray(metric(one(pos, 0), one(vel, 0), pos, vel, scale))
        assert np.all(forward >= 0)
        assert forward[0] == pytest.approx(0.0, abs=1e-3)

        ab = float(
            metric(one(pos, 0), one(vel, 0), many(pos, -1), many(vel, -1), scale)[0]
        )
        ba = float(
            metric(one(pos, -1), one(vel, -1), many(pos, 0), many(vel, 0), scale)[0]
        )
        assert ab == pytest.approx(ba, rel=1e-4, abs=1e-4)


class TestOrdererContract:
    """Invariants every orderer owes its caller, whatever the curve."""

    @SETTINGS
    @given(
        n=st.integers(min_value=30, max_value=120),
        turns=st.floats(min_value=0.25, max_value=1.0, width=32),
        n_prototypes=st.integers(min_value=5, max_value=20),
    )
    def test_ordering_is_a_permutation_of_the_visited_set(
        self, helix, n, turns, n_prototypes
    ):
        """No observation is dropped, duplicated, or invented."""
        pos, vel, _ = helix(n=n, turns=turns)
        result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=n_prototypes))
        idx = np.asarray(result.ordering)
        assert np.array_equal(np.sort(idx), np.unique(idx))
        assert idx.min() >= 0
        assert idx.max() < n
        assert int(result.n_visited) == int((np.asarray(result.indices) >= 0).sum())

    @SETTINGS
    @given(
        n=st.integers(min_value=30, max_value=120),
        n_prototypes=st.integers(min_value=5, max_value=20),
    )
    def test_chord_is_finite_on_every_visited_observation(self, helix, n, n_prototypes):
        """`chord` is `nan` only where a stage declined to visit."""
        pos, vel, _ = helix(n=n)
        result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=n_prototypes))
        chord = np.asarray(result.chord)
        visited = np.asarray(result.indices) >= 0
        assert np.isfinite(chord[np.asarray(result.ordering)]).all()
        assert not visited.all() or np.isfinite(chord).all()

    @SETTINGS
    @given(n=st.integers(min_value=40, max_value=120))
    def test_backbone_spans_the_data(self, helix, n):
        """The backbone lies within the data's bounding box, not off in space.

        A prototype that no datum reaches used to be moved to the coordinate
        origin, which dragged the spline through a point on no part of the
        curve. This asserts the property that regression violated.
        """
        pos, vel, _ = helix(n=n, radius=3.0, height=5.0)
        result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=12))
        for key, data in pos.items():
            lo, hi = float(jnp.min(data)), float(jnp.max(data))
            span = hi - lo
            bb = np.asarray(result.backbone[key])
            assert bb.min() >= lo - 0.25 * span
            assert bb.max() <= hi + 0.25 * span
