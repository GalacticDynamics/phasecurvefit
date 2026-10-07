"""Tests for the MST-backbone orderer (``pcf.orderers.MSTOrderer``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.abstract_result import AbstractResult


def _open_arc(n=200, seed=0):
    """Points sampled along a known open 1-D curve, then shuffled.

    Evenly spaced along the true parameter so the kNN graph stays connected
    (random-uniform sampling can leave gaps wider than ``k`` can bridge).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    x = 10.0 * t
    y = np.sin(3.0 * t)
    noise = rng.normal(0.0, 0.02, size=(n, 2))
    pos = {"x": jnp.asarray(x + noise[:, 0]), "y": jnp.asarray(y + noise[:, 1])}
    vel = {"x": jnp.ones(n), "y": jnp.asarray(3.0 * np.cos(3.0 * t))}
    # shuffle so input order carries no ordering information
    perm = rng.permutation(n)
    pos = {k: v[perm] for k, v in pos.items()}
    vel = {k: v[perm] for k, v in vel.items()}
    t_shuffled = t[perm]
    return pos, vel, jnp.asarray(t_shuffled)


def _open_ring(n=240, gap_deg=20.0, seed=0):
    """Points on a near-closed ring with a small angular gap (an open loop).

    Evenly spaced in angle so the along-ring kNN graph stays connected while the
    gap chord exceeds ``jump_cap`` and is severed.
    """
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, (360.0 - gap_deg), n) * np.pi / 180.0
    x, y = np.cos(ang), np.sin(ang)
    perm = rng.permutation(n)
    pos = {"x": jnp.asarray(x[perm]), "y": jnp.asarray(y[perm])}
    vel = {"x": jnp.asarray(-np.sin(ang)[perm]), "y": jnp.asarray(np.cos(ang)[perm])}
    return pos, vel, jnp.asarray(ang[perm])


class TestMSTNamespace:
    """Tests for MST namespace."""

    def test_exported(self):
        """Exported."""
        assert hasattr(pcf.orderers, "MSTOrderer")


class TestMSTConformance:
    """Tests for MST conformance."""

    def test_returns_orderingresult_with_backbone(self):
        """Returns orderingresult with backbone."""
        pos, vel, _ = _open_arc()
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        assert isinstance(res, AbstractResult)
        assert res.backbone is not None
        assert res.gamma_range == (-1.0, 1.0)
        # every tracer ordered (no -1) for a connected graph
        assert jnp.all(res.indices >= 0)
        assert set(res.indices.tolist()) == set(range(len(res.indices)))
        out = res(jnp.linspace(-1.0, 1.0, 11))
        assert jnp.all(jnp.isfinite(out["x"]))

    def test_mismatched_component_keys_raise(self):
        """Positions and velocities must share the same component keys."""
        pos = {"x": jnp.arange(4.0), "y": jnp.arange(4.0)}
        vel = {"x": jnp.ones(4), "z": jnp.ones(4)}  # 'z' != 'y'
        with pytest.raises(ValueError, match="same component keys"):
            pcf.orderers.MSTOrderer(k=2, jump_cap=10.0).order(pos, vel)


class TestMSTOrderingCorrectness:
    """Tests for MST ordering correctness."""

    def test_monotone_in_true_arclength(self):
        """Monotone in true arclength."""
        pos, vel, t = _open_arc(n=300)
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        t_ordered = np.asarray(t)[np.asarray(res.ordering)]
        # rank correlation with position-in-order is ~ +/-1 for a clean arc
        rho = np.corrcoef(np.arange(t_ordered.size), t_ordered)[0, 1]
        assert abs(rho) > 0.99

    def test_endpoints_are_extremes(self):
        """Endpoints are extremes."""
        pos, vel, t = _open_arc(n=300)
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        order = np.asarray(res.ordering)
        t = np.asarray(t)
        # the two tips sit at the two ends of the true parameter range
        tip_t = sorted([t[order[0]], t[order[-1]]])
        assert tip_t[0] < 0.02
        assert tip_t[1] > 0.98


class TestMSTLoop:
    """Tests for MST loop."""

    def test_full_coverage_and_tips_at_gap(self):
        """Full coverage and tips at gap."""
        pos, vel, ang = _open_ring(n=240, gap_deg=20.0)
        res = pcf.orderers.MSTOrderer(k=10, jump_cap=0.1).order(pos, vel)
        assert jnp.all(res.indices >= 0)  # ~100% coverage
        order = np.asarray(res.ordering)
        ang = np.asarray(ang)
        # the two ends of the ordering sit at the two sides of the gap
        tip_angs = sorted([ang[order[0]], ang[order[-1]]])
        assert tip_angs[0] < np.deg2rad(20.0)  # near angle 0
        assert tip_angs[1] > np.deg2rad(320.0)  # near angle 340 (other gap edge)


class TestMSTDeterminism:
    """Tests for MST determinism."""

    def test_deterministic(self):
        """Deterministic."""
        pos, vel, _ = _open_arc()
        o = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)
        assert jnp.array_equal(o.order(pos, vel).indices, o.order(pos, vel).indices)


def _two_clusters(n_a=80, n_b=30):
    """Two well-separated line clusters -> a disconnected kNN graph."""
    xa = np.linspace(0.0, 1.0, n_a)
    xb = np.linspace(0.0, 1.0, n_b) + 100.0
    x = np.concatenate([xa, xb])
    y = np.zeros_like(x)
    pos = {"x": jnp.asarray(x), "y": jnp.asarray(y)}
    vel = {"x": jnp.ones_like(jnp.asarray(x)), "y": jnp.zeros_like(jnp.asarray(x))}
    return pos, vel


class TestMSTDisconnected:
    """Tests for MST disconnected."""

    def test_raise_on_disconnected(self):
        """Raise on disconnected."""
        pos, vel, _ = _open_arc(n=100)
        # jump_cap far below inter-point spacing -> disconnected graph
        with pytest.raises(ValueError, match=r"disconnected|connected"):
            pcf.orderers.MSTOrderer(k=8, jump_cap=1e-6, on_disconnected="raise").order(
                pos, vel
            )

    def test_largest_orders_bigger_component(self):
        """Largest orders bigger component."""
        pos, vel = _two_clusters(n_a=80, n_b=30)
        res = pcf.orderers.MSTOrderer(
            k=5, jump_cap=0.5, on_disconnected="largest"
        ).order(pos, vel)
        assert int((res.indices >= 0).sum()) == 80  # bigger cluster ordered
        assert int((res.indices < 0).sum()) == 30  # smaller left unvisited

    def test_warn_on_disconnected(self):
        """Warn on disconnected."""
        pos, vel = _two_clusters()
        with pytest.warns(UserWarning, match="disconnected"):
            pcf.orderers.MSTOrderer(k=5, jump_cap=0.5, on_disconnected="warn").order(
                pos, vel
            )

    def test_connect_orders_every_component_without_warning(self):
        """``"connect"`` bridges the pieces: nothing is dropped, nothing warns."""
        pos, vel = _two_clusters(n_a=80, n_b=30)
        res = pcf.orderers.MSTOrderer(
            k=5, jump_cap=0.5, on_disconnected="connect"
        ).order(pos, vel)  # `filterwarnings = error`: a warning would fail this
        assert int(res.n_visited) == 110
        x = np.asarray(pos["x"])[np.asarray(res.ordering)]
        # Tip to tip across the gap: one cluster, then the other, monotonically.
        assert np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0)

    def test_connect_handles_many_components(self):
        """Several pieces, found and joined over more than one bridging round."""
        centers = np.array([0.0, 50.0, 120.0, 130.0, 400.0])
        x = np.concatenate([np.linspace(c, c + 1.0, 20) for c in centers])
        pos = {"x": jnp.asarray(x), "y": jnp.zeros(x.size)}
        vel = {"x": jnp.ones(x.size), "y": jnp.zeros(x.size)}
        res = pcf.orderers.MSTOrderer(
            k=5, jump_cap=2.0, on_disconnected="connect"
        ).order(pos, vel)
        assert int(res.n_visited) == x.size
        xs = x[np.asarray(res.ordering)]
        assert np.all(np.diff(xs) >= 0) or np.all(np.diff(xs) <= 0)

    def test_connect_is_a_no_op_on_a_connected_graph(self):
        """With nothing to bridge, ``"connect"`` gives the ``"raise"`` answer."""
        pos, vel, _ = _open_arc(n=100)
        kw = {"k": 8, "jump_cap": 2.0}
        a = pcf.orderers.MSTOrderer(**kw, on_disconnected="connect").order(pos, vel)
        b = pcf.orderers.MSTOrderer(**kw, on_disconnected="raise").order(pos, vel)
        assert jnp.array_equal(a.indices, b.indices)

    def test_connect_ignores_jump_cap_for_the_bridge_only(self):
        """The bridge may be longer than ``jump_cap``; ordinary edges may not."""
        pos, vel = _two_clusters(n_a=80, n_b=30)  # the gap is ~99, jump_cap is 0.5
        res = pcf.orderers.MSTOrderer(
            k=5, jump_cap=0.5, on_disconnected="connect"
        ).order(pos, vel)
        assert int(res.n_visited) == 110

    def test_disconnected_message_does_not_blame_an_infinite_jump_cap(self):
        """``jump_cap=inf`` cannot be "too small": the message must not say so."""
        pos, vel = _two_clusters(n_a=80, n_b=30)
        with pytest.raises(ValueError, match="disconnected") as exc:
            pcf.orderers.MSTOrderer(
                k=5, jump_cap=float("inf"), on_disconnected="raise"
            ).order(pos, vel)
        msg = str(exc.value)
        assert "jump_cap=inf" not in msg
        assert "connect" in msg  # names the way out

    def test_unknown_on_disconnected_raises(self):
        """An unknown policy is rejected at construction, not treated as 'largest'."""
        with pytest.raises(ValueError, match="on_disconnected"):
            pcf.orderers.MSTOrderer(on_disconnected="nonsense")


def _arc_with_interlopers(n_arc=200, n_out=15, seed=0):
    """Build a noisy open arc plus interlopers scattered across its footprint.

    Returns ``(pos, vel, is_outlier)`` with the interlopers appended after the
    arc points, each carrying a velocity uncorrelated with the arc.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_arc)
    x = 10.0 * t + rng.normal(0.0, 0.02, n_arc)
    y = np.sin(3.0 * t) + rng.normal(0.0, 0.02, n_arc)
    xi = rng.uniform(x.min(), x.max(), n_out)
    yi = rng.uniform(y.min() - 3.0, y.max() + 3.0, n_out)
    pos = {"x": jnp.asarray(np.r_[x, xi]), "y": jnp.asarray(np.r_[y, yi])}
    vel = {
        "x": jnp.asarray(np.r_[np.ones(n_arc), rng.normal(0.0, 1.0, n_out)]),
        "y": jnp.asarray(np.r_[3.0 * np.cos(3.0 * t), rng.normal(0.0, 1.0, n_out)]),
    }
    is_outlier = np.r_[np.zeros(n_arc, bool), np.ones(n_out, bool)]
    return pos, vel, is_outlier


class TestMSTEdgeClip:
    """Optional sigma-clipping of MST edge lengths (``edge_clip_sigma``)."""

    def test_none_is_a_noop(self):
        """``edge_clip_sigma=None`` (default) leaves every point visited."""
        pos, vel, _ = _arc_with_interlopers()
        res = pcf.orderers.MSTOrderer(k=10, jump_cap=20.0).order(pos, vel)
        assert int(res.n_skipped) == 0

    def test_rejects_interlopers_keeps_stream(self):
        """Clipping rejects the scattered interlopers, keeps the arc."""
        pos, vel, is_outlier = _arc_with_interlopers(n_arc=200, n_out=15)
        res = pcf.orderers.MSTOrderer(k=10, jump_cap=20.0, edge_clip_sigma=3.0).order(
            pos, vel
        )
        rejected = np.asarray(res.indices)
        visited = {int(i) for i in rejected if i >= 0}
        rej = set(range(is_outlier.size)) - visited
        n_out_rej = sum(is_outlier[i] for i in rej)
        n_in_rej = sum(1 for i in rej if not is_outlier[i])
        assert n_out_rej >= 0.6 * int(is_outlier.sum())  # most interlopers gone
        assert n_in_rej <= 0.05 * int((~is_outlier).sum())  # few genuine cut

    def test_clean_stream_not_clipped(self):
        """A clean arc with no interlopers loses no genuine points."""
        pos, vel, _t = _open_arc(n=200)
        res = pcf.orderers.MSTOrderer(k=10, jump_cap=20.0, edge_clip_sigma=3.0).order(
            pos, vel
        )
        assert int(res.n_skipped) == 0

    def test_composes_with_velocity_weight(self):
        """Clipping uses spatial length, so it still works with velocity weights."""
        pos, vel, is_outlier = _arc_with_interlopers(n_arc=200, n_out=15)
        res = pcf.orderers.MSTOrderer(
            k=10, jump_cap=20.0, velocity_weight=1.0, edge_clip_sigma=3.0
        ).order(pos, vel)
        visited = {int(i) for i in np.asarray(res.indices) if i >= 0}
        rej = set(range(is_outlier.size)) - visited
        assert sum(is_outlier[i] for i in rej) >= 0.6 * int(is_outlier.sum())

    @pytest.mark.parametrize("dup", [False, True], ids=["plain", "duplicates"])
    @pytest.mark.parametrize("subset", [False, True], ids=["all", "subset"])
    def test_matches_reference_loop(self, dup, subset):
        """The mask-based clip keeps exactly the nodes the per-slice loop kept."""
        from scipy.sparse import csr_matrix  # noqa: PLC0415
        from scipy.sparse.csgraph import (  # noqa: PLC0415
            connected_components,
            minimum_spanning_tree,
        )
        from scipy.spatial import cKDTree  # noqa: PLC0415

        from phasecurvefit._src.orderers import mst  # noqa: PLC0415

        def reference(tree, P, nodes, *, sigma, max_iters):
            log_floor = np.log(mst._EDGE_CLIP_MIN_RATIO)
            current = nodes
            for _ in range(max_iters):
                sub = tree[current][:, current].tocoo()
                upper = sub.row < sub.col
                ei, ej = sub.row[upper], sub.col[upper]
                length = np.linalg.norm(P[current[ei]] - P[current[ej]], axis=1)
                pos = length > 0.0
                if not pos.any():
                    break
                loglen = np.log(length[pos])
                med = float(np.median(loglen))
                scale = 1.4826 * float(np.median(np.abs(loglen - med)))
                cut = np.zeros_like(pos)
                cut[pos] = loglen > med + max(sigma * scale, log_floor)
                if not cut.any():
                    break
                m = current.size
                keep = ~cut
                g = csr_matrix(
                    (np.ones(int(keep.sum())), (ei[keep], ej[keep])), shape=(m, m)
                )
                _, labels = connected_components(g.maximum(g.T), directed=False)
                sizes = np.bincount(labels)
                size_min = max(2, int(np.ceil(mst._EDGE_CLIP_SMALL_FRAC * m)))
                small = np.isin(labels, np.flatnonzero(sizes < size_min))
                if not small.any():
                    break
                current = current[~small]
            return current

        pos, _, _ = _arc_with_interlopers(n_arc=400, n_out=30)
        P = np.stack([np.asarray(pos[c]) for c in sorted(pos)], axis=1)
        if dup:
            P = np.r_[P, P[::3]]
        d, i = cKDTree(P).query(P, k=11)
        # Self by index, as MSTOrderer does: with duplicates cKDTree may list a
        # coincident copy before the point itself.
        not_self = i != np.arange(len(P))[:, None]
        rows = np.nonzero(not_self)[0]
        w = np.maximum(d[not_self], np.finfo(d.dtype).tiny)
        tree = minimum_spanning_tree(
            csr_matrix((w, (rows, i[not_self])), shape=(len(P),) * 2)
        )
        tree = tree + tree.T
        nodes = np.arange(len(P))
        if subset:
            nodes = nodes[np.asarray(P[:, 0]) > 2.0]
        for sigma in (1.5, 3.0):
            got = mst._sigma_clip_edges(tree, P, nodes, sigma=sigma, max_iters=5)
            want = reference(tree, P, nodes, sigma=sigma, max_iters=5)
            np.testing.assert_array_equal(got, want)
            assert got.size < nodes.size  # the comparison exercised a real clip

    def test_invalid_sigma_raises(self):
        """A non-positive ``edge_clip_sigma`` is rejected at construction."""
        with pytest.raises(ValueError, match="edge_clip_sigma"):
            pcf.orderers.MSTOrderer(edge_clip_sigma=0.0)

    def test_invalid_max_iters_raises(self):
        """``edge_clip_max_iters < 1`` is rejected at construction."""
        with pytest.raises(ValueError, match="edge_clip_max_iters"):
            pcf.orderers.MSTOrderer(edge_clip_sigma=3.0, edge_clip_max_iters=0)


class TestMSTJaxTraceability:
    """``order()`` stays traceable under jit/vmap/grad.

    Host algorithms run via ``jax.pure_callback`` when traced, returning only
    indices; backbone coordinates are gathered from positions/velocities in
    ordinary JAX, so gradient flows through them like any other data-dependent
    gather. See the module docstring.
    """

    def test_jit(self):
        """``order()`` composed with interpolation runs under ``jax.jit``."""
        pos, vel, _t = _open_arc(n=60)
        orderer = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)

        @jax.jit
        def run(pos, vel):
            return orderer.order(pos, vel)(jnp.linspace(-1.0, 1.0, 5))

        eager = orderer.order(pos, vel)(jnp.linspace(-1.0, 1.0, 5))
        jitted = run(pos, vel)
        assert jnp.allclose(jitted["x"], eager["x"])
        assert jnp.allclose(jitted["y"], eager["y"])

    def test_vmap(self):
        """``order()`` runs one host call per batch element under ``jax.vmap``."""
        pos, vel, _t = _open_arc(n=60)
        orderer = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)
        pos_batch = {k: jnp.stack([v, v]) for k, v in pos.items()}
        vel_batch = {k: jnp.stack([v, v]) for k, v in vel.items()}

        def single(p, v):
            return orderer.order(p, v)(jnp.array(0.0))

        out = jax.vmap(single)(pos_batch, vel_batch)
        assert out["x"].shape == (2,)
        assert jnp.all(jnp.isfinite(out["x"]))

    def test_grad_through_backbone_gather_is_nonzero(self):
        """Backbone coordinates are a gather of positions, so gradient flows.

        ``Cb = P[backbone_nodes]`` is a literal gather, so -- away from the
        measure-zero set of configurations where the selected nodes themselves
        would change -- its true derivative is the ordinary gather Jacobian,
        not zero (unlike the discrete ``indices``/``backbone_size``, which
        stay gradient-free; see ``test_grad_of_indices_is_zero``).
        """
        pos, vel, _t = _open_arc(n=60)

        def loss(x):
            res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(
                {"x": x, "y": pos["y"]}, vel
            )
            return jnp.sum(res(jnp.array(0.3))["x"] ** 2)

        assert jnp.any(jax.grad(loss)(pos["x"]) != 0.0)

    def test_jit_does_not_crash_the_process_on_unlucky_knn_shapes(self):
        """Regression test for a real segfault, not just a failing assertion.

        ``jax.pure_callback``'s host-dispatch thread was found to have too
        small a stack for some (unlucky, not adversarial) ``cKDTree`` query
        shapes -- this exact input reliably crashed the whole process (not a
        catchable exception) before ``_run_in_thread`` gave the host
        computation a normal thread's stack instead. If that regresses, this
        test does not fail cleanly -- it takes the interpreter down.
        """
        xs = jnp.concatenate([jnp.linspace(0.0, 9.0, 40), jnp.array([30.0])])
        ys = jnp.concatenate([jnp.zeros(40), jnp.array([30.0])])
        vel = {"x": jnp.ones(41), "y": jnp.zeros(41)}
        clipper = pcf.orderers.MSTOrderer(k=10, jump_cap=50.0, edge_clip_sigma=3.0)

        @jax.jit
        def run(pos, vel):
            return clipper.order(pos, vel).indices

        out = run({"x": xs, "y": ys}, vel)
        assert int((out >= 0).sum()) == 40  # the lone interloper still rejected

    def test_grad_of_indices_is_zero(self):
        """The discrete ordering/backbone_size have no gradient (backbone coords do)."""
        pos, vel, _t = _open_arc(n=60)

        def loss(x):
            res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(
                {"x": x, "y": pos["y"]}, vel
            )
            # indices/backbone_size are int32; grad needs a float cotangent
            # target, so combine them into one float loss to check both.
            return jnp.sum(res.indices.astype(jnp.float32)) + res.backbone_size.astype(
                jnp.float32
            )

        assert jnp.all(jax.grad(loss)(pos["x"]) == 0.0)

    def test_grad_through_direct_positions_is_nonzero(self):
        """A loss on the untouched ``positions`` passthrough still differentiates."""
        pos, vel, _t = _open_arc(n=60)

        def loss(x):
            res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(
                {"x": x, "y": pos["y"]}, vel
            )
            return jnp.sum(res.positions["x"] ** 2)

        assert jnp.any(jax.grad(loss)(pos["x"]) != 0.0)

    def test_grad_wrt_velocity_only_is_traced(self):
        """Positions concrete, velocities traced: still takes the traced path.

        The two are independent tracer checks (``P`` and ``V``), so this covers
        the case the ``positions``-only grad test above does not: ``V`` traced
        while ``P`` stays a plain, concrete array.
        """
        pos, vel, _t = _open_arc(n=60)

        def loss(vx):
            res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(
                pos, {"x": vx, "y": vel["y"]}
            )
            return jnp.sum(res(jnp.array(0.3))["x"] ** 2)

        assert jnp.all(jax.grad(loss)(vel["x"]) == 0.0)


def _with_copies(pos, vel, idx, n_copies):
    """Append ``n_copies`` exact copies of the points at ``idx`` (repeat obs)."""
    rep = np.repeat(np.atleast_1d(idx), n_copies)
    pos = {k: jnp.concatenate([v, v[rep]]) for k, v in pos.items()}
    vel = {k: jnp.concatenate([v, v[rep]]) for k, v in vel.items()}
    return pos, vel


class TestMSTDuplicates:
    """Coincident points (zero-length edges) must stay in the graph."""

    def test_clump_larger_than_k_stays_connected(self):
        """More copies than ``k``: the clump's kNN edges are all zero-length."""
        pos, vel, _ = _open_arc(n=100)
        pos, vel = _with_copies(pos, vel, 50, n_copies=10)
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=20.0).order(pos, vel)
        assert int(res.n_skipped) == 0

    def test_edge_clip_with_duplicates(self):
        """Zero-length edges neither poison the clip statistic nor get cut."""
        pos, vel, is_outlier = _arc_with_interlopers(n_arc=200, n_out=15)
        pos, vel = _with_copies(pos, vel, np.arange(0, 200, 2), n_copies=1)
        is_outlier = np.r_[is_outlier, np.zeros(100, bool)]
        res = pcf.orderers.MSTOrderer(k=10, jump_cap=20.0, edge_clip_sigma=3.0).order(
            pos, vel
        )
        visited = {int(i) for i in np.asarray(res.indices) if i >= 0}
        rej = set(range(is_outlier.size)) - visited
        assert sum(is_outlier[i] for i in rej) >= 0.6 * int(is_outlier.sum())
        assert sum(1 for i in rej if not is_outlier[i]) <= 0.05 * 300

    def test_edge_clip_all_coincident(self):
        """Only zero-length edges: nothing to clip, every point kept."""
        pos = {"x": jnp.ones(12), "y": jnp.zeros(12)}
        vel = {"x": jnp.ones(12), "y": jnp.zeros(12)}
        res = pcf.orderers.MSTOrderer(k=4, edge_clip_sigma=3.0).order(pos, vel)
        assert int(res.n_skipped) == 0


def _cases():
    """Return reference configs: (name, positions, velocities, MSTOrderer kwargs)."""
    rng = np.random.default_rng(11)

    def stream(n, interlopers=0.0, dup=0):
        t = np.linspace(0, 1, n)
        p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(
            0, 0.02, (n, 3)
        )
        v = np.c_[np.ones(n), 3 * np.cos(3 * t), -0.6 * np.sin(2 * t)]
        m = int(interlopers * n)
        if m:
            i = rng.choice(n, m, replace=False)
            pad = np.array([0.0, 3.0, 3.0])
            p[i] = rng.uniform(p.min(0) - pad, p.max(0) + pad, (m, 3))
            v[i] = rng.normal(size=(m, 3))
        if dup:
            src = rng.choice(n, dup // 25, replace=False)
            p = np.r_[p, np.repeat(p[src], 25, axis=0)]
            v = np.r_[v, np.repeat(v[src], 25, axis=0)]
        perm = rng.permutation(len(p))
        return p[perm].astype(np.float32), v[perm].astype(np.float32)

    def loop(n):
        a = np.linspace(0, 2 * np.pi * 0.95, n)
        p = np.c_[np.cos(a), np.sin(a), 0 * a] + rng.normal(0, 0.01, (n, 3))
        v = np.c_[-np.sin(a), np.cos(a), 0 * a]
        perm = rng.permutation(n)
        return p[perm].astype(np.float32), v[perm].astype(np.float32)

    two = stream(1500)
    two[0][:500, 0] += 50.0
    return [
        ("stream", *stream(1500), {"jump_cap": 2.0}),
        ("clip", *stream(1500, 0.01), {"jump_cap": 50.0, "edge_clip_sigma": 3.0}),
        ("dup", *stream(1500, dup=250), {"jump_cap": 2.0}),
        (
            "loop_vw",
            *loop(1000),
            {"jump_cap": 0.5, "velocity_weight": 1.0, "orient_by_velocity": True},
        ),
        (
            "loop_sever",
            *loop(1000),
            {"jump_cap": 0.5, "sever_cos_threshold": 0.0, "on_disconnected": "largest"},
        ),
        ("connect", *two, {"jump_cap": 2.0, "on_disconnected": "connect"}),
    ]


def _as_dicts(p, v):
    comps = "xyz"[: p.shape[1]]
    return (
        {c: jnp.asarray(p[:, i]) for i, c in enumerate(comps)},
        {c: jnp.asarray(v[:, i]) for i, c in enumerate(comps)},
    )


def _distinct_backbone(result):
    """Backbone coordinates (valid prefix) with consecutive repeats collapsed."""
    n = int(result.backbone_size)
    bb = np.stack([np.asarray(c)[:n] for c in result.backbone.values()], axis=1)
    return bb[np.r_[True, np.any(np.diff(bb, axis=0) != 0, axis=1)]]


class TestMSTBackends:
    """``MSTOrderer(neighbors=...)``: every backend gives the scipy result."""

    @pytest.mark.parametrize(
        "backend",
        [
            pcf.neighbors.BucketKDTree(),
            pcf.neighbors.BruteForce(),
            pcf.neighbors.Jaxkd(),
        ],
        ids=["bucket", "brute", "jaxkd"],
    )
    @pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
    def test_matches_scipy(self, case, backend):
        """Same ordering and backbone as the scipy backend on reference configs."""
        _, p, v, kw = case
        pos, vel = _as_dicts(p, v)
        want = pcf.orderers.MSTOrderer(
            k=10, neighbors=pcf.neighbors.Scipy(), **kw
        ).order(pos, vel)
        got = pcf.orderers.MSTOrderer(k=10, neighbors=backend, **kw).order(pos, vel)
        np.testing.assert_array_equal(np.asarray(got.indices), np.asarray(want.indices))
        # Coincident points tie in distance, so backends may thread the MST
        # through a clump in a different order; compare the backbone with
        # consecutive repeats collapsed.
        np.testing.assert_array_equal(_distinct_backbone(got), _distinct_backbone(want))

    def test_default_is_bucket_kdtree(self):
        """The kd-tree is the default backend."""
        assert isinstance(
            pcf.orderers.MSTOrderer().neighbors, pcf.neighbors.BucketKDTree
        )

    def test_scipy_raises_under_jit(self):
        """The scipy backend is eager-only, with a pointer to BucketKDTree."""
        pos, vel, _ = _open_arc(n=60)
        orderer = pcf.orderers.MSTOrderer(
            k=8, jump_cap=2.0, neighbors=pcf.neighbors.Scipy()
        )
        with pytest.raises(TypeError, match="BucketKDTree"):
            jax.jit(lambda p, v: orderer.order(p, v).indices)(pos, vel)

    def test_default_jit_matches_eager(self):
        """The default backend traces, and jit equals eager."""
        pos, vel, _ = _open_arc(n=200)
        orderer = pcf.orderers.MSTOrderer(k=10, jump_cap=2.0)
        eager = orderer.order(pos, vel).indices
        jitted = jax.jit(lambda p, v: orderer.order(p, v).indices)(pos, vel)
        np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 9])
    def test_tiny_inputs_match_scipy(self, n):
        """Review Focus 3: tiny n (and n <= k) match the scipy backend."""
        rng = np.random.default_rng(n)
        pos, vel = _as_dicts(
            rng.normal(size=(n, 3)).astype(np.float32),
            rng.normal(size=(n, 3)).astype(np.float32),
        )
        kw = {"k": 10, "jump_cap": 100.0}
        want = pcf.orderers.MSTOrderer(neighbors=pcf.neighbors.Scipy(), **kw).order(
            pos, vel
        )
        got = pcf.orderers.MSTOrderer(**kw).order(pos, vel)
        np.testing.assert_array_equal(np.asarray(got.indices), np.asarray(want.indices))

    def test_one_dimensional_matches_scipy(self):
        """Review Focus 5: a single position component."""
        rng = np.random.default_rng(5)
        x = rng.permutation(np.linspace(0, 10, 400)).astype(np.float32)
        pos, vel = {"x": jnp.asarray(x)}, {"x": jnp.ones(400)}
        want = pcf.orderers.MSTOrderer(
            k=8, jump_cap=1.0, neighbors=pcf.neighbors.Scipy()
        ).order(pos, vel)
        got = pcf.orderers.MSTOrderer(k=8, jump_cap=1.0).order(pos, vel)
        np.testing.assert_array_equal(np.asarray(got.indices), np.asarray(want.indices))

    def test_non_finite_positions_raise(self):
        """Review Focus 2: NaN positions raise instead of mis-ordering."""
        pos, vel, _ = _open_arc(n=60)
        pos = {**pos, "x": pos["x"].at[5].set(jnp.nan)}
        with pytest.raises(Exception, match="finite"):
            jax.block_until_ready(
                pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel).indices
            )

    def test_vmap_default_backend(self):
        """Vmap over two clouds equals two separate eager calls."""
        a, b = _open_arc(n=120, seed=0), _open_arc(n=120, seed=1)
        orderer = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)
        batch_p = {c: jnp.stack([a[0][c], b[0][c]]) for c in a[0]}
        batch_v = {c: jnp.stack([a[1][c], b[1][c]]) for c in a[1]}
        got = jax.vmap(lambda p, v: orderer.order(p, v).indices)(batch_p, batch_v)
        for i, (p, v, _) in enumerate((a, b)):
            np.testing.assert_array_equal(
                np.asarray(got[i]), np.asarray(orderer.order(p, v).indices)
            )
