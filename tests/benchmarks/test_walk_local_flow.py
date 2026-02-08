"""Benchmarks for walk_local_flow function."""

import localflowwalk as lfw


class TestWalkLocalFlowBenchmarks:
    """Benchmarks for walk_local_flow function."""

    def test_walk_local_flow_simple_2d(self, benchmark, simple_2d_stream):
        """Benchmark walk_local_flow on 50-point 2D stream."""
        pos, vel = simple_2d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=0, lam=1.0)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_medium_2d(self, benchmark, medium_2d_stream):
        """Benchmark walk_local_flow on 100-point 2D stream."""
        pos, vel = medium_2d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=0, lam=1.0)
        assert result.indices.shape == (100,)

    def test_walk_local_flow_large_2d(self, benchmark, large_2d_stream):
        """Benchmark walk_local_flow on 500-point 2D stream."""
        pos, vel = large_2d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=0, lam=1.0)
        assert result.indices.shape == (500,)

    def test_walk_local_flow_simple_3d(self, benchmark, simple_3d_stream):
        """Benchmark walk_local_flow on 50-point 3D stream."""
        pos, vel = simple_3d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=0, lam=1.0)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_spatial_only(self, benchmark, simple_2d_stream):
        """Benchmark walk_local_flow with spatial metric only (lam=0)."""
        pos, vel = simple_2d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=0, lam=0.0)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_high_momentum_weight(self, benchmark, simple_2d_stream):
        """Benchmark walk_local_flow with high momentum weight (lam=10)."""
        pos, vel = simple_2d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=0, lam=10.0)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_different_start(self, benchmark, simple_2d_stream):
        """Benchmark walk_local_flow starting from different point."""
        pos, vel = simple_2d_stream
        result = benchmark(lfw.walk_local_flow, pos, vel, start_idx=25, lam=1.0)
        assert result.indices.shape == (50,)
