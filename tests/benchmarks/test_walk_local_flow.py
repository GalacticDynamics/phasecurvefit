"""Benchmarks for the local-flow walk (via `LocalFlowOrderer`)."""

import phasecurvefit as pcf


class TestWalkLocalFlowBenchmarks:
    """Benchmarks for the local-flow walk (via `LocalFlowOrderer`)."""

    def test_walk_local_flow_simple_2d(self, benchmark, simple_2d_stream):
        """Benchmark the local-flow walk on 50-point 2D stream."""
        pos, vel = simple_2d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_medium_2d(self, benchmark, medium_2d_stream):
        """Benchmark the local-flow walk on 100-point 2D stream."""
        pos, vel = medium_2d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (100,)

    def test_walk_local_flow_large_2d(self, benchmark, large_2d_stream):
        """Benchmark the local-flow walk on 500-point 2D stream."""
        pos, vel = large_2d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (500,)

    def test_walk_local_flow_simple_3d(self, benchmark, simple_3d_stream):
        """Benchmark the local-flow walk on 50-point 3D stream."""
        pos, vel = simple_3d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_spatial_only(self, benchmark, simple_2d_stream):
        """Benchmark the local-flow walk with spatial metric only (metric_scale=0)."""
        pos, vel = simple_2d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=0.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_high_momentum_weight(self, benchmark, simple_2d_stream):
        """Benchmark the local-flow walk with high momentum weight (metric_scale=10)."""
        pos, vel = simple_2d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=10.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (50,)

    def test_walk_local_flow_different_start(self, benchmark, simple_2d_stream):
        """Benchmark the local-flow walk starting from a different point."""
        pos, vel = simple_2d_stream
        orderer = pcf.orderers.LocalFlowOrderer(start_idx=25, metric_scale=1.0)
        result = benchmark(orderer.order, pos, vel)
        assert result.indices.shape == (50,)
