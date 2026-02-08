"""Benchmarks for metrics and query strategies."""

import localflowwalk as lfw


class TestMetricsAndStrategiesBenchmarks:
    """Benchmarks for metrics and query strategies."""

    def test_spatial_distance_metric_brute_force(self, benchmark, simple_2d_stream):
        """Benchmark with spatial distance metric using brute force."""
        pos, vel = simple_2d_stream

        # Default strategy is BruteForce, spatial metric via lam=0
        result = benchmark(
            lfw.walk_local_flow,
            pos,
            vel,
            start_idx=0,
            lam=0.0,
        )

        assert result.indices.shape == (50,)

    def test_full_phase_space_distance_metric_brute_force(
        self, benchmark, simple_2d_stream
    ):
        """Benchmark with full phase-space distance metric using brute force."""
        pos, vel = simple_2d_stream

        # Full phase-space metric is the default with lam > 0
        result = benchmark(
            lfw.walk_local_flow,
            pos,
            vel,
            start_idx=0,
            lam=1.0,
        )

        assert result.indices.shape == (50,)

    def test_custom_metric_with_config(self, benchmark, simple_2d_stream):
        """Benchmark with custom metric via WalkConfig."""
        pos, vel = simple_2d_stream

        config = lfw.WalkConfig(metric=lfw.metrics.AlignedMomentumDistanceMetric())
        result = benchmark(
            lfw.walk_local_flow,
            pos,
            vel,
            start_idx=0,
            lam=1.0,
            config=config,
        )

        assert result.indices.shape == (50,)
