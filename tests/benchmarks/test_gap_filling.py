"""Benchmarks for gap filling operations."""

import localflowwalk as lfw


class TestGapFillingBenchmarks:
    """Benchmarks for gap filling operations."""

    def test_fill_ordering_gaps_simple(self, benchmark, trained_autoencoder):
        """Benchmark gap filling on simple stream."""
        trained_ae, walk_result = trained_autoencoder
        result = benchmark(lfw.nn.fill_ordering_gaps, trained_ae, walk_result)
        assert result is not None
