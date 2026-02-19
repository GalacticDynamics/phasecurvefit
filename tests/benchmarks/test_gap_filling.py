"""Benchmarks for gap filling operations."""

import phasecurvefit as pcf


class TestGapFillingBenchmarks:
    """Benchmarks for gap filling operations."""

    def test_fill_ordering_gaps_simple(self, benchmark, trained_autoencoder):
        """Benchmark gap filling on simple stream."""
        trained_ae, walk_result = trained_autoencoder
        result = benchmark(pcf.nn.fill_ordering_gaps, trained_ae, walk_result)
        assert result is not None
