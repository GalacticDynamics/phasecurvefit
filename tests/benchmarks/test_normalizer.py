"""Benchmarks for data normalization."""

import phasecurvefit as pcf


class TestNormalizerBenchmarks:
    """Benchmarks for data normalization."""

    def test_normalizer_creation_simple(self, benchmark, simple_2d_stream):
        """Benchmark StandardScalerNormalizer creation."""
        pos, vel = simple_2d_stream
        normalizer = benchmark(pcf.nn.StandardScalerNormalizer, pos, vel)
        assert normalizer is not None

    def test_normalizer_creation_medium(self, benchmark, medium_2d_stream):
        """Benchmark StandardScalerNormalizer on medium dataset."""
        pos, vel = medium_2d_stream
        normalizer = benchmark(pcf.nn.StandardScalerNormalizer, pos, vel)
        assert normalizer is not None

    def test_normalizer_normalization_simple(self, benchmark, simple_2d_stream):
        """Benchmark normalization operations."""
        pos, vel = simple_2d_stream
        normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)

        normalized_pos, normalized_vel = benchmark(normalizer.transform, pos, vel)

        assert normalized_pos is not None
        assert normalized_vel is not None
