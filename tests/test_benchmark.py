import pytest

import ckmeans  # Rust implementation


def samples_label(num_samples):
    """Format the number of samples for a benchmark group name."""
    if num_samples >= 1e6:
        return f"{int(num_samples / 1e6)} M"
    return f"{int(num_samples / 1000)} k"


def test_rust_ckmeans(benchmark, test_data, num_samples, num_clusters):
    """Benchmark Rust ckmeans implementation."""
    benchmark.group = f"{samples_label(num_samples)}_{num_clusters} clusters"

    result = benchmark(ckmeans.ckmeans, test_data, num_clusters)
    assert len(result) == num_clusters


def test_cpp_ckmeans(benchmark, test_data, num_samples, num_clusters):
    """Benchmark C++ ckmeans implementation."""
    try:
        import ckmeans_1d_dp  # C++ implementation
    except ImportError:
        pytest.skip("ckmeans_1d_dp not available on this platform")

    benchmark.group = f"{samples_label(num_samples)}_{num_clusters} clusters"

    result = benchmark(ckmeans_1d_dp.ckmeans, test_data, num_clusters)
    assert len(result.centers) == num_clusters


def test_rust_ckmeans_optimal(benchmark, test_data, num_samples):
    """Benchmark Rust ckmeans_optimal implementation, for k from 1 to 9."""
    benchmark.group = f"{samples_label(num_samples)}_optimal"

    result = benchmark(ckmeans.ckmeans_optimal, test_data)
    assert result.sizes.sum() == num_samples


def test_cpp_ckmeans_optimal(benchmark, test_data, num_samples):
    """Benchmark C++ ckmeans implementation with BIC selection, for k from 1 to 9."""
    try:
        import ckmeans_1d_dp  # C++ implementation
    except ImportError:
        pytest.skip("ckmeans_1d_dp not available on this platform")

    benchmark.group = f"{samples_label(num_samples)}_optimal"

    result = benchmark(ckmeans_1d_dp.ckmeans, test_data, (1, 9))
    assert result.size.sum() == num_samples
