import ckmeans  # Rust implementation
import ckmeans_1d_dp  # C++ implementation


def test_rust_ckmeans(benchmark, test_data, num_samples, num_clusters):
    """Benchmark Rust ckmeans implementation."""
    # Format group name based on parameters
    if num_samples >= 1e6:
        samples_str = f"{int(num_samples / 1e6)} M"
    else:
        samples_str = f"{int(num_samples / 1000)} k"
    benchmark.group = f"{samples_str}_{num_clusters} clusters"

    result = benchmark(ckmeans.ckmeans, test_data, num_clusters)
    assert len(result) == num_clusters


def test_cpp_ckmeans(benchmark, test_data, num_samples, num_clusters):
    """Benchmark C++ ckmeans implementation."""
    # Format group name based on parameters
    if num_samples >= 1e6:
        samples_str = f"{int(num_samples / 1e6)} M"
    else:
        samples_str = f"{int(num_samples / 1000)} k"
    benchmark.group = f"{samples_str}_{num_clusters} clusters"

    result = benchmark(ckmeans_1d_dp.ckmeans, test_data, num_clusters)
    assert len(result.centers) == num_clusters
