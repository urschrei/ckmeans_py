import ckmeans  # Rust implementation
import ckmeans_1d_dp  # C++ implementation
import numpy as np
import pytest


@pytest.fixture
def test_data():
    """Generate 110k random uniform samples, 1.0 >= samples <= 3.0."""
    return np.random.default_rng().uniform(1.0, 3.0, 110000)


def test_rust_ckmeans(benchmark, test_data):
    """Benchmark Rust ckmeans implementation with 110k samples, 5 clusters."""
    result = benchmark(ckmeans.ckmeans, test_data, 5)
    assert len(result) == 5


def test_cpp_ckmeans(benchmark, test_data):
    """Benchmark C++ ckmeans implementation with 110k samples, 5 clusters."""
    result = benchmark(ckmeans_1d_dp.ckmeans, test_data, 5)
    assert len(result.centers) == 5
