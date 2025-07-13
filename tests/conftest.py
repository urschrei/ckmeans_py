import numpy as np
import pytest


def pytest_generate_tests(metafunc):
    """Generate test parameters for benchmarks."""
    if (
        "num_samples" in metafunc.fixturenames
        and "num_clusters" in metafunc.fixturenames
    ):
        metafunc.parametrize(
            "num_samples,num_clusters",
            [
                (110000, 5),
                (110000, 20),
                (int(1e6), 5),
                (int(1e6), 20),
            ],
            ids=["110k_5clusters", "110k_20clusters", "1M_5clusters", "1M_20clusters"],
        )


@pytest.fixture
def test_data(num_samples):
    """Generate random uniform samples, 1.0 >= samples <= 3.0."""
    return np.random.default_rng().uniform(1.0, 3.0, num_samples)
