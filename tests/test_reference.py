import numpy as np
import numpy.typing as npt
import pytest

from ckmeans import ckmeans_optimal

ckmeans_1d_dp = pytest.importorskip("ckmeans_1d_dp")


def reference_sizes(
    data: npt.NDArray[np.float64], k_min: int, k_max: int
) -> npt.NDArray[np.float64]:
    """Get the cluster sizes from Ckmeans.1d.dp, which pads them with zeros to k_max."""
    sizes = ckmeans_1d_dp.ckmeans(data, k=(k_min, k_max)).size
    return sizes[sizes > 0]


def mixture(seed: int) -> npt.NDArray[np.float64]:
    """Make a Gaussian mixture with 1 to 5 components of different sizes and spreads."""
    rng = np.random.default_rng(seed)
    groups = int(rng.integers(1, 6))
    return np.concatenate(
        [
            rng.normal(10.0 * i, rng.uniform(0.5, 2.0), int(rng.integers(10, 60)))
            for i in range(groups)
        ]
    )


@pytest.mark.parametrize("seed", range(50))
def test_k_matches_reference(seed: int) -> None:
    data = mixture(seed)
    expected = reference_sizes(data, 1, 9)
    result = ckmeans_optimal(data)
    assert result.k == expected.size
    np.testing.assert_array_equal(result.sizes, expected)


@pytest.mark.parametrize("seed", range(10))
def test_k_range_matches_reference(seed: int) -> None:
    data = mixture(seed)
    expected = reference_sizes(data, 2, 6)
    result = ckmeans_optimal(data, k_min=2, k_max=6)
    assert result.k == expected.size
