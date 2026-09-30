import numpy as np
import pytest

from ckmeans import CkmeansError, OptimalResult, ckmeans, ckmeans_optimal

DATA = [1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0]


def test_three_groups() -> None:
    result = ckmeans_optimal(DATA)
    assert isinstance(result, OptimalResult)
    assert result.k == 3
    assert len(result.clusters) == 3
    np.testing.assert_array_equal(result.centers, [1.0, 50.0, 100.0])
    np.testing.assert_array_equal(result.sizes, [3, 3, 3])
    np.testing.assert_array_equal(result.withinss, [0.0, 0.0, 0.0])
    assert result.sizes.dtype == np.intp


def test_k_max_capped_at_distinct_values() -> None:
    result = ckmeans_optimal(DATA, k_max=9)
    np.testing.assert_array_equal(result.ks, [1, 2, 3])
    assert result.bic.shape == result.ks.shape
    assert result.ks.dtype == np.intp


def test_k_range() -> None:
    rng = np.random.default_rng(2)
    data = rng.uniform(0.0, 100.0, 500)
    result = ckmeans_optimal(data, k_min=2, k_max=6)
    np.testing.assert_array_equal(result.ks, [2, 3, 4, 5, 6])
    assert result.k in result.ks
    assert result.k == result.ks[np.argmin(result.bic)]
    assert result.sizes.sum() == data.size


def test_clusters_match_ckmeans() -> None:
    rng = np.random.default_rng(3)
    data = np.concatenate([rng.normal(loc, 1.0, 50) for loc in (0.0, 20.0, 40.0)])
    result = ckmeans_optimal(data)
    expected = ckmeans(data, result.k)
    for got, want in zip(result.clusters, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(result.centers, [c.mean() for c in expected])


@pytest.mark.parametrize(
    ("k_min", "k_max"),
    [(0, 9), (5, 2), (1, 256), (4, 9)],
    ids=["k_min_zero", "k_min_above_k_max", "k_max_above_255", "k_min_above_distinct"],
)
def test_invalid_range(k_min: int, k_max: int) -> None:
    with pytest.raises(CkmeansError):
        ckmeans_optimal(DATA, k_min=k_min, k_max=k_max)


def test_nan() -> None:
    with pytest.raises(CkmeansError, match="NaN"):
        ckmeans_optimal([1.0, np.nan, 3.0])


def test_frozen() -> None:
    result = ckmeans_optimal(DATA)
    with pytest.raises(AttributeError):
        result.k = 2  # ty: ignore[invalid-assignment]


def test_repr() -> None:
    assert repr(ckmeans_optimal(DATA)).startswith("OptimalResult(k=3, centers=")
