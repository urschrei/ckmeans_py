import numpy as np
import pytest

from ckmeans import CkmeansError, breaks, ckmeans

FUNCTIONS = [ckmeans, breaks]


@pytest.mark.parametrize("func", FUNCTIONS)
def test_zero_clusters(func) -> None:
    with pytest.raises(CkmeansError):
        func(np.array([1.0, 2.0, 3.0]), 0)


@pytest.mark.parametrize("func", FUNCTIONS)
def test_more_clusters_than_values(func) -> None:
    with pytest.raises(CkmeansError):
        func(np.array([1.0, 2.0, 3.0]), 5)


@pytest.mark.parametrize("func", FUNCTIONS)
@pytest.mark.parametrize("k", [256, -1])
def test_cluster_count_out_of_range(func, k: int) -> None:
    with pytest.raises(CkmeansError, match="from 1 to 255"):
        func(np.arange(300, dtype=np.float64), k)


@pytest.mark.parametrize("func", FUNCTIONS)
def test_nan(func) -> None:
    with pytest.raises(CkmeansError, match="NaN"):
        func(np.array([1.0, np.nan, 3.0]), 2)


def test_error_is_value_error() -> None:
    assert issubclass(CkmeansError, ValueError)


@pytest.mark.parametrize("func", FUNCTIONS)
def test_strided_input(func) -> None:
    data = np.array([1.0, -9.0, 2.0, -9.0, 100.0, -9.0, 101.0, -9.0])
    strided = data[::2]
    assert not strided.flags.c_contiguous
    result = func(strided, 2)
    expected = func(np.ascontiguousarray(strided), 2)
    assert len(result) == len(expected)
    for got, want in zip(result, expected, strict=True):
        np.testing.assert_array_equal(got, want)
