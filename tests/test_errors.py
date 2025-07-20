import numpy as np
import pytest

from ckmeans import InvalidClusterCountError, InvalidDataError, breaks, ckmeans


def test_empty_array_error():
    """Test that empty arrays raise InvalidDataError"""
    empty_data = np.array([])

    with pytest.raises(InvalidDataError, match="Input data array is empty"):
        ckmeans(empty_data, 2)

    with pytest.raises(InvalidDataError, match="Input data array is empty"):
        breaks(empty_data, 2)


def test_zero_clusters_error():
    """Test that k=0 raises InvalidClusterCountError"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(
        InvalidClusterCountError,
        match="Number of clusters \\(k\\) must be greater than 0",
    ):
        ckmeans(data, 0)


def test_too_many_clusters_error():
    """Test that k > len(data) raises InvalidClusterCountError"""
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        InvalidClusterCountError,
        match="Number of clusters \\(5\\) cannot exceed the number of data points \\(3\\)",
    ):
        ckmeans(data, 5)


def test_nan_values_error():
    """Test that NaN values raise InvalidDataError"""
    data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    with pytest.raises(
        InvalidDataError, match="Input data contains NaN at index 2"
    ):
        breaks(data_with_nan, 2)


def test_infinite_values_error():
    """Test that infinite values raise InvalidDataError"""
    data_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

    with pytest.raises(
        InvalidDataError, match="Input data contains infinite value at index 2"
    ):
        breaks(data_with_inf, 2)

    # Test negative infinity as well
    data_with_neginf = np.array([1.0, 2.0, -np.inf, 4.0, 5.0])

    with pytest.raises(
        InvalidDataError, match="Input data contains infinite value at index 2"
    ):
        ckmeans(data_with_neginf, 2)


def test_large_k_value_error():
    """Test that very large k values raise InvalidClusterCountError"""
    # Create enough data points so 256 doesn't exceed the count
    data = np.arange(300.0)

    # k must fit in u8 (max 255)
    with pytest.raises(
        InvalidClusterCountError, match="Number of clusters is too large"
    ):
        ckmeans(data, 256)


def test_edge_case_single_element():
    """Test clustering with single element"""
    data = np.array([42.0])

    # Should work with k=1
    result = ckmeans(data, 1)
    assert len(result) == 1
    assert list(result[0]) == [42.0]

    # Should fail with k>1
    with pytest.raises(InvalidClusterCountError):
        ckmeans(data, 2)


def test_edge_case_all_same_values():
    """Test clustering when all values are identical"""
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    # When all values are identical, ckmeans may return fewer clusters than requested
    # because there's no meaningful way to split identical values
    result = ckmeans(data, 2)

    # All values should still be 5.0
    all_values = []
    for cluster in result:
        all_values.extend(list(cluster))
    assert all(v == 5.0 for v in all_values)
    assert len(all_values) == 5
