from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ckmeans import breaks, ckmeans

DATA = [1.0, 2.0, 3.0, 100.0, 101.0, 102.0]


@pytest.mark.parametrize(
    "data",
    [
        DATA,
        tuple(DATA),
        np.array(DATA, dtype=np.int64),
        np.array(DATA, dtype=np.int32),
        np.array(DATA, dtype=np.float32),
    ],
    ids=["list", "tuple", "int64", "int32", "float32"],
)
def test_array_like_input(data) -> None:
    result = ckmeans(data, 2)
    assert all(cluster.dtype == np.float64 for cluster in result)
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result[1], [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(breaks(data, 2), breaks(np.array(DATA), 2))


@pytest.mark.parametrize("func", [ckmeans, breaks])
def test_two_dimensional_input(func) -> None:
    with pytest.raises(TypeError):
        func(np.ones((3, 2)), 2)


def test_threads() -> None:
    rng = np.random.default_rng(1)
    inputs = [rng.uniform(0.0, 100.0, 10_000) for _ in range(8)]
    expected = [ckmeans(data, 5) for data in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda data: ckmeans(data, 5), inputs))
    for got, want in zip(results, expected, strict=True):
        for got_cluster, want_cluster in zip(got, want, strict=True):
            np.testing.assert_array_equal(got_cluster, want_cluster)


@pytest.mark.parametrize("func", [ckmeans, breaks])
def test_k_as_keyword(func) -> None:
    by_position = func(DATA, 2)
    by_keyword = func(DATA, k=2)
    assert len(by_position) == len(by_keyword)
    for got, want in zip(by_keyword, by_position, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("func", [ckmeans, breaks])
def test_data_as_keyword(func) -> None:
    by_keyword = func(data=DATA, k=2)
    by_position = func(DATA, 2)
    assert len(by_keyword) == len(by_position)
    for got, want in zip(by_keyword, by_position, strict=True):
        np.testing.assert_array_equal(got, want)
