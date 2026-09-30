import numpy as np

from ckmeans import ckmeans


def test_ckmeans() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    clusters = 2
    result = ckmeans(data, clusters)
    assert list(result[0]) == [1.0, 2.0, 3.0, 4.0]
    assert list(result[1]) == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_clusters_are_contiguous_views() -> None:
    rng = np.random.default_rng(4)
    data = rng.uniform(0.0, 100.0, 1_000)
    result = ckmeans(data, 4)
    assert sum(len(cluster) for cluster in result) == data.size
    np.testing.assert_array_equal(np.concatenate(result), np.sort(data))
    for cluster in result:
        assert cluster.dtype == np.float64
        assert cluster.flags.c_contiguous
        assert not np.shares_memory(cluster, data)
