from ckmeans import ckmeans


def test_ckmeans() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0]
    clusters = 2
    result = ckmeans(data, clusters)
    assert result == [[1.0, 2.0, 3.0, 4.0], [100.0, 101.0, 102.0, 103.0]]
