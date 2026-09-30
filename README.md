# CKmeans: Optimal Univariate Clustering

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimising the difference within groups – what Wang & Song refer to as `withinss`, or within sum-of-squares – means that groups are optimally homogeneous within and the data is split into representative groups. This is useful for visualisation, where one may wish to represent a continuous variable in discrete colour or style groups.

Being a dynamic approach, this algorithm is based on two matrices that store incrementally-computed values for squared deviations and backtracking indexes.

If you do not know the number of clusters, `ckmeans_optimal` chooses it with the Bayesian Information Criterion (BIC), following Song & Zhong (2020).

## Implementation
This library uses the [`ckmeans`](https://crates.io/crates/ckmeans) Rust crate, by the same author.

All functions accept any one-dimensional array-like input (a NumPy array of any numeric dtype, a list or a tuple), and convert it to `float64`. They release the GIL during the calculation.

### `ckmeans(data, k)`
Cluster `data` into `k` groups with the least within-group sum of squares. Returns a list of `float64` arrays, one for each cluster, in ascending order of value. Each cluster is sorted.

If `data` has fewer than `k` distinct values, there is one cluster for each distinct value.

### `breaks(data, k)`
Calculate the breaks between `k` clusters, for labels and legends. Returns one break fewer than the number of clusters.

The lower bounds of the clusters from `ckmeans` can have many decimal places, which makes them unsuitable for a legend. Rounding them can be too loose (spurious decimal places) or too strict (classes ranging "from `x` to `x`"). Each break `b` is instead the roundest number between the highest value of a cluster (`last`) and the lowest value of the next cluster (`first`), so that `last < b <= first`. It is a multiple of the largest power of ten that has a multiple in that interval; of those multiples, it is the one nearest the midpoint.

This method is based on the [visionscarto](https://observablehq.com/@visionscarto/natural-breaks#round) method of the same name.

### `ckmeans_optimal(data, k_min=1, k_max=9)`
Find the best number of clusters for `data`, and cluster the data. Use this function when you do not know how many clusters the data has.

The function clusters `data` once for each number of clusters `k` from `k_min` to `k_max`. It then chooses the `k` with the lowest Bayesian Information Criterion (BIC), which balances how closely the clusters fit the data against the number of clusters. `k_max` is capped at the number of distinct values in `data`.

The default range of 1 to 9 is the same as in the R package [Ckmeans.1d.dp](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html). If the data can have more than 9 clusters, set a higher `k_max` (up to 255).

Returns an `OptimalResult`, with these attributes:

| Attribute | Type | Content |
| --- | --- | --- |
| `k` | `int` | The chosen number of clusters |
| `clusters` | `list` of `float64` arrays | The clusters, as returned by `ckmeans` |
| `centers` | `float64` array | The mean of each cluster |
| `sizes` | `intp` array | The number of values in each cluster |
| `withinss` | `float64` array | The within-cluster sum of squares of each cluster |
| `ks` | `intp` array | The candidate numbers of clusters |
| `bic` | `float64` array | The BIC of each candidate number of clusters |

### Errors
All functions raise `CkmeansError`, a subclass of `ValueError`, if:

- `k` (or `k_min`) is less than 1
- `k`, `k_min` or `k_max` is greater than 255
- `k` is greater than the number of values, or `k_min` is greater than the number of distinct values
- `k_min` is greater than `k_max`
- `data` contains NaN

### Numerical limits
- Infinite values are accepted, but a cluster that contains one has no finite sum of squares. The result is then a partition of the input with no optimality guarantee. `ckmeans_optimal` returns NaN BIC values for such input and uses `k_min`.
- The costs are calculated in `float64` from cumulative sums. If the data spans a very large range (for example, values that differ by 1 alongside values of order 1e8), cost differences below `float64` resolution are lost. The result can then be sub-optimal by that amount, and equal values can be put in adjacent clusters. `breaks` then returns the first value of the upper cluster as the break.

# Upgrading from 0.2
- Invalid input raises `CkmeansError`, a subclass of `ValueError`. Version 0.2 raised `RuntimeError`, or `PanicException` for NaN input, `k` greater than 255 and non-contiguous arrays.
- `breaks` uses the method of version 2.0 of the crate, and can return different values. For example, `breaks([1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0], 2)` returns `[100.0]`, not `[50.0]`.
- The clusters from `ckmeans` are views of one sorted copy of the input.
- `data` and `k` can be given as keyword arguments.
- Python 3.11 or later is required.

# Install for **local** development
1. Ensure that `maturin` and a recent Rust toolchain are installed
2. `maturin develop --release` (you will need to re-run this command if you're hacking on the source and want to e.g. benchmark your changes). `uv run` rebuilds the extension when the Rust sources change.

## Benchmarks
Benchmarks can be run using `uv run pytest --benchmark-only`.

The benchmarks compare this Rust implementation against [ckmeans-1d-dp](https://pypi.org/project/ckmeans-1d-dp/) (C++ implementation) across different data sizes and cluster counts:
- 110k samples with 5 or 20 clusters
- 1M samples with 5 or 20 clusters

Results are grouped for meaningful comparison between implementations with identical parameters. Benchmark results also generate histogram visualisations saved as SVG files.

### Results
`ckmeans` is around 10 % faster than `ckmeans.1d.dp`, and around 20 % faster than the [original R package](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html) that wraps the same CPP library `ckmeans.1d.dp`.

Note: The ckmeans-1d-dp Python package only returns _indices_ identifying each cluster to which the input belongs. If you want to cluster your data, you need to do that yourself.

# Examples
```python
import numpy as np
from ckmeans import ckmeans

data = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0])
result = ckmeans(data, 2)
np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0, 4.0])
np.testing.assert_array_equal(result[1], [100.0, 101.0, 102.0, 103.0])
```

```python
from ckmeans import breaks

result = breaks([0.12, 0.13, 0.21, 0.24, 0.87, 0.91], k=2)
assert result.tolist() == [0.6]
```

```python
from ckmeans import ckmeans_optimal

data = [1.0, 1.1, 1.2, 50.0, 51.0, 52.0, 100.0, 101.0, 102.0]
result = ckmeans_optimal(data)
assert result.k == 3
assert result.sizes.tolist() == [3, 3, 3]
```

# License
[Blue Oak Model License 1.0.0](license.txt)
