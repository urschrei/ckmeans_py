# CKmeans: Optimal Univariate Clustering

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimizing the difference within groups – what Wang & Song refer to as `withinss`, or within sum-of-squares – means that groups are optimally homogenous within and the data is split into representative groups. This is very useful for visualization, where one may wish to represent a continuous variable in discrete colour or style groups. This function can provide groups that emphasize differences between data.

Being a dynamic approach, this algorithm is based on two matrices that store incrementally-computed values for squared deviations and backtracking indexes.

Unlike the [original implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html), this implementation does not include any code to automatically determine the optimal number of clusters: this information needs to be explicitly provided. It **does** provide the `roundbreaks` method to aid labelling, however.

## Implementation
This library uses the [`ckmeans`](https://crates.io/crates/ckmeans) Rust crate, by the same author, implementing the `ckmeans` and `breaks` methods.

### `ckmeans(data, k)`
Cluster data into `k` bins

Minimizing the difference within groups – what Wang & Song refer to as `withinss`,
or within sum-of-squares, means that groups are optimally homogenous within groups and the data are
split into representative groups. This is very useful for visualization, where one may wish to
represent a continuous variable in discrete colour or style groups. This function can provide
groups – or “classes” – that emphasize differences between data.


### `breaks(data, k)`
Calculate `k - 1` breaks in the data, distinguishing classes for labelling or visualisation

The boundaries of the classes returned by `ckmeans` are “ugly” in the sense that the values
returned are the lower bound of each cluster, which aren't always practical for labelling, since they
may have many decimal places. To create a legend, the values should be rounded — however the
rounding might be either too loose (and would thus result in spurious decimal places), or too
strict, resulting in classes ranging “from `x` to `x`”. A better approach is to choose the roundest
number that separates the lowest point from a class from the highest point in the preceding
class — thus giving just enough precision to distinguish the classes.
This function is closer to what Jenks returns: `k - 1` “breaks” in the data, useful for labelling.

This method is a port of the [visionscarto](https://observablehq.com/@visionscarto/natural-breaks#round) method of the same name.

# Install for **local** development
1. Ensure that `maturin` and a recent Rust toolchain are installed
2. `maturin develop --release` (you will need to re-run this command if you're hacking on the source and want to e.g. benchmark your changes)

## Benchmarks
Benchmarks can be run using `uv run pytest --benchmark-only`.

The benchmarks compare this Rust implementation against [ckmeans-1d-dp](https://pypi.org/project/ckmeans-1d-dp/) (C++ implementation) across different data sizes and cluster counts:
- 110k samples with 5 or 20 clusters
- 1M samples with 5 or 20 clusters

Results are grouped for meaningful comparison between implementations with identical parameters. Benchmark results also generate histogram visualisations saved as SVG files.

### Results
`ckmeans` is around 10 % faster than `ckmeans.1d.dp`, and around 20 % faster than the [original R package](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html) that wraps the same CPP library `ckmeans.1d.dp`.

Note: The ckmeans-1d-dp Python package only returns _indices_ identifying each cluster to which the input belongs; if you actually want to cluster your data you need to do that yourself.

# Examples
```python
from ckmeans import ckmeans
import numpy as np


data = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0])
clusters = 2
result = ckmeans(data, clusters)
assert result == [
    np.array([1.0, 2.0, 3.0, 4.0]),
    np.array([100.0, 101.0, 102.0, 103.0])
]
```

```python
from ckmeans import breaks
import numpy as np


data = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0])
clusters = 2
result = breaks(data, clusters)
assert result == [50.0,]
```
# License
[Blue Oak Model License 1.0.0](license.txt)
