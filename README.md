# CKmeans: Optimal Univariate Clustering

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimizing the difference within groups – what Wang & Song refer to as `withinss`, or within sum-of-squares – means that groups are optimally homogenous within and the data is split into representative groups. This is very useful for visualization, where one may wish to represent a continuous variable in discrete colour or style groups. This function can provide groups that emphasize differences between data.

Being a dynamic approach, this algorithm is based on two matrices that store incrementally-computed values for squared deviations and backtracking indexes.

Unlike the [original implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html), this implementation does not include any code to automatically determine the optimal number of clusters: this information needs to be explicitly provided. It **does** provide the `roundbreaks` method to aid labelling, however.

## Implementation
This library uses the [`ckmeans`](https://crates.io/crates/ckmeans) Rust crate, by the same author.

# Example
```python
from ckmeans import ckmeans


data = [1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0]
clusters = 2
result = ckmeans(data, clusters)
assert result == [[1.0, 2.0, 3.0, 4.0], [100.0, 101.0, 102.0, 103.0]]
```
