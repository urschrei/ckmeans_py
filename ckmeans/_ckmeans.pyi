from typing import final

import numpy as np
import numpy.typing as npt

class CkmeansError(ValueError):
    """Raised when the input data or the number of clusters is not valid."""

@final
class OptimalResult:
    """The result of `ckmeans_optimal`."""

    @property
    def k(self) -> int:
        """The chosen number of clusters."""
    @property
    def clusters(self) -> list[npt.NDArray[np.float64]]:
        """The clusters, in ascending order of value."""
    @property
    def centers(self) -> npt.NDArray[np.float64]:
        """The mean of each cluster."""
    @property
    def sizes(self) -> npt.NDArray[np.intp]:
        """The number of values in each cluster."""
    @property
    def withinss(self) -> npt.NDArray[np.float64]:
        """The within-cluster sum of squares of each cluster."""
    @property
    def ks(self) -> npt.NDArray[np.intp]:
        """The candidate numbers of clusters."""
    @property
    def bic(self) -> npt.NDArray[np.float64]:
        """The BIC of each candidate number of clusters."""

def ckmeans(data: npt.ArrayLike, /, k: int) -> list[npt.NDArray[np.float64]]:
    """Cluster data into k groups with the least within-group sum of squares."""

def breaks(data: npt.ArrayLike, /, k: int) -> npt.NDArray[np.float64]:
    """Calculate the breaks between k clusters, for labels and legends."""

def ckmeans_optimal(
    data: npt.ArrayLike, /, k_min: int = 1, k_max: int = 9
) -> OptimalResult:
    """Cluster data with the number of clusters that has the lowest BIC."""
