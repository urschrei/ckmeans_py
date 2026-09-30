import numpy as np
import numpy.typing as npt

class CkmeansError(ValueError):
    """Raised when the input data or the number of clusters is not valid."""

def ckmeans(data: npt.ArrayLike, /, k: int) -> list[npt.NDArray[np.float64]]:
    """Cluster data into k groups with the least within-group sum of squares."""

def breaks(data: npt.ArrayLike, /, k: int) -> npt.NDArray[np.float64]:
    """Calculate the breaks between k clusters, for labels and legends."""
