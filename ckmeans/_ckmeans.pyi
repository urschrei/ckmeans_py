import numpy as np
import numpy.typing as npt

class CkmeansError(ValueError):
    """Raised when the input data or the number of clusters is not valid."""

def ckmeans(data: npt.ArrayLike, k: int, /) -> list[npt.NDArray[np.float64]]:
    """Cluster data into k bins."""

def breaks(data: npt.ArrayLike, k: int, /) -> npt.NDArray[np.float64]:
    """Calculate k - 1 breaks in the data."""
