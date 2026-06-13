import numpy as np
import numpy.typing as npt

def ckmeans(data: npt.NDArray[np.float64], k: int, /) -> list[npt.NDArray[np.float64]]:
    """Cluster data into k bins."""

def breaks(data: npt.NDArray[np.float64], k: int, /) -> npt.NDArray[np.float64]:
    """Calculate k - 1 breaks in the data."""
