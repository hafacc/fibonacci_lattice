import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._irrational import golden_powers

_GOLDEN = (np.sqrt(5) - 1) / 2  # golden ratio minus one


@nb.jit(
    nb.float64[:, :](nb.int64, nb.int64),
    cache=True,
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def cube_lattice(dim: int, num_points: int) -> NDArray[np.float64]:  # pragma: nocover
    """Generate num_points points over the dim dimensional cube.

    Generates `num_points` points spread roughly evenly over `[0, 1]^dim`.

    Parameters
    ----------
    dim : dimension of cube to generate points in
    num_points : the number of points to generate

    Examples
    --------
    >>> cube_lattice(2, 4)
    array([[0.125     , 0.11803399],
           [0.375     , 0.73606798],
           [0.625     , 0.35410197],
           [0.875     , 0.97213595]])
    """
    if dim < 1:
        raise ValueError(f"dimension must be greater than zero: {dim}")
    elif num_points < 1:
        raise ValueError(f"must request at least one point: {num_points}")
    mults = np.concatenate((np.full(1, 1 / num_points), golden_powers(dim - 1)))
    offsets = (0.5 + _GOLDEN * np.arange(dim)) % 1.0
    offsets[0] = 0.5 / num_points
    return (offsets + mults * np.arange(num_points)[:, None]) % 1.0  # type: ignore[return-value]
