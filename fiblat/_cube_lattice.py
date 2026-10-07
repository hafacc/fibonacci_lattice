import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._irrational import n_primes


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
    """
    if dim < 1:
        raise ValueError(f"dimension must be greater than zero: {dim}")
    elif num_points < 1:
        raise ValueError(f"must request at least one point: {num_points}")
    rest = np.sqrt(n_primes(dim - 1))
    mults = np.concatenate((np.full(1, 1 / num_points), rest))
    return (mults * np.arange(num_points)[:, None]) % 1.0  # type: ignore[return-value]
