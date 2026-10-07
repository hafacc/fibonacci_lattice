import numba as nb
import numpy as np
from numpy.typing import NDArray

_ROOT_STEPS = 64  # each step shrinks the error at least threefold


@nb.jit(nb.float64[:](nb.int64), cache=True, nogil=True)
def golden_powers(count: int) -> NDArray[np.float64]:  # pragma: nocover
    """Create the first count inverse powers of the root of x^(count + 1) = x + 1."""
    root = 2.0
    for _ in range(_ROOT_STEPS):
        root = (1 + root) ** (1 / (count + 1))
    return root ** -np.arange(1.0, count + 1)
