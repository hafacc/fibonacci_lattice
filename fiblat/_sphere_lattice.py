import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._cube_lattice import cube_lattice

_MAX_STEPS = 16  # newton steps before settling for the current angle
_ANGLE_TOL = 1e-13  # newton step small enough to stop at
_INTEGRAL_TOL = 1e-15  # integral error small enough to stop at


@nb.jit(
    nb.float64(nb.float64, nb.int64, nb.float64[:], nb.float64[:], nb.float64[:]),
    cache=True,
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def inv_int_sin(
    target: float,
    order: int,
    integrals: NDArray[np.float64],
    densities: NDArray[np.float64],
    cotangents: NDArray[np.float64],
) -> float:  # pragma: no cover
    """Invert the integral of sin(t) ** order dt from 0 to x.

    The arrays hold that integral, `sin(x) ** order`, and `cot(x)` on an even
    grid from 0 to pi / 2.
    """
    cells = integrals.size - 1
    width = np.pi / 2 / cells
    left = min(max(int(np.searchsorted(integrals, target)) - 1, 0), cells - 1)
    lower = left * width
    base = integrals[left]
    rise = integrals[left + 1] - base
    if rise <= 0:
        return lower
    else:
        base_density = densities[left]
        if left == 0:
            base_slope = 1.0 if order == 1 else 0.0
            base_curve = 2.0 if order == 2 else 0.0  # noqa: PLR2004
        else:
            base_cot = cotangents[left]
            base_slope = order * base_density * base_cot
            base_curve = order * base_density * ((order - 1) * base_cot**2 - 1)

        angle = lower + width * min(max((target - base) / rise, 0.0), 1.0)
        for _ in range(_MAX_STEPS):
            sine = np.sin(angle)
            density = sine**order
            if density <= 0:
                break
            cotangent = np.cos(angle) / sine
            slope = order * density * cotangent
            # multiplied in this order so a huge cotangent can't overflow
            curve = order * ((order - 1) * (density * cotangent) * cotangent - density)
            # integrates the quintic matching value, slope and curve at both ends
            offset = angle - lower
            integral = base + offset * (
                (base_density + density) / 2
                + offset
                * ((base_slope - slope) / 10 + offset * (base_curve + curve) / 120)
            )
            error = integral - target
            if abs(error) <= _INTEGRAL_TOL:
                break
            step = error / density
            angle = min(max(angle - step, lower), lower + width)
            if abs(step) <= _ANGLE_TOL:
                break
        return angle


@nb.jit(
    nb.float64[:, :](nb.int64, nb.int64),
    cache=True,
    nogil=True,
    fastmath=True,
    error_model="numpy",
)
def sphere_lattice(
    dim: int, num_points: int
) -> NDArray[np.float64]:  # pragma: no cover
    """Generate num_points points over the dim - 1 dimensional hypersphere.

    Generate a `num_points` length list of `dim`-dimensional tuples such that
    each element has an l2 norm of 1, and their nearest neighbor is roughly
    identical for each point.

    Parameters
    ----------
    dim : the dimension of points to sample, i.e. the length of tuples in the
        returned list
    num_points : the number of points to generate

    Examples
    --------
    >>> sphere_lattice(3, 4)
    array([[ 0.44679483,  0.48772367,  0.75      ],
           [-0.96453846, -0.08464959,  0.25      ],
           [ 0.76840062, -0.58911839, -0.25      ],
           [-0.11521053,  0.65132675, -0.75      ]])
    """
    if dim < 2:  # noqa: PLR2004
        raise ValueError(f"dimension must be greater than one: {dim}")
    elif num_points < 1:
        raise ValueError(f"must request at least one point: {num_points}")
    cube = cube_lattice(dim - 1, num_points)
    cols = dim - 1

    # columns one and up hold angles until the final loop
    output = np.empty((num_points, dim), "f8")
    output[:, 1] = 2 * np.pi * cube[:, 1 % cols]
    if dim > 2:  # noqa: PLR2004
        output[:, 2] = 2 * np.arcsin(np.sqrt(cube[:, 2 % cols]))
    if dim > 3:  # noqa: PLR2004
        # cells must stay much narrower than sin ** order, about 1 / sqrt(order)
        cells = 64 + int(40 * np.sqrt(dim))
        angles = np.linspace(0, np.pi / 2, cells + 1)
        sines = np.sin(angles)
        cosines = np.cos(angles)
        cotangents = np.zeros(cells + 1)
        cotangents[1:] = cosines[1:] / sines[1:]
        densities = sines.copy()
        integrals = 2 * np.sin(angles / 2) ** 2
        others = angles.copy()
        for order in range(2, dim - 1):
            highest = 0.0
            for index in range(cells + 1):
                reduced = (order - 1) * others[index]
                value = (reduced - cosines[index] * densities[index]) / order
                # rounding must not break the ordering the search relies on
                highest = max(highest, value)
                others[index] = highest
                densities[index] *= sines[index]
            integrals, others = others, integrals

            half = integrals[cells]
            for point in range(num_points):
                frac = cube[point, (order + 1) % cols]
                if frac > 0.5:  # noqa: PLR2004
                    mirrored = inv_int_sin(
                        2 * half * (1 - frac), order, integrals, densities, cotangents
                    )
                    output[point, order + 1] = np.pi - mirrored
                else:
                    output[point, order + 1] = inv_int_sin(
                        2 * half * frac, order, integrals, densities, cotangents
                    )

    for point in range(num_points):
        scale = 1.0
        for index in range(dim - 1, 0, -1):
            angle = output[point, index]
            output[point, index] = scale * np.cos(angle)
            scale *= np.sin(angle)
        output[point, 0] = scale
    return output
