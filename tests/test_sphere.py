"""Test sphere lattice."""

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest_benchmark.fixture import BenchmarkFixture

from fiblat import cube_lattice, sphere_lattice


def test_on_unit_sphere(benchmark: BenchmarkFixture) -> None:
    """Test points are on the unit sphere."""
    lattice = benchmark(sphere_lattice, 27, 1000)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


def test_on_unit_sphere_large(benchmark: BenchmarkFixture) -> None:
    """Test high dimensional points are on the unit sphere."""
    lattice = benchmark(sphere_lattice, 600, 3)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


@pytest.mark.long
def test_on_large_case(benchmark: BenchmarkFixture) -> None:
    """Test many high dimensional points on the unit sphere."""
    lattice = benchmark(sphere_lattice, 300, 30_000)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


def test_evenly_distributed() -> None:
    """Test that points are close to evenly distributed."""
    lattice = sphere_lattice(27, 100)
    dists = 1 - (lattice @ lattice.T) + 2 * np.eye(100)
    min_dists = dists.min(-1)
    errors = np.sum(min_dists < 0.3)  # noqa: PLR2004
    assert errors < 9  # noqa: PLR2004


def int_sin(angles: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """Compute the integral of sin(t) ** order dt from 0 to each angle."""
    cosines = np.cos(angles)
    sines = np.sin(angles)
    start = order % 2
    result = angles if start == 0 else 2 * np.sin(angles / 2) ** 2
    powers = sines if start == 0 else sines**2
    for num in range(start + 1, order, 2):
        result = (num * result - cosines * powers) / (num + 1)
        powers = powers * sines**2
    return result


@pytest.mark.parametrize("dim, num_points", [(3, 1000), (4, 1000), (27, 100), (400, 5)])
def test_angles_follow_cube(dim: int, num_points: int) -> None:
    """Test each angle sits at the cube's fraction of its sine power integral."""
    lattice = sphere_lattice(dim, num_points)
    cube = cube_lattice(dim - 1, num_points)
    for col in range(2, dim):
        angles = np.arctan2(np.linalg.norm(lattice[:, :col], axis=1), lattice[:, col])
        fracs = int_sin(angles, col - 1) / int_sin(np.array([np.pi]), col - 1)
        assert np.allclose(fracs, cube[:, col % (dim - 1)], rtol=0, atol=1e-12)


def test_fibonacci() -> None:
    """Test three dimensions give the classic Fibonacci sphere."""
    num_points = 100
    lattice = sphere_lattice(3, num_points)
    heights = 1 - (2 * np.arange(num_points) + 1) / num_points
    assert np.allclose(lattice[:, 2], heights)
    turns = np.arctan2(lattice[:, 0], lattice[:, 1]) / (2 * np.pi)
    assert np.allclose(np.diff(turns) % 1, (np.sqrt(5) - 1) / 2)


@pytest.mark.parametrize("ratio", [2, 3, 5])
def test_multiples_unrelated(ratio: int) -> None:
    """Test a point isn't pulled toward the points at multiples of its index."""
    lattice = sphere_lattice(50, 3000)
    base = np.arange(1, 3000 // ratio)
    dots = np.sum(lattice[base] * lattice[ratio * base], axis=1)
    assert abs(dots.mean()) < 0.05  # noqa: PLR2004


def test_invalid_inputs() -> None:
    """Test sphere raises for invalid inputs."""
    with pytest.raises(ValueError):
        sphere_lattice(1, 2)
    with pytest.raises(ValueError):
        sphere_lattice(2, 0)
