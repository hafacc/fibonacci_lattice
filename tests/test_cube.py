"""Test cube lattice."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from fiblat import cube_lattice


def test_one_dim() -> None:
    """Test cube with single dimension (line)."""
    one = cube_lattice(1, 4)
    assert np.allclose(one, (np.arange(4)[:, None] + 0.5) / 4)


def test_in_unit_cube(benchmark: BenchmarkFixture) -> None:
    """Test that points are in the unit cube."""
    lattice = benchmark(cube_lattice, 100, 1000)
    assert np.all((0 <= lattice) & (lattice <= 1))


def test_evenly_distributed() -> None:
    """Test that points are roughly evenly distributed."""
    lattice = cube_lattice(27, 100)
    dists = np.linalg.norm(lattice[:, None] - lattice, 2, -1) + 27 * np.eye(100)
    min_dists = dists.min(-1)
    errors = np.sum(min_dists < 1.2)  # noqa: PLR2004
    assert errors < 3  # noqa: PLR2004


def test_fibonacci() -> None:
    """Test two dimensions step by the golden ratio."""
    lattice = cube_lattice(2, 100)
    steps = np.diff(lattice[:, 1]) % 1
    assert np.allclose(steps, (np.sqrt(5) - 1) / 2)


def test_off_faces() -> None:
    """Test no point sits on a face of the cube."""
    lattice = cube_lattice(4, 1000)
    assert np.all((0 < lattice) & (lattice < 1))


def test_invalid_inputs() -> None:
    """Test cube raises for invalid inputs."""
    with pytest.raises(ValueError):
        cube_lattice(0, 2)
    with pytest.raises(ValueError):
        cube_lattice(2, 0)
