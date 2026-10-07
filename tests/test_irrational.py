"""Test irrational module."""

import numpy as np

from fiblat._irrational import golden_powers


def test_golden_powers() -> None:
    """Test that the powers come from the root of x^(count + 1) = x + 1."""
    assert golden_powers(0).size == 0
    assert np.allclose(golden_powers(1), (np.sqrt(5) - 1) / 2)
    for count in [2, 5, 40]:
        powers = golden_powers(count)
        root = 1 / powers[0]
        assert np.isclose(root ** (count + 1), root + 1)
        assert np.allclose(powers, root ** -np.arange(1, count + 1))
