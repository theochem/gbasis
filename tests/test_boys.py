"""Tests for the Boys function utility."""

import numpy as np
import pytest
from gbasis.boys import boys_function


def test_boys_m0_x0():
    """F_0(0) should equal 1."""
    assert np.isclose(boys_function(0, 0.0), 1.0)


def test_boys_m1_x0():
    """F_1(0) should equal 1/3."""
    assert np.isclose(boys_function(1, 0.0), 1.0 / 3.0)


def test_boys_m2_x0():
    """F_2(0) should equal 1/5."""
    assert np.isclose(boys_function(2, 0.0), 1.0 / 5.0)


def test_boys_m0_large_x():
    """F_0(x) -> sqrt(pi) / (2*sqrt(x)) as x -> infinity."""
    x = 100.0
    expected = np.sqrt(np.pi) / (2.0 * np.sqrt(x))
    assert np.isclose(boys_function(0, x), expected, rtol=1e-4)


def test_boys_array_input():
    """Boys function should handle array input and return correct shape."""
    x = np.array([0.0, 1.0, 5.0, 30.0])
    result = boys_function(0, x)
    assert result.shape == (4,)


def test_boys_positive_values():
    """Boys function should return positive values for non-negative x."""
    x = np.linspace(0.1, 50, 50)
    for m in range(5):
        assert np.all(boys_function(m, x) > 0)


def test_boys_decreasing():
    """F_m(x) should be strictly decreasing in x for fixed m."""
    x = np.linspace(0.1, 10.0, 50)
    for m in range(4):
        vals = boys_function(m, x)
        assert np.all(np.diff(vals) < 0)