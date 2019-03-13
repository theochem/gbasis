"""Utility functions for running tests."""
import itertools as it

import numpy as np


def skip_init(class_obj):
    """Return instance of the given class without initialization.

    Parameters
    ----------
    class_obj : type
        Class.

    Returns
    -------
    instance : class_obj
        Instance of the given class without intialization.

    """

    class NoInitClass(class_obj):
        """Class {} without the __init__."""

        def __init__(self):
            """Null initialization."""
            pass

    NoInitClass.__name__ = "NoInit{}".format(class_obj.__name__)
    NoInitClass.__doc__ = NoInitClass.__doc__.format(class_obj.__name__)
    return NoInitClass()


def partial_deriv_finite_diff(func, x, order, epsilon=1e-8, num_points=1):
    """Return the first order partial derivative of the given function at the given value.

    Parameters
    ----------
    func : func
        Function to derivatize with respect to each variable in the input.
    x : np.ndarray(K,)
        Input to the function.
    order : K-list of int
        Order of the differentiation with respect to each axis/variable.
    epsilon : float
        Step size between any two points in any two dimensions.
        Note that any two adjacent points along one dimension are equally spaced apart.
        Too small a step size will cause problems for higher order derivatization. Too large of a
        step size will decrease accuracy.
    num_points : int
        Number of points used to construct the neighbourhood of the given point in each direction in
        each dimension.
        For higher order derivatives, you may need to provide more points for greater accuracy.

    Returns
    -------
    partial_deriv : K-tuple of floats
        First-order partial derivative of the function with respect to a variable in the input,
        evaluated at the given input.

    Raises
    ------
    ValueError
        If the given order is not the same shape

    """
    order = np.array(order)

    step = np.arange(-num_points, num_points + 1) * epsilon
    samples = np.array([func(x + np.array(steps)) for steps in it.product(*[step] * x.size)])
    samples = samples.reshape(*[step.size] * x.size)

    while np.sum(order) > 0:
        index = np.nonzero(order)[0][0]
        samples = np.gradient(samples, epsilon)[index]
        order[index] -= 1

    return samples[tuple(num_points * np.ones(x.size, dtype=int))]


def test_finite_diff():
    """Test finite_diff."""

    def func(x):
        """Test function."""
        return np.sum(np.exp(x))

    x = np.arange(4)
    assert np.allclose(partial_deriv_finite_diff(func, x, [1, 0, 0, 0]), np.exp(0))
    assert np.allclose(partial_deriv_finite_diff(func, x, [0, 1, 0, 0]), np.exp(1))
    assert np.allclose(partial_deriv_finite_diff(func, x, [0, 0, 1, 0]), np.exp(2))
    assert np.allclose(partial_deriv_finite_diff(func, x, [0, 0, 0, 1]), np.exp(3))
    assert np.allclose(partial_deriv_finite_diff(func, x, [1, 1, 0, 0], epsilon=1e-5), 0)
    assert np.allclose(
        partial_deriv_finite_diff(func, x, [0, 2, 0, 0], epsilon=1e-5, num_points=2), np.exp(1)
    )
    assert np.allclose(partial_deriv_finite_diff(func, x, [1, 1, 1, 0], epsilon=1e-3), 0)

    def func(x):
        """Test function."""
        return np.prod(np.exp(x))

    x = np.arange(2)
    test = partial_deriv_finite_diff(func, x, [1, 0])
    assert np.allclose(test, func(x))
    test = partial_deriv_finite_diff(func, x, [0, 1])
    assert np.allclose(test, func(x))
    test = partial_deriv_finite_diff(func, x, [1, 1], epsilon=1e-5)
    assert np.allclose(test, func(x))
    test = partial_deriv_finite_diff(func, x, [2, 0], epsilon=1e-5, num_points=2)
    assert np.allclose(test, func(x))
    test = partial_deriv_finite_diff(func, x, [0, 2], epsilon=1e-5, num_points=2)
    assert np.allclose(test, func(x))
