"""Utility functions for gbasis."""

import numpy as np
import scipy.special


def factorial2(n):
    """Wrap scipy.special.factorial2 to return 1.0 when the input is not positive.

    This is a temporary workaround to address issue #129, while we wait for Scipy's update.
    To learn more, see https://github.com/scipy/scipy/issues/18409.

    Parameters
    ----------
    n : int or np.ndarray
        Values to calculate n!! for. If n <= 0, the return value is 1.
    """
    # Scipy  1.11.x returns an integer when n is an integer, but 1.10.x returns an array,
    # so np.array(n) is passed to make sure the output is always an array.
    out = scipy.special.factorial2(np.array(n))
    out[out <= 0] = 1.0
    return out
