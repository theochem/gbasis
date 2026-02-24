import numpy as np

from .overlap_n import arbitrary_order_overlap


def compute_intracule(shells):

    n = len(shells)

    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            tensor = arbitrary_order_overlap(
                [shells[i], shells[j]]
            )

            # extract scalar value
            value = tensor.data[0] if tensor.nnz > 0 else 0.0

            result[i, j] = value

    return result


def compute_extracule(shells):

    n = len(shells)

    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            tensor = arbitrary_order_overlap(
                [shells[i], shells[j]]
            )

            value = tensor.data[0] if tensor.nnz > 0 else 0.0

            result[i, j] = value

    return result