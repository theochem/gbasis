"""Linear combinations of evaluations of contractions and its derivatives."""
import itertools as it

from gbasis.contractions import ContractedCartesianGaussians
import numpy as np


def lincomb_blocks_evals(contractions, eval_func, trans_blocks):
    """Return linear transformation of the evaluations of the given contractions.

    Parameters
    ----------
    contractions : list/tuple of ContractedCartesianGaussians
        Contractions that will be used to generate the function evaluations.
    eval_func : function
        Function for evaluating the Cartesian contraction or its derivative.
        Input of the function is a ContractedCartesianGaussians instance.
        Output of the function is an array whose first index corresponds to the contractions of the
        given ContractedCartesianGaussians instance.
    trans_blocks : iterable of np.ndarray
        Transformations of each set of contractions that corresponds to a
        ContractedCartesianGaussians instance.
        Each transformation matrix applies from the left, i.e. the number of columns of the
        transformation matrix should match the number of contractions for the corresponding
        ContractedCartesianGaussians instance.

    Returns
    -------
    func_output : np.ndarray
        Output of the function.
        Output has the same shape as the output of a `eval_func` call past the 0th dimension. The
        0th dimension has the same size as the total number of contractions.

    Raises
    ------
    TypeError
        If `contractions` is not a list/tuple of ContractedCartesianGaussians instances.
        If `trans_blocks` is not an interable of np.ndarray instances.
    ValueError
        If the number of elements in `contractions` and `trans_blocks` are not equal.
        If a ContractedCartesianGaussians instance in `contractions` does not have the same number
        of contractions as there are columns in the corresponding transformation matrix.
        If the output of `eval_func` for all of the ContractedCartesianGaussians instances do not
        have the same shape past the first dimension, i.e. `output.shape[1:]`.

    """
    if not isinstance(contractions, (list, tuple)):
        raise TypeError(
            "Argument `contractions` must be given as a list or tuple of "
            "ContractedCartesianGaussians instance"
        )
    if not hasattr(trans_blocks, "__iter__"):
        raise TypeError("Argument `trans_blocks` must be given as an iterable of numpy arrays.")

    for contraction in contractions:
        if not isinstance(contraction, ContractedCartesianGaussians):
            raise TypeError(
                "Given contractions must be instances of the ContractedCartesianGaussians class."
            )
    test_trans_blocks, trans_blocks = it.tee(trans_blocks)
    for i, block in enumerate(test_trans_blocks):
        if not (isinstance(block, np.ndarray) and block.ndim == 2):
            raise TypeError("Each transformation matrix must be a two-dimensional numpy array.")
        try:
            if contractions[i].num_contr != block.shape[1]:
                raise ValueError(
                    "The {0}th ContractedCartesianGaussians instance of the `contractions` must"
                    " have the same number of contractions as there are columns in the {0}th "
                    "numpy array of `trans_blocks`".format(i)
                )
        except IndexError:
            raise ValueError("Number of contractions and transformation blocks must be the same.")
    if len(contractions) != i + 1:  # pylint: disable=W0631
        raise ValueError("Number of contractions and transformation blocks must be the same.")

    # FIXME: einsum not used efficiently
    matrices = [
        np.einsum("ij,j...->i...", block, eval_func(contraction))
        for contraction, block in zip(contractions, trans_blocks)
    ]
    if not all(matrix.shape[1:] == matrices[0].shape[1:] for matrix in matrices):
        raise ValueError(
            "The output of `eval_func` for all of the ContractedCartesianGaussians instances must "
            "have the same shape past the first dimension, i.e. `output.shape[1:]`."
        )
    return np.concatenate(matrices, axis=0)
