"""Base class for arrays that depend on two contracted Gaussians."""
import abc

from gbasis.base import BaseGaussianRelatedArray
from gbasis.spherical import generate_transformation
import numpy as np


# pylint: disable=W0235
class BaseTwoIndexAsymmetric(BaseGaussianRelatedArray):
    """Base class for constructing arrays with two indices associated with two sets of contractions.

    The first index of the returned array is associated with the first set of contracted Gaussians
    (or some linear combination of them). The second index of the returned array is associated with
    the second set of contracted Gaussians (or some linear combination of them).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Sets of contractions associated with each axis of the array.

    Properties
    ----------
    contractions_one : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first index of the array.
    contractions_two : tuple of ContractedCartesianGaussians
        Contractions that are associated with the second index of the array.

    Methods
    -------
    __init__(self, contractions_one, contractions_two)
        Initialize.
    construct_array_contraction(self, contraction_one, contraction_two, **kwargs) : np.ndarray
        Return the array associated with a `ContractedCartesianGaussians` instance.
    construct_array_cartesian(self, **kwargs) : np.ndarray
        Return the array associated with Cartesian Gaussians.
    construct_array_spherical(self, **kwargs) : np.ndarray
        Return the array associated with spherical Gaussians (atomic orbitals).
    construct_array_spherical_lincomb(self, transform_one, transform_two, **kwargs) : np.ndarray
        Return the array associated with linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).

    """

    def __init__(self, contractions_one, contractions_two):
        """Initialize.

        Parameters
        ----------
        contractions_one : list/tuple of ContractedCartesianGaussians
            Contractions that are associated with the first index of the array.
        contractions_two : list/tuple of ContractedCartesianGaussians
            Contractions that are associated with the second index of the array.

        """
        super().__init__(contractions_one, contractions_two)

    @property
    def contractions_one(self):
        """Contractions that are associated with the first index of the array.

        Returns
        -------
        contractions_one : tuple of ContractedCartesianGaussians
            Contractions that are associated with the first index of the array.

        """
        return self._axes_contractions[0]

    @property
    def contractions_two(self):
        """Contractions that are associated with the second index of the array.

        Returns
        -------
        contractions_two : tuple of ContractedCartesianGaussians
            Contractions that are associated with the second index of the array.

        """
        return self._axes_contractions[1]

    @abc.abstractmethod
    def construct_array_contraction(self, contractions_one, contractions_two, **kwargs):
        """Return the array associated with two sets of contracted Cartesian Gaussians.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array_contraction : np.ndarray
            Array associated with the given instance(s) of ContractedCartesianGaussians.

        Notes
        -----
        The next level of classes will be the construction of specific arrays given a set of
        contracted Cartesian Gaussians. The keyword arguments will be different depending on its
        functionality. You should use explicit keyword arguments when defining this function, rather
        than the arbitrary number of keywords (as is done here).

        """

    def construct_array_cartesian(self, **kwargs):
        """Return the array associated with the given set(s) of contracted Cartesian Gaussians.

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contrations`. See
            `construct_array_contractions` for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray
            Array associated with the given set of contracted Cartesian Gaussians.
            First and second indices of the array is associated with the contracted Cartesian
            Gaussians.

        """
        matrices = [
            np.concatenate(
                [
                    self.construct_array_contraction(cont_one, cont_two, **kwargs)
                    for cont_two in self.contractions_two
                ],
                axis=1,
            )
            for cont_one in self.contractions_one
        ]
        return np.concatenate(matrices, axis=0)

    def construct_array_spherical(self, **kwargs):
        """Return the array associated with two contracted spherical Gaussians (atomic orbitals).

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contrations`. See
            `construct_array_contractions` for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray
            Array associated with the atomic orbitals associated with the given set(s) of contracted
            Cartesian Gaussians.
            First and second indices of the array is associated with two contracted spherical
            Gaussians (atomic orbitals).

        """
        matrices_spherical = []
        for cont_one in self.contractions_one:
            # get transformation from cartesian to spherical for the first index (applied to left)
            transform_one = generate_transformation(
                cont_one.angmom, cont_one.angmom_components, "left"
            )
            matrices_spherical_cols = []
            for cont_two in self.contractions_two:
                # get transformation from cartesian to spherical for the first index (applied to
                # left)
                transform_two = generate_transformation(
                    cont_two.angmom, cont_two.angmom_components, "left"
                )
                # evaluate
                matrix_contraction = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # transform
                matrix_contraction = np.tensordot(transform_one, matrix_contraction, (1, 0))
                matrix_contraction = np.tensordot(transform_two, matrix_contraction, (1, 1))
                matrix_contraction = np.swapaxes(matrix_contraction, 0, 1)
                # store
                matrices_spherical_cols.append(matrix_contraction)
            matrices_spherical.append(np.concatenate(matrices_spherical_cols, axis=1))
        return np.concatenate(matrices_spherical, axis=0)

    def construct_array_spherical_lincomb(self, transform_one, transform_two, **kwargs):
        r"""Return the array associated with linear combinations of spherical Gaussians (LCAO's).

        .. math::

            \sum_{j} T^{one}_{i_1 j} \sum_{k} T^{two}_{i_2 k} M_{jklm...}
            = M^{trans}_{i_1 i_2 lm...}

        Parameters
        ----------
        transform_one : np.ndarray
            Array associated with the linear combinations of spherical Gaussians (LCAO's) associated
            with the first index.
        transform_two : np.ndarray
            Array associated with the linear combinations of spherical Gaussians (LCAO's) associated
            with the second index.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed directly to `construct_array_spherical`, which will
            then pass it down to `construct_array_contrations`. See `construct_array_contractions`
            for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray
            Array associated with the given two sets of contracted Cartesian Gaussians.
            First index of the array corresponds to the linear combination of contracted spherical
            Gaussians associated with the first set of contractions, `contractions_one`.
            Second index of the array corresponds to the linear combination of contracted spherical
            Gaussians associated with the second set of contractions, `contractions_two`.

        """
        array_transformed = self.construct_array_spherical(**kwargs)
        array_transformed = np.tensordot(transform_one, array_transformed, (1, 0))
        array_transformed = np.tensordot(transform_two, array_transformed, (1, 1))
        array_transformed = np.swapaxes(array_transformed, 0, 1)
        return array_transformed
