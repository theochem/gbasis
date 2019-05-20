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
    construct_array_contraction(self, contractions_one, contractions_two, **kwargs) :
    np.ndarray(M_1, L_cart_1, M_2, L_cart_2, ...)
        Return the array associated with a `ContractedCartesianGaussians` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self, **kwargs) : np.ndarray(K_cart_1, K_cart_2, ...)
        Return the array associated with Cartesian Gaussians.
        `K_cart_1` is the total number of Cartesian contractions within the `contractions_one`.
        `K_cart_2` is the total number of Cartesian contractions within the `contractions_two`.
    construct_array_spherical(self, **kwargs) : np.ndarray(K_sph_1, K_sph_2, ...)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph_1` is the total number of spherical contractions within the `contractions_one`.
        `K_sph_2` is the total number of spherical contractions within the `contractions_two`.
    construct_array_spherical_lincomb(self, transform_one, transform_two, **kwargs) :
    np.ndarray(K_orbs_1, K_orbs_2, ...)
        Return the array associated with linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).
        `K_orbs_1` is the number of basis functions produced after the linear combinations of the
        spherical contractions associated with `contractions_one`.
        `K_orbs_2` is the number of basis functions produced after the linear combinations of the
        spherical contractions associated with `contractions_two`.

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
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, ...)
            Array associated with the given instance(s) of ContractedCartesianGaussians.
            First axis corresponds to the segmented contraction within `contractions_one`. `M_1` is
            the number of segmented contractions with the same exponents (and angular momentum)
            associated with the first index.
            Second axis corresponds to the angular momentum vector of the `contractions_one`.
            `L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Third axis corresponds to the segmented contraction within `contractions_two`. `M_2` is
            the number of segmented contractions with the same exponents (and angular momentum)
            associated with the second index.
            Fourth axis corresponds to the angular momentum vector of the `contractions_two`.
            `L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.

        Notes
        -----
        The next level of classes will be the construction of specific arrays given a set of
        contracted Cartesian Gaussians. The keyword arguments will be different depending on its
        functionality. You should use explicit keyword arguments when defining this function, rather
        than the arbitrary number of keywords (as is done here).

        The methods `construct_array_cartesian`, `construct_array_spherical`, and
        `construct_array_spherical_lincomb` depend on this function to produce an array whose first
        and second indices correspond to the contraction (within a generalized contraction) and
        the angular momentum vector of `contractions_one`, and third and fourth indices correspond
        to the contraction (within a generalized contraction) and the angular momentum vector of
        `contractions_two`,. These other methods **will** fail with little warning if the shape of
        the output is different. Even if both `contractions_one` and `contractions_two` are
        segmented contractions, the first and third indices must correspond to the contraction.
        In other words, the shape must still be (1, L_1, 1, L_2).

        """

    def construct_array_cartesian(self, **kwargs):
        """Return the array associated with the given set(s) of contracted Cartesian Gaussians.

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contraction`. See
            `construct_array_contraction` for details on the keyword arguments.

        Returns
        -------
        array :  np.ndarray(K_cart_1, K_cart_2, ...)
            Array associated with the given set of contracted Cartesian Gaussians.
            First index corresponds to the Cartesian contraction within the `contractions_one`.
            `K_cart_1` is the total number of Cartesian contractions within the `contractions_one`.
            Second index corresponds to the Cartesian contraction within the `contractions_two`.
            `K_cart_2` is the total number of Cartesian contractions within the `contractions_two`.

        """
        matrices = []
        for cont_one in self.contractions_one:
            matrices_cols = []
            for cont_two in self.contractions_two:
                block = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for i in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for i in block.shape[4:]]
                )
                # assume array always has shape (M_1, L_1, M_2, L_2, ...)
                if block.shape[0] == 1:
                    block = np.squeeze(block, axis=0)
                else:
                    block = np.concatenate(block, axis=0)
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                if block.shape[1] == 1:
                    block = np.squeeze(block, axis=1)
                else:
                    block = np.swapaxes(np.swapaxes(block, 0, 1), 1, 2)
                    block = np.concatenate(block, axis=0)
                    block = np.swapaxes(block, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
                matrices_cols.append(block)
            matrices.append(np.concatenate(matrices_cols, axis=1))
        return np.concatenate(matrices, axis=0)

    def construct_array_spherical(self, **kwargs):
        """Return the array associated with two contracted spherical Gaussians (atomic orbitals).

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contraction`. See
            `construct_array_contraction` for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_sph_1, K_sph_2, ...)
            Array associated with the atomic orbitals associated with the given set(s) of contracted
            Cartesian Gaussians.
            First index corresponds to the spherical contraction within the `contractions_one`.
            `K_sph_1` is the total number of spherical contractions within the `contractions_one`.
            Second index corresponds to the spherical contraction within the `contractions_two`.
            `K_sph_2` is the total number of spherical contractions within the `contractions_two`.

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
                # normalize contractions
                matrix_contraction *= cont_one.norm_cont.reshape(
                    *matrix_contraction.shape[:2], *[1 for i in matrix_contraction.shape[2:]]
                )
                matrix_contraction *= cont_two.norm_cont.reshape(
                    1, 1, *matrix_contraction.shape[2:4], *[1 for i in matrix_contraction.shape[4:]]
                )
                # transform
                # assume array always has shape (M_1, L_1, M_2, L_2, ...)
                if matrix_contraction.shape[0] == 1:
                    matrix_contraction = np.squeeze(matrix_contraction, axis=0)
                    matrix_contraction = np.tensordot(transform_one, matrix_contraction, (1, 0))
                else:
                    matrix_contraction = np.tensordot(transform_one, matrix_contraction, (1, 1))
                    matrix_contraction = np.concatenate(
                        np.swapaxes(matrix_contraction, 0, 1), axis=0
                    )
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                if matrix_contraction.shape[1] == 1:
                    matrix_contraction = np.squeeze(matrix_contraction, axis=1)
                    matrix_contraction = np.tensordot(transform_two, matrix_contraction, (1, 1))
                    matrix_contraction = np.swapaxes(matrix_contraction, 0, 1)
                else:
                    matrix_contraction = np.tensordot(transform_two, matrix_contraction, (1, 2))
                    matrix_contraction = np.swapaxes(np.swapaxes(matrix_contraction, 0, 1), 0, 2)
                    matrix_contraction = np.concatenate(matrix_contraction, axis=0)
                    matrix_contraction = np.swapaxes(matrix_contraction, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
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
            then pass it down to `construct_array_contraction`. See `construct_array_contraction`
            for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray
            Array associated with the given two sets of contracted Cartesian Gaussians.
            First index of the array corresponds to the linear combination of contracted spherical
            Gaussians associated with the first set of contractions, `contractions_one`. `K_orbs_1`
            is the number of basis functions produced after the linear combinations of the spherical
            contractions associated with `contractions_one`.
            Second index of the array corresponds to the linear combination of contracted spherical
            Gaussians associated with the second set of contractions, `contractions_two`. `K_orbs_2`
            is the number of basis functions produced after the linear combinations of the spherical
            contractions associated with `contractions_two`.

        """
        array_transformed = self.construct_array_spherical(**kwargs)
        array_transformed = np.tensordot(transform_one, array_transformed, (1, 0))
        array_transformed = np.tensordot(transform_two, array_transformed, (1, 1))
        array_transformed = np.swapaxes(array_transformed, 0, 1)
        return array_transformed
