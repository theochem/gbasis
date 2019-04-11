"""Base class for arrays that depend on one contracted Gaussian."""
import abc

from gbasis.base import BaseGaussianRelatedArray
from gbasis.spherical import generate_transformation
import numpy as np


# pylint: disable=W0235
class BaseOneIndex(BaseGaussianRelatedArray):
    """Base class for constructing arrays associated with one contracted Gaussian.

    The first dimension (axis 0) of the returned array is associated with a contracted Gaussian (or
    a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Contractions that are associated with each index of the array.
        Each tuple of ContractedCartesianGaussians corresponds to an index of the array.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first index of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, contraction) : np.ndarray(M, L_cart, ...)
        Return the array associated with a `ContractedCartesianGaussians` instance.
        `M` is the number of segmented contractions with the same exponents (and angular momentum).
        `L_cart` is the number of Cartesian contractions for the given angular momentum.
    construct_array_cartesian(self) : np.ndarray(K_cart, ...)
        Return the array associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, ...)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform) : np.ndarray(K_orbs, ...)
        Return the array associated with linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    def __init__(self, contractions):
        """Initialize.

        Parameters
        ----------
        contractions : list/tuple of ContractedCartesianGaussians
            Contractions that are associated with the first index of the array.

        """
        super().__init__(contractions)

    @property
    def contractions(self):
        """Contractions that are associated with the first index of the array.

        Returns
        -------
        contractions : tuple of ContractedCartesianGaussians
            Contractions that are associated with the first index of the array.

        """
        return self._axes_contractions[0]

    @abc.abstractmethod
    def construct_array_contraction(self, contractions, **kwargs):
        """Return the array associated with a set of contracted Cartesian Gaussians.

        Parameters
        ----------
        contractions : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) that will be used to construct an
            array.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array_contraction : np.ndarray(M, L_cart, ...)
            Array associated with the given instance(s) of ContractedCartesianGaussians.
            First index corresponds to segmented contractions within the given generalized
            contraction (same exponents and angular momentum, but different coefficients). `M` is
            the number of segmented contractions with the same exponents (and angular momentum).
            Second index corresponds to angular momentum vector. `L_cart` is the number of Cartesian
            contractions for the given angular momentum.

        Notes
        -----
        The next level of classes will be the construction of specific arrays given a set of
        contracted Cartesian Gaussians. The keyword arguments will be different depending on its
        functionality. You should use explicit keyword arguments when defining this function, rather
        than the arbitrary number of keywords (as is done here).

        The methods `construct_array_cartesian`, `construct_array_spherical`, and
        `construct_array_spherical_lincomb` depend on this function to produce an array whose first
        index corresponds to the contraction (within a generalized contraction) and second index
        corresponds to the angular momentum vector. These other methods **will** fail with little
        warning if the shape of the output is different. Even if there is only one contraction (i.e.
        segmented contraction), the first index must correspond to the contraction. In other words,
        the shape must still be (1, L, N).

        """

    def construct_array_cartesian(self, **kwargs):
        """Return the array associated with the given set of contracted Cartesian Gaussians.

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contraction`. See
            `construct_array_contraction` for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_cart, ...)
            Array associated with the given set of contracted Cartesian Gaussians.
            First index of the array is associated with the contracted Cartesian Gaussian. `K_cart`
            is the total number of Cartesian contractions within the instance.

        """
        matrices = []
        for contraction in self.contractions:
            array = self.construct_array_contraction(contraction, **kwargs)
            # normalize contractions
            norm_cont = contraction.norm_cont
            array *= (norm_cont ** (-0.5)).reshape(*array.shape[:2], *[1 for i in array.shape[2:]])
            # ASSUME array always has shape (M, L, ...)
            if array.shape[0] == 1:
                matrices.append(np.squeeze(array, axis=0))
            else:
                matrices.append(np.concatenate(array, axis=0))
        return np.concatenate(matrices, axis=0)

    def construct_array_spherical(self, **kwargs):
        """Return the array associated with contracted spherical Gaussians (atomic orbitals).

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contraction`. See
            `construct_array_contraction` for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_sph, ...)
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.
            First index of the array is associated with the contracted spherical Gaussian. `K_sph`
            is the total number of Cartesian contractions within the instance.

        """
        matrices_spherical = []
        for cont in self.contractions:
            # get transformation from cartesian to spherical (applied to left)
            transform = generate_transformation(cont.angmom, cont.angmom_components, "left")
            # evaluate the function at the given points
            matrix_contraction = self.construct_array_contraction(cont, **kwargs)
            # transform
            # ASSUME array always has shape (M, L, ...)
            if matrix_contraction.shape[0] == 1:
                matrix_contraction = np.squeeze(matrix_contraction, axis=0)
                matrix_contraction = np.tensordot(transform, matrix_contraction, (1, 0))
            else:
                matrix_contraction = np.tensordot(transform, matrix_contraction, (1, 1))
                matrix_contraction = np.concatenate(np.swapaxes(matrix_contraction, 0, 1), axis=0)
            # store
            matrices_spherical.append(matrix_contraction)

        return np.concatenate(matrices_spherical, axis=0)

    def construct_array_spherical_lincomb(self, transform, **kwargs):
        r"""Return the array associated with linear combinations of spherical Gaussians (LCAO's).

        .. math::

            \sum_{j} T_{i j} M_{jklm...} = M^{trans}_{iklm...}


        Parameters
        ----------
        transform : np.ndarray(K_orbs, K_sph)
            Array associated with the linear combinations of spherical Gaussians (LCAO's).
            Transformation is applied to the left, i.e. the sum is over the second index of
            `transform` and first index of the array for contracted spherical Gaussians.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed directly to `construct_array_spherical`, which will
            then pass it down to `construct_array_contraction`. See `construct_array_contraction`
            for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_orbs, ...)
            Array whose first index is associated with the linear combinations of the contracted
            spherical Gaussians.
            `K_orbs` is the number of basis functions produced after the linear combinations.

        """
        array_spherical = self.construct_array_spherical(**kwargs)
        return np.tensordot(transform, array_spherical, (1, 0))
