"""Base class for arrays that depend on two contracted Gaussians."""
import abc

from gbasis.base import BaseGaussianRelatedArray
from gbasis.spherical import generate_transformation
import numpy as np


# pylint: disable=W0235
class BaseTwoIndexSymmetric(BaseGaussianRelatedArray):
    """Base class for constructing arrays associated with two contracted Gaussian.

    The first two axes of the returned array is associated with the given set of contracted Gaussian
    (or a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Sets of contractions associated with each axis of the array.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first and second indices of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, contraction, **kwargs) :
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
    construct_array_cartesian(self, **kwargs) : np.ndarray(K_cart, K_cart, ...)
        Return the array associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self, **kwargs) : np.ndarray(K_sph, K_sph, ...)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform, **kwargs) : np.ndarray(K_orbs, K_orbs, ...)
        Return the array associated with linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    def __init__(self, contractions):
        """Initialize.

        Parameters
        ----------
        contractions : list/tuple of ContractedCartesianGaussians
            Contractions that are associated with the first and second indices of the array.

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

    # NOTE: check if output is symmetric when contractions_one = contractions_two?
    @abc.abstractmethod
    def construct_array_contraction(self, contractions_one, contractions_two, **kwargs):
        """Return the array associated with a set of contracted Cartesian Gaussians.

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
            Array associated with the given instances of ContractedCartesianGaussians.
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
        array : np.ndarray(K_cart, K_cart, ...)
            Array associated with the given set of contracted Cartesian Gaussians.
            First and second indices of the array are associated with the contracted Cartesian
            Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

        Notes
        -----
        The blocks along the diagonal, i.e. blocks where the first two axes belong to the same
        set of contractions, are transposed in the process of constructing the whole array.
        It is assumed that the array returned from `construct_array_contraction` is symmetric with
        respect to the swapping of the first and second axes.

        """
        triu_blocks = []
        for i, cont_one in enumerate(self.contractions):
            for cont_two in self.contractions[i:]:
                block = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for i in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for i in block.shape[4:]]
                )
                # assume array always has shape (M_1, L_1, M_2, L_2, ...)
                block = np.concatenate(block, axis=0)
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                block = np.swapaxes(np.swapaxes(block, 0, 1), 1, 2)
                block = np.concatenate(block, axis=0)
                block = np.swapaxes(block, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
                triu_blocks.append(block)
        # use numpy triu and tril indices to create blocks
        num_blocks_side = len(self.contractions)
        all_blocks = np.zeros((num_blocks_side, num_blocks_side), dtype=object)
        all_blocks[np.triu_indices(num_blocks_side)] = triu_blocks
        all_blocks[np.tril_indices(num_blocks_side)] = [
            np.swapaxes(block, 0, 1) for block in all_blocks.T[np.tril_indices(num_blocks_side)]
        ]
        # concatenate
        return np.concatenate(
            [np.concatenate(row_blocks, axis=1) for row_blocks in all_blocks], axis=0
        )

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
        array : np.ndarray(K_sph, K_sph, ...)
            Array associated with the atomic orbitals associated with the given set(s) of contracted
            Cartesian Gaussians.
            First and second indices of the array are associated with two contracted spherical
            Gaussians (atomic orbitals). `K_sph` is the total number of spherical contractions
            within the instance.

        Notes
        -----
        The blocks along the diagonal, i.e. blocks where the first two axes belong to the same
        set of contractions, are transposed in the process of constructing the whole array.
        It is assumed that the array returned from `construct_array_contraction` is symmetric with
        respect to the swapping of the first and second axes.

        """
        triu_blocks = []
        for i, cont_one in enumerate(self.contractions):
            # get transformation from cartesian to spherical (applied to left)
            transform_one = generate_transformation(
                cont_one.angmom, cont_one.angmom_components, "left"
            )
            for cont_two in self.contractions[i:]:
                transform_two = generate_transformation(
                    cont_two.angmom, cont_two.angmom_components, "left"
                )
                # evaluate
                block_sph = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block_sph *= cont_one.norm_cont.reshape(
                    *block_sph.shape[:2], *[1 for i in block_sph.shape[2:]]
                )
                block_sph *= cont_two.norm_cont.reshape(
                    1, 1, *block_sph.shape[2:4], *[1 for i in block_sph.shape[4:]]
                )
                # assume array has shape (M_1, L_1, M_2, L_2, ...)
                # transform
                block_sph = np.tensordot(transform_one, block_sph, (1, 1))
                block_sph = np.concatenate(np.swapaxes(block_sph, 0, 1), axis=0)
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                block_sph = np.tensordot(transform_two, block_sph, (1, 2))
                block_sph = np.swapaxes(np.swapaxes(block_sph, 0, 1), 0, 2)
                block_sph = np.concatenate(block_sph, axis=0)
                block_sph = np.swapaxes(block_sph, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
                # store
                triu_blocks.append(block_sph)
        # use numpy triu and tril indices to create blocks
        num_blocks_side = len(self.contractions)
        all_blocks = np.zeros((num_blocks_side, num_blocks_side), dtype=object)
        all_blocks[np.triu_indices(num_blocks_side)] = triu_blocks
        all_blocks[np.tril_indices(num_blocks_side)] = [
            np.swapaxes(block, 0, 1) for block in all_blocks.T[np.tril_indices(num_blocks_side)]
        ]
        # concatenate
        return np.concatenate(
            [np.concatenate(row_blocks, axis=1) for row_blocks in all_blocks], axis=0
        )

    def construct_array_spherical_lincomb(self, transform, **kwargs):
        r"""Return the array associated with linear combinations of spherical Gaussians (LCAO's).

        .. math::

            \sum_{j} T_{i_1 j} \sum_{k} T_{i_2 k} M_{jklm...} = M^{trans}_{i_1 i_2 lm...}

        Parameters
        ----------
        transform : np.ndarray
            Array associated with the linear combinations of spherical Gaussians (LCAO's) associated
            with the first and second index.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed directly to `construct_array_spherical`, which will
            then pass it down to `construct_array_contraction`. See `construct_array_contraction`
            for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_orbs, K_orbs, ...)
            Array whose first and second indices are associated with the linear combinations of the
            contracted spherical Gaussians.
            First and second indices of the array correspond to the linear combination of contracted
            spherical Gaussians. `K_orbs` is the number of basis functions produced after the linear
            combinations.

        """
        array_transformed = self.construct_array_spherical(**kwargs)
        array_transformed = np.tensordot(transform, array_transformed, (1, 0))
        array_transformed = np.tensordot(transform, array_transformed, (1, 1))
        array_transformed = np.swapaxes(array_transformed, 0, 1)
        return array_transformed

    def construct_array_mix(self, coord_types, **kwargs):
        """Return the array associated with set of Gaussians of the given coordinate systems.

        Parameters
        ----------
        coord_types : list/tuple of str
            Types of the coordinate system for each ContractedCartesianGaussians.
            Each entry must be one of "cartesian" or "spherical".
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.

        """
        raise NotImplementedError
