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
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first and second indices of the array.
        Property of `BaseTwoIndexSymmetric`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, contraction, **kwargs) :
    np.ndarray(M_1, L_cart_1, M_2, L_cart_2, ...)
        Return the array associated with a `GeneralizedContractionShell` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self, **kwargs) : np.ndarray(K_cart, K_cart, ...)
        Return the array associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self, **kwargs) : np.ndarray(K_sph, K_sph, ...)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_lincomb(self, transform, coord_type) : np.ndarray(K_orbs, K_orbs, ...)
        Return the array associated with linear combinations of contractions in the given coordinate
        system.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    def __init__(self, contractions):
        """Initialize.

        Parameters
        ----------
        contractions : list/tuple of GeneralizedContractionShell
            Contractions that are associated with the first and second indices of the array.

        """
        super().__init__(contractions)

    @property
    def contractions(self):
        """Contractions that are associated with the first two indices of the array.

        Returns
        -------
        contractions : tuple of GeneralizedContractionShell
            Contractions that are associated with the first two indices of the array.

        """
        return self._axes_contractions[0]

    # NOTE: check if output is symmetric when contractions_one = contractions_two?
    @abc.abstractmethod
    def construct_array_contraction(self, contractions_one, contractions_two, **kwargs):
        """Return the array associated with a set of contracted Cartesian Gaussians.

        Parameters
        ----------
        contractions_one : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, ...)
            Array associated with the given instances of GeneralizedContractionShell.
            Dimension 0 corresponds to the segmented contraction within `contractions_one`.
            `M_1` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the first index.
            Dimension 1 corresponds to the angular momentum vector of the `contractions_one`.
            `L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Dimension 2 corresponds to the segmented contraction within `contractions_two`.
            `M_2` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the second index.
            Dimension 3 corresponds to the angular momentum vector of the `contractions_two`.
            `L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.

        Notes
        -----
        The next level of classes will be the construction of specific arrays given a set of
        contracted Cartesian Gaussians. The keyword arguments will be different depending on its
        functionality. You should use explicit keyword arguments when defining this function, rather
        than the arbitrary number of keywords (as is done here).

        The methods `construct_array_cartesian`, `construct_array_spherical`, and
        `construct_array_lincomb` depend on this function to produce an array with dimensions 0, 1
        corresponding to the contraction (within a generalized contraction) and the angular momentum
        vector of `contractions_one`, and dimensions 2, 3 corresponding to the contraction (within a
        generalized contraction) and the angular momentum vector of `contractions_two`,.
        These other methods **will** fail with little warning if the shape of
        the output is different. Even if both `contractions_one` and `contractions_two` are
        segmented contractions, the dimensions 0 and 2 must correspond to the contraction.
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
            Dimensions 0 and 1 of the array are associated with the contracted Cartesian Gaussians.
            `K_cart` is the total number of Cartesian contractions within the instance.

        Notes
        -----
        The blocks along the diagonal, i.e. blocks where the first two axes belong to the same
        set of contractions, are transposed in the process of constructing the whole array.
        It is assumed that the array returned from `construct_array_contraction` is symmetric with
        respect to the swapping of the first two axes.

        """
        triu_blocks = []
        for i, cont_one in enumerate(self.contractions):
            for cont_two in self.contractions[i:]:
                block = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for _ in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for _ in block.shape[4:]]
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
            Dimensions 0 and 1 are associated with two contracted spherical Gaussians (atomic
            orbitals). `K_sph` is the total number of spherical contractions within the
            instance.

        Notes
        -----
        The blocks along the diagonal, i.e. blocks where the first two axes belong to the same
        set of contractions, are transposed in the process of constructing the whole array.
        It is assumed that the array returned from `construct_array_contraction` is symmetric with
        respect to the swapping of the first two axes.

        """
        triu_blocks = []
        for i, cont_one in enumerate(self.contractions):
            # get transformation from cartesian to spherical (applied to left)
            transform_one = generate_transformation(
                cont_one.angmom,
                cont_one.angmom_components_cart,
                cont_one.angmom_components_sph,
                "left",
            )
            for cont_two in self.contractions[i:]:
                transform_two = generate_transformation(
                    cont_two.angmom,
                    cont_two.angmom_components_cart,
                    cont_two.angmom_components_sph,
                    "left",
                )
                # evaluate
                block_sph = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block_sph *= cont_one.norm_cont.reshape(
                    *block_sph.shape[:2], *[1 for _ in block_sph.shape[2:]]
                )
                block_sph *= cont_two.norm_cont.reshape(
                    1, 1, *block_sph.shape[2:4], *[1 for _ in block_sph.shape[4:]]
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

    def construct_array_mix(self, coord_types, **kwargs):
        """Return the array associated with a set of Gaussians of the given coordinate systems.

        Parameters
        ----------
        coord_types : list/tuple of str
            Types of the coordinate system for each `GeneralizedContractionShell`.
            Each entry must be one of "cartesian" or "spherical".
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray(K_cont, K_cont, ...)
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.
            Dimensions 0 and 1 are associated with two contractions in the given coordinate system.
            `K_cont` is the total number of contractions within the instance.

        Raises
        ------
        TypeError
            If `coord_types` is not a list/tuple.
        ValueError
            If `coord_types` has an entry that is not "cartesian" or "spherical".
            If `coord_types` has different number of entries as the number of
            `GeneralizedContractionShell` (`contractions`) in instance.

        """
        if not isinstance(coord_types, (list, tuple)):
            raise TypeError("`coord_types` must be a list or a tuple.")
        if not all(i in ["cartesian", "spherical"] for i in coord_types):
            raise ValueError(
                "Each entry of `coord_types` must be one of 'cartesian' or 'spherical'."
            )
        if len(coord_types) != len(self.contractions):
            raise ValueError(
                "`coord_types` must have the same number of entries as the number of "
                "`GeneralizedContractionShell` in the instance."
            )

        triu_blocks = []
        for i, (cont_one, type_one) in enumerate(zip(self.contractions, coord_types)):
            # get transformation from cartesian to spherical (applied to left)
            transform_one = generate_transformation(
                cont_one.angmom,
                cont_one.angmom_components_cart,
                cont_one.angmom_components_sph,
                "left",
            )
            for cont_two, type_two in zip(self.contractions[i:], coord_types[i:]):
                transform_two = generate_transformation(
                    cont_two.angmom,
                    cont_two.angmom_components_cart,
                    cont_two.angmom_components_sph,
                    "left",
                )
                # evaluate
                block = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for _ in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for _ in block.shape[4:]]
                )
                # assume array has shape (M_1, L_1, M_2, L_2, ...)
                if type_one == "spherical":
                    # transform
                    block = np.tensordot(transform_one, block, (1, 1))
                    block = np.swapaxes(block, 0, 1)
                block = np.concatenate(block, axis=0)
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                if type_two == "spherical":
                    block = np.tensordot(transform_two, block, (1, 2))
                    block = np.swapaxes(np.swapaxes(block, 0, 1), 0, 2)
                else:
                    block = np.swapaxes(np.swapaxes(block, 0, 1), 1, 2)
                block = np.concatenate(block, axis=0)
                block = np.swapaxes(block, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
                # store
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

    def construct_array_lincomb(self, transform, coord_type, **kwargs):
        r"""Return the array associated with linear combinations of contractions.

        .. math::

            \sum_{j} T_{i_1 j} \sum_{k} T_{i_2 k} M_{jklm...} = M^{trans}_{i_1 i_2 lm...}

        Parameters
        ----------
        transform : np.ndarray(K_orbs, K_cont)
            Transformation matrix from contractions in the given coordinate system (e.g. AO) to
            linear combinations of contractions (e.g. MO).
            Transformation is applied to the left.
            Rows correspond to the linear combinations (i.e. MO) and the columns correspond to the
            contractions (i.e. AO).
        coord_type : {"cartesian", "spherical", list/tuple of "cartesian" or "spherical}
            Types of the coordinate system for the contractions.
            If "cartesian", then all of the contractions are treated as Cartesian contractions.
            If "spherical", then all of the contractions are treated as spherical contractions.
            If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
            coordinate type of each `GeneralizedContractionShell` instance.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed directly to `construct_array_spherical`, which will
            then pass it down to `construct_array_contraction`. See `construct_array_contraction`
            for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_orbs, K_orbs, ...)
            Array whose first two indices are associated with the linear combinations of the
            contractions.
            Dimensions 0 and 1 correspond to the linear combination of contracted spherical
            Gaussians. `K_orbs` is the number of basis functions produced after the linear
            combinations.

        Raises
        ------
        TypeError
            If `coord_type` is not one of "cartesian", "spherical", or a list/tuple of these
            strings.

        """
        if coord_type == "cartesian":
            array = self.construct_array_cartesian(**kwargs)
        elif coord_type == "spherical":
            array = self.construct_array_spherical(**kwargs)
        elif isinstance(coord_type, (list, tuple)):
            array = self.construct_array_mix(coord_type, **kwargs)
        else:
            raise TypeError(
                "`coord_type` must be one of 'cartesian', 'spherical', or a list/tuple of these "
                "strings."
            )
        array = np.tensordot(transform, array, (1, 0))
        array = np.tensordot(transform, array, (1, 1))
        array = np.swapaxes(array, 0, 1)
        return array
