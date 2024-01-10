"""Base class for arrays that depend on four contracted Gaussians."""
import abc
import itertools as it

from gbasis.base import BaseGaussianRelatedArray
from gbasis.spherical import generate_transformation
import numpy as np


# pylint: disable=W0235
class BaseFourIndexSymmetric(BaseGaussianRelatedArray):
    """Base class for constructing arrays associated with four contracted Gaussian.

    The first four axes of the returned array are associated with the given set of contracted
    Gaussian (or a linear combination of a set of Gaussians).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first four indices of the array.
        Property of `BaseFourIndexSymmetric`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, cont1, cont2, cont3, cont4, **kwargs) :
    np.ndarray(M_1, L_cart_1, M_2, L_cart_2, M_3, L_cart_3, M_4, L_cart_4, ...)
        Return the array associated with a `GeneralizedContractionShell` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
        `M_3` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the third index.
        `L_cart_3` is the number of Cartesian contractions for the given angular momentum
        associated with the third index.
        `M_4` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the fourth index.
        `L_cart_4` is the number of Cartesian contractions for the given angular momentum
        associated with the fourth index.
    construct_array_cartesian(self, **kwargs) : np.ndarray(K_cart, K_cart, K_cart, K_cart, ...)
        Return the array associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self, **kwargs) : np.ndarray(K_sph, K_sph, K_sph, K_sph, ...)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_mix(self, coord_types, **kwargs) :
    np.ndarray(K_cont, K_cont, K_cont, K_cont, ...)
        Return the array associated with all of the contraction in the given coordinate
        system.
        `K_cont` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform, coord_type) :
    np.ndarray(K_orbs, K_orbs, K_orbs, K_orbs, ...)
        Return the array associated with linear combinations of contractions in the given coordinate
        system.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    def __init__(self, contractions):
        """Initialize.

        Parameters
        ----------
        contractions : list/tuple of GeneralizedContractionShell
            Contractions that are associated with the first four indices of the array.

        """
        super().__init__(contractions)

    @property
    def contractions(self):
        """Contractions that are associated with the first four indices of the array.

        Returns
        -------
        contractions : tuple of GeneralizedContractionShell
            Generalized contraction shell that is associated with the first four indices of the
            array.

        """
        return self._axes_contractions[0]

    @abc.abstractmethod
    def construct_array_contraction(self, cont1, cont2, cont3, cont4, **kwargs):
        """Return the array associated with a set of contracted Cartesian Gaussians.

        Parameters
        ----------
        cont1 : GeneralizedContractionShell
            Generalized contracted (Cartesian) shell associated with the first index of the array.
        cont2 : GeneralizedContractionShell
            Generalized contracted (Cartesian) shell associated with the second index of the array.
        cont3 : GeneralizedContractionShell
            Generalized contracted (Cartesian) shell associated with the three index of the array.
        cont4 : GeneralizedContractionShell
            Generalized contracted (Cartesian) shell associated with the four index of the array.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array_cont : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, M_3, L_cart_3, M_4, L_cart_4, ...)
            Return the array associated with a `GeneralizedContractionShell` instances.
            Dimension 0 corresponds to the segmented contraction within `cont_one`. `M_1` is
            the number of segmented contractions with the same exponents (and angular momentum)
            associated with the first index.
            Dimension 1 corresponds to the angular momentum vector of the `cont_one`.
            `L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Dimension 2 corresponds to the segmented contraction within `cont_two`.
            `M_2` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the second index.
            Dimension 3 corresponds to the angular momentum vector of the `cont_two`.
            `L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.
            Dimension 4 corresponds to the segmented contraction within `cont_three`.
            `M_3` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the third index.
            Dimension 5 corresponds to the angular momentum vector of the `cont_three`.
            `L_cart_3` is the number of Cartesian contractions for the given angular momentum
            associated with the third index.
            Dimension 6 corresponds to the segmented contraction within `cont_four`.
            `M_4` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the fourth index.
            Dimension 7 corresponds to the angular momentum vector of the `cont_four`.
            `L_cart_4` is the number of Cartesian contractions for the given angular momentum
            associated with the fourth index.

        Notes
        -----
        The next level of classes will be the construction of specific arrays given a set of
        contracted Cartesian Gaussians. The keyword arguments will be different depending on its
        functionality. You should use explicit keyword arguments when defining this function, rather
        than the arbitrary number of keywords (as is done here).

        The methods `construct_array_cartesian`, `construct_array_spherical`, and
        `construct_array_lincomb` depend on this function to produce an array with dimensions 0, 1
        corresponding to the contraction (within a generalized contraction) and angular momentum
        vector of `cont_one`, dimensions 2, 3 corresponding to the contraction
        (within a generalized contraction) and angular momentum vector of `cont_two`, and so
        on for `cont_three` and `cont_four`. These other methods **will** fail with little warning
        if the shape of the output is different.
        Even if all of `cont1`, `cont2`, `cont3`, and `cont4` are segmented contractions,
        dimensions 0, 2, 4, and 6 must correspond to the contraction. In other words, the shape must
        still be (1, L_1, 1, L_2, 1, L_3, 1, L_4).

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
        array : np.ndarray(K_cart, K_cart, K_cart, K_cart, ...)
            Array associated with the given set of contracted Cartesian Gaussians.
            Dimensions 0, 1, 2, and 3 of the array are associated with the contracted Cartesian
            Gaussians. `K_cart` is the total number of Cartesian contractions within the
            instance.

        Notes
        -----
        The blocks along the diagonal, i.e. blocks where the first two axes belong to the same
        set of contractions, are transposed in the process of constructing the whole array.
        It is assumed that the array returned from `construct_array_contraction` is symmetric with
        respect to the swapping of the axes 0 and 1.

        """
        # pylint: disable=C0103,R0914
        all_blocks = np.zeros((len(self.contractions),) * 4, dtype=object)

        pair_i_cont = list(it.combinations_with_replacement(enumerate(self.contractions), 2))
        for pair_ind, ((i, cont_one), (j, cont_two)) in enumerate(pair_i_cont):
            for (k, cont_three), (l, cont_four) in pair_i_cont[pair_ind:]:
                block = self.construct_array_contraction(
                    cont_one, cont_two, cont_three, cont_four, **kwargs
                )
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for _ in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for _ in block.shape[4:]]
                )
                block *= cont_three.norm_cont.reshape(
                    1, 1, 1, 1, *block.shape[4:6], *[1 for _ in block.shape[6:]]
                )
                block *= cont_four.norm_cont.reshape(
                    1, 1, 1, 1, 1, 1, *block.shape[6:8], *[1 for _ in block.shape[8:]]
                )
                # assume array always has shape (M_1, L_1, M_2, L_2, M_3, L_3, M_4, L_4, ..)
                block = block.reshape(
                    block.shape[0] * block.shape[1],
                    block.shape[2] * block.shape[3],
                    block.shape[4] * block.shape[5],
                    block.shape[6] * block.shape[7],
                    *block.shape[8:],
                )
                # array now has shape (M_1 L_1, M_2 L_2, M_3, L_3, M_4, L_4, ...)
                all_blocks[i, j, k, l] = block
                all_blocks[i, j, l, k] = np.swapaxes(block, 2, 3)
                all_blocks[j, i, k, l] = np.swapaxes(block, 0, 1)
                all_blocks[j, i, l, k] = np.swapaxes(np.swapaxes(block, 2, 3), 0, 1)
                all_blocks[k, l, i, j] = np.swapaxes(np.swapaxes(block, 1, 3), 0, 2)
                all_blocks[l, k, i, j] = np.swapaxes(
                    np.swapaxes(np.swapaxes(block, 1, 3), 0, 2), 0, 1
                )
                all_blocks[k, l, j, i] = np.swapaxes(
                    np.swapaxes(np.swapaxes(block, 1, 3), 0, 2), 2, 3
                )
                all_blocks[l, k, j, i] = np.swapaxes(np.swapaxes(block, 1, 2), 0, 3)

        # concatenate
        return np.concatenate(
            [
                np.concatenate(
                    [
                        np.concatenate(
                            [np.concatenate(blocks_three, axis=3) for blocks_three in blocks_two],
                            axis=2,
                        )
                        for blocks_two in blocks_one
                    ],
                    axis=1,
                )
                for blocks_one in all_blocks
            ],
            axis=0,
        )

    def construct_array_spherical(self, **kwargs):
        """Return the array associated with four contracted spherical Gaussians (atomic orbitals).

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed entirely to `construct_array_contraction`. See
            `construct_array_contraction` for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_sph, K_sph, K_sph, K_sph, ...)
            Array associated with the atomic orbitals associated with the given set(s) of contracted
            Cartesian Gaussians.
            Dimensions 0, 1, 2 and 3 of the array are associated with four contracted spherical
            Gaussians (atomic orbitals). `K_sph` is the total number of spherical contractions
            within the instance.

        Notes
        -----
        The blocks along the diagonal, i.e. blocks where the first two axes belong to the same
        set of contractions, are transposed in the process of constructing the whole array.
        It is assumed that the array returned from `construct_array_contraction` is symmetric with
        respect to the swapping of the axes 0 and 1.

        """
        # pylint: disable=C0103,R0914
        all_blocks = np.zeros((len(self.contractions),) * 4, dtype=object)
        # NOTE: we get list of unique pairs of (index, contraction_instance) to avoid double
        # counting the ij and kl. e.g. 0, 1, 0, 0 and 0, 0, 0, 1 will both be present with the
        # previous approach
        pair_i_cont = list(it.combinations_with_replacement(enumerate(self.contractions), 2))
        for pair_ind, ((i, cont_one), (j, cont_two)) in enumerate(pair_i_cont):
            transform_one = generate_transformation(
                cont_one.angmom,
                cont_one.angmom_components_cart,
                cont_one.angmom_components_sph,
                "left",
            )
            transform_two = generate_transformation(
                cont_two.angmom,
                cont_two.angmom_components_cart,
                cont_two.angmom_components_sph,
                "left",
            )
            for (k, cont_three), (l, cont_four) in pair_i_cont[pair_ind:]:
                transform_three = generate_transformation(
                    cont_three.angmom,
                    cont_three.angmom_components_cart,
                    cont_three.angmom_components_sph,
                    "left",
                )
                transform_four = generate_transformation(
                    cont_four.angmom,
                    cont_four.angmom_components_cart,
                    cont_four.angmom_components_sph,
                    "left",
                )

                block = self.construct_array_contraction(
                    cont_one, cont_two, cont_three, cont_four, **kwargs
                )
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for _ in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for _ in block.shape[4:]]
                )
                block *= cont_three.norm_cont.reshape(
                    1, 1, 1, 1, *block.shape[4:6], *[1 for _ in block.shape[6:]]
                )
                block *= cont_four.norm_cont.reshape(
                    1, 1, 1, 1, 1, 1, *block.shape[6:8], *[1 for _ in block.shape[8:]]
                )

                # transform
                block = np.tensordot(transform_one, block, (1, 1))
                block = np.swapaxes(block, 0, 1)
                block = np.tensordot(transform_two, block, (1, 3))
                block = np.swapaxes(np.swapaxes(np.swapaxes(block, 0, 1), 1, 2), 2, 3)
                block = np.tensordot(transform_three, block, (1, 5))
                block = np.swapaxes(
                    np.swapaxes(
                        np.swapaxes(np.swapaxes(np.swapaxes(block, 0, 1), 1, 2), 2, 3), 3, 4
                    ),
                    4,
                    5,
                )
                block = np.tensordot(transform_four, block, (1, 7))
                block = np.swapaxes(
                    np.swapaxes(
                        np.swapaxes(
                            np.swapaxes(
                                np.swapaxes(np.swapaxes(np.swapaxes(block, 0, 1), 1, 2), 2, 3), 3, 4
                            ),
                            4,
                            5,
                        ),
                        5,
                        6,
                    ),
                    6,
                    7,
                )

                # array has shape (M_1, L_1, M_2, L_2, M_3, L_3, M_4, L_4, ..)
                block = block.reshape(
                    block.shape[0] * block.shape[1],
                    block.shape[2] * block.shape[3],
                    block.shape[4] * block.shape[5],
                    block.shape[6] * block.shape[7],
                    *block.shape[8:],
                )
                # array now has shape (M_1 L_1, M_2 L_2, M_3, L_3, M_4, L_4, ...)
                all_blocks[i, j, k, l] = block
                all_blocks[i, j, l, k] = np.swapaxes(block, 2, 3)
                all_blocks[j, i, k, l] = np.swapaxes(block, 0, 1)
                all_blocks[j, i, l, k] = np.swapaxes(np.swapaxes(block, 2, 3), 0, 1)
                all_blocks[k, l, i, j] = np.swapaxes(np.swapaxes(block, 1, 3), 0, 2)
                all_blocks[l, k, i, j] = np.swapaxes(
                    np.swapaxes(np.swapaxes(block, 1, 3), 0, 2), 0, 1
                )
                all_blocks[k, l, j, i] = np.swapaxes(
                    np.swapaxes(np.swapaxes(block, 1, 3), 0, 2), 2, 3
                )
                all_blocks[l, k, j, i] = np.swapaxes(np.swapaxes(block, 1, 2), 0, 3)

        # concatenate
        return np.concatenate(
            [
                np.concatenate(
                    [
                        np.concatenate(
                            [np.concatenate(blocks_three, axis=3) for blocks_three in blocks_two],
                            axis=2,
                        )
                        for blocks_two in blocks_one
                    ],
                    axis=1,
                )
                for blocks_one in all_blocks
            ],
            axis=0,
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
        array : np.ndarray(K_cont, K_cont, K_cont, K_cont, ...)
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.
            Dimensions 0, 1, 2 and 3 of the array are associated with two contractions in the given
            coordinate system. `K_cont` is the total number of contractions within the
            instance.

        Raises
        ------
        TypeError
            If `coord_types` is not a list/tuple.
        ValueError
            If `coord_types` has an entry that is not "cartesian" or "spherical".
            If `coord_types` has different number of entries as the number of
            `GeneralizedContractionShell` (`contractions`) in instance.

        """
        # pylint: disable=C0103,R0914
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

        all_blocks = np.zeros((len(self.contractions),) * 4, dtype=object)
        pair_i_cont = list(
            it.combinations_with_replacement(
                zip(range(len(self.contractions)), self.contractions, coord_types), 2
            )
        )
        for pair_ind, ((i, cont_one, type_one), (j, cont_two, type_two)) in enumerate(pair_i_cont):
            if type_one == "spherical":
                transform_one = generate_transformation(
                    cont_one.angmom,
                    cont_one.angmom_components_cart,
                    cont_one.angmom_components_sph,
                    "left",
                )
            if type_two == "spherical":
                transform_two = generate_transformation(
                    cont_two.angmom,
                    cont_two.angmom_components_cart,
                    cont_two.angmom_components_sph,
                    "left",
                )
            for (k, cont_three, type_three), (l, cont_four, type_four) in pair_i_cont[pair_ind:]:
                block = self.construct_array_contraction(
                    cont_one, cont_two, cont_three, cont_four, **kwargs
                )
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for _ in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for _ in block.shape[4:]]
                )
                block *= cont_three.norm_cont.reshape(
                    1, 1, 1, 1, *block.shape[4:6], *[1 for _ in block.shape[6:]]
                )
                block *= cont_four.norm_cont.reshape(
                    1, 1, 1, 1, 1, 1, *block.shape[6:8], *[1 for _ in block.shape[8:]]
                )

                # transform
                if type_one == "spherical":
                    block = np.tensordot(transform_one, block, (1, 1))
                    block = np.swapaxes(block, 0, 1)
                if type_two == "spherical":
                    block = np.tensordot(transform_two, block, (1, 3))
                    block = np.swapaxes(np.swapaxes(np.swapaxes(block, 0, 1), 1, 2), 2, 3)
                if type_three == "spherical":
                    transform_three = generate_transformation(
                        cont_three.angmom,
                        cont_three.angmom_components_cart,
                        cont_three.angmom_components_sph,
                        "left",
                    )
                    block = np.tensordot(transform_three, block, (1, 5))
                    block = np.swapaxes(
                        np.swapaxes(
                            np.swapaxes(np.swapaxes(np.swapaxes(block, 0, 1), 1, 2), 2, 3), 3, 4
                        ),
                        4,
                        5,
                    )
                if type_four == "spherical":
                    transform_four = generate_transformation(
                        cont_four.angmom,
                        cont_four.angmom_components_cart,
                        cont_four.angmom_components_sph,
                        "left",
                    )
                    block = np.tensordot(transform_four, block, (1, 7))
                    block = np.swapaxes(
                        np.swapaxes(
                            np.swapaxes(
                                np.swapaxes(
                                    np.swapaxes(np.swapaxes(np.swapaxes(block, 0, 1), 1, 2), 2, 3),
                                    3,
                                    4,
                                ),
                                4,
                                5,
                            ),
                            5,
                            6,
                        ),
                        6,
                        7,
                    )

                # array has shape (M_1, L_1, M_2, L_2, M_3, L_3, M_4, L_4, ..)
                block = block.reshape(
                    block.shape[0] * block.shape[1],
                    block.shape[2] * block.shape[3],
                    block.shape[4] * block.shape[5],
                    block.shape[6] * block.shape[7],
                    *block.shape[8:],
                )
                # array now has shape (M_1 L_1, M_2 L_2, M_3, L_3, M_4, L_4, ...)
                all_blocks[i, j, k, l] = block
                all_blocks[i, j, l, k] = np.swapaxes(block, 2, 3)
                all_blocks[j, i, k, l] = np.swapaxes(block, 0, 1)
                all_blocks[j, i, l, k] = np.swapaxes(np.swapaxes(block, 2, 3), 0, 1)
                all_blocks[k, l, i, j] = np.swapaxes(np.swapaxes(block, 1, 3), 0, 2)
                all_blocks[l, k, i, j] = np.swapaxes(
                    np.swapaxes(np.swapaxes(block, 1, 3), 0, 2), 0, 1
                )
                all_blocks[k, l, j, i] = np.swapaxes(
                    np.swapaxes(np.swapaxes(block, 1, 3), 0, 2), 2, 3
                )
                all_blocks[l, k, j, i] = np.swapaxes(np.swapaxes(block, 1, 2), 0, 3)

        # concatenate
        return np.concatenate(
            [
                np.concatenate(
                    [
                        np.concatenate(
                            [np.concatenate(blocks_three, axis=3) for blocks_three in blocks_two],
                            axis=2,
                        )
                        for blocks_two in blocks_one
                    ],
                    axis=1,
                )
                for blocks_one in all_blocks
            ],
            axis=0,
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
        coord_type : list/tuple of str
            Types of the coordinate system for each GeneralizedContractionShell.
            Each entry must be one of "cartesian" or "spherical". If multiple
            instances of GeneralizedContractionShell are given but only one string
            ("cartesian" or "spherical") is provided in the list/tuple, all of the
            contractions will be treated according to that string.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.
            These keyword arguments are passed directly to `construct_array_spherical`, which will
            then pass it down to `construct_array_contraction`. See `construct_array_contraction`
            for details on the keyword arguments.

        Returns
        -------
        array : np.ndarray(K_orbs, K_orbs, K_orbs, K_orbs, ...)
            Array whose first four indices are associated with the linear combinations of the
            contractions.
            Dimensions 0, 1, 2 and 3 of the array correspond to the linear combination of contracted
            spherical Gaussians. `K_orbs` is the number of basis functions produced after the
            linear combinations.

        Raises
        ------
        TypeError
            If `coord_type` is not a list/tuple of the strings 'cartesian' or 'spherical'.

        """
        if all(ct == "cartesian" for ct in coord_type):
            array = self.construct_array_cartesian(**kwargs)
        elif all(ct == "spherical" for ct in coord_type):
            array = self.construct_array_spherical(**kwargs)
        elif isinstance(coord_type, (list, tuple)):
            array = self.construct_array_mix(coord_type, **kwargs)
        else:
            raise TypeError(
                "`coord_type` must be a list/tuple of the strings 'cartesian' or 'spherical'"
            )
        array = np.tensordot(transform, array, (1, 0))
        array = np.tensordot(transform, array, (1, 1))
        array = np.tensordot(transform, array, (1, 2))
        array = np.tensordot(transform, array, (1, 3))
        array = np.swapaxes(np.swapaxes(array, 0, 3), 1, 2)
        return array
