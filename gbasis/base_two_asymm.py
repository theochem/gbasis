"""Base class for arrays that depend on two contracted Gaussians."""
import abc

import numpy as np

from gbasis.base import BaseGaussianRelatedArray
from gbasis.spherical import generate_transformation


# pylint: disable=W0235
class BaseTwoIndexAsymmetric(BaseGaussianRelatedArray):
    """Base class for constructing arrays with two indices associated with two sets of contractions.

    The first index of the returned array is associated with the first set of contracted Gaussians
    (or some linear combination of them). The second index of the returned array is associated with
    the second set of contracted Gaussians (or some linear combination of them).

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions_one : tuple of GeneralizedContractionShell
        Contractions that are associated with the first index of the array.
        Property of `BaseTwoIndexAsymmetric`.
    contractions_two : tuple of GeneralizedContractionShell
        Contractions that are associated with the second index of the array.
        Property of `BaseTwoIndexAsymmetric`.

    Methods
    -------
    __init__(self, contractions_one, contractions_two)
        Initialize.
    construct_array_contraction(self, contractions_one, contractions_two, **kwargs) :
        **np.ndarray(M_1, L_cart_1, M_2, L_cart_2, ...)**

        Return the array associated with a `GeneralizedContractionShell` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self, **kwargs) : np.ndarray(K_cart_1, K_cart_2, ...)
        Return the array associated with Cartesian Gaussians.
        `K_cart_1` is the total number of Cartesian contractions within `contractions_one`.
        `K_cart_2` is the total number of Cartesian contractions within `contractions_two`.
    construct_array_spherical(self, **kwargs) : np.ndarray(K_sph_1, K_sph_2, ...)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph_1` is the total number of spherical contractions within `contractions_one`.
        `K_sph_2` is the total number of spherical contractions within `contractions_two`.
    construct_array_lincomb(self, transform_one, transform_two, coord_type, **kwargs) :
        **np.ndarray(K_orbs_1, K_orbs_2, ...)**

        Return the array associated with linear combinations of contractions in the given coordinate
        system.
        `K_orbs_1` is the number of basis functions produced by linear combinations of the
        spherical contractions associated with `contractions_one`.
        `K_orbs_2` is the number of basis functions produced by linear combinations of the
        spherical contractions associated with `contractions_two`.

    """

    def __init__(self, contractions_one, contractions_two):
        """Initialize.

        Parameters
        ----------
        contractions_one : list/tuple of GeneralizedContractionShell
            Contractions that are associated with the first index of the array.
        contractions_two : list/tuple of GeneralizedContractionShell
            Contractions that are associated with the second index of the array.

        """
        super().__init__(contractions_one, contractions_two)

    @property
    def contractions_one(self):
        """Contractions that are associated with the first index of the array.

        Returns
        -------
        contractions_one : tuple of GeneralizedContractionShell
            Contractions that are associated with the first index of the array.

        """
        return self._axes_contractions[0]

    @property
    def contractions_two(self):
        """Contractions that are associated with the second index of the array.

        Returns
        -------
        contractions_two : tuple of GeneralizedContractionShell
            Contractions that are associated with the second index of the array.

        """
        return self._axes_contractions[1]

    @abc.abstractmethod
    def construct_array_contraction(self, contractions_one, contractions_two, **kwargs):
        """Return the array associated with two sets of contracted Cartesian Gaussians.

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
            Array associated with the given instance(s) of GeneralizedContractionShell.
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
        generalized contraction) and the angular momentum vector of `contractions_two`.
        These other methods **will** fail with little warning if the shape of the output is
        different. Even if both `contractions_one` and `contractions_two` are segmented
        contractions, the first and third indices must correspond to the contraction. In other
        words, the shape must still be (1, L_1, 1, L_2).

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
            Dimension 0 corresponds to the Cartesian contraction within `contractions_one`.
            `K_cart_1` is the total number of Cartesian contractions within
            `contractions_one`.
            Dimension 1 corresponds to the Cartesian contraction within `contractions_two`.
            `K_cart_2` is the total number of Cartesian contractions within
            `contractions_two`.

        """
        matrices = []
        for cont_one in self.contractions_one:
            matrices_cols = []
            for cont_two in self.contractions_two:
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
            Dimension 0 corresponds to the Cartesian contraction within `contractions_one`.
            `K_sph_1` is the total number of Cartesian contractions within `contractions_one`.
            Dimension 1 corresponds to the Cartesian contraction within `contractions_two`.
            `K_sph_2` is the total number of Cartesian contractions within `contractions_two`.

        """
        matrices_spherical = []
        for cont_one in self.contractions_one:
            # get transformation from cartesian to spherical for the first index (applied to left)
            transform_one = generate_transformation(
                cont_one.angmom,
                cont_one.angmom_components_cart,
                cont_one.angmom_components_sph,
                "left",
            )
            matrices_spherical_cols = []
            for cont_two in self.contractions_two:
                # get transformation from cartesian to spherical for the first index (applied to
                # left)
                transform_two = generate_transformation(
                    cont_two.angmom,
                    cont_two.angmom_components_cart,
                    cont_two.angmom_components_sph,
                    "left",
                )
                # evaluate
                matrix_contraction = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                matrix_contraction *= cont_one.norm_cont.reshape(
                    *matrix_contraction.shape[:2], *[1 for _ in matrix_contraction.shape[2:]]
                )
                matrix_contraction *= cont_two.norm_cont.reshape(
                    1, 1, *matrix_contraction.shape[2:4], *[1 for _ in matrix_contraction.shape[4:]]
                )
                # transform
                # assume array always has shape (M_1, L_1, M_2, L_2, ...)
                matrix_contraction = np.tensordot(transform_one, matrix_contraction, (1, 1))
                matrix_contraction = np.concatenate(np.swapaxes(matrix_contraction, 0, 1), axis=0)
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                matrix_contraction = np.tensordot(transform_two, matrix_contraction, (1, 2))
                matrix_contraction = np.swapaxes(np.swapaxes(matrix_contraction, 0, 1), 0, 2)
                matrix_contraction = np.concatenate(matrix_contraction, axis=0)
                matrix_contraction = np.swapaxes(matrix_contraction, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
                # store
                matrices_spherical_cols.append(matrix_contraction)
            matrices_spherical.append(np.concatenate(matrices_spherical_cols, axis=1))
        return np.concatenate(matrices_spherical, axis=0)

    def construct_array_mix(self, coord_types_one, coord_types_two, **kwargs):
        """Return the array associated with a set of Gaussians of the given coordinate systems.

        Parameters
        ----------
        coord_types_one : list/tuple of str
            Types of the coordinate system for `GeneralizedContractionShell` associated with the
            first index of the array.
            Each entry must be one of "cartesian" or "spherical".
        coord_types_two : list/tuple of str
            Types of the coordinate system for GeneralizedContractionShell associated with the
            second index of the array.
            Each entry must be one of "cartesian" or "spherical".
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray(K_cont, K_cont, ...)
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.
            First two indices of the array are associated with two contractions in the given
            coordinate system. `K_cont` is the total number of contractions within the
            instance.

        Raises
        ------
        TypeError
            If `coord_types_one` is not a list/tuple.
            If `coord_types_two` is not a list/tuple.
        ValueError
            If `coord_types_one` has an entry that is not "cartesian" or "spherical".
            If `coord_types_one` has different number of entries as the number of
            `GeneralizedContractionShell` (`contractions`) in instance.
            If `coord_types_two` has an entry that is not "cartesian" or "spherical".
            If `coord_types_two` has different number of entries as the number of
            `GeneralizedContractionShell` (`contractions`) in instance.

        """
        if not isinstance(coord_types_one, (list, tuple)):
            raise TypeError("`coord_types_one` must be a list or a tuple.")
        if not isinstance(coord_types_two, (list, tuple)):
            raise TypeError("`coord_types_two` must be a list or a tuple.")
        if not all(i in ["cartesian", "spherical"] for i in coord_types_one):
            raise ValueError(
                "Each entry of `coord_types_one` must be one of 'cartesian' or 'spherical'."
            )
        if not all(i in ["cartesian", "spherical"] for i in coord_types_two):
            raise ValueError(
                "Each entry of `coord_types_two` must be two of 'cartesian' or 'spherical'."
            )
        if len(coord_types_one) != len(self.contractions_one):
            raise ValueError(
                "`coord_types_one` must have the same number of entries as the number of "
                "`GeneralizedContractionShell` in the instance."
            )
        if len(coord_types_two) != len(self.contractions_two):
            raise ValueError(
                "`coord_types_two` must have the same number of entries as the number of "
                "`GeneralizedContractionShell` in the instance."
            )

        matrices_spherical = []
        for cont_one, type_one in zip(self.contractions_one, coord_types_one):
            if type_one == "spherical":
                # get transformation from cartesian to spherical for the first index (applied to
                # left)
                transform_one = generate_transformation(
                    cont_one.angmom,
                    cont_one.angmom_components_cart,
                    cont_one.angmom_components_sph,
                    "left",
                )
            matrices_spherical_cols = []
            for cont_two, type_two in zip(self.contractions_two, coord_types_two):
                # evaluate
                block = self.construct_array_contraction(cont_one, cont_two, **kwargs)
                # normalize contractions
                block *= cont_one.norm_cont.reshape(*block.shape[:2], *[1 for _ in block.shape[2:]])
                block *= cont_two.norm_cont.reshape(
                    1, 1, *block.shape[2:4], *[1 for _ in block.shape[4:]]
                )
                # transform
                # assume array always has shape (M_1, L_1, M_2, L_2, ...)
                if type_one == "spherical":
                    block = np.tensordot(transform_one, block, (1, 1))
                    block = np.swapaxes(block, 0, 1)
                block = np.concatenate(block, axis=0)
                # array now has shape (M_1 L_1, M_2, L_2, ...)
                if type_two == "spherical":
                    # get transformation from cartesian to spherical for the first index (applied to
                    # left)
                    transform_two = generate_transformation(
                        cont_two.angmom,
                        cont_two.angmom_components_cart,
                        cont_two.angmom_components_sph,
                        "left",
                    )
                    block = np.tensordot(transform_two, block, (1, 2))
                    block = np.swapaxes(np.swapaxes(block, 0, 1), 0, 2)
                else:
                    block = np.swapaxes(np.swapaxes(block, 0, 1), 1, 2)
                block = np.concatenate(block, axis=0)
                block = np.swapaxes(block, 0, 1)
                # array now has shape (M_1 L_1, M_2 L_2, ...)
                # store
                matrices_spherical_cols.append(block)
            matrices_spherical.append(np.concatenate(matrices_spherical_cols, axis=1))
        return np.concatenate(matrices_spherical, axis=0)

    def construct_array_lincomb(
        self, transform_one, transform_two, coord_type_one, coord_type_two, **kwargs
    ):
        r"""Return the array associated with linear combinations of contractions.

        .. math::

            \sum_{j} T^{one}_{i_1 j} \sum_{k} T^{two}_{i_2 k} M_{jklm...}
            = M^{trans}_{i_1 i_2 lm...}

        Parameters
        ----------
        transform_one : np.ndarray
            Array associated with the linear combinations of spherical Gaussians (LCAO's) associated
            with the first index.
            If None, then transformation is skipped.
        transform_two : np.ndarray
            Array associated with the linear combinations of spherical Gaussians (LCAO's) associated
            with the second index.
            If None, then transformation is skipped.
        coord_type_one : list/tuple of string
            Types of the coordinate system for the contractions associated with the first index.
            Each entry must be one of "cartesian" or "spherical". If multiple
            instances of GeneralizedContractionShell are given but only one string
            ("cartesian" or "spherical") is provided in the list/tuple, all of the
            contractions will be treated according to that string.
        coord_type_two : list/tuple of string
            Types of the coordinate system for the contractions associated with the second index.
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
        array : np.ndarray(K_orbs_1, K_orbs_2, ...)
            Array associated with the linear combinations of given two sets of contractions.
            Dimension 0 corresponds to the linear combination of contractions with the first set of
            contractions, `contractions_one`. `K_orbs_1` is the number of basis functions
            produced by linear combination of the spherical contractions associated with
            `contractions_one`.
            Dimension 1 corresponds to the linear combination of contractions associated with the
            second set of contractions, `contractions_two`. `K_orbs_2` is the number of basis
            functions produced by linear combinations of the spherical contractions associated with
            `contractions_two`.

        Raises
        ------
        TypeError
            If `coord_type` is not a list/tuple of the strings 'cartesian' or 'spherical'.

        """
        if all(ct_one == "cartesian" for ct_one in coord_type_one) and all(
            ct_two == "cartesian" for ct_two in coord_type_two
        ):
            array = self.construct_array_cartesian(**kwargs)
        elif all(ct_one == "spherical" for ct_one in coord_type_one) and all(
            ct_two == "spherical" for ct_two in coord_type_two
        ):
            array = self.construct_array_spherical(**kwargs)
        else:
            if coord_type_one in ["cartesian", "spherical"]:
                coord_type_one = [coord_type_one] * len(self.contractions_one)
            if coord_type_two in ["cartesian", "spherical"]:
                coord_type_two = [coord_type_two] * len(self.contractions_two)
            if not (
                isinstance(coord_type_one, (list, tuple))
                and isinstance(coord_type_two, (list, tuple))
            ):
                raise TypeError(
                    "`coord_type` must be a list/tuple of the strings 'cartesian' or 'spherical'"
                )
            array = self.construct_array_mix(coord_type_one, coord_type_two, **kwargs)
        if transform_one is not None:
            array = np.tensordot(transform_one, array, (1, 0))
        if transform_two is not None:
            array = np.tensordot(transform_two, array, (1, 1))
            array = np.swapaxes(array, 0, 1)
        return array
