"""Functions for computing overlap of a basis set."""

import numpy as np

from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._moment_int import _compute_multipole_moment_integrals
from gbasis.screening import is_two_index_overlap_screened


class Overlap(BaseTwoIndexSymmetric):
    """Class for obtaining the overlap for a set of Gaussian contractions.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first and second indices of the array.
        Property of `Overlap`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(contractions_one, contractions_two) :
        **np.ndarray(M_1, L_cart_1, M_2, L_cart_2)**

        Return the overlap associated with a `GeneralizedContractionShell` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the overlap integrals associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the overlap integrals associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_mix(self, coord_types, **kwargs) : np.ndarray(K_cont, K_cont)
        Return the overlap integrals associated with the contraction in the given coordinate system.
        `K_cont` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform, coord_type) : np.ndarray(K_orbs, K_orbs)
        Return the overlap integrals associated with the linear combinations of contractions in the
        given coordinate system.
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def construct_array_contraction(
        contractions_one, contractions_two, screen_basis=True, tol_screen=1e-8
    ):
        """Return the evaluations of the given contractions at the given coordinates.

        Parameters
        ----------
        contractions_one : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the overlap.
        contractions_two : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the overlap.
        screen_basis : bool, optional
            A toggle to enable or disable screening. Default value is True to enable screening.
        tol_screen : float, optional
            The tolerance used for screening overlap integrals. `tol_screen` is combined with the
            minimum contraction exponents to compute a cutoff which is compared to the distance
            between the contraction centers to decide whether the overlap integral should be
            set to zero. The default value for `tol_screen` is 1e-8.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
            Overlap associated with the given instances of `GeneralizedContractionShell`.
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

        Raises
        ------
        TypeError
            If contractions_one is not a `GeneralizedContractionShell` instance.
            If contractions_two is not a `GeneralizedContractionShell` instance.

        """
        if not isinstance(contractions_one, GeneralizedContractionShell):
            raise TypeError("`contractions_one` must be a `GeneralizedContractionShell` instance.")
        if not isinstance(contractions_two, GeneralizedContractionShell):
            raise TypeError("`contractions_two` must be a `GeneralizedContractionShell` instance.")

        # return zero if screening is enabled, and the integral is screened
        if screen_basis and is_two_index_overlap_screened(
            contractions_one, contractions_two, tol_screen
        ):
            return np.zeros(
                (
                    contractions_one.num_seg_cont,
                    len(contractions_one.norm_prim_cart),
                    contractions_two.num_seg_cont,
                    len(contractions_two.norm_prim_cart),
                ),
                dtype=np.float64,
            )
        # compute not-screened integrals
        return _compute_multipole_moment_integrals(
            np.zeros(3),
            np.zeros((1, 3), dtype=int),
            # contraction on the left hand side
            contractions_one.coord,
            contractions_one.angmom_components_cart,
            contractions_one.exps,
            contractions_one.coeffs,
            contractions_one.norm_prim_cart,
            # contraction on the right hand side
            contractions_two.coord,
            contractions_two.angmom_components_cart,
            contractions_two.exps,
            contractions_two.coeffs,
            contractions_two.norm_prim_cart,
        )[0]


def overlap_integral(basis, transform=None, screen_basis=True, tol_screen=1e-8):
    r"""Return overlap integral of the given basis set.

    .. math::

        \int \phi_a (\mathbf{r}) \phi_b (\mathbf{r}) d\mathbf{r}

    where :math:`\phi_a(\mathbf{r})` and :math:`\phi_b(\mathbf{r})` are the basis functions

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    transform : np.ndarray(K_orbs, K_cont), optional
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    screen_basis : bool, optional
        A toggle to enable or disable screening. Default value is True to enable screening.
    tol_screen : float, optional
        The tolerance used for screening overlap integrals. `tol_screen` is combined with the
        minimum contraction exponents to compute a cutoff which is compared to the distance
        between the contraction centers to decide whether the overlap integral should be
        set to zero. The default value for `tol_screen` is 1e-8.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Overlap integral of the given basis set.
        Dimensions 0 and 1 of the array correspond to the basis functions. `K_orbs` is the
        number of basis functions in the basis set.

    """
    coord_type = [shell.coord_type for shell in basis]
    kwargs = {"tol_screen": tol_screen, "screen_basis": screen_basis}

    if transform is not None:
        return Overlap(basis).construct_array_lincomb(transform, coord_type, **kwargs)
    if all(ct == "cartesian" for ct in coord_type):
        return Overlap(basis).construct_array_cartesian(**kwargs)
    if all(ct == "spherical" for ct in coord_type):
        return Overlap(basis).construct_array_spherical(**kwargs)
    return Overlap(basis).construct_array_mix(coord_type, **kwargs)
