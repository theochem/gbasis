"""Functions for computing overlap of a basis set."""
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.moment_int import _compute_multipole_moment_integrals
import numpy as np


class Overlap(BaseTwoIndexSymmetric):
    """Class for obtaining the overlap for a set of Gaussian contractions.

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
    construct_array_contraction(self, contraction) : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
        Return the array associated with a `ContractedCartesianGaussians` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the array associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the array associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform) : np.ndarray(K_orbs, K_orbs)
        Return the array associated with linear combinations of spherical Gaussians (linear
        combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def construct_array_contraction(contractions_one, contractions_two):
        """Return the evaluations of the given contractions at the given coordinates.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
            Overlap associated with the given instances of ContractedCartesianGaussians.
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
            This array should be symmetric with respect to the swapping of the first and second
            axes.

        Raises
        ------
        TypeError
            If contractions_one is not a ContractedCartesianGaussians instance.
            If contractions_two is not a ContractedCartesianGaussians instance.

        """
        if not isinstance(contractions_one, ContractedCartesianGaussians):
            raise TypeError("`contractions_one` must be a ContractedCartesianGaussians instance.")
        if not isinstance(contractions_two, ContractedCartesianGaussians):
            raise TypeError("`contractions_two` must be a ContractedCartesianGaussians instance.")

        coord_a = contractions_one.coord
        angmoms_a = contractions_one.angmom_components
        alphas_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        norm_a = contractions_one.norm
        coord_b = contractions_two.coord
        angmoms_b = contractions_two.angmom_components
        alphas_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs
        norm_b = contractions_two.norm
        return _compute_multipole_moment_integrals(
            np.zeros(3),
            np.zeros((1, 3), dtype=int),
            coord_a,
            angmoms_a,
            alphas_a,
            coeffs_a,
            norm_a,
            coord_b,
            angmoms_b,
            alphas_b,
            coeffs_b,
            norm_b,
        )[0]


def overlap_basis_cartesian(basis):
    """Return the overlap of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return Overlap(basis).construct_array_cartesian()


def overlap_basis_spherical(basis):
    """Return the overlap of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.

    Returns
    -------
    array : np.ndarray(K_sph, K_sph)
        Array associated with the atomic orbitals associated with the given set(s) of contracted
        Cartesian Gaussians.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    """
    return Overlap(basis).construct_array_spherical()


def overlap_basis_spherical_lincomb(basis, transform):
    """Return the overlap of the linear combination of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    transform : np.ndarray(K_orbs, K_sph)
        Array associated with the linear combinations of spherical Gaussians (LCAO's).
        Transformation is applied to the left, i.e. the sum is over the second index of `transform`
        and first index of the array for contracted spherical Gaussians.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Array whose first and second indices are associated with the linear combinations of the
        contracted spherical Gaussians.
        First and second indices of the array correspond to the linear combination of contracted
        spherical Gaussians. `K_orbs` is the number of basis functions produced after the linear
        combinations.

    """
    return Overlap(basis).construct_array_spherical_lincomb(transform)
