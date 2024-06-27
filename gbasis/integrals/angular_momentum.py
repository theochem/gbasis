"""Module for evaluating the integral over the angular momentum operator."""
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._diff_operator_int import (
    _compute_differential_operator_integrals_intermediate,
)
from gbasis.integrals._moment_int import _compute_multipole_moment_integrals_intermediate
import numpy as np


# TODO: need to test against reference
class AngularMomentumIntegral(BaseTwoIndexSymmetric):
    """Class for obtaining the angular momentum integral for a set of Gaussian contractions.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first and second indices of the array.
        Property of `AngularMomentumIntegral`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(contractions_one, contractions_two) :
        **np.ndarray(M_1, L_cart_1, M_2, L_cart_2, 3)**

        Return the integral over the angular momentum operator associated with a
        `GeneralizedContractionShell` instance.
        :math:`M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        :math:`L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        :math:`M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        :math:`L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart, 3)
        Return the integral over the angular momentum operator associated with Cartesian Gaussians.
        :math:`K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph, 3)
        Return the integral over the angular momentum operators associated with spherical Gaussians
        (atomic orbitals).
        :math:`K_sph` is the total number of spherical contractions within the instance.
    construct_array_mix(self, coord_types) : np.ndarray(K_cont, K_cont, 3)
        Return the integral over the angular momentum operators associated with the contraction in
        the given coordinate system.
        :math:`K_cont` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform, coord_type) : np.ndarray(K_orbs, K_orbs, 3)
        Return the integral over the angular momentum operator associated with linear combinations
        of spherical Gaussians (linear combinations of atomic orbitals).
        :math:`K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def construct_array_contraction(contractions_one, contractions_two):
        """Return the integrals over the angular momentum operator of the given contractions.

        Parameters
        ----------
        contractions_one : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the kinetic energy integral.
        contractions_two : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the kinetic energy integral.

        Returns
        -------
        array_contraction : np.ndarray(M_1, L_cart_1, M_2, L_cart_2, 3)
            Integral over than angular momentum operator associated with the given instances of
            GeneralizedContractionShell.
            Dimension 0 corresponds to the segmented contraction within `contractions_one`.
            :math:`M_1` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the first index.
            Dimension 1 corresponds to the angular momentum vector of the `contractions_one`.
            :math:`L_cart_1` is the number of Cartesian contractions for the given angular momentum
            associated with the first index.
            Dimension 2 corresponds to the segmented contraction within `contractions_two`.
            :math:`M_2` is the number of segmented contractions with the same exponents (and angular
            momentum) associated with the second index.
            Dimension 3 corresponds to the angular momentum vector of the `contractions_two`.
            :math:`L_cart_2` is the number of Cartesian contractions for the given angular momentum
            associated with the second index.
            Dimension 4 corresponds to the direction of the angular momentum :math:`(x, y, z)`.

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

        diff_integrals = _compute_differential_operator_integrals_intermediate(
            1,
            contractions_one.coord,
            np.max(contractions_one.angmom_components_cart),
            contractions_one.exps,
            contractions_two.coord,
            np.max(contractions_two.angmom_components_cart),
            contractions_two.exps,
        )
        moment_integrals = _compute_multipole_moment_integrals_intermediate(
            np.zeros(3),
            1,
            contractions_one.coord,
            np.max(contractions_one.angmom_components_cart),
            contractions_one.exps,
            contractions_two.coord,
            np.max(contractions_two.angmom_components_cart),
            contractions_two.exps,
        )

        angmoms_a = contractions_one.angmom_components_cart
        angmoms_b = contractions_two.angmom_components_cart
        output = np.array(
            [
                moment_integrals[0, angmoms_b[:, None, 0], angmoms_a[None, :, 0], 0, :, :]
                * (
                    moment_integrals[1, angmoms_b[:, None, 1], angmoms_a[None, :, 1], 1, :, :]
                    * diff_integrals[1, angmoms_b[:, None, 2], angmoms_a[None, :, 2], 2, :, :]
                    - moment_integrals[1, angmoms_b[:, None, 2], angmoms_a[None, :, 2], 2, :, :]
                    * diff_integrals[1, angmoms_b[:, None, 1], angmoms_a[None, :, 1], 1, :, :]
                ),
                moment_integrals[0, angmoms_b[:, None, 1], angmoms_a[None, :, 1], 1, :, :]
                * (
                    moment_integrals[1, angmoms_b[:, None, 2], angmoms_a[None, :, 2], 2, :, :]
                    * diff_integrals[1, angmoms_b[:, None, 0], angmoms_a[None, :, 0], 0, :, :]
                    - moment_integrals[1, angmoms_b[:, None, 0], angmoms_a[None, :, 0], 0, :, :]
                    * diff_integrals[1, angmoms_b[:, None, 2], angmoms_a[None, :, 2], 2, :, :]
                ),
                moment_integrals[0, angmoms_b[:, None, 2], angmoms_a[None, :, 2], 2, :, :]
                * (
                    moment_integrals[1, angmoms_b[:, None, 0], angmoms_a[None, :, 0], 0, :, :]
                    * diff_integrals[1, angmoms_b[:, None, 1], angmoms_a[None, :, 1], 1, :, :]
                    - moment_integrals[1, angmoms_b[:, None, 1], angmoms_a[None, :, 1], 1, :, :]
                    * diff_integrals[1, angmoms_b[:, None, 0], angmoms_a[None, :, 0], 0, :, :]
                ),
            ]
        )

        # normalize and contract
        norm_a = contractions_one.norm_prim_cart[np.newaxis, np.newaxis, :, np.newaxis, :]
        output = np.tensordot(output * norm_a, contractions_one.coeffs, (4, 0))
        norm_b = contractions_two.norm_prim_cart[np.newaxis, :, np.newaxis, :, np.newaxis]
        output = np.tensordot(output * norm_b, contractions_two.coeffs, (3, 0))

        return -1j * np.transpose(output, (3, 2, 4, 1, 0))


def angular_momentum_integral(basis, transform=None):
    r"""Return the integral over :math:`hat{L}` of the given basis set.

    .. math::

            \left< \hat{\mathbf{L}} \right>
            &= \int \phi_a(\mathbf{r}) \left( -i \mathbf{r} \times \nabla \right) \phi_b(\mathbf{r}) d\mathbf{r}\\
            &= -i
            \begin{bmatrix}
            \int \phi_a(\mathbf{r}) y\frac{\partial}{\partial z} \phi_b(\mathbf{r}) d\mathbf{r}
            - \int \phi_a(\mathbf{r}) z\frac{\partial}{\partial y} \phi_b(\mathbf{r}) d\mathbf{r}\\\\
            \int \phi_a(\mathbf{r}) z\frac{\partial}{\partial x} \phi_b(\mathbf{r}) d\mathbf{r}
            - \int \phi_a(\mathbf{r}) x\frac{\partial}{\partial z} \phi_b(\mathbf{r}) d\mathbf{r}\\\\
            \int \phi_a(\mathbf{r}) x\frac{\partial}{\partial y} \phi_b(\mathbf{r}) d\mathbf{r}
            - \int \phi_a(\mathbf{r}) y\frac{\partial}{\partial x} \phi_b(\mathbf{r}) d\mathbf{r}\\\\
            \end{bmatrix}

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    transform : np.ndarray(K, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.

    Returns
    -------
    array : np.ndarray(K, K, 3)
        Array associated with the basis functions in the given shells of generalized contractions.
        Dimensions 0 and 1 of the array are associated with the basis functions in the basis set,
        where :math:`K` is the total number of basis functions in the basis set.
        Dimension 2 corresponds to the direction of the angular momentum :math:`(x, y, z)`.

    """
    coord_type = [ct for ct in [shell.coord_type for shell in basis]]

    if transform is not None:
        return AngularMomentumIntegral(basis).construct_array_lincomb(transform, coord_type)
    if all(ct == "cartesian" for ct in coord_type):
        return AngularMomentumIntegral(basis).construct_array_cartesian()
    if all(ct == "spherical" for ct in coord_type):
        return AngularMomentumIntegral(basis).construct_array_spherical()
    return AngularMomentumIntegral(basis).construct_array_mix(coord_type)
