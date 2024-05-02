"""Module for evaluating the integral over the momentum operator."""
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._diff_operator_int import _compute_differential_operator_integrals
import numpy as np


# TODO: need to test against reference
class MomentumIntegral(BaseTwoIndexSymmetric):
    """Class for obtaining the momentum integral.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions : tuple of GeneralizedContractionShell
        Contractions that are associated with the first and second indices of the array.
        Property of `MomentumIntegral`.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(contractions_one, contractions_two) :
        **np.ndarray(M_1, L_cart_1, M_2, L_cart_2, 3)**

        Return the integral over the momentum operator associated with a
        `GeneralizedContractionShell` instance.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart, 3)
        Return the integral over the momentum operator associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph, 3)
        Return the integral over the momentum operator associated with spherical Gaussians (atomic
        orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_mix(self, coord_types) : np.ndarray(K_cont, K_cont, 3)
        Return the integral over the momentum operator associated with the contraction in the given
        coordinate system.
        `K_cont` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform, coord_type) : np.ndarray(K_orbs, K_orbs, 3)
        Return the integral over the momentum operator associated with linear combinations of
        spherical Gaussians (linear combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def construct_array_contraction(contractions_one, contractions_two):
        """Return the integrals over the momentum operator of the given contractions.

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
            Integral over than momentum operator associated with the given instances of
            `GeneralizedContractionShell`.
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
            Dimension 4 corresponds to the dimension of the momentum (x, y, z).

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

        output = _compute_differential_operator_integrals(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            contractions_one.coord,
            contractions_one.angmom_components_cart,
            contractions_one.exps,
            contractions_one.coeffs,
            contractions_one.norm_prim_cart,
            contractions_two.coord,
            contractions_two.angmom_components_cart,
            contractions_two.exps,
            contractions_two.coeffs,
            contractions_two.norm_prim_cart,
        )
        return -1j * np.transpose(output, (1, 2, 3, 4, 0))


def momentum_integral(basis, transform=None):
    r"""Return integral over momentum operator of the given basis set.

    .. math::

        \left< \hat{\mathbf{p}} \right>
        &= \int \phi_a(\mathbf{r}) \left( -i \nabla \right) \phi_b(\mathbf{r}) d\mathbf{r}\\
        &= -i
        \begin{bmatrix}
        \int \phi_a(\mathbf{r}) \frac{\partial}{\partial x} \phi_b(\mathbf{r}) d\mathbf{r}\\\\
        \int \phi_a(\mathbf{r}) \frac{\partial}{\partial y} \phi_b(\mathbf{r}) d\mathbf{r}\\\\
        \int \phi_a(\mathbf{r}) \frac{\partial}{\partial z} \phi_b(\mathbf{r}) d\mathbf{r}
        \end{bmatrix}

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Momentum integral of the given basis set.
        Dimensions 0 and 1 of the array correspond to the basis functions. `K_orbs` is the
        number of basis functions in the basis set.

    """
    coord_type = [ct for ct in [shell.coord_type for shell in basis]]

    if transform is not None:
        return MomentumIntegral(basis).construct_array_lincomb(transform, coord_type)
    if all(ct == "cartesian" for ct in coord_type):
        return MomentumIntegral(basis).construct_array_cartesian()
    if all(ct == "spherical" for ct in coord_type):
        return MomentumIntegral(basis).construct_array_spherical()
    return MomentumIntegral(basis).construct_array_mix(coord_type)
