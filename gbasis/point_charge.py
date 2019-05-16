"""Module for computing point charge integrals."""
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
from scipy.special import hyp1f1  # pylint: disable=E0611


class PointChargeIntegral(BaseTwoIndexSymmetric):
    r"""General class for calculating one-electron integrals for interaction with a point charge.

    One electron integrals are assumed to have the following form:
    .. math::

        \int \phi_a(\mathbf{r}) \phi_b(\mathbf{r}) g(|\mathbf{r} - \mathbf{R}_C|, Z) d\mathbf{r}

    where :math:`\phi_a` is Gaussian contraction on the left, :math:`\phi_b` is the Gaussian
    contraction on the right, :math:`\mathbf{R}_C` is the position of the point charge, :math:`Z` is
    the charge at the point charge, and :math:`g(|\mathbf{r} - \mathbf{R}_C|, Z)` determines the
    type of the one-electorn integral.


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
    boys_func : np.ndarray(np.ndarray(M, K_b, K_a))
        Boys function used to evaluate the one-electron integral.
        `M` is the number of orders that will be evaluated. `K_a` and `K_b` are the number of
        primitives on the left and right side, respectively.
    construct_array_contraction(self, contractions_one, contractions_two, coord_point, charge_point
        Return the point charge integrals for the given `ContractedCartesianGaussians` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the one-electron integrals associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the one-electron integrals associated with spherical Gaussians (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform) : np.ndarray(K_orbs, K_orbs)
        Return the one-electron integrals associated with linear combinations of spherical Gaussians
        (linear combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    # pylint: disable=W0613
    @staticmethod
    def boys_func(orders, weighted_dist):
        r"""Return the value of Boys function for the given orders and weighted distances.

        Parameters
        ----------
        orders : np.ndarray(M, 1, 1)
            Differentiation order of the helper function.
            Same as m in eq. 23, Aldrichs, R. Phys. Chem. Chem. Phys., 2006, 8, 3072-3077.
            `M` is the number of orders that will be evaluated.
        weighted_dist : np.ndarray(1, K_b, K_a)
            Weighted interatomic distance
            .. math::

                \frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{AB}||^2

            where :math:`\alpha_i` is the exponent of the ith primitive on the left side and the
            :math:`\beta_j` is the exponent of the jth primitive on the right side.
            `K_a` and `K_b` are the number of primitives on the left and right side, respectively.
            Note that the index 1 corresponds to the primitive on the right side and the index 2
            corresponds to the primitive on the left side.

        Returns
        -------
        boys_eval : np.ndarray(M, K_b, K_a)
            Output is the Boys function evaluated for each order and the weighted interactomic
            distance.

        """
        return None

    # pylint: disable=R0914
    @staticmethod
    def construct_array_contraction(contractions_one, contractions_two, coord_point, boys_func):
        """Return the evaluations of the point charge interaction for the given contractions.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        coord_point : np.ndarray(3,)
            Center of the point charge.
        boys_func : function(order : int, weighted_dist : np.ndarray(L_b, L_a))
            Boys function used to evaluate the one-electron integral.
            `L_a` and `L_b` are the angular momentum of contraction one and two respectively.

        """

        if not isinstance(contractions_one, ContractedCartesianGaussians):
            raise TypeError("`contractions_one` must be a ContractedCartesianGaussians instance.")
        if not isinstance(contractions_two, ContractedCartesianGaussians):
            raise TypeError("`contractions_two` must be a ContractedCartesianGaussians instance.")
        if not (
            isinstance(coord_point, np.ndarray) and coord_point.ndim == 1 and coord_point.size == 3
        ):
            raise TypeError("`coord_point` must be a one-dimensional numpy array with 3 elements.")
        try:
            boys_func(order=1, weighted_dist=np.arange(4).reshape((2, 2)))
        except ValueError:
            raise ValueError("`boys_func` must accept two arguments, `order` and `weighted_dist`")

        # TODO: Overlap screening

        coord_a = contractions_one.coord
        angmom_a = contractions_one.angmom
        angmoms_a = contractions_one.angmom_components
        exps_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        coord_b = contractions_two.coord
        angmom_b = contractions_two.angmom
        angmoms_b = contractions_two.angmom_components
        exps_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs

        # Enforce L_a >= L_b
        if angmom_a < angmom_b:
            integrals = _compute_one_elec_integrals(
                coord_point,
                boys_func,
                coord_b,
                angmom_b,
                exps_b,
                coeffs_b,
                coord_a,
                angmom_a,
                exps_a,
                coeffs_a,
            )
            integrals = np.transpose(integrals, (2, 3, 0, 1))
        else:
            integrals = _compute_one_elec_integrals(
                coord_point,
                boys_func,
                coord_a,
                angmom_a,
                exps_a,
                coeffs_a,
                coord_b,
                angmom_b,
                exps_b,
                coeffs_b,
            )

        # Generate output array
        # Ordering for output array:
        # axis 0 : index for segmented contractions of contraction one
        # axis 1 : angular momentum vector of contraction one (in the same order as angmoms_a)
        # axis 2 : index for segmented contractions of contraction two
        # axis 3 : angular momentum vector of contraction two (in the same order as angmoms_b)
        output = np.zeros(
            (
                len(coeffs_a[0]),
                contractions_one.num_contr,
                len(coeffs_b[0]),
                contractions_two.num_contr,
            )
        )

        for i, comp_a in enumerate(angmoms_a):
            a_x, a_y, a_z = comp_a
            for j, comp_b in enumerate(angmoms_b):
                b_x, b_y, b_z = comp_b
                output[:, i, :, j] = integrals[a_x, a_y, a_z, b_x, b_y, b_z, :, :]

        return output


def one_electron_integral_basis_cartesian(basis, coord_point, boys_func):
    """Return the one-electron integrals of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coord_point : np.ndarray(3,)
        Center of the point charge.
    boys_func : function(order : int, weighted_dist : np.ndarray(L_b, L_a))
        Boys function used to evaluate the one-electron integral.
        `L_a` and `L_b` are the angular momentum of contraction one and two respectively.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart, D)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return PointChargeIntegral(basis).construct_array_cartesian(
        coord_point=coord_point, boys_func=boys_func
    )


def one_electron_integral_basis_spherical(basis, coord_point, boys_func):
    """Return the one-electron integrals of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coord_point : np.ndarray(3,)
        Center of the point charge.
    boys_func : function(order : int, weighted_dist : np.ndarray(L_b, L_a))
        Boys function used to evaluate the one-electron integral.
        `L_a` and `L_b` are the angular momentum of contraction one and two respectively.

    Returns
    -------
    array : np.ndarray(K_sph, K_sph, D)
        Array associated with the atomic orbitals associated with the given set(s) of contracted
        Cartesian Gaussians.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    """
    return PointChargeIntegral(basis).construct_array_spherical(
        coord_point=coord_point, boys_func=boys_func
    )


def one_electron_integral_spherical_lincomb(basis, transform, coord_point, boys_func):
    """Return the one-electron integrals of the LCAO's in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    transform : np.ndarray(K_orbs, K_sph)
        Array associated with the linear combinations of spherical Gaussians (LCAO's).
        Transformation is applied to the left, i.e. the sum is over the second index of `transform`
        and first index of the array for contracted spherical Gaussians.
    coord_point : np.ndarray(3,)
        Center of the point charge.
    boys_func : function(order : int, weighted_dist : np.ndarray(L_b, L_a))
        Boys function used to evaluate the one-electron integral.
        `L_a` and `L_b` are the angular momentum of contraction one and two respectively.

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs, D)
        Array whose first and second indices are associated with the linear combinations of the
        contracted spherical Gaussians.
        First and second indices of the array correspond to the linear combination of contracted
        spherical Gaussians. `K_orbs` is the number of basis functions produced after the linear
        combinations.

    """
    return PointChargeIntegral(basis).construct_array_spherical_lincomb(
        transform, coord_point=coord_point, boys_func=boys_func
    )


class ElectroStaticPotential(PointChargeIntegral):
    """Class for evaluating the electrostatic potential.

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
    boys_func : np.ndarray(np.ndarray(M, K_b, K_a))
        Boys function used to evaluate the one-electron integral.
        `M` is the number of orders that will be evaluated. `K_a` and `K_b` are the number of
        primitives on the left and right side, respectively.
    construct_array_contraction(self, contraction) : np.ndarray(M_1, L_cart_1, M_2, L_cart_2)
        Return the one-electron Coulomb interaction integrals for the given
        `ContractedCartesianGaussians` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum associated
        with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular momentum)
        associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum associated
        with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart, K_cart)
        Return the one-electron Coulomb interaction integrals associated with Cartesian Gaussians.
        `K_cart` is the total number of Cartesian contractions within the instance.
    construct_array_spherical(self) : np.ndarray(K_sph, K_sph)
        Return the one-electron Coulomb interaction integrals associated with spherical Gaussians
        (atomic orbitals).
        `K_sph` is the total number of spherical contractions within the instance.
    construct_array_spherical_lincomb(self, transform) : np.ndarray(K_orbs, K_orbs)
        Return the one-electron Coulomb interaction integrals associated with linear combinations
        of spherical Gaussians
        (linear combinations of atomic orbitals).
        `K_orbs` is the number of basis functions produced after the linear combinations.

    """

    @staticmethod
    def boys_func(orders, weighted_dist):
        r"""Return the value of Boys function for the given orders and weighted distances.

        The Coulombic Boys function can be written as a renormalized special case of the Kummer
        confluent hypergeometric function, as derived in Helgaker (eq. 9.8.39).

        Parameters
        ----------
        orders : np.ndarray(M, 1, 1)
            Differentiation order of the helper function.
            Same as m in eq. 23, Aldrichs, R. Phys. Chem. Chem. Phys., 2006, 8, 3072-3077.
            `M` is the number of orders that will be evaluated.
        weighted_dist : np.ndarray(1, K_b, K_a)
            Weighted interatomic distance
            .. math::

                \frac{\alpha_i \beta_j}{\alpha_i + \beta_j} * ||R_{AB}||^2

            where :math:`\alpha_i` is the exponent of the ith primitive on the left side and the
            :math:`\beta_j` is the exponent of the jth primitive on the right side.
            `K_a` and `K_b` are the number of primitives on the left and right side, respectively.
            Note that the index 1 corresponds to the primitive on the right side and the index 2
            corresponds to the primitive on the left side.

        Returns
        -------
        boys_eval : np.ndarray(M, K_b, K_a)
            Output is the Boys function evaluated for each order and the weighted interactomic
            distance.

        Notes
        -----
        There's some documented instability for hyp1f1, mainly for large values or complex numbers.
        In this case it seems fine, since m should be less than 10 in most cases, and except for
        exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
        problem becomes a precision error in most cases where it was an overflow error before, so
        the values should be close even when they are wrong.
        This function cannot be vectorized for both m and x.

        """
        return hyp1f1(orders + 1 / 2, orders + 3 / 2, -weighted_dist) / (2 * orders + 1)
