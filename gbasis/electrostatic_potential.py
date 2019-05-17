"""Module for computing the electrostatic potential integrals."""
from gbasis.point_charge import PointChargeIntegral
from scipy.special import hyp1f1  # pylint: disable=E0611


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

    @classmethod
    def construct_array_contraction(self, contractions_one, contractions_two, coord_point):
        """Return electrostatic potential for the given contractions.

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

        Raises
        ------
        TypeError
            If `contractions_one` is not a ContractedCartesianGaussians instance.
            If `contractions_two` is not a ContractedCartesianGaussians instance.
            If `coord_point` is not a one-dimensional numpy array with three elements.
            If `charge_point` is not an integer or a float.

        """
        return super().construct_array_contraction(
            contractions_one, contractions_two, coord_point, -1
        )


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
