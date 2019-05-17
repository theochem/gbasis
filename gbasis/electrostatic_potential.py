"""Module for computing the electrostatic potential integrals."""
from gbasis.point_charge import PointChargeIntegral


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
