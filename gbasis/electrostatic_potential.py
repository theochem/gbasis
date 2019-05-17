"""Module for computing the electrostatic potential integrals."""
from gbasis.point_charge import PointChargeIntegral


def electrostatic_potential_basis_cartesian(basis, coord_point):
    """Return the electrostatic potentials of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coord_point : np.ndarray(3,)
        Center of the point charge.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart)
        Array associated with the given set of contracted Cartesian Gaussians.
        First and second indices of the array are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return PointChargeIntegral(basis).construct_array_cartesian(
        coord_point=coord_point, charge_point=-1
    )


def electrostatic_potential_basis_spherical(basis, coord_point):
    """Return the electrostatic potentials of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    coord_point : np.ndarray(3,)
        Center of the point charge.

    Returns
    -------
    array : np.ndarray(K_sph, K_sph)
        Array associated with the atomic orbitals associated with the given set(s) of contracted
        Cartesian Gaussians.
        First and second indices of the array are associated with two contracted spherical Gaussians
        (atomic orbitals). `K_sph` is the total number of spherical contractions within the
        instance.

    """
    return PointChargeIntegral(basis).construct_array_spherical(
        coord_point=coord_point, charge_point=-1
    )


def electrostatic_potential_spherical_lincomb(basis, transform, coord_point):
    """Return the electrostatic potentials of the LCAO's in the spherical form.

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

    Returns
    -------
    array : np.ndarray(K_orbs, K_orbs)
        Array whose first and second indices are associated with the linear combinations of the
        contracted spherical Gaussians.
        First and second indices of the array correspond to the linear combination of contracted
        spherical Gaussians. `K_orbs` is the number of basis functions produced after the linear
        combinations.

    """
    return PointChargeIntegral(basis).construct_array_spherical_lincomb(
        transform, coord_point=coord_point, charge_point=-1
    )
