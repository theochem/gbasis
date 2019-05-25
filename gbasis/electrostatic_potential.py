"""Module for computing the electrostatic potential integrals."""
from gbasis.point_charge import point_charge_cartesian, point_charge_spherical
import numpy as np


# FIXME: check if density_matrix has the right shape. otherwise, hartree potential needs to be
# obtained first, which is quite expensive
def _electrostatic_potential_base(
    base_func,
    basis,
    density_matrix,
    coords_points,
    nuclear_coords,
    nuclear_charges,
    threshold_dist=0.0,
):
    """Return the electrostatic potentials of the basis set in the Cartesian form.

    Parameters
    ----------
    base_func : {point_charge_cartesian, point_charge_spherical}
        Function.
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    density_matrix : np.ndarray(K, K)
        Density matrix constructed using the given basis set.
    coords_points : np.ndarray(N, 3)
        Points at which the electrostatic potential is evaluated.
        Rows correspond to the points and columns correspond to the x, y, and z components.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
        Rows correspond to the atoms and columns correspond to the x, y, and z components.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.
    threshold_dist : {float, 0.0}
        Threshold for rejecting nuclei whose distances to the points are less than the provided
        value. i.e. nuclei that are closer to the point than the threshold are discarded when
        computing the electrostatic potential of the point.
        Default value is 0.0, i.e. no nuclei are discarded.

    Returns
    -------
    array : np.ndarray(K, N)
        Array associated with the given basis set.
        First index of the array is associated with the contraction. `K` is the number of
        contractions.
        Second index of the array is associated with the points at whcih the electrostatic potential
        is evaluated. `N` is the number of points.

    Raises
    ------
    TypeError
        If `density_matrix_cart` is not a two-dimensional numpy array.
        If `nuclear_coords` is not a two-dimensional numpy array with 3 columns.
        If `nuclear_charges` is not a one-dimensional numpy array.
        If `threshold_dist` is not a int/float.
    ValueError
        If `density_matrix_cart` must be a symmetric (square) matrix.
        If bumber of rows in `nuclear_coords` is not equal to the number of elements in
        `nuclear_charges`.
        If `threshold_dist` is less than 0.

    Notes
    -----
    The density matrix here is expressed with respect to Cartesian contractions. If your density
    matrix is expressed with respect to spherical contractions, see
    `gbasis.electrostatic_potential_electrostatic_potential_sphrical`.

    """
    if not (isinstance(density_matrix, np.ndarray) and density_matrix.ndim == 2):
        raise TypeError("`density_matrix_cart` must be given as a two-dimensional numpy array.")
    if not (
        isinstance(nuclear_coords, np.ndarray)
        and nuclear_coords.ndim == 2
        and nuclear_coords.shape[1] == 3
    ):
        raise TypeError("`nuclear_coords` must be a two-dimensional numpy array with 3 columns.")
    if not (isinstance(nuclear_charges, np.ndarray) and nuclear_charges.ndim == 1):
        raise TypeError("`nuclear_charges` must be a one-dimensional numpy array.")

    if not (
        density_matrix.shape[0] == density_matrix.shape[1]
        and np.allclose(density_matrix, density_matrix.T)
    ):
        raise ValueError("`density_matrix_cart` must be a symmetric (square) matrix.")
    if nuclear_coords.shape[0] != nuclear_charges.size:
        raise ValueError(
            "Number of rows in `nuclear_coords` must be equal to the number of elements in "
            "`nuclear_charges`."
        )
    if not isinstance(threshold_dist, (int, float)):
        raise TypeError("`threshold_dist` must be a int/float.")
    if threshold_dist < 0:
        raise ValueError("`threshold_dist` must be greater than or equal to zero.")

    hartree_potential = base_func(basis, coords_points, -np.ones(coords_points.shape[0]))
    hartree_potential *= density_matrix[:, :, None]
    hartree_potential = np.sum(hartree_potential, axis=(0, 1))

    # silence warning for dividing by zero
    old_settings = np.seterr(divide="ignore")
    external_potential = (
        nuclear_charges[None, :]
        / np.sum((coords_points[:, :, None] - nuclear_coords.T[None, :, :]) ** 2, axis=1) ** 0.5
    )
    # zero out potentials of elements that are too close to the nucleus
    external_potential[external_potential > 1.0 / np.array(threshold_dist)] = 0
    # restore old settings
    np.seterr(**old_settings)
    # sum over potentials for each dimension
    external_potential = -np.sum(external_potential, axis=1)

    return -(external_potential + hartree_potential)


def electrostatic_potential_cartesian(
    basis, density_matrix_cart, coords_points, nuclear_coords, nuclear_charges
):
    """Return the electrostatic potentials of the basis set in the Cartesian form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    density_matrix_cart : np.ndarray(K_cart, K_cart)
        Density matrix constructed using the Cartesian forms of the given basis set.
    coords_points : np.ndarray(N, 3)
        Points at which the electrostatic potential is evaluated.
        Rows correspond to the points and columns correspond to the x, y, and z components.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
        Rows correspond to the atoms and columns correspond to the x, y, and z components.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.

    Returns
    -------
    array : np.ndarray(K_cart, N)
        Array associated with the given set of Cartesian contractions.
        First index of the array is associated with the contraction. `K_cart` is the number of
        Cartesian contractions.
        Second index of the array is associated with the points at whcih the electrostatic potential
        is evaluated. `N` is the number of points.

    Raises
    ------
    TypeError
        If `density_matrix_cart` is not a two-dimensional numpy array.
        If `nuclear_coords` is not a two-dimensional numpy array with 3 columns.
        If `nuclear_charges` is not a one-dimensional numpy array.
    ValueError
        If `density_matrix_cart` must be a symmetric (square) matrix.
        If bumber of rows in `nuclear_coords` is not equal to the number of elements in
        `nuclear_charges`.

    Notes
    -----
    The density matrix here is expressed with respect to Cartesian contractions. If your density
    matrix is expressed with respect to spherical contractions, see
    `gbasis.electrostatic_potential_electrostatic_potential_sphrical`.

    """
    return _electrostatic_potential_base(
        point_charge_cartesian,
        basis,
        density_matrix_cart,
        coords_points,
        nuclear_coords,
        nuclear_charges,
    )


def electrostatic_potential_spherical(
    basis, density_matrix_sph, coords_points, nuclear_coords, nuclear_charges
):
    """Return the electrostatic potentials of the basis set in the spherical form.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    density_matrix_sph : np.ndarray(K_sph, K_sph)
        Density matrix constructed using the Cartesian forms of the given basis set.
    coords_points : np.ndarray(N, 3)
        Points at which the electrostatic potential is evaluated.
        Rows correspond to the points and columns correspond to the x, y, and z components.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
        Rows correspond to the atoms and columns correspond to the x, y, and z components.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.

    Returns
    -------
    array : np.ndarray(K_sph, N)
        Array associated with the given set of spherical contractions.
        First index of the array is associated with the spherical contraction. `K_cont` is the
        number of spherical contractions.
        Second index of the array is associated with the points at whcih the electrostatic potential
        is evaluated. `N` is the number of points.

    Raises
    ------
    TypeError
        If `density_matrix_cart` is not a two-dimensional numpy array.
        If `nuclear_coords` is not a two-dimensional numpy array with 3 columns.
        If `nuclear_charges` is not a one-dimensional numpy array.
    ValueError
        If `density_matrix_cart` must be a symmetric (square) matrix.
        If bumber of rows in `nuclear_coords` is not equal to the number of elements in
        `nuclear_charges`.

    Notes
    -----
    The density matrix here is expressed with respect to spherical contractions. If your density
    matrix is expressed with respect to Cartesian contractions, see
    `gbasis.electrostatic_potential_electrostatic_potential_cartesian`.

    """
    return _electrostatic_potential_base(
        point_charge_spherical,
        basis,
        density_matrix_sph,
        coords_points,
        nuclear_coords,
        nuclear_charges,
    )
