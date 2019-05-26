"""Module for computing the electrostatic potential integrals."""
from gbasis.point_charge import point_charge_cartesian, point_charge_mix, point_charge_spherical
import numpy as np


def _electrostatic_potential_base(
    basis,
    density_matrix,
    coords_points,
    nuclear_coords,
    nuclear_charges,
    coord_type,
    threshold_dist=0.0,
):
    """Return the electrostatic potentials of the basis set in the Cartesian form.

    Parameters
    ----------
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
    coord_type : {"cartesian", "spherical", list/tuple of "cartesian" or "spherical"}
        Types of the coordinate system for the contractions.
        If "cartesian", then all of the contractions are treated as Cartesian contractions.
        If "spherical", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
        coordinate type of each ContractedCartesianGaussians instance.
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
    # pylint: disable=R0912
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

    if coord_type == "cartesian":
        if sum(cont.num_cart * cont.num_seg_cont for cont in basis) != density_matrix.shape[0]:
            raise ValueError(
                "`density_matrix` does not have number of rows/columns that is equal to the total "
                "number of Cartesian contractions (atomic orbitals)."
            )
        hartree_potential = point_charge_cartesian(
            basis, coords_points, -np.ones(coords_points.shape[0])
        )
    elif coord_type == "spherical":
        if sum(cont.num_sph * cont.num_seg_cont for cont in basis) != density_matrix.shape[0]:
            raise ValueError(
                "`density_matrix` does not have number of rows/columns that is equal to the total "
                "number of spherical contractions (atomic orbitals)."
            )
        hartree_potential = point_charge_spherical(
            basis, coords_points, -np.ones(coords_points.shape[0])
        )
    elif isinstance(coord_type, (list, tuple)):
        if (
            sum(
                cont.num_sph * cont.num_seg_cont
                if j == "spherical"
                else cont.num_cart * cont.num_seg_cont
                for cont, j in zip(basis, coord_type)
            )
            != density_matrix.shape[0]
        ):
            raise ValueError(
                "`density_matrix` does not have number of rows/columns that is equal to the total "
                "number of contractions in the given coordinate systems (atomic orbitals)."
            )
        hartree_potential = point_charge_mix(
            basis, coords_points, -np.ones(coords_points.shape[0]), coord_types=coord_type
        )
    else:
        raise TypeError(
            "`coord_type` must be 'spherical', 'cartesian', or a list/tuple of these strings."
        )
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
    basis, density_matrix_cart, coords_points, nuclear_coords, nuclear_charges, threshold_dist=0.0
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
    threshold_dist : {float, 0.0}
        Threshold for rejecting nuclei whose distances to the points are less than the provided
        value. i.e. nuclei that are closer to the point than the threshold are discarded when
        computing the electrostatic potential of the point.
        Default value is 0.0, i.e. no nuclei are discarded.

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
    `gbasis.electrostatic_potential_electrostatic_potential_spherical`. Otherwise, see
    `gbasis.electrostatic_potential_electrostatic_potential_mix`.

    """
    return _electrostatic_potential_base(
        basis,
        density_matrix_cart,
        coords_points,
        nuclear_coords,
        nuclear_charges,
        "cartesian",
        threshold_dist=threshold_dist,
    )


def electrostatic_potential_spherical(
    basis, density_matrix_sph, coords_points, nuclear_coords, nuclear_charges, threshold_dist=0.0
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
    threshold_dist : {float, 0.0}
        Threshold for rejecting nuclei whose distances to the points are less than the provided
        value. i.e. nuclei that are closer to the point than the threshold are discarded when
        computing the electrostatic potential of the point.
        Default value is 0.0, i.e. no nuclei are discarded.

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
    `gbasis.electrostatic_potential_electrostatic_potential_cartesian`. Otherwise, see
    `gbasis.electrostatic_potential_electrostatic_potential_mix`.

    """
    return _electrostatic_potential_base(
        basis,
        density_matrix_sph,
        coords_points,
        nuclear_coords,
        nuclear_charges,
        "spherical",
        threshold_dist=threshold_dist,
    )


def electrostatic_potential_mix(
    basis,
    density_matrix_mix,
    coords_points,
    nuclear_coords,
    nuclear_charges,
    coord_types,
    threshold_dist=0.0,
):
    """Return the electrostatic potentials of the basis set in the given coordinate systems.

    Parameters
    ----------
    basis : list/tuple of ContractedCartesianGaussians
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    density_matrix_mix : np.ndarray(K_cont, K_cont)
        Density matrix constructed using the Cartesian forms of the given basis set.
    coords_points : np.ndarray(N, 3)
        Points at which the electrostatic potential is evaluated.
        Rows correspond to the points and columns correspond to the x, y, and z components.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
        Rows correspond to the atoms and columns correspond to the x, y, and z components.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.
    coord_types : list/tuple of str
        Types of the coordinate system for each ContractedCartesianGaussians.
        Each entry must be one of "cartesian" or "spherical".
    threshold_dist : {float, 0.0}
        Threshold for rejecting nuclei whose distances to the points are less than the provided
        value. i.e. nuclei that are closer to the point than the threshold are discarded when
        computing the electrostatic potential of the point.
        Default value is 0.0, i.e. no nuclei are discarded.

    Returns
    -------
    array : np.ndarray(K_cont, N)
        Array associated with the given set of contractions in the given coordinate systems.
        First index of the array is associated with the contraction. `K_cont` is the number of
        contractions.
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
    The density matrix here is expressed with respect to contractions of the given coordinate
    systems. If your density matrix is expressed with respect to Cartesian contractions, see
    `gbasis.electrostatic_potential_electrostatic_potential_cartesian`. If your density
    matrix is expressed with respect to spherical contractions, see
    `gbasis.electrostatic_potential_electrostatic_potential_spherical`.

    """
    return _electrostatic_potential_base(
        basis,
        density_matrix_mix,
        coords_points,
        nuclear_coords,
        nuclear_charges,
        coord_types,
        threshold_dist=threshold_dist,
    )
