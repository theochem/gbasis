"""Module for computing electrostatic potential integrals."""
from gbasis.integrals.point_charge import point_charge_integral
import numpy as np


def electrostatic_potential(
    basis,
    one_density_matrix,
    points,
    nuclear_coords,
    nuclear_charges,
    transform=None,
    threshold_dist=0.0,
):
    r"""Return the electrostatic potentials of the basis set in the Cartesian form.

    .. math::

        - \left(
            - \sum_A \frac{Z_A}{|\mathbf{R}_C - \mathbf{R}_A|}
            + \sum_{ab} \gamma_{ab} \int \phi_a(\mathbf{r}) \frac{-1}{|\mathbf{r} - \mathbf{R}_C|} \phi_b(\mathbf{r}) d\mathbf{r}
        \right)

    where :math:`\mathbf{R}_C` is the coordinate of a unitary point charge, :math:`\mathbf{R}_A` is the
    coordinate of the nucleus :math:`A`, :math:`Z_A` its charge, and :math:`\gamma_{ab}` is the
    one-electron density matrix.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    one_density_matrix : np.ndarray(K_orbs, K_orbs)
        One-electron density matrix in terms of the given basis set.
        If the basis is transformed using `transform` keyword, then the density matrix is assumed
        to be expressed with respect to the transformed basis set.
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the basis functions
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Cartesian coordinates of each atom.
        Rows correspond to the atoms and columns correspond to the :math:`x, y, \text{and} z`
        components.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.
    transform : np.ndarray(K_orbs, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    threshold_dist : {float, 0.0}
        Threshold for rejecting nuclei whose distances to the points are less than the provided
        value. i.e. nuclei that are closer to the point than the threshold are discarded when
        computing the electrostatic potential of the point.
        Default value is 0.0, i.e. no nuclei are discarded.

    Returns
    -------
    array : np.ndarray(N)
        Electrostatic potential evaluated at the given points.

    Raises
    ------
    TypeError
        If `one_density_matrix` is not a two-dimensional `numpy` array.
        If `nuclear_coords` is not a two-dimensional `numpy` array with 3 columns.
        If `nuclear_charges` is not a one-dimensional `numpy` array.
        If `threshold_dist` is not an integer or float.
    ValueError
        If `one_density_matrix` is not a symmetric (square) matrix.
        If number of rows in `nuclear_coords` is not equal to the number of elements in
        `nuclear_charges`.
        If `threshold_dist` is less than 0.

    """
    # pylint: disable=R0912
    if not (isinstance(one_density_matrix, np.ndarray) and one_density_matrix.ndim == 2):
        raise TypeError(
            "`one_density_matrix_cart` must be given as a two-dimensional `numpy` array."
        )
    if not (
        isinstance(nuclear_coords, np.ndarray)
        and nuclear_coords.ndim == 2
        and nuclear_coords.shape[1] == 3
    ):
        raise TypeError("`nuclear_coords` must be a two-dimensional `numpy` array with 3 columns.")
    if not (isinstance(nuclear_charges, np.ndarray) and nuclear_charges.ndim == 1):
        raise TypeError("`nuclear_charges` must be a one-dimensional `numpy` array.")

    if not (
        one_density_matrix.shape[0] == one_density_matrix.shape[1]
        and np.allclose(one_density_matrix, one_density_matrix.T)
    ):
        raise ValueError("`one_density_matrix_cart` must be a symmetric (square) matrix.")
    if nuclear_coords.shape[0] != nuclear_charges.size:
        raise ValueError(
            "Number of rows in `nuclear_coords` must be equal to the number of elements in "
            "`nuclear_charges`."
        )
    if not isinstance(threshold_dist, (int, float)):
        raise TypeError("`threshold_dist` must be an integer or float.")
    if threshold_dist < 0:
        raise ValueError("`threshold_dist` must be greater than or equal to zero.")

    coord_type = [ct for ct in [shell.coord_type for shell in basis]]

    if all(ct == "cartesian" for ct in coord_type):
        if sum(cont.num_cart * cont.num_seg_cont for cont in basis) != one_density_matrix.shape[0]:
            raise ValueError(
                "`one_density_matrix` does not have number of rows/columns that is equal to the "
                "total number of Cartesian contractions (atomic orbitals)."
            )
    elif all(ct == "spherical" for ct in coord_type):
        if sum(cont.num_sph * cont.num_seg_cont for cont in basis) != one_density_matrix.shape[0]:
            raise ValueError(
                "`one_density_matrix` does not have number of rows/columns that is equal to the "
                "total number of spherical contractions (atomic orbitals)."
            )
    elif isinstance(coord_type, (list, tuple)):
        if (
            sum(
                cont.num_sph * cont.num_seg_cont
                if j == "spherical"
                else cont.num_cart * cont.num_seg_cont
                for cont, j in zip(basis, coord_type)
            )
            != one_density_matrix.shape[0]
        ):
            raise ValueError(
                "`one_density_matrix` does not have number of rows/columns that is equal to the "
                "total number of contractions in the given coordinate systems (atomic orbitals)."
            )
    else:
        raise TypeError(
            "`coord_type` must be a list/tuple of the strings 'spherical' or 'cartesian'."
        )
    hartree_potential = point_charge_integral(
        basis, points, -np.ones(points.shape[0]), transform=transform
    )
    hartree_potential *= one_density_matrix[:, :, None]
    hartree_potential = np.sum(hartree_potential, axis=(0, 1))

    # silence warning for dividing by zero
    old_settings = np.seterr(divide="ignore")
    external_potential = (
        nuclear_charges[None, :]
        / np.sum((points[:, :, None] - nuclear_coords.T[None, :, :]) ** 2, axis=1) ** 0.5
    )
    # zero out potentials of elements that are too close to the nucleus
    external_potential[external_potential > 1.0 / np.array(threshold_dist)] = 0
    # restore old settings
    np.seterr(**old_settings)
    # sum over potentials for each dimension
    external_potential = -np.sum(external_potential, axis=1)

    return -(external_potential + hartree_potential)
