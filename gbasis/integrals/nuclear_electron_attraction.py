"""Module for computing the nuclear electron attraction."""
from gbasis.integrals.point_charge import point_charge_integral
import numpy as np


def nuclear_electron_attraction_integral(basis, nuclear_coords, nuclear_charges, transform=None):
    """Return the nuclear electron attraction integrals of the basis set in the Cartesian form.

    .. math::

        \sum_C \int \phi_a(\mathbf{r}) \frac{-Z_C}{|\mathbf{r} - \mathbf{R}_C|} \phi_b(\mathbf{r})
    d\mathbf{r}

    where :math:`\mathbf{R}_C` is the coordinate of the nucleus :math:`C` and :math:`Z_C` its charge.

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Contracted Cartesian Gaussians (of the same shell) that will be used to construct an array.
    nuclear_coords : np.ndarray(N_nuc, 3)
        Coordinates of each atom.
    nuclear_charges : np.ndarray(N_nuc)
        Charges of each atom.
    transform : np.ndarray(K, K_cont)
        Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.

    Returns
    -------
    array : np.ndarray(K_cart, K_cart)
        Array associated with the given set of contracted Cartesian Gaussians.
        Dimensions 0 and 1 are associated with the contracted Cartesian
        Gaussians. `K_cart` is the total number of Cartesian contractions within the instance.

    """
    return np.sum(
        point_charge_integral(basis, nuclear_coords, nuclear_charges, transform=transform),
        axis=2,
    )
