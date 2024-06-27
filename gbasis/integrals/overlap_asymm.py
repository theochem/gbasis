"""Functions for computing overlap between two basis sets."""
from gbasis.base_two_asymm import BaseTwoIndexAsymmetric
from gbasis.integrals.overlap import Overlap


class OverlapAsymmetric(BaseTwoIndexAsymmetric):
    """Class for obtaining the overlap between two sets of Gaussian contractions.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Sets of contractions associated with each axis of the array.
    contractions_one : tuple of GeneralizedContractionShell
        Contractions that are associated with the first index of the array.
        Property of `OverlapAsymmetric`.
    contractions_two : tuple of GeneralizedContractionShell
        Contractions that are associated with the second index of the array.
        Property of `OverlapAsymmetric`.

    Methods
    -------
    __init__(self, contractions_one, contractions_two)
        Initialize.
    construct_array_contraction(contractions_one, contractions_two) :
        **np.ndarray(M_1, L_cart_1, M_2, L_cart_2)**
        Return the overlap associated with the two `GeneralizedContractionShell` instances.
        `M_1` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the first index.
        `L_cart_1` is the number of Cartesian contractions for the given angular momentum
        associated with the first index.
        `M_2` is the number of segmented contractions with the same exponents (and angular
        momentum) associated with the second index.
        `L_cart_2` is the number of Cartesian contractions for the given angular momentum
        associated with the second index.
    construct_array_cartesian(self) : np.ndarray(K_cart_1, K_cart_2)
        Return the overlap integrals between Cartesian Gaussians of the two basis sets.
        `K_cart_1` is the total number of Cartesian contractions within `contractions_one`.
        `K_cart_2` is the total number of Cartesian contractions within `contractions_two`.
    construct_array_spherical(self) : np.ndarray(K_sph_1, K_sph_2)
        Return the overlap integrals associated with spherical Gaussians (atomic orbitals) of the
        two basis sets.
        `K_sph_1` is the total number of spherical contractions within `contractions_one`.
        `K_sph_2` is the total number of spherical contractions within `contractions_two`.
    construct_array_mix(self, coord_types_one, coord_types_two) :
        **np.ndarray(K_cont_1, K_cont_2)**

        Return the overlap integrals associated with the contractions in the given coordinate
        systems of the two basis sets.
        `K_cont_1` is the total number of contractions within the given basis set.
        `K_cont_2` is the total number of contractions within the given basis set.
    construct_array_lincomb(self, transform_one, transform_two, coord_type_one, coord_type_two) :
        **np.ndarray(K_orbs_1, K_orbs_2)**

        Return the overlap integrals associated with the linear combinations of contractions of the
        two basis sets.
        `K_orbs_1` is the number of basis functions produced after the linear combinations of
        the spherical contractions associated with `contractions_one`.
        `K_orbs_2` is the number of basis functions produced after the linear combinations of
        the spherical contractions associated with `contractions_two`.

    """

    construct_array_contraction = staticmethod(Overlap.construct_array_contraction)


def overlap_integral_asymmetric(
    basis_one,
    basis_two,
    transform_one=None,
    transform_two=None,
):
    """Return overlap integrals between two basis sets.

    .. math::

        \int \phi_a (\mathbf{r}) \psi_b (\mathbf{r}) d\mathbf{r}

    where :math:`\phi_a` and :math:`\psi_b` are the basis functions in `basis_one` and `basis_two`.

    Parameters
    ----------
    basis_one : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    basis_two : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    transform_one : np.ndarray(K_orbs, K_cont)
        Transformation matrix of the `basis_one` in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.
    transform_one : np.ndarray(K_orbs, K_cont)
        Transformation matrix of the `basis_two` in the given coordinate system (e.g. AO) to linear
        combinations of contractions (e.g. MO).
        Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
        and index 0 of the array for contractions.
        Default is no transformation.

    Returns
    -------
    array : np.ndarray(K_orbs_1, K_orbs_2)
        Overlap integral of the given basis set.
        Dimensions 0 of the array correspond to the basis functions in `basis_one`.
        `K_orbs_1` is the number of basis functions in the `basis_one`.
        Dimensions 1 of the array correspond to the basis functions in `basis_two`.
        `K_orbs_2` is the number of basis functions in the `basis_two`.

    """
    coord_type_one = [ct for ct in [shell.coord_type for shell in basis_one]]
    coord_type_two = [ct for ct in [shell.coord_type for shell in basis_two]]

    return OverlapAsymmetric(basis_one, basis_two).construct_array_lincomb(
        transform_one, transform_two, coord_type_one, coord_type_two
    )
