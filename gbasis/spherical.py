"""Convert from Cartesian Gaussians to Spherical Gaussians."""
from gbasis.deriv import eval_shell
from gbasis.lincomb import lincomb_blocks_evals
import numpy as np
from scipy.special import comb, factorial


def shift_factor(mag):
    """Calculate the shift factor for solid harmonics.

    shift_factor = 0 if mag >= 0 and shift_factor = 1/2 if mag < 0.

    Parameters
    ----------
    mag : int
        The magnetic quantum number of the basis function.

    Returns
    -------
    shift_factor : float
        The shift factor for calculating solid harmonics.

    Raises
    ------
    TypeError
        If mag is not an integer.

    """
    if not isinstance(mag, int):
        raise TypeError("The magnetic quantum number must be an integer.")

    return np.piecewise(float(mag), [mag < 0, mag >= 0], [0.5, 0])


def expansion_coeff(angmom, mag, i, j, k):
    r"""Calculate the real solid harmonic expansion coefficient.

    .. math::

        C^angmom,mag_i,j,k = -1^{i + k - shift_factor} * (1/4)^i * {angmom \choose i}
        * {(angmom - i) \choose (\abs{mag} + i)} * {i \choose j} * {\abs{mag} \choose 2 * k},

    where shift_factor = 0 if mag >= 0 and shift_factor = 1/2 if mag < 0.

    Parameters
    ----------
    angmom : int
        The angular momentum of the Gaussian primitive(s).
    mag : int
        The magnetic quantum number(s) of the Gaussian primitive(s).
    i, j : int
        The generator indices for the expansion coefficient.
    k : float
        The generator indices for the expansion coefficient.

    Returns
    -------
    coeff : float
        The real solid harmonic expansion coefficient.

    Raises
    ------
    TypeError
        If angmom is not an integer.
        If mag is not an integer.
        If i or j is not an integer.
        If k is not a float.
    ValueError
        If angmom is negative.
        If mag has a greater magnitude than angmom.
        If k is not either an integer (mag >= 0) or a half integer (mag < 0).

    """
    if not isinstance(angmom, int):
        raise TypeError("Angular momentum must be an integer.")
    if angmom < 0:
        raise ValueError("Angular momentum must be a non-negative integer.")
    if not isinstance(mag, int):
        raise TypeError("The magnetic quantum number must be an integer.")
    if np.abs(mag) > angmom:
        raise ValueError("The magnetic quantum number must be between -(angmom) and angmom.")
    if not isinstance(i, int):
        raise TypeError("Index i must be an integer")
    if not isinstance(j, int):
        raise TypeError("Index j must be an integer")
    if isinstance(k, int):
        k = float(k)
    if not isinstance(k, float):
        raise TypeError("Index k must be a float.")
    if k != int(k) and mag >= 0:
        raise ValueError("Index k must be an integer for non-negative magnetic quantum numbers.")
    if k != int(k) + 0.5 and mag < 0:
        raise ValueError("Index k must be a half integer for negative magnetic quantum numbers.")

    if mag < 0:
        return np.real(
            (complex(-1)) ** (i + k - shift_factor(mag))
            * (1 / 4) ** i
            * comb(angmom, i)
            * comb(angmom - i, np.abs(mag) + i)
            * comb(i, j)
            * comb(np.abs(mag), 2 * k)
        )
    return (
        (-1) ** (i + k - shift_factor(mag))
        * (1 / 4) ** i
        * comb(angmom, i)
        * comb(angmom - i, np.abs(mag) + i)
        * comb(i, j)
        * comb(np.abs(mag), 2 * k)
    )


def harmonic_norm(angmom, mag):
    """Calculate the normalization constant of a real solid harmonic.

        .. math::

            `N^S_angmom,m = 1/(2^abs{m} * angmom!) * sqrt{2 * (angmom + abs{m})!
            * (angmom - abs{m})! / 2^del{0, m})}`,
            where :math: `del{0, m}` is the Kronecker delta of 0 and m.

    Parameters
    ----------
    angmom : int
        The angular momentum of the basis function(s).
    mag : int
        The magnetic quantum number(s) of the basis function(s).

    Returns
    -------
    norm : float
        The normalization constant of the real solid harmonic.

    Raises
    ------
    TypeError
        If angmom is not an integer.
        If mag is not an integer.
    ValueError
        If angmom is negative.
        If mag has a greater magnitude than angmom.

    """
    if not isinstance(angmom, int):
        raise TypeError("Angular momentum must be an integer.")
    if angmom < 0:
        raise ValueError("Angular momentum must be a non-negative integer.")
    if not isinstance(mag, int):
        raise TypeError("The magnetic quantum number must be an integer.")
    if np.abs(mag) > angmom:
        raise ValueError("The magnetic quantum number must be between -(angmom) and angmom.")

    return (1 / (2 ** np.abs(mag) * factorial(angmom))) * np.sqrt(
        (2 * factorial(angmom + np.abs(mag)) * factorial(angmom - np.abs(mag)))
        / (2 ** int(mag == 0))
    )


def real_solid_harmonic(angmom, mag):
    r"""Calculate a real solid harmonic.

    .. math::

        S_angmom,mag = N^S_angmom,mag \sum_i=0^[(angmom-abs{mag})/2] \sum_j=0^i
        \sum_k=shift_factor^([(abs{mag}-1)/2] + 0.5) C^angmom,m_i,j,k * x^a_x * y^a_y * z^a_z,

    where :math:`a_x = 2t + abs{mag} - 2(u + v)`, :math:`a_y = 2(u + v)`,
    and :math:`a_z = angmom - 2t - abs{mag}`.

    Parameters
    ----------
    angmom : int
        The angular momentum of the basis function(s).
    mag : int
        The magnetic quantum number(s) of the basis function(s).

    Returns
    -------
    harmonic : dict
        The Cartesian components of a Gaussian primitive, linked to the associated coefficient.

    Raises
    ------
    TypeError
        If angmom is not an integer.
        If mag is not an integer.
    ValueError
        If angmom is negative.
        If mag has a greater magnitude than angmom.

    """
    if not isinstance(angmom, int):
        raise TypeError("Angular momentum must be an integer.")
    if angmom < 0:
        raise ValueError("Angular momentum must be a non-negative integer.")
    if not isinstance(mag, int):
        raise TypeError("The magnetic quantum number must be an integer.")
    if np.abs(mag) > angmom:
        raise ValueError("The magnetic quantum number must be between -(angmom) and angmom.")

    harmonic = {}
    norm = harmonic_norm(angmom, mag)

    for i, j, k in [
        (x, y, z + shift_factor(mag))
        for x in range((angmom - np.abs(mag)) // 2 + 1)
        for y in range(x + 1)
        for z in range(np.abs(mag) // 2 + 1)
    ]:
        # Multiply by n to easily make transform. matrix
        coeff = expansion_coeff(angmom, mag, i, j, k) * norm
        # a_x, a_y, a_z are the components of a Cartesian Gaussian primitive
        a_x = int(2 * i + np.abs(mag) - (2 * (j + k)))
        a_y = int(2 * (j + k))
        a_z = int(angmom - 2 * i - np.abs(mag))
        if coeff != 0:
            harmonic[(a_x, a_y, a_z)] = coeff

    return harmonic


def generate_transformation(angmom, cartesian_order, apply_from):
    r"""Generate the transformation matrix for a given shell.

    The Cartesian primitives are ordered as given in cartesian_order. The spherical
    Gaussians are ordered in ascending order from -(angmom) to angmom.

    Parameters
    ----------
    angmom : int
        The angular momentum of the basis function(s).
    cartesian_order : np.ndarray((angmom + 1) * (angmom + 2) / 2, 3)
        The x, y, and z components of the angular momentum for each primitive,
        (:math:`\vec{a} = (a_x, a_y, a_z)`, where :math:`a_x + a_y + a_z = l`).
        The order in which the primitives are given defines the order in which
        they will appear in the transformation matrix.
    apply_from : str
        The side on which the transformation matrix is applied. One of {"left", "right"}.

    Returns
    -------
    transform : np.ndarray((angmom + 1)*(angmom + 2) / 2, 2 * angmom + 1)
        The transformation matrix from Cartesian primitives to spherical primitives.

    Raises
    ------
    TypeError
        If angmom is not an integer.
        If cartesian_order is not an array.
        If each member of cartesian_order is not a tuple.
    ValueError
        If angmom is negative.
        If cartesian_order does not have shape (angmom, 3).
        If the components of any primitive do not sum to angmom.

    """
    if not isinstance(angmom, int):
        raise TypeError("Angular momentum must be an integer.")
    if angmom < 0:
        raise ValueError("Angular momentum must be a non-negative integer.")
    if not isinstance(cartesian_order, np.ndarray):
        raise TypeError("The order of the Cartesian primitives must be an array.")
    if not cartesian_order.shape == ((angmom + 1) * (angmom + 2) // 2, 3):
        raise ValueError(
            "The order of the Cartesian primitives must have dimensions {}".format(
                ((angmom + 1) * (angmom + 2) // 2, 3)
            )
        )
    if not np.all(np.sum(cartesian_order, axis=1) == angmom):
        raise ValueError("Each primitive's components must sum to the angular momentum.")
    if not isinstance(apply_from, str):
        raise TypeError("Specify the side on which the transformation is applied as a string")
    if apply_from not in ["left", "right"]:
        raise ValueError(
            "Specify the side of application for the transformation using 'left' or 'right'"
        )

    order = {
        components: index for index, components in enumerate(list(map(tuple, cartesian_order)))
    }
    transform = np.zeros(((angmom + 1) * (angmom + 2) // 2, 2 * angmom + 1))

    for mag in range(-angmom, angmom + 1):
        harmonic = real_solid_harmonic(angmom, mag)
        for components, coeff in harmonic.items():
            transform[order[components], mag + angmom] = coeff

    if apply_from == "left":
        return transform.T
    return transform


def eval_spherical_shell(*, coords, shell):
    """Return the set of spherical contractions evaluated at the given coordinates.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Point in space where the Gaussian primitive is evaluated.
    shell : ContractedCartesianGaussians
        Set of contracted Cartesian Gaussians with the same angular momentum.

    Returns
    -------
    derivative : np.ndarray(L, N)
        Evaluation of the (zeroth-order) derivative.
        :math:`L` is the number of contractions associated with the given `shell`.

    Raises
    ------
    TypeError
        If the arguments are given as positional arguments.
        If coords is not a numpy array.
        If shell is not a ContractedCartesianGaussians.
    ValueError
        If coords is not a two-dimensional numpy array with 3 columns.

    Notes
    -----
    When calling this function, the arguments must be given via keywords and not positional
    arguments. This feature is used to catch problems that arise due to a change in API.

    """
    return lincomb_blocks_evals(
        [shell],
        eval_shell,
        [generate_transformation(shell.angmom, shell.angmom_components, apply_from="left")],
        coords=coords,
    )
