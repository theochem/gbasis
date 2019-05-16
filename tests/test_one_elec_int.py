"""Test gbasis._one_elec_int."""
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
import pytest
from scipy.special import hyp1f1


def boys_func(order, weighted_dist):
    r"""Boys function for evaluating the one-electron Coulomb interaction integral.

    The Coulombic Boys function can be written as a renormalized special case of the Kummer
    confluent hypergeometric function, as derived in Helgaker (eq. 9.8.39).

    Parameters
    ----------
    order : int
        Differentiation order of the helper function.
        Same as m in eq. 23, Aldrichs, R. Phys. Chem. Chem. Phys., 2006, 8, 3072-3077.
    weighted_dist : np.ndarray(L_b, L_a)
        The weighted interatomic distance, :math:`\\mu * R_{AB}^{2}`, where `\\mu` is the
        harmonic mean of the exponents for primitive one and two,
        :math:`\\mu = \\alpha * \\beta / (\\alpha + \\beta)`.
        `L_a` and `L_b` are the angular momentum of contraction one and two respectively.

    Returns
    -------
    boys_eval : np.ndarray(L_b, L_a)
        The evaluation of the Boys function for the given values. This output corresponds
        to the evaluation for every Gaussian primitive at the given differentiation order.

    Notes
    -----
    There's some documented instability for hyp1f1, mainly for large values or complex numbers.
    In this case it seems fine, since m should be less than 10 in most cases, and except for
    exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
    problem becomes a precision error in most cases where it was an overflow error before, so
    the values should be close even when they are wrong.
    This function cannot be vectorized for both m and x.

    """

    return hyp1f1(order + 1 / 2, order + 3 / 2, -weighted_dist) / (2 * order + 1)


# FIXME: test fails because reference/answer does not normalize contractions
@pytest.mark.skip(
    reason="The answer that were hand-computed were not using normalized contractions."
)
def test_compute_one_elec_int_v_recursion():
    """Test vertical recursion in _one_elec_int._compute_one_elec_integrals."""
    contr_one = ContractedCartesianGaussians(
        3, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.25])
    )
    contr_two = ContractedCartesianGaussians(
        2, np.array([1.5, 2, 3]), 0, np.array([3.0, 4.0]), np.array([0.02, 0.01])
    )
    coord_a = contr_one.coord
    angmom_a = contr_one.angmom
    exps_a = contr_one.exps
    coeffs_a = contr_one.coeffs
    coord_b = contr_two.coord
    angmom_b = contr_two.angmom
    exps_b = contr_two.exps
    coeffs_b = contr_two.coeffs
    answer = _compute_one_elec_integrals(
        np.array([0.0, 0.0, 0.0]),  # coord_point
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
    # Test V(0)(000|000) using hand-calculated values
    assert np.allclose(15.454923601063644, answer[0, 0, 0, 0, 0, 0])

    # Test vertical recursion for one nonzero index
    # Base case for recursion
    assert np.allclose(-1.1698638244929143, answer[1, 0, 0, 0, 0, 0])
    # Recursion loop for all a
    assert np.allclose(32.000027447778216, answer[2, 0, 0, 0, 0, 0])
    assert np.allclose(-1.0616103065740603, answer[3, 0, 0, 0, 0, 0])

    # Test vertical recursion for two nonzero indices
    # Base case for recursion
    assert np.allclose(-3.2916645250193146, answer[2, 1, 0, 0, 0, 0])
    # Recursion loop for all a
    assert np.allclose(88.24749934654083, answer[2, 2, 0, 0, 0, 0])

    # Test vertical recursion for three nonzero indices
    # Base case for recursion
    assert np.allclose(2.9480682730654943, answer[2, 1, 1, 0, 0, 0])
    # Recursion loop for all a
    assert np.allclose(-5.795589744177589, answer[2, 1, 2, 0, 0, 0])


# FIXME: test fails because reference/answer does not normalize contractions
@pytest.mark.skip(
    reason="The answer that were hand-computed were not using normalized contractions."
)
def test_compute_one_elec_int_s_type():
    """Test _one_elec_int._compute_one_electron_integrals for s-type primitives."""
    # ContractedCartesianGaussians(angmom, coord, charge, coeffs, exps)
    s_type_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0]), np.array([0.1])
    )
    s_type_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0]), np.array([0.02])
    )
    coord_a = s_type_one.coord
    angmom_a = s_type_one.angmom
    exps_a = s_type_one.exps
    coeffs_a = s_type_one.coeffs
    coord_b = s_type_two.coord
    angmom_b = s_type_two.angmom
    exps_b = s_type_two.exps
    coeffs_b = s_type_two.coeffs
    s_test = _compute_one_elec_integrals(
        np.array([0.0, 0.0, 0.0]),  # coord_point
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
    # Test output array using hand-calculated values
    p = 0.12
    x_pa = 1 / 6
    y_pa = 1 / 6
    z_pa = 1 / 4
    x_ab = -1
    y_ab = -1
    z_ab = -1.5
    x_pc = 2 / 3
    y_pc = 14 / 12
    z_pc = 7 / 4
    coeff = 3.0

    v0 = np.exp(-17 / 240) * hyp1f1(1 / 2, 3 / 2, -701 / 1200)
    v1 = np.exp(-17 / 240) * hyp1f1(1 + 1 / 2, 1 + 3 / 2, -701 / 1200) / (2 * 1 + 1)
    v2 = np.exp(-17 / 240) * hyp1f1(2 + 1 / 2, 2 + 3 / 2, -701 / 1200) / (2 * 2 + 1)

    v_100 = x_pa * v0 - x_pc * v1
    v1_100 = x_pa * v1 - x_pc * v2
    v_200 = x_pa * v_100 - x_pc * v1_100 + (1 / (2 * p)) * (v0 - v1)
    v_010 = y_pa * v0 - y_pc * v1
    v1_010 = y_pa * v1 - y_pc * v2
    v_020 = y_pa * v_010 - y_pc * v1_010 + (1 / (2 * p)) * (v0 - v1)
    v_001 = z_pa * v0 - z_pc * v1
    v1_001 = z_pa * v1 - z_pc * v2
    v_002 = z_pa * v_001 - z_pc * v1_001 + (1 / (2 * p)) * (v0 - v1)
    v_011 = z_pa * v_010 - z_pc * v1_010
    v_110 = y_pa * v_100 - y_pc * v1_100
    v_101 = z_pa * v_100 - z_pc * v1_100

    assert np.allclose(s_test[1, 0, 0, 1, 0, 0], coeff * v_200 + x_ab * coeff * v_100)
    assert np.allclose(s_test[1, 0, 0, 0, 1, 0], coeff * v_110 + y_ab * coeff * v_100)
    assert np.allclose(s_test[1, 0, 0, 0, 0, 1], coeff * v_101 + z_ab * coeff * v_100)
    assert np.allclose(s_test[0, 1, 0, 1, 0, 0], coeff * v_110 + x_ab * coeff * v_010)
    assert np.allclose(s_test[0, 1, 0, 0, 1, 0], coeff * v_020 + y_ab * coeff * v_010)
    assert np.allclose(s_test[0, 1, 0, 0, 0, 1], coeff * v_011 + z_ab * coeff * v_010)
    assert np.allclose(s_test[0, 0, 1, 1, 0, 0], coeff * v_101 + x_ab * coeff * v_001)
    assert np.allclose(s_test[0, 0, 1, 0, 1, 0], coeff * v_011 + y_ab * coeff * v_001)
    assert np.allclose(s_test[0, 0, 1, 0, 0, 1], coeff * v_002 + z_ab * coeff * v_001)


# FIXME: test fails because reference/answer does not normalize contractions
@pytest.mark.skip(
    reason="The answer that were hand-computed were not using normalized contractions."
)
def test_compute_one_elec_int_multiple_contractions():
    """Test _one_elec_int._compute_one_electron_integrals for s-type contractions."""
    # ContractedCartesianGaussians(angmom, coord, charge, coeffs, exps)
    contr_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.25])
    )
    contr_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0, 4.0]), np.array([0.02, 0.01])
    )
    coord_a = contr_one.coord
    angmom_a = contr_one.angmom
    exps_a = contr_one.exps
    coeffs_a = contr_one.coeffs
    coord_b = contr_two.coord
    angmom_b = contr_two.angmom
    exps_b = contr_two.exps
    coeffs_b = contr_two.coeffs
    answer = _compute_one_elec_integrals(
        np.array([0.0, 0.0, 0.0]),  # coord_point
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
    # Test output array using hand-calculated values
    assert np.allclose(answer[1, 0, 0, 1, 0, 0], 33.16989127227113)
    assert np.allclose(answer[1, 0, 0, 0, 1, 0], 2.131647187292573)
    assert np.allclose(answer[1, 0, 0, 0, 0, 1], 3.1974707809388594)
    assert np.allclose(answer[0, 1, 0, 1, 0, 0], 4.1837386542199715)
    assert np.allclose(answer[0, 1, 0, 0, 1, 0], 36.74056538017357)
    assert np.allclose(answer[0, 1, 0, 0, 0, 1], 7.831446742304523)
    assert np.allclose(answer[0, 0, 1, 1, 0, 0], 6.275607981329958)
    assert np.allclose(answer[0, 0, 1, 0, 1, 0], 7.831446742304523)
    assert np.allclose(answer[0, 0, 1, 0, 0, 1], 43.26677099876067)


# FIXME: test fails because reference/answer does not normalize contractions
# FIXME: test doesn't actually test anything
@pytest.mark.skip(
    reason="The answer that were hand-computed were not using normalized contractions."
)
def test_compute_one_elec_int_generalized_contraction():
    """Test _one_elec_int._compute_one_electron_integrals for generalized contractions."""
    contr_one = ContractedCartesianGaussians(
        3, np.array([0.5, 1, 1.5]), 0, np.array([[1.0, 2.0], [1.5, 2.5]]), np.array([0.1, 0.25])
    )
    contr_two = ContractedCartesianGaussians(
        2, np.array([1.5, 2, 3]), 0, np.array([[3.0, 4.0], [3.5, 4.5]]), np.array([0.02, 0.01])
    )
    coord_a = contr_one.coord
    angmom_a = contr_one.angmom
    exps_a = contr_one.exps
    coeffs_a = contr_one.coeffs
    coord_b = contr_two.coord
    angmom_b = contr_two.angmom
    exps_b = contr_two.exps
    coeffs_b = contr_two.coeffs
    _compute_one_elec_integrals(
        np.array([0.0, 0.0, 0.0]),  # coord_point
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
