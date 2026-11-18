"""Test gbasis.integrals.electron_repulsion improved OS+HGP integration.

Tests for Week 6: Integration of the improved OS+HGP algorithm into
the electron_repulsion module via ElectronRepulsionIntegralImproved class.
"""

import numpy as np
import pytest
from utils import HortonContractions, find_datafile

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals.electron_repulsion import (
    ElectronRepulsionIntegral,
    ElectronRepulsionIntegralImproved,
    _schwarz_screener_cache,
    electron_repulsion_integral,
    electron_repulsion_integral_improved,
)
from gbasis.parsers import make_contractions, parse_nwchem


class TestImprovedClassConstruction:
    """Tests for ElectronRepulsionIntegralImproved class."""

    def test_type_checks(self):
        """Test that type checks raise TypeError for invalid inputs."""
        coord = np.array([0.0, 0.0, 0.0])
        cont = GeneralizedContractionShell(0, coord, np.array([1.0]), np.array([0.1]), "spherical")
        with pytest.raises(TypeError):
            ElectronRepulsionIntegralImproved.construct_array_contraction(None, cont, cont, cont)
        with pytest.raises(TypeError):
            ElectronRepulsionIntegralImproved.construct_array_contraction(cont, None, cont, cont)
        with pytest.raises(TypeError):
            ElectronRepulsionIntegralImproved.construct_array_contraction(cont, cont, None, cont)
        with pytest.raises(TypeError):
            ElectronRepulsionIntegralImproved.construct_array_contraction(cont, cont, cont, None)

    def test_ssss_contraction(self):
        """Test (ss|ss) contraction with improved class."""
        coord_a = np.array([0.5, 1.0, 1.5])
        coord_b = np.array([1.5, 2.0, 3.0])
        coord_c = np.array([2.5, 3.0, 4.0])
        coord_d = np.array([3.5, 4.0, 5.0])

        cont_a = GeneralizedContractionShell(
            0, coord_a, np.array([1.0]), np.array([0.1]), "spherical"
        )
        cont_b = GeneralizedContractionShell(
            0, coord_b, np.array([3.0]), np.array([0.2]), "spherical"
        )
        cont_c = GeneralizedContractionShell(
            0, coord_c, np.array([3.0]), np.array([0.2]), "spherical"
        )
        cont_d = GeneralizedContractionShell(
            0, coord_d, np.array([3.0]), np.array([0.2]), "spherical"
        )

        result = ElectronRepulsionIntegralImproved.construct_array_contraction(
            cont_a, cont_b, cont_c, cont_d
        )
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_output_shape_s_orbitals(self):
        """Test output shape for s-orbital contractions."""
        coord = np.array([0.0, 0.0, 0.0])
        # coeffs shape (K, M) = (2, 1), exps shape (K,) = (2,)
        cont = GeneralizedContractionShell(
            0, coord, np.array([[0.1], [0.2]]), np.array([1.0, 0.5]), "spherical"
        )
        result = ElectronRepulsionIntegralImproved.construct_array_contraction(
            cont, cont, cont, cont
        )
        # Shape: (M_1, L_1, M_2, L_2, M_3, L_3, M_4, L_4)
        # s-orbital: L=1, M depends on coeffs
        assert result.shape[1] == 1  # L_cart for s-orbital
        assert result.shape[3] == 1
        assert result.shape[5] == 1
        assert result.shape[7] == 1


class TestImprovedMatchesOld:
    """Test that improved implementation matches old implementation."""

    def test_ssss_matches(self):
        """Test (ss|ss) integrals match between old and improved."""
        coord_a = np.array([0.5, 1.0, 1.5])
        coord_b = np.array([1.5, 2.0, 3.0])
        coord_c = np.array([2.5, 3.0, 4.0])
        coord_d = np.array([3.5, 4.0, 5.0])

        cont_a = GeneralizedContractionShell(
            0, coord_a, np.array([1.0]), np.array([0.1]), "spherical"
        )
        cont_b = GeneralizedContractionShell(
            0, coord_b, np.array([3.0]), np.array([0.2]), "spherical"
        )
        cont_c = GeneralizedContractionShell(
            0, coord_c, np.array([3.0]), np.array([0.2]), "spherical"
        )
        cont_d = GeneralizedContractionShell(
            0, coord_d, np.array([3.0]), np.array([0.2]), "spherical"
        )

        result_old = ElectronRepulsionIntegral.construct_array_contraction(
            cont_a, cont_b, cont_c, cont_d
        )
        result_new = ElectronRepulsionIntegralImproved.construct_array_contraction(
            cont_a, cont_b, cont_c, cont_d
        )

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="(ss|ss) integrals don't match between old and improved class",
        )

    def test_sssp_matches(self):
        """Test (ss|sp) integrals match between old and improved."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])
        coord_d = np.array([1.0, 1.0, 0.0])

        cont_s = GeneralizedContractionShell(
            0, coord_a, np.array([[1.0], [0.5]]), np.array([1.0, 0.5]), "spherical"
        )
        cont_s2 = GeneralizedContractionShell(
            0, coord_b, np.array([1.0]), np.array([0.8]), "spherical"
        )
        cont_s3 = GeneralizedContractionShell(
            0, coord_c, np.array([1.0]), np.array([1.2]), "spherical"
        )
        cont_p = GeneralizedContractionShell(
            1, coord_d, np.array([1.0]), np.array([0.9]), "spherical"
        )

        result_old = ElectronRepulsionIntegral.construct_array_contraction(
            cont_s, cont_s2, cont_s3, cont_p
        )
        result_new = ElectronRepulsionIntegralImproved.construct_array_contraction(
            cont_s, cont_s2, cont_s3, cont_p
        )

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="(ss|sp) integrals don't match between old and improved class",
        )

    def test_spsp_matches(self):
        """Test (sp|sp) integrals match between old and improved."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.5, 0.0, 0.0])
        coord_c = np.array([0.0, 1.5, 0.0])
        coord_d = np.array([1.5, 1.5, 0.0])

        cont_s = GeneralizedContractionShell(
            0, coord_a, np.array([[1.0], [0.5]]), np.array([1.0, 0.5]), "spherical"
        )
        cont_p1 = GeneralizedContractionShell(
            1, coord_b, np.array([1.0]), np.array([0.8]), "spherical"
        )
        cont_s2 = GeneralizedContractionShell(
            0, coord_c, np.array([1.0]), np.array([1.2]), "spherical"
        )
        cont_p2 = GeneralizedContractionShell(
            1, coord_d, np.array([1.0]), np.array([0.9]), "spherical"
        )

        result_old = ElectronRepulsionIntegral.construct_array_contraction(
            cont_s, cont_p1, cont_s2, cont_p2
        )
        result_new = ElectronRepulsionIntegralImproved.construct_array_contraction(
            cont_s, cont_p1, cont_s2, cont_p2
        )

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="(sp|sp) integrals don't match between old and improved class",
        )

    def test_pppp_matches(self):
        """Test (pp|pp) integrals match between old and improved."""
        coord_a = np.array([0.0, 0.0, 0.0])
        coord_b = np.array([1.0, 0.0, 0.0])
        coord_c = np.array([0.0, 1.0, 0.0])
        coord_d = np.array([1.0, 1.0, 0.0])

        cont_p1 = GeneralizedContractionShell(
            1, coord_a, np.array([1.0]), np.array([1.0]), "spherical"
        )
        cont_p2 = GeneralizedContractionShell(
            1, coord_b, np.array([1.0]), np.array([0.8]), "spherical"
        )
        cont_p3 = GeneralizedContractionShell(
            1, coord_c, np.array([1.0]), np.array([1.2]), "spherical"
        )
        cont_p4 = GeneralizedContractionShell(
            1, coord_d, np.array([1.0]), np.array([0.9]), "spherical"
        )

        result_old = ElectronRepulsionIntegral.construct_array_contraction(
            cont_p1, cont_p2, cont_p3, cont_p4
        )
        result_new = ElectronRepulsionIntegralImproved.construct_array_contraction(
            cont_p1, cont_p2, cont_p3, cont_p4
        )

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="(pp|pp) integrals don't match between old and improved class",
        )


class TestImprovedFullBasis:
    """Test improved implementation with full basis sets."""

    def test_sto6g_bec_cartesian_matches(self):
        """Test improved matches old for Be-C with STO-6G basis (Cartesian)."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        coords = np.array([[0, 0, 0], [1.0, 0, 0]])
        basis = make_contractions(basis_dict, ["Be", "C"], coords, "cartesian")
        basis = [
            HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis
        ]

        result_old = electron_repulsion_integral(basis, notation="chemist")
        result_new = electron_repulsion_integral_improved(basis, notation="chemist")

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="STO-6G Be-C Cartesian integrals don't match",
        )

    def test_sto6g_bec_horton_reference(self):
        """Test improved matches Horton reference for Be-C with STO-6G."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        coords = np.array([[0, 0, 0], [1.0, 0, 0]])
        basis = make_contractions(basis_dict, ["Be", "C"], coords, "cartesian")
        basis = [
            HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis
        ]

        horton_ref = np.load(find_datafile("data_horton_bec_cart_elec_repulsion.npy"))
        result_new = electron_repulsion_integral_improved(basis)

        np.testing.assert_allclose(
            result_new,
            horton_ref,
            rtol=1e-10,
            atol=1e-15,
            err_msg="Improved integrals don't match Horton reference",
        )

    def test_sto6g_carbon_spherical_matches(self):
        """Test improved matches old for Carbon with STO-6G (spherical)."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "spherical")

        result_old = electron_repulsion_integral(basis, notation="chemist")
        result_new = electron_repulsion_integral_improved(basis, notation="chemist")

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="STO-6G Carbon spherical integrals don't match",
        )


class TestImprovedNotation:
    """Test that notation conversions work correctly."""

    def test_chemist_notation(self):
        """Test that Chemists' notation works correctly."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "cartesian")

        erep_obj = ElectronRepulsionIntegralImproved(basis)
        assert np.allclose(
            erep_obj.construct_array_cartesian(),
            electron_repulsion_integral_improved(basis, notation="chemist"),
        )

    def test_physicist_notation(self):
        """Test that Physicists' notation works correctly."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "cartesian")

        erep_obj = ElectronRepulsionIntegralImproved(basis)
        assert np.allclose(
            np.einsum("ijkl->ikjl", erep_obj.construct_array_cartesian()),
            electron_repulsion_integral_improved(basis, notation="physicist"),
        )

    def test_invalid_notation_raises(self):
        """Test that invalid notation raises ValueError."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "cartesian")

        with pytest.raises(ValueError):
            electron_repulsion_integral_improved(basis, notation="bad")

    def test_physicist_matches_old_physicist(self):
        """Test that physicist notation matches between old and improved."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "cartesian")

        result_old = electron_repulsion_integral(basis, notation="physicist")
        result_new = electron_repulsion_integral_improved(basis, notation="physicist")

        np.testing.assert_allclose(
            result_new,
            result_old,
            rtol=1e-10,
            err_msg="Physicist notation doesn't match between old and improved",
        )


class TestImprovedSymmetries:
    """Test that improved integrals satisfy expected symmetries."""

    def test_chemist_symmetry_ijkl_jilk(self):
        """Test (ij|kl) = (ji|lk) symmetry in Chemists' notation."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "spherical")

        integrals = electron_repulsion_integral_improved(basis, notation="chemist")
        # (ij|kl) = (ji|lk)
        np.testing.assert_allclose(
            integrals,
            np.transpose(integrals, (1, 0, 3, 2)),
            rtol=1e-10,
            err_msg="(ij|kl) != (ji|lk) symmetry violated",
        )

    def test_chemist_symmetry_ijkl_klij(self):
        """Test (ij|kl) = (kl|ij) symmetry in Chemists' notation."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "spherical")

        integrals = electron_repulsion_integral_improved(basis, notation="chemist")
        # (ij|kl) = (kl|ij)
        np.testing.assert_allclose(
            integrals,
            np.transpose(integrals, (2, 3, 0, 1)),
            rtol=1e-10,
            err_msg="(ij|kl) != (kl|ij) symmetry violated",
        )

    def test_positive_diagonal(self):
        """Test that diagonal integrals (ii|ii) are positive."""
        basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
        basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]), "spherical")

        integrals = electron_repulsion_integral_improved(basis, notation="chemist")
        n = integrals.shape[0]
        for i in range(n):
            assert integrals[i, i, i, i] > 0, f"Diagonal (ii|ii) not positive for i={i}"


H_COORD = np.array([0.0, 0.0, 0.0])
CL_COORD = np.array([0.3, -0.2, 2.4])


@pytest.mark.parametrize(
    "cls", [ElectronRepulsionIntegral, ElectronRepulsionIntegralImproved], ids=["old", "improved"]
)
def test_tight_s_in_bra_and_diffuse_d_in_ket_primitives(cls):
    """Test a quartet whose bra has a much larger exponent sum than its ket.

    The primitives are d on H, the tightest s on Cl (cc-pVTZ), and two d on Cl. The reference value
    is the largest element of the same normalized integral from PySCF. The electron transfer
    recursion used to be off by 4e3 for this quartet.
    """
    conts = [
        GeneralizedContractionShell(2, H_COORD, np.array([1.0]), np.array([1.057]), "cartesian"),
        GeneralizedContractionShell(0, CL_COORD, np.array([1.0]), np.array([4.561e5]), "cartesian"),
        GeneralizedContractionShell(2, CL_COORD, np.array([1.0]), np.array([0.344]), "cartesian"),
        GeneralizedContractionShell(2, CL_COORD, np.array([1.0]), np.array([0.344]), "cartesian"),
    ]
    integrals = cls.construct_array_contraction(*conts)
    assert integrals.shape == (1, 6, 1, 1, 1, 6, 1, 6)
    assert np.max(np.abs(integrals)) == pytest.approx(2.332134867280155e-06, rel=1e-8)
    swapped = cls.construct_array_contraction(conts[2], conts[3], conts[0], conts[1])
    np.testing.assert_allclose(
        integrals, np.transpose(swapped, (4, 5, 6, 7, 0, 1, 2, 3)), rtol=1e-8, atol=1e-18
    )


@pytest.mark.parametrize(
    "cls", [ElectronRepulsionIntegral, ElectronRepulsionIntegralImproved], ids=["old", "improved"]
)
def test_schwarz_inequality_with_tight_and_diffuse_primitives(cls):
    """Test |(ab|cd)| <= sqrt((ab|ab) (cd|cd)) for very different exponents."""
    exps_s = np.array([4.561e5, 6.833e4, 1.555e4, 4.405e3, 1.439e3, 5.204e2, 2.031e2, 83.96])
    coeffs_s = np.array([4.0e-4, 2.0e-3, 8.4e-3, 2.95e-2, 8.78e-2, 2.115e-1, 3.654e-1, 3.409e-1])
    basis = [
        GeneralizedContractionShell(2, H_COORD, np.array([1.0]), np.array([1.057]), "cartesian"),
        GeneralizedContractionShell(0, CL_COORD, coeffs_s, exps_s, "cartesian"),
        GeneralizedContractionShell(
            1, CL_COORD, np.array([1.0, 0.5]), np.array([20.0, 0.5]), "cartesian"
        ),
        GeneralizedContractionShell(2, CL_COORD, np.array([1.0]), np.array([0.344]), "cartesian"),
    ]
    integrals = cls(basis).construct_array_cartesian()
    diagonal = np.sqrt(np.abs(np.einsum("abab->ab", integrals)))
    bound = diagonal[:, :, None, None] * diagonal[None, None, :, :]
    assert np.all(np.abs(integrals) <= bound * (1 + 1e-8) + 1e-14)
    np.testing.assert_allclose(
        integrals, np.transpose(integrals, (2, 3, 0, 1)), rtol=1e-8, atol=1e-14
    )


def test_schwarz_bounds_with_tight_and_diffuse_primitives():
    """Test that the Schwarz bounds are the square roots of the largest (ab|ab) integrals."""

    def normalized(shell):
        """Return the shell with contraction coefficients that give a normalized contraction."""
        return GeneralizedContractionShell(
            shell.angmom,
            shell.coord,
            shell.coeffs[:, 0] * shell.norm_cont[0, 0],
            shell.exps,
            "cartesian",
        )

    # the s contraction of Cl in cc-pVTZ, from the tightest to the most diffuse primitive
    exps_s = np.array(
        [
            4.561e5, 6.833e4, 1.555e4, 4.405e3, 1.439e3, 5.204e2, 2.031e2,
            83.96, 36.2, 15.83, 6.334, 2.694, 0.4313,
        ]
    )  # fmt: skip
    coeffs_s = np.array(
        [
            2.0e-5, 4.0e-4, 2.0e-3, 8.4e-3, 2.95e-2, 8.78e-2, 2.115e-1,
            3.654e-1, 3.409e-1, 1.021e-1, 3.1e-3, 1.1e-3, 2.0e-4,
        ]
    )  # fmt: skip
    basis = [
        GeneralizedContractionShell(2, H_COORD, np.array([1.0]), np.array([1.057]), "cartesian"),
        normalized(GeneralizedContractionShell(0, CL_COORD, coeffs_s, exps_s, "cartesian")),
        normalized(
            GeneralizedContractionShell(
                1, CL_COORD, np.array([1.0, 0.5]), np.array([20.0, 0.5]), "cartesian"
            )
        ),
        GeneralizedContractionShell(2, CL_COORD, np.array([1.0]), np.array([0.344]), "cartesian"),
    ]
    integrals = ElectronRepulsionIntegralImproved(basis).construct_array_cartesian()
    offsets = np.cumsum([0] + [len(shell.angmom_components_cart) for shell in basis])
    expected = np.zeros((len(basis), len(basis)))
    for i in range(len(basis)):
        for j in range(len(basis)):
            block_i = slice(offsets[i], offsets[i + 1])
            block_j = slice(offsets[j], offsets[j + 1])
            expected[i, j] = np.sqrt(np.max(np.abs(integrals[block_i, block_j, block_i, block_j])))

    # the screener that is made by electron_repulsion_integral_improved
    electron_repulsion_integral_improved(basis, schwarz_threshold=1e-12)
    screener = _schwarz_screener_cache[(tuple(id(shell) for shell in basis), 1e-12)]
    np.testing.assert_allclose(screener.bounds, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "cls", [ElectronRepulsionIntegral, ElectronRepulsionIntegralImproved], ids=["old", "improved"]
)
@pytest.mark.parametrize(
    "quartet", [(0, 1, 3, 3), (1, 0, 1, 3), (0, 2, 2, 3)], ids=["d_s_d_d", "s_d_s_d", "d_p_p_d"]
)
def test_swapped_contractions_that_need_splitting(cls, quartet):
    """Test (ab|cd) = (cd|ab) for contractions whose primitives have very different exponents.

    The second contraction has two contractions with the same exponents and the third has two
    primitives, with weights that are not small on the tightest and the most diffuse primitive.
    """
    coeffs_s = np.array([[1.0, 0.5], [0.8, -1.0], [1.0, 0.3]])
    exps_s = np.array([4.561e5, 83.96, 0.4313])
    basis = [
        GeneralizedContractionShell(2, H_COORD, np.array([1.0]), np.array([1.057]), "cartesian"),
        GeneralizedContractionShell(0, CL_COORD, coeffs_s, exps_s, "cartesian"),
        GeneralizedContractionShell(
            1, CL_COORD, np.array([1.0, 0.5]), np.array([20.0, 0.5]), "cartesian"
        ),
        GeneralizedContractionShell(2, CL_COORD, np.array([1.0]), np.array([0.344]), "cartesian"),
    ]
    a, b, c, d = (basis[i] for i in quartet)
    direct = cls.construct_array_contraction(a, b, c, d)
    swapped = cls.construct_array_contraction(c, d, a, b)
    np.testing.assert_allclose(
        direct,
        np.transpose(swapped, (4, 5, 6, 7, 0, 1, 2, 3)),
        rtol=1e-7,
        atol=1e-12 * np.max(np.abs(direct)),
    )
