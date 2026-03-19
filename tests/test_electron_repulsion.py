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
