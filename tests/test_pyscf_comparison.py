"""Direct comparison of GBasis ERIs with PySCF reference.

This test validates that both the original and improved (OS+HGP) GBasis
implementations produce the same electron repulsion integrals as PySCF.

Notation conventions:
- PySCF mol.intor('int2e') returns integrals in chemist's notation (ij|kl).
- GBasis electron_repulsion_integral() defaults to physicist's notation <ij|kl>.
- GBasis electron_repulsion_integral(notation="chemist") returns chemist's (ij|kl).
- Conversion: <ij|kl> = (ik|jl), so physicist = chemist.transpose(0, 2, 1, 3).

We use manual basis definitions to ensure both codes use exactly the same
basis parameters, avoiding normalization conversion issues.
"""

import numpy as np
import pytest

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals.electron_repulsion import (
    electron_repulsion_integral,
    electron_repulsion_integral_improved,
)

gto = pytest.importorskip("pyscf.gto", reason="pyscf not installed")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_h2_sto3g():
    """Create H2 with STO-3G basis (same parameters for GBasis and PySCF).

    STO-3G for H: 3 primitives contracted to 1 s-function.
    Returns GBasis basis list and PySCF mol object.
    """
    exps = np.array([3.42525091, 0.62391373, 0.16885540])
    coeffs = np.array([[0.15432897], [0.53532814], [0.44463454]])

    coord1 = np.array([0.0, 0.0, 0.0])
    coord2 = np.array([1.4, 0.0, 0.0])  # 1.4 Bohr apart

    basis = [
        GeneralizedContractionShell(0, coord1, coeffs, exps, "cartesian"),
        GeneralizedContractionShell(0, coord2, coeffs, exps, "cartesian"),
    ]

    mol = gto.M(
        atom="H 0 0 0; H 1.4 0 0",
        unit="bohr",
        basis={"H": gto.basis.parse("""
            H  S
                3.42525091    0.15432897
                0.62391373    0.53532814
                0.16885540    0.44463454
        """)},
        verbose=0,
    )
    return basis, mol


def make_h2_primitive():
    """Create H2 with a single primitive s-function per atom.

    Simplest possible case: no contraction complexity.
    """
    exps = np.array([1.0])
    coeffs = np.array([[1.0]])

    coord1 = np.array([0.0, 0.0, 0.0])
    coord2 = np.array([2.0, 0.0, 0.0])

    basis = [
        GeneralizedContractionShell(0, coord1, coeffs, exps, "cartesian"),
        GeneralizedContractionShell(0, coord2, coeffs, exps, "cartesian"),
    ]

    mol = gto.M(
        atom="H 0 0 0; H 2.0 0 0",
        unit="bohr",
        basis={"H": gto.basis.parse("H  S\n    1.0    1.0")},
        verbose=0,
    )
    return basis, mol


def make_h2_sp():
    """Create H2 with s + p basis (one s and one p per atom).

    This tests angular momentum handling, which is critical because
    chemist/physicist notation differences matter for mixed angular momentum.
    """
    s_exps = np.array([1.0])
    s_coeffs = np.array([[1.0]])
    p_exps = np.array([0.8])
    p_coeffs = np.array([[1.0]])

    coord1 = np.array([0.0, 0.0, 0.0])
    coord2 = np.array([2.0, 0.0, 0.0])

    basis = [
        GeneralizedContractionShell(0, coord1, s_coeffs, s_exps, "cartesian"),
        GeneralizedContractionShell(1, coord1, p_coeffs, p_exps, "cartesian"),
        GeneralizedContractionShell(0, coord2, s_coeffs, s_exps, "cartesian"),
        GeneralizedContractionShell(1, coord2, p_coeffs, p_exps, "cartesian"),
    ]

    mol = gto.M(
        atom="H 0 0 0; H 2.0 0 0",
        unit="bohr",
        basis={"H": gto.basis.parse("""
            H  S
                1.0    1.0
            H  P
                0.8    1.0
        """)},
        verbose=0,
    )
    return basis, mol


# ---------------------------------------------------------------------------
# Tests: Original implementation vs PySCF
# ---------------------------------------------------------------------------


class TestPySCFOriginal:
    """Compare original GBasis ERIs with PySCF reference."""

    def test_h2_sto3g_chemist(self):
        """Test H2/STO-3G in chemist notation: GBasis (ij|kl) == PySCF (ij|kl)."""
        basis, mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")  # PySCF returns chemist (ij|kl)

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf,
            rtol=1e-6,
            atol=1e-10,
            err_msg="H2/STO-3G chemist notation doesn't match PySCF",
        )

    def test_h2_sto3g_physicist(self):
        """Test H2/STO-3G in physicist notation.

        GBasis physicist <ij|kl> == PySCF chemist (ij|kl).transpose(0,2,1,3).
        """
        basis, mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral(basis, notation="physicist")
        eri_pyscf = mol.intor("int2e")
        eri_pyscf_physicist = eri_pyscf.transpose(0, 2, 1, 3)

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf_physicist,
            rtol=1e-6,
            atol=1e-10,
            err_msg="H2/STO-3G physicist notation doesn't match PySCF",
        )

    def test_h2_primitive(self):
        """Test single-primitive H2 (simplest possible case)."""
        basis, mol = make_h2_primitive()

        eri_gbasis = electron_repulsion_integral(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Single primitive H2 ERIs don't match PySCF",
        )

    def test_h2_sp_basis(self):
        """Test H2 with s+p basis (tests angular momentum handling)."""
        basis, mol = make_h2_sp()

        eri_gbasis = electron_repulsion_integral(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf,
            rtol=1e-6,
            atol=1e-10,
            err_msg="H2 s+p basis ERIs don't match PySCF",
        )

    def test_h2_specific_integrals(self):
        """Test specific physically meaningful integrals for H2/STO-3G."""
        basis, mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        # (00|00) - Coulomb integral, both electrons on atom 1
        assert np.isclose(
            eri_gbasis[0, 0, 0, 0], eri_pyscf[0, 0, 0, 0], rtol=1e-6
        ), f"(00|00) mismatch: GBasis={eri_gbasis[0,0,0,0]:.10f}, PySCF={eri_pyscf[0,0,0,0]:.10f}"

        # (11|11) - Coulomb integral, both electrons on atom 2
        assert np.isclose(
            eri_gbasis[1, 1, 1, 1], eri_pyscf[1, 1, 1, 1], rtol=1e-6
        ), f"(11|11) mismatch: GBasis={eri_gbasis[1,1,1,1]:.10f}, PySCF={eri_pyscf[1,1,1,1]:.10f}"

        # By symmetry, (00|00) == (11|11) for identical atoms
        assert np.isclose(
            eri_gbasis[0, 0, 0, 0], eri_gbasis[1, 1, 1, 1], rtol=1e-6
        ), "Identical atom diagonal integrals should be equal"

        # (01|01) - exchange-type integral
        assert np.isclose(
            eri_gbasis[0, 1, 0, 1], eri_pyscf[0, 1, 0, 1], rtol=1e-6
        ), "(01|01) mismatch"

    def test_h2_symmetries(self):
        """Test that PySCF and GBasis agree on 8-fold ERI symmetries."""
        basis, _mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral(basis, notation="chemist")

        # Chemist notation symmetries: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)
        n = eri_gbasis.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for m in range(n):
                        val = eri_gbasis[i, j, k, m]
                        assert np.isclose(
                            val, eri_gbasis[j, i, k, m], rtol=1e-10
                        ), f"(ij|kl)!=(ji|kl) for ({i}{j}|{k}{m})"
                        assert np.isclose(
                            val, eri_gbasis[i, j, m, k], rtol=1e-10
                        ), f"(ij|kl)!=(ij|lk) for ({i}{j}|{k}{m})"
                        assert np.isclose(
                            val, eri_gbasis[k, m, i, j], rtol=1e-10
                        ), f"(ij|kl)!=(kl|ij) for ({i}{j}|{k}{m})"


# ---------------------------------------------------------------------------
# Tests: Improved (OS+HGP) implementation vs PySCF
# ---------------------------------------------------------------------------


class TestPySCFImproved:
    """Compare improved OS+HGP GBasis ERIs with PySCF reference."""

    def test_h2_sto3g_chemist(self):
        """Test improved implementation H2/STO-3G in chemist notation."""
        basis, mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral_improved(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Improved H2/STO-3G chemist notation doesn't match PySCF",
        )

    def test_h2_sto3g_physicist(self):
        """Test improved implementation H2/STO-3G in physicist notation."""
        basis, mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral_improved(basis, notation="physicist")
        eri_pyscf = mol.intor("int2e")
        eri_pyscf_physicist = eri_pyscf.transpose(0, 2, 1, 3)

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf_physicist,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Improved H2/STO-3G physicist notation doesn't match PySCF",
        )

    def test_h2_primitive(self):
        """Test improved implementation with single primitive."""
        basis, mol = make_h2_primitive()

        eri_gbasis = electron_repulsion_integral_improved(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Improved single primitive H2 ERIs don't match PySCF",
        )

    def test_h2_sp_basis(self):
        """Test improved implementation H2 with s+p basis."""
        basis, mol = make_h2_sp()

        eri_gbasis = electron_repulsion_integral_improved(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        np.testing.assert_allclose(
            eri_gbasis,
            eri_pyscf,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Improved H2 s+p basis ERIs don't match PySCF",
        )

    def test_improved_matches_original(self):
        """Test that improved and original implementations agree on PySCF test case."""
        basis, _ = make_h2_sto3g()

        eri_old = electron_repulsion_integral(basis, notation="chemist")
        eri_new = electron_repulsion_integral_improved(basis, notation="chemist")

        np.testing.assert_allclose(
            eri_new, eri_old, rtol=1e-10, err_msg="Improved doesn't match original for H2/STO-3G"
        )

    def test_improved_matches_original_sp(self):
        """Test that improved and original agree for s+p basis."""
        basis, _ = make_h2_sp()

        eri_old = electron_repulsion_integral(basis, notation="chemist")
        eri_new = electron_repulsion_integral_improved(basis, notation="chemist")

        np.testing.assert_allclose(
            eri_new, eri_old, rtol=1e-10, err_msg="Improved doesn't match original for H2 s+p"
        )


# ---------------------------------------------------------------------------
# Tests: Physical sanity checks
# ---------------------------------------------------------------------------


class TestPySCFPhysicalProperties:
    """Verify physical properties of ERIs using PySCF as cross-check."""

    def test_positive_diagonal(self):
        """Test that Coulomb integrals (ii|ii) are positive."""
        basis, mol = make_h2_sto3g()

        eri_gbasis = electron_repulsion_integral_improved(basis, notation="chemist")
        eri_pyscf = mol.intor("int2e")

        n = eri_gbasis.shape[0]
        for i in range(n):
            assert eri_gbasis[i, i, i, i] > 0, f"GBasis (ii|ii) not positive for i={i}"
            assert eri_pyscf[i, i, i, i] > 0, f"PySCF (ii|ii) not positive for i={i}"

    def test_coulomb_greater_than_exchange(self):
        """Test that Coulomb integral >= exchange integral for H2.

        For same-atom shells: (00|00) >= (01|01) because the exchange
        integral involves orbital overlap which reduces the value.
        """
        basis, _mol = make_h2_sto3g()

        eri = electron_repulsion_integral_improved(basis, notation="chemist")

        # Coulomb (00|00) should be > exchange (01|01)
        coulomb = eri[0, 0, 0, 0]
        exchange = eri[0, 1, 0, 1]
        assert coulomb > exchange, f"Coulomb ({coulomb:.6f}) should be > exchange ({exchange:.6f})"

    def test_eri_values_physically_reasonable(self):
        """Test that ERI values are in a physically reasonable range.

        For H2 with STO-3G, typical values are 0.1-1.0 in atomic units.
        """
        basis, _ = make_h2_sto3g()

        eri = electron_repulsion_integral_improved(basis, notation="chemist")

        # All values should be finite
        assert np.all(np.isfinite(eri)), "ERIs contain NaN or Inf"

        # Diagonal integrals should be in reasonable range (0.1 - 2.0 a.u.)
        for i in range(eri.shape[0]):
            val = eri[i, i, i, i]
            assert (
                0.01 < val < 10.0
            ), f"Diagonal integral ({i}{i}|{i}{i}) = {val} outside reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
