"""Test gbasis.overlap."""
from gbasis._moment_int import _compute_multipole_moment_integrals
from gbasis.contractions import ContractedCartesianGaussians, make_contractions
from gbasis.overlap import (
    Overlap,
    overlap_basis_cartesian,
    overlap_basis_spherical,
    overlap_basis_spherical_lincomb,
)
from gbasis.parsers import parse_nwchem
import numpy as np
import pytest
from scipy.special import factorial2
from utils import find_datafile


def test_overlap_construct_array_contraction():
    """Test gbasis.overlap.Overlap.construct_array_contraction."""
    test_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.01])
    )
    test_two = ContractedCartesianGaussians(
        2, np.array([1.5, 2, 3]), 0, np.array([3.0, 4.0]), np.array([0.2, 0.02])
    )
    answer = np.array(
        [
            [
                _compute_multipole_moment_integrals(
                    np.array([0, 0, 0]),
                    np.array([[0, 0, 0]]),
                    np.array([0.5, 1, 1.5]),
                    np.array([angmom_comp_one]),
                    np.array([0.1, 0.01]),
                    np.array([[1], [2]]),
                    np.array(
                        [
                            [
                                (2 * 0.1 / np.pi) ** (3 / 4)
                                * (4 * 0.1) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                                (2 * 0.01 / np.pi) ** (3 / 4)
                                * (4 * 0.01) ** (1 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_one - 1))),
                            ]
                        ]
                    ),
                    np.array([1.5, 2, 3]),
                    np.array([angmom_comp_two]),
                    np.array([0.2, 0.02]),
                    np.array([[3], [4]]),
                    np.array(
                        [
                            [
                                (2 * 0.2 / np.pi) ** (3 / 4)
                                * (4 * 0.2) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                                (2 * 0.02 / np.pi) ** (3 / 4)
                                * (4 * 0.02) ** (2 / 2)
                                / np.sqrt(np.prod(factorial2(2 * angmom_comp_two - 1))),
                            ]
                        ]
                    ),
                )
                for angmom_comp_two in test_two.angmom_components
            ]
            for angmom_comp_one in test_one.angmom_components
        ]
    )
    assert np.allclose(
        np.squeeze(Overlap.construct_array_contraction(test_one, test_two)), np.squeeze(answer)
    )

    with pytest.raises(TypeError):
        Overlap.construct_array_contraction(test_one, None)
    with pytest.raises(TypeError):
        Overlap.construct_array_contraction(None, test_two)


def test_overlap_basis_cartesian():
    """Test gbasis.eval.overlap_basis_cartesian."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(overlap_obj.construct_array_cartesian(), overlap_basis_cartesian(basis))


def test_overlap_basis_spherical():
    """Test gbasis.eval.overlap_basis_spherical."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(overlap_obj.construct_array_spherical(), overlap_basis_spherical(basis))


def test_overlap_basis_spherical_lincomb():
    """Test gbasis.eval.overlap_basis_spherical_lincomb."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    transform = np.random.rand(14, 18)
    assert np.allclose(
        overlap_obj.construct_array_spherical_lincomb(transform),
        overlap_basis_spherical_lincomb(basis, transform),
    )


def test_overlap_basis_cartesian_norm_anorcc():
    """Test the norm of gbasis.eval.overlap_basis_cartesian on the ANO-RCC basis set.

    The contraction coefficients in ANO-RCC is such that the cartesian contractions are normalized.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_basis_spherical_norm_sto6g():
    """Test the norm of gbasis.eval.overlap_basis_spherical on the STO-6G basis set.

    The contraction coefficients in STO-6G is such that the spherical contractions are not
    normalized to past 3rd decimal places.

    """
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_spherical()), 1)


def test_overlap_basis_spherical_norm_anorcc():
    """Test the norm of gbasis.eval.overlap_basis_spherical on the ANO-RCC basis set.

    The contraction coefficients in ANO-RCC is such that the Cartesian contractions are normalized.

    """
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["C"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)

    basis = make_contractions(basis_dict, ["Xe"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)


def test_overlap_basis_cartesian_norm_sto6g():
    """Test the norm of gbasis.eval.overlap_basis_cartesian on the STO-6G basis set.

    The contraction coefficients in STO-6G is such that the Cartesian contractions are not
    normalized to past 3rd decimal places.

    """
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)

    basis = make_contractions(basis_dict, ["Kr"], np.array([[0, 0, 0]]))
    overlap_obj = Overlap(basis)
    assert np.allclose(np.diag(overlap_obj.construct_array_cartesian()), 1)
