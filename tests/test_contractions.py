"""Test gbasis.contractions."""
from gbasis.contractions import ContractedCartesianGaussians, make_contractions
from gbasis.parsers import parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, skip_init


def test_charge_setter():
    """Test setter for ContractedCartesianGaussians.charge."""
    test = skip_init(ContractedCartesianGaussians)
    test.charge = 2
    assert isinstance(test._charge, float) and test._charge == 2
    test.charge = -2
    assert isinstance(test._charge, float) and test._charge == -2
    test.charge = 2.5
    assert isinstance(test._charge, float) and test._charge == 2.5
    with pytest.raises(TypeError):
        test.charge = "0"
    with pytest.raises(TypeError):
        test.charge = None


def test_charge_getter():
    """Test getter for ContractedCartesianGaussians.charge."""
    test = skip_init(ContractedCartesianGaussians)
    test._charge = 2
    assert test.charge == 2


def test_coord_setter():
    """Test setter for ContractedCartesianGaussians.coord."""
    test = skip_init(ContractedCartesianGaussians)
    test.coord = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coord, np.ndarray)
        and test._coord.dtype == float
        and np.allclose(test._coord, np.array([1, 2, 3]))
    )
    test.coord = np.array([1, 2, 3])
    assert (
        isinstance(test._coord, np.ndarray)
        and test._coord.dtype == float
        and np.allclose(test._coord, np.array([1, 2, 3]))
    )

    with pytest.raises(TypeError):
        test.coord = [1, 2, 3]
    with pytest.raises(TypeError):
        test.coord = np.array([1, 2])
    with pytest.raises(TypeError):
        test.coord = np.array([1, 2, 3], dtype=bool)


def test_coord_getter():
    """Test getter for ContractedCartesianGaussians.coord."""
    test = skip_init(ContractedCartesianGaussians)
    test._coord = 2
    assert test.coord == 2


def test_angmom_setter():
    """Test setter for ContractedCartesianGaussians.angmom."""
    test = skip_init(ContractedCartesianGaussians)
    test.angmom = 1
    assert isinstance(test._angmom, int) and test._angmom == 1
    test.angmom = 0
    assert isinstance(test._angmom, int) and test._angmom == 0
    with pytest.raises(ValueError):
        test.angmom = -2
    with pytest.raises(TypeError):
        test.angmom = "0"
    with pytest.raises(TypeError):
        test.angmom = 0.0
    with pytest.raises(TypeError):
        test.angmom = None


def test_angmom_getter():
    """Test getter for ContractedCartesianGaussians.angmom."""
    test = skip_init(ContractedCartesianGaussians)
    test._angmom = 1
    assert test.angmom == 1


def test_exps_setter():
    """Test setter for ContractedCartesianGaussians.exps."""
    test = skip_init(ContractedCartesianGaussians)
    test.exps = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._exps, np.ndarray)
        and test._exps.dtype == float
        and np.allclose(test._exps, np.array([1, 2, 3]))
    )

    test = skip_init(ContractedCartesianGaussians)
    test.coeffs = np.array([1.0, 2.0, 3.0])
    test.exps = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._exps, np.ndarray)
        and test._exps.dtype == float
        and np.allclose(test._exps, np.array([1, 2, 3]))
    )

    test = skip_init(ContractedCartesianGaussians)
    with pytest.raises(TypeError):
        test.exps = [1, 2, 3]
    with pytest.raises(TypeError):
        test.exps = np.array([1, 2, 3], dtype=bool)
    with pytest.raises(ValueError):
        test.coeffs = np.array([1.0, 2.0, 3.0])
        test.exps = np.array([4.0, 5.0])
    with pytest.raises(ValueError):
        test.coeffs = np.array([[1.0], [2.0], [3.0]])
        test.exps = np.array([4.0, 5.0])


def test_exps_getter():
    """Test getter for ContractedCartesianGaussians.exps."""
    test = skip_init(ContractedCartesianGaussians)
    test._exps = [2.0, 3.0]
    assert test.exps == [2.0, 3.0]


def test_coeffs_setter():
    """Test setter for ContractedCartesianGaussians.coeffs."""
    test = skip_init(ContractedCartesianGaussians)
    test.coeffs = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1], [2], [3]]))
    )

    test = skip_init(ContractedCartesianGaussians)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([1.0, 2.0, 3.0])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1], [2], [3]]))
    )

    test = skip_init(ContractedCartesianGaussians)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([[1.0], [2.0], [3.0]])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1], [2], [3]]))
    )

    test = skip_init(ContractedCartesianGaussians)
    test.exps = np.array([4.0, 5.0, 6.0])
    test.coeffs = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert (
        isinstance(test._coeffs, np.ndarray)
        and test._coeffs.dtype == float
        and np.allclose(test._coeffs, np.array([[1, 4], [2, 5], [3, 6]]))
    )

    test = skip_init(ContractedCartesianGaussians)
    with pytest.raises(TypeError):
        test.coeffs = [1, 2, 3]
    with pytest.raises(TypeError):
        test.coeffs = np.array([1, 2, 3], dtype=bool)
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0])
        test.coeffs = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0, 6.0])
        test.coeffs = np.array([[[1.0, 2.0, 3.0]]])
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0, 6.0])
        test.coeffs = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        test.exps = np.array([4.0, 5.0])
        test.coeffs = np.array([[1.0], [2.0], [3.0]])


def test_coeffs_getter():
    """Test getter for ContractedCartesianGaussians.coeffs."""
    test = skip_init(ContractedCartesianGaussians)
    test._coeffs = [2.0, 3.0]
    assert test.coeffs == [2.0, 3.0]


def tests_init():
    """Test ContractedCartesianGaussians.__init__."""
    test = ContractedCartesianGaussians(
        1,
        np.array([0, 1, 2]),
        0,
        np.array([1, 2, 3, 4], dtype=float),
        np.array([5, 6, 7, 8], dtype=float),
    )
    assert test._angmom == 1
    assert np.allclose(test._coord, np.array([0, 1, 2]))
    assert test._charge == 0
    assert np.allclose(test._coeffs, np.array([[1], [2], [3], [4]]))
    assert np.allclose(test._exps, np.array([5, 6, 7, 8]))


def test_angmom_components():
    """Test ContractedCartesianGaussians.angmom_components."""
    test = skip_init(ContractedCartesianGaussians)
    test._angmom = 0
    assert np.allclose(test.angmom_components, [(0, 0, 0)])
    test._angmom = 1
    assert np.allclose(test.angmom_components, [(0, 0, 1), (0, 1, 0), (1, 0, 0)])
    test._angmom = 2
    assert np.allclose(
        test.angmom_components, [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    )
    test._angmom = 3
    assert np.allclose(
        test.angmom_components,
        [
            (0, 0, 3),
            (0, 1, 2),
            (0, 2, 1),
            (0, 3, 0),
            (1, 0, 2),
            (1, 1, 1),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (3, 0, 0),
        ],
    )
    test._angmom = 10
    assert len(test.angmom_components) == 11 * 12 / 2


# TODO: Test norm using actual integrals
# TODO: add more tests
def test_norm():
    """Test ContractedCartesianGaussians.norm."""
    test = ContractedCartesianGaussians(
        0, np.array([0, 0, 0]), 0, np.array([1.0]), np.array([0.25])
    )
    assert np.isclose(test.norm, 0.2519794355383807303479140)
    test = ContractedCartesianGaussians(3, np.array([0, 0, 0]), 0, np.array([1.0]), np.array([0.5]))
    assert np.isclose(test.norm[7], 0.6920252830162908851679097)


def test_num_contr():
    """Test ContractedCartesianGaussians.num_contr."""
    test = skip_init(ContractedCartesianGaussians)
    last_num_contr = 0
    for i in range(100):
        test._angmom = i
        assert test.num_contr == last_num_contr + i + 1
        last_num_contr = test.num_contr


def test_make_contractions():
    """Test gbasis.contractions.make_contractions."""
    with open(find_datafile("data_sto6g.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    with pytest.raises(TypeError):
        make_contractions(basis_dict, {"H", "H"}, np.array([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, [0, 0], np.array([[0, 0, 0], [1, 1, 1]]))

    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], [[0, 0, 0], [1, 1, 1]])
    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([0, 0, 0, 1, 1, 1]))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0, 2], [1, 1, 1, 2]]))

    with pytest.raises(ValueError):
        make_contractions(basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))

    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), [0, 0])
    with pytest.raises(TypeError):
        make_contractions(
            basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), np.array([[0, 0]])
        )

    with pytest.raises(ValueError):
        make_contractions(
            basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), np.array([0, 0, 0])
        )

    test = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    assert isinstance(test, tuple)
    assert len(test) == 2
    assert test[0].angmom == 0
    assert np.allclose(test[0].coord, np.array([0, 0, 0]))
    assert test[0].charge == 0
    assert np.allclose(
        test[0].coeffs,
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )
    assert np.allclose(
        test[0].exps,
        np.array([35.52322122, 6.513143725, 1.822142904, 0.625955266, 0.243076747, 0.100112428]),
    )
    assert test[1].angmom == 0
    assert np.allclose(test[1].coord, np.array([1, 1, 1]))
    assert test[1].charge == 0
    assert np.allclose(
        test[1].coeffs,
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )
    assert np.allclose(
        test[1].exps,
        np.array([35.52322122, 6.513143725, 1.822142904, 0.625955266, 0.243076747, 0.100112428]),
    )
