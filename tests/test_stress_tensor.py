"""Test gbasis.stress_tensor."""
from gbasis.density import eval_density_laplacian, eval_deriv_density, eval_deriv_density_matrix
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.stress_tensor import eval_stress_tensor
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


def test_eval_stress_tensor():
    """Test gbasis.stress_tensor.eval_stress_tensor."""
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    coords = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], coords)
    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        eval_stress_tensor(np.identity(40), basis, points, np.identity(40), np.array(0), 0)
    with pytest.raises(TypeError):
        eval_stress_tensor(np.identity(40), basis, points, np.identity(40), 1, None)
    with pytest.raises(TypeError):
        eval_stress_tensor(np.identity(40), basis, points, np.identity(40), 1, 0j)

    test_a = eval_stress_tensor(np.identity(40), basis, points, np.identity(40), 0, 0)
    test_b = eval_stress_tensor(np.identity(40), basis, points, np.identity(40), 1, 0)
    test_c = eval_stress_tensor(np.identity(40), basis, points, np.identity(40), 1, 2)
    test_d = eval_stress_tensor(np.identity(40), basis, points, np.identity(40), 0.5, 2)
    for i in range(3):
        for j in range(3):
            orders_i = np.array([0, 0, 0])
            orders_i[i] += 1
            orders_j = np.array([0, 0, 0])
            orders_j[j] += 1

            temp1 = eval_deriv_density_matrix(
                orders_i, orders_j, np.identity(40), basis, points, np.identity(40)
            )
            temp2 = eval_deriv_density_matrix(
                orders_j, orders_i, np.identity(40), basis, points, np.identity(40)
            )
            temp3 = eval_deriv_density_matrix(
                orders_i + orders_j,
                np.array([0, 0, 0]),
                np.identity(40),
                basis,
                points,
                np.identity(40),
            )
            temp4 = eval_deriv_density_matrix(
                orders_i + orders_j,
                np.array([0, 0, 0]),
                np.identity(40),
                basis,
                points,
                np.identity(40),
            )
            if i == j:
                temp5 = eval_density_laplacian(np.identity(40), basis, points, np.identity(40))
            else:
                temp5 = 0
            assert np.allclose(test_a[:, i, j], 0.5 * temp3 + 0.5 * temp4)
            assert np.allclose(test_b[:, i, j], -0.5 * temp1 - 0.5 * temp2)
            assert np.allclose(test_c[:, i, j], -0.5 * temp1 - 0.5 * temp2 - temp5)
            assert np.allclose(
                test_d[:, i, j], -0.25 * temp1 - 0.25 * temp2 + 0.25 * temp3 + 0.25 * temp4 - temp5
            )

    assert np.allclose(test_a, np.swapaxes(test_a, 1, 2))
    assert np.allclose(test_b, np.swapaxes(test_b, 1, 2))
    assert np.allclose(test_c, np.swapaxes(test_c, 1, 2))
    assert np.allclose(test_d, np.swapaxes(test_d, 1, 2))
