"""Test gbasis.stress_tensor."""
from gbasis.density import eval_density_laplacian, eval_deriv_density, eval_deriv_density_matrix
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.stress_tensor import eval_ehrenfest_force, eval_ehrenfest_hessian, eval_stress_tensor
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


def test_eval_ehrenfest_force():
    """Test gbasis.stress_tensor.eval_ehrenfest_force."""
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    coords = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], coords)
    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), np.array(0), 0)
    with pytest.raises(TypeError):
        eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), 1, None)
    with pytest.raises(TypeError):
        eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), 1, 0j)

    test_a = eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), 0, 0)
    test_b = eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), 1, 0)
    test_c = eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), 0.5, 0)
    test_d = eval_ehrenfest_force(np.identity(40), basis, points, np.identity(40), 0, 2)
    for j in range(3):
        ref_a = np.zeros(points.shape[0])
        ref_b = np.zeros(points.shape[0])
        ref_c = np.zeros(points.shape[0])
        ref_d = np.zeros(points.shape[0])

        orders_j = np.array([0, 0, 0])
        orders_j[j] += 1
        for i in range(3):
            orders_i = np.array([0, 0, 0])
            orders_i[i] += 1

            temp1 = eval_deriv_density_matrix(
                2 * orders_i, orders_j, np.identity(40), basis, points, np.identity(40)
            )
            temp2 = eval_deriv_density_matrix(
                orders_i, orders_i + orders_j, np.identity(40), basis, points, np.identity(40)
            )
            temp3 = eval_deriv_density_matrix(
                2 * orders_i + orders_j,
                np.array([0, 0, 0]),
                np.identity(40),
                basis,
                points,
                np.identity(40),
            )
            temp4 = eval_deriv_density_matrix(
                orders_i + orders_j, orders_i, np.identity(40), basis, points, np.identity(40)
            )
            temp5 = eval_deriv_density(
                2 * orders_i + orders_j, np.identity(40), basis, points, np.identity(40)
            )

            ref_a += temp3 + temp4
            ref_b += -temp1 - temp2
            ref_c += -0.5 * temp1 - 0.5 * temp2 + 0.5 * temp3 + 0.5 * temp4
            ref_d += temp3 + temp4 - temp5
        assert np.allclose(test_a[:, j], ref_a)
        assert np.allclose(test_b[:, j], ref_b)
        assert np.allclose(test_c[:, j], ref_c)
        assert np.allclose(test_d[:, j], ref_d)


def test_eval_ehrenfest_hessian():
    """Test gbasis.stress_tensor.eval_ehrenfest_hessian."""
    with open(find_datafile("data_anorcc.nwchem"), "r") as f:
        test_basis = f.read()
    basis_dict = parse_nwchem(test_basis)
    coords = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], coords)
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps) for i in basis]

    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), np.array(0), 0)
    with pytest.raises(TypeError):
        eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 1, None)
    with pytest.raises(TypeError):
        eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 1, 0j)

    test_a = eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 0, 0)
    test_b = eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 1, 0)
    test_c = eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 0.5, 0)
    test_d = eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 0, 2)
    for j in range(3):
        for k in range(3):
            ref_a = np.zeros(points.shape[0])
            ref_b = np.zeros(points.shape[0])
            ref_c = np.zeros(points.shape[0])
            ref_d = np.zeros(points.shape[0])

            orders_j = np.array([0, 0, 0])
            orders_j[j] += 1
            orders_k = np.array([0, 0, 0])
            orders_k[k] += 1
            for i in range(3):
                orders_i = np.array([0, 0, 0])
                orders_i[i] += 1

                temp1 = eval_deriv_density_matrix(
                    2 * orders_i + orders_k,
                    orders_j,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp2 = eval_deriv_density_matrix(
                    2 * orders_i,
                    orders_j + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp3 = eval_deriv_density_matrix(
                    orders_i + orders_k,
                    orders_i + orders_j,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp4 = eval_deriv_density_matrix(
                    orders_i,
                    orders_i + orders_j + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp5 = eval_deriv_density_matrix(
                    2 * orders_i + orders_j + orders_k,
                    np.array([0, 0, 0]),
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp6 = eval_deriv_density_matrix(
                    2 * orders_i + orders_j,
                    orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp7 = eval_deriv_density_matrix(
                    orders_i + orders_j + orders_k,
                    orders_i,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp8 = eval_deriv_density_matrix(
                    orders_i + orders_j,
                    orders_i + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )
                temp9 = eval_deriv_density(
                    2 * orders_i + orders_j + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                )

                ref_a += temp5 + temp6 + temp7 + temp8
                ref_b += -temp1 - temp2 - temp3 - temp4
                ref_c += (
                    -0.5 * temp1
                    - 0.5 * temp2
                    - 0.5 * temp3
                    - 0.5 * temp4
                    + 0.5 * temp5
                    + 0.5 * temp6
                    + 0.5 * temp7
                    + 0.5 * temp8
                )
                ref_d += temp3 + temp4 + temp5 + temp6 - temp9
            assert np.allclose(test_a[:, j, k], ref_a)
            assert np.allclose(test_b[:, j, k], ref_b)
            assert np.allclose(test_c[:, j, k], ref_c)
            assert np.allclose(test_d[:, j, k], ref_d)
    assert np.allclose(
        eval_ehrenfest_hessian(
            np.identity(40), basis, points, np.identity(40), 0, 0, symmetric=True
        ),
        (
            eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 0, 0)
            + np.swapaxes(
                eval_ehrenfest_hessian(np.identity(40), basis, points, np.identity(40), 0, 0), 1, 2
            )
        )
        / 2,
    )
