"""Test gbasis.evals.stress_tensor."""

from gbasis.evals.density import (
    evaluate_density_laplacian,
    evaluate_deriv_density,
    evaluate_deriv_reduced_density_matrix,
)
from gbasis.evals.stress_tensor import (
    evaluate_ehrenfest_force,
    evaluate_ehrenfest_hessian,
    evaluate_stress_tensor,
)
from gbasis.parsers import make_contractions, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile, HortonContractions


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
def test_evaluate_stress_tensor(screen_basis, tol_screen):
    """Test gbasis.evals.stress_tensor.evaluate_stress_tensor."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    coords = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], coords, "spherical")
    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        evaluate_stress_tensor(np.identity(40), basis, points, np.array(0), 0, np.identity(40))
    with pytest.raises(TypeError):
        evaluate_stress_tensor(np.identity(40), basis, points, 1, None, np.identity(40))
    with pytest.raises(TypeError):
        evaluate_stress_tensor(np.identity(40), basis, points, 1, 0j, np.identity(40))

    test_a = evaluate_stress_tensor(
        np.identity(40),
        basis,
        points,
        0,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_b = evaluate_stress_tensor(
        np.identity(40),
        basis,
        points,
        1,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_c = evaluate_stress_tensor(
        np.identity(40),
        basis,
        points,
        1,
        2,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_d = evaluate_stress_tensor(
        np.identity(40),
        basis,
        points,
        0.5,
        2,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    # compute reference without screening
    for i in range(3):
        for j in range(3):
            orders_i = np.array([0, 0, 0])
            orders_i[i] += 1
            orders_j = np.array([0, 0, 0])
            orders_j[j] += 1

            temp1 = evaluate_deriv_reduced_density_matrix(
                orders_i,
                orders_j,
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp2 = evaluate_deriv_reduced_density_matrix(
                orders_j,
                orders_i,
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp3 = evaluate_deriv_reduced_density_matrix(
                orders_i + orders_j,
                np.array([0, 0, 0]),
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp4 = evaluate_deriv_reduced_density_matrix(
                orders_i + orders_j,
                np.array([0, 0, 0]),
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            if i == j:
                temp5 = evaluate_density_laplacian(
                    np.identity(40), basis, points, np.identity(40), screen_basis=False
                )
            else:
                temp5 = 0
            # check that non screened reference matches screened result within tol_screen
            assert np.allclose(test_a[:, i, j], 0.5 * temp3 + 0.5 * temp4, atol=tol_screen)
            assert np.allclose(test_b[:, i, j], -0.5 * temp1 - 0.5 * temp2, atol=tol_screen)
            assert np.allclose(test_c[:, i, j], -0.5 * temp1 - 0.5 * temp2 - temp5, atol=tol_screen)
            assert np.allclose(
                test_d[:, i, j],
                -0.25 * temp1 - 0.25 * temp2 + 0.25 * temp3 + 0.25 * temp4 - temp5,
                atol=tol_screen,
            )


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
def test_evaluate_ehrenfest_force(screen_basis, tol_screen):
    """Test gbasis.evals.stress_tensor.evaluate_ehrenfest_force."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    coords = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], coords, "spherical")
    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        evaluate_ehrenfest_force(np.identity(40), basis, points, np.array(0), 0, np.identity(40))
    with pytest.raises(TypeError):
        evaluate_ehrenfest_force(np.identity(40), basis, points, 1, None, np.identity(40))
    with pytest.raises(TypeError):
        evaluate_ehrenfest_force(np.identity(40), basis, points, 1, 0j, np.identity(40))

    test_a = evaluate_ehrenfest_force(
        np.identity(40),
        basis,
        points,
        0,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_b = evaluate_ehrenfest_force(
        np.identity(40),
        basis,
        points,
        1,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_c = evaluate_ehrenfest_force(
        np.identity(40),
        basis,
        points,
        0.5,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_d = evaluate_ehrenfest_force(
        np.identity(40),
        basis,
        points,
        0,
        2,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    for j in range(3):
        ref_a = np.zeros(points.shape[0])
        ref_b = np.zeros(points.shape[0])
        ref_c = np.zeros(points.shape[0])
        ref_d = np.zeros(points.shape[0])

        # compute reference without screening
        orders_j = np.array([0, 0, 0])
        orders_j[j] += 1
        for i in range(3):
            orders_i = np.array([0, 0, 0])
            orders_i[i] += 1

            temp1 = evaluate_deriv_reduced_density_matrix(
                2 * orders_i,
                orders_j,
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp2 = evaluate_deriv_reduced_density_matrix(
                orders_i,
                orders_i + orders_j,
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp3 = evaluate_deriv_reduced_density_matrix(
                2 * orders_i + orders_j,
                np.array([0, 0, 0]),
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp4 = evaluate_deriv_reduced_density_matrix(
                orders_i + orders_j,
                orders_i,
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )
            temp5 = evaluate_deriv_density(
                2 * orders_i + orders_j,
                np.identity(40),
                basis,
                points,
                np.identity(40),
                screen_basis=False,
            )

            ref_a += temp3 + temp4
            ref_b += -temp1 - temp2
            ref_c += -0.5 * temp1 - 0.5 * temp2 + 0.5 * temp3 + 0.5 * temp4
            ref_d += temp3 + temp4 - temp5
        # check that non screened reference matches screened result within tol_screen
        assert np.allclose(test_a[:, j], -ref_a, atol=tol_screen)
        assert np.allclose(test_b[:, j], -ref_b, atol=tol_screen)
        assert np.allclose(test_c[:, j], -ref_c, atol=tol_screen)
        assert np.allclose(test_d[:, j], -ref_d, atol=tol_screen)


@pytest.mark.parametrize("screen_basis", [True, False])
@pytest.mark.parametrize("tol_screen", [1e-8])
def test_evaluate_ehrenfest_hessian(screen_basis, tol_screen):
    """Test gbasis.evals.stress_tensor.evaluate_ehrenfest_hessian."""
    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    coords = np.array([[0, 0, 0]])
    basis = make_contractions(basis_dict, ["H"], coords, "spherical")
    basis = [HortonContractions(i.angmom, i.coord, i.coeffs, i.exps, i.coord_type) for i in basis]

    points = np.random.rand(10, 3)

    with pytest.raises(TypeError):
        evaluate_ehrenfest_hessian(np.identity(40), basis, points, np.array(0), 0, np.identity(40))
    with pytest.raises(TypeError):
        evaluate_ehrenfest_hessian(np.identity(40), basis, points, 1, None, np.identity(40))
    with pytest.raises(TypeError):
        evaluate_ehrenfest_hessian(np.identity(40), basis, points, 1, 0j, np.identity(40))

    test_a = evaluate_ehrenfest_hessian(
        np.identity(40),
        basis,
        points,
        0,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_b = evaluate_ehrenfest_hessian(
        np.identity(40),
        basis,
        points,
        1,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_c = evaluate_ehrenfest_hessian(
        np.identity(40),
        basis,
        points,
        0.5,
        0,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )
    test_d = evaluate_ehrenfest_hessian(
        np.identity(40),
        basis,
        points,
        0,
        2,
        np.identity(40),
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )

    # compute reference without screening
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

                temp1 = evaluate_deriv_reduced_density_matrix(
                    2 * orders_i + orders_k,
                    orders_j,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp2 = evaluate_deriv_reduced_density_matrix(
                    2 * orders_i,
                    orders_j + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp3 = evaluate_deriv_reduced_density_matrix(
                    orders_i + orders_k,
                    orders_i + orders_j,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp4 = evaluate_deriv_reduced_density_matrix(
                    orders_i,
                    orders_i + orders_j + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp5 = evaluate_deriv_reduced_density_matrix(
                    2 * orders_i + orders_j + orders_k,
                    np.array([0, 0, 0]),
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp6 = evaluate_deriv_reduced_density_matrix(
                    2 * orders_i + orders_j,
                    orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp7 = evaluate_deriv_reduced_density_matrix(
                    orders_i + orders_j + orders_k,
                    orders_i,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp8 = evaluate_deriv_reduced_density_matrix(
                    orders_i + orders_j,
                    orders_i + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
                )
                temp9 = evaluate_deriv_density(
                    2 * orders_i + orders_j + orders_k,
                    np.identity(40),
                    basis,
                    points,
                    np.identity(40),
                    screen_basis=False,
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
            assert np.allclose(test_a[:, j, k], -ref_a, atol=tol_screen)
            assert np.allclose(test_b[:, j, k], -ref_b, atol=tol_screen)
            assert np.allclose(test_c[:, j, k], -ref_c, atol=tol_screen)
            assert np.allclose(test_d[:, j, k], -ref_d, atol=tol_screen)
    # check symmetry if requested
    assert np.allclose(
        evaluate_ehrenfest_hessian(
            np.identity(40),
            basis,
            points,
            0,
            0,
            np.identity(40),
            symmetric=True,
            screen_basis=False,
        ),
        (
            evaluate_ehrenfest_hessian(
                np.identity(40), basis, points, 0, 0, np.identity(40), screen_basis=False
            )
            + np.swapaxes(
                evaluate_ehrenfest_hessian(
                    np.identity(40), basis, points, 0, 0, np.identity(40), screen_basis=False
                ),
                1,
                2,
            )
        )
        / 2,
    )
