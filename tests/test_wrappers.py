"""Test gbasis.wrapper."""

from gbasis.contractions import GeneralizedContractionShell
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.wrappers import from_iodata, from_pyscf
import numpy as np
import pytest
from utils import find_datafile


def test_from_iodata():
    """Test gbasis.wrapper.from_iodata."""
    pytest.importorskip("iodata")
    from iodata import load_one

    mol = load_one(find_datafile("data_iodata_water_sto3g_hf_g03.fchk"))

    basis = from_iodata(mol)
    coord_types = [type for type in [shell.coord_type for shell in basis]]

    assert coord_types == ["cartesian"] * 5
    assert all(isinstance(i, GeneralizedContractionShell) for i in basis)
    assert basis[0].angmom == 0
    assert np.allclose(basis[0].coord, mol.atcoords[0])
    assert np.allclose(basis[0].exps, np.array([130.7093214, 23.80886605, 6.443608313]))
    assert np.allclose(
        basis[0].coeffs, np.array([0.1543289673, 0.5353281423, 0.4446345422]).reshape(-1, 1)
    )
    assert np.allclose(basis[0].norm_cont, 1.0)

    assert basis[1].angmom == 0
    assert np.allclose(basis[1].coord, mol.atcoords[0])
    assert np.allclose(basis[1].exps, np.array([5.033151319, 1.169596125, 0.3803889600]))
    assert np.allclose(
        basis[1].coeffs, np.array([-0.09996722919, 0.3995128261, 0.7001154689]).reshape(-1, 1)
    )
    assert np.allclose(basis[1].norm_cont, 1.0)

    assert basis[2].angmom == 1
    assert np.allclose(basis[2].coord, mol.atcoords[0])
    assert np.allclose(basis[2].exps, np.array([5.033151319, 1.169596125, 0.3803889600]))
    assert np.allclose(
        basis[2].coeffs, np.array([0.1559162750, 0.6076837186, 0.3919573931]).reshape(-1, 1)
    )
    assert np.allclose(basis[2].norm_cont, 1.0)

    assert basis[3].angmom == 0
    assert np.allclose(basis[3].coord, mol.atcoords[1])
    assert np.allclose(basis[3].exps, np.array([3.425250914, 0.6239137298, 0.1688554040]))
    assert np.allclose(
        basis[3].coeffs, np.array([0.1543289673, 0.5353281423, 0.4446345422]).reshape(-1, 1)
    )
    assert np.allclose(basis[3].norm_cont, 1.0)

    assert basis[4].angmom == 0
    assert np.allclose(basis[4].coord, mol.atcoords[2])
    assert np.allclose(basis[4].exps, np.array([3.425250914, 0.6239137298, 0.1688554040]))
    assert np.allclose(
        basis[4].coeffs, np.array([0.1543289673, 0.5353281423, 0.4446345422]).reshape(-1, 1)
    )
    assert np.allclose(basis[4].norm_cont, 1.0)

    # Artificially change angular momentum.
    # The following few lines are commented out deliberately. The file
    # "data_iodata_water_sto3g_hf_g03.fchk" does not contain "spherical"
    # functions for angular momenta 0 and 1, so there is no need to have
    # conventions for them. (This is a fairly common pattern in most QC codes.)
    # -- BEGIN COMMENTED ASSERTS
    # basis[2].angmom = 0
    # assert basis[2].angmom_components_sph == (0,)
    # basis[2].angmom = 1
    # assert basis[2].angmom_components_sph == (1, -1, 0)
    # -- END COMMENTED ASSERTS
    basis[2].angmom = 2
    assert basis[2].angmom_components_sph == ("c0", "c1", "s1", "c2", "s2")
    basis[2].angmom = 3
    assert basis[2].angmom_components_sph == ("c0", "c1", "s1", "c2", "s2", "c3", "s3")

    # NOTE: you shouldn't actually change the magnetic quantum number that is not compatible with
    # the angular momentum, but we do so here to check that user input is accepted
    mol.obasis.conventions[(0, "p")] = ["c1"]
    basis = from_iodata(mol)
    coord_types = [type for type in [shell.coord_type for shell in basis]]
    basis[2].angmom = 0
    assert coord_types == ["cartesian"] * 5
    assert basis[2].angmom_components_sph == ("c1",)
    assert np.allclose(basis[2].norm_cont, 1.0)

    mol.obasis.conventions[(1, "p")] = ["c1", "c0", "s1"]
    basis = from_iodata(mol)
    coord_types = [type for type in [shell.coord_type for shell in basis]]
    basis[2].angmom = 1
    assert coord_types == ["cartesian"] * 5
    assert basis[2].angmom_components_sph == ("c1", "c0", "s1")
    assert np.allclose(basis[2].norm_cont, 1.0)

    mol.obasis.conventions[(1, "c")] = ["z", "y", "x"]
    basis = from_iodata(mol)
    basis[2].angmom = 1
    assert np.allclose(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), basis[2].angmom_components_cart)

    # Test Cartesian convention generation for missing angmom
    # Needed for cases when only spherical basis is used in basis set
    del mol.obasis.conventions[(1, "c")]
    basis = from_iodata(mol)
    basis[2].angmom = 1
    assert np.allclose(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), basis[2].angmom_components_cart)

    with pytest.raises(ValueError):
        basis[2].angmom = 10
        basis[2].angmom_components_sph

    with pytest.raises(ValueError):
        mol.obasis.primitive_normalization = "L1"
        basis, coord_types = from_iodata(mol)


def test_from_pyscf():
    """Test gbasis.wrapper.from_pyscf."""
    pytest.importorskip("pyscf")
    from pyscf import gto

    mol = gto.Mole()
    mol.build(atom="""Kr 1.0 2.0 3.0""", basis="ano-rcc", unit="Bohr")
    test = from_pyscf(mol)

    basis_dict = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    basis = make_contractions(basis_dict, ["Kr"], np.array([[1, 2, 3]]), "spherical")

    with pytest.raises(ValueError):

        class OtherName(gto.Mole):
            pass

        test = from_pyscf(OtherName())

    assert len(test) == len(basis)
    for i, j in zip(test, basis):
        assert np.allclose(i.coord, j.coord)
        assert i.angmom == j.angmom
        assert np.allclose(i.exps, j.exps)
        assert np.allclose(i.coeffs, j.coeffs)
        assert np.allclose(i.norm_cont, j.norm_cont)

    assert test[0].angmom_components_sph == ("c0",)
    assert test[1].angmom_components_sph == ("c1", "s1", "c0")
    assert test[2].angmom_components_sph == ("s2", "s1", "c0", "c1", "c2")
