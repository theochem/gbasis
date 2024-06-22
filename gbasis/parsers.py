"""Parsers for reading basis set files."""

import re

from gbasis.contractions import GeneralizedContractionShell
import numpy as np


def parse_nwchem(nwchem_basis_file):
    """Parse nwchem basis set file.

    Parameters
    ----------
    nwchem_basis_file : str
        Path to the nwchem basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater than "k", an error will be raised.

    """
    # pylint: disable=R0914
    with open(nwchem_basis_file, "r") as basis_fh:
        nwchem_basis = basis_fh.read()

    data = re.split(r"\n\s*(\w[\w]?)[ ]+(\w+)\s*\n", nwchem_basis)
    dict_angmom = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    # remove first part
    if "\n" in data[0]:  # pragma: no branch
        data = data[1:]
    atoms = data[::3]
    angmoms = data[1::3]
    exps_coeffs_all = data[2::3]
    # trim out headers at the end
    output = {}
    for atom, angmom_gen, exps_coeffs in zip(atoms, angmoms, exps_coeffs_all):
        output.setdefault(atom, [])
        angmom_seg = [dict_angmom[i.lower()] for i in angmom_gen]
        exps_coeffs = exps_coeffs.split("\n")
        exps = []
        coeffs_gen = []
        for line in exps_coeffs:
            test = re.search(
                r"^\s*([0-9\.DE\+\-]+)\s+((?:(?:[0-9\.DE\+\-]+)\s+)*(?:[0-9\.DE\+\-]+))\s*$", line
            )
            try:
                exp, coeff_gen = test.groups()
                coeff_gen = re.split(r"\s+", coeff_gen)
            except AttributeError:
                continue
            # clean up
            exp = float(exp.lower().replace("d", "e"))
            coeff_gen = [float(i.lower().replace("d", "e")) for i in coeff_gen if i is not None]
            exps.append(exp)
            coeffs_gen.append(coeff_gen)
        exps = np.array(exps)
        coeffs_gen = np.array(coeffs_gen)

        if len(angmom_seg) == 1:
            output[atom].append((angmom_seg[0], exps, coeffs_gen))
        else:
            for i, angmom in enumerate(angmom_seg):
                output[atom].append((angmom, exps, coeffs_gen[:, i]))

    return output


def parse_gbs(gbs_basis_file):
    """Parse Gaussian94 basis set file.

    Parameters
    ----------
    gbs_basis_file : str
        Path to the Gaussian94 basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater than "k", an error will be raised.

    Since Gaussian94 basis format does not explicitly state which contractions are generalized, we
    infer that subsequent contractions belong to the same generalized shell if they have the same
    exponents and angular momentum. If two contractions are not one after another or if they are
    associated with more than one angular momentum, they are treated to be segmented contractions.

    """
    # pylint: disable=R0914
    with open(gbs_basis_file) as basis_fh:
        gbs_basis = basis_fh.read()
    # splits file into 'element', 'basis stuff', 'element',' basis stuff'
    # e.g., ['H','stuff with exponents & coefficients\n', 'C', 'stuff with etc\n']
    data = re.split(r"\n\s*(\w[\w]?)\s+\w+\s*\n", gbs_basis)
    dict_angmom = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    # remove first part
    if "\n" in data[0]:  # pragma: no branch
        data = data[1:]
    # atoms: stride of 2 get the ['H','C', etc]. basis: take strides of 2 to skip elements
    atoms = data[::2]
    basis = data[1::2]
    # trim out headers at the end
    output = {}
    for atom, shells in zip(atoms, basis):
        output.setdefault(atom, [])

        shells = re.split(r"\n?\s*(\w+)\s+\w+\s+\w+\.\w+\s*\n", shells)
        # remove the ends
        atom_basis = shells[1:]
        # get angular momentums
        angmom_shells = atom_basis[::2]
        # get exponents and coefficients
        exps_coeffs_shells = atom_basis[1::2]

        for angmom_seg, exp_coeffs in zip(angmom_shells, exps_coeffs_shells):
            angmom_seg = [dict_angmom[i.lower()] for i in angmom_seg]
            exps = []
            coeffs_seg = []
            exp_coeffs = exp_coeffs.split("\n")
            for line in exp_coeffs:
                test = re.search(
                    r"^\s*([0-9\.DE\+\-]+)\s+((?:(?:[0-9\.DE\+\-]+)\s+)*(?:[0-9\.DE\+\-]+))\s*$",
                    line,
                )
                try:
                    exp, coeff_seg = test.groups()
                    coeff_seg = re.split(r"\s+", coeff_seg)
                except AttributeError:
                    continue
                # clean up
                exp = float(exp.lower().replace("d", "e"))
                coeff_seg = [float(i.lower().replace("d", "e")) for i in coeff_seg if i is not None]
                exps.append(exp)
                coeffs_seg.append(coeff_seg)
            exps = np.array(exps)
            coeffs_seg = np.array(coeffs_seg)
            # if len(angmom_seg) == 1:
            #     coeffs_seg = coeffs_seg[:, None]
            for i, angmom in enumerate(angmom_seg):
                # ensure previous and current exps are same length before using np.allclose()
                if output[atom] and len(output[atom][-1][1]) == len(exps):
                    # check if current exp's should be added to previous generalized contraction
                    hstack = np.allclose(output[atom][-1][1], exps)
                else:
                    hstack = False
                if output[atom] and output[atom][-1][0] == angmom and hstack:
                    output[atom][-1] = (
                        angmom,
                        exps,
                        np.hstack([output[atom][-1][2], coeffs_seg[:, i : i + 1]]),
                    )
                else:
                    output[atom].append((angmom, exps, coeffs_seg[:, i : i + 1]))

    return output


def make_contractions(basis_dict, atoms, coords, coord_types):
    """Return the contractions that correspond to the given atoms for the given basis.

    Parameters
    ----------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Output of the parsers from gbasis.parsers.
    atoms : N-list/tuple of str
        Atoms at which the contractions are centered.
    coords : np.ndarray(N, 3)
        Coordinates of each atom.
    coord_types : {"cartesian"/"c", list/tuple of "cartesian"/"c" or "spherical"/"p", "spherical"/"p"}
        Types of the coordinate system for the contractions.
        If "cartesian" or "c", then all of the contractions are treated as Cartesian contractions.
        If "spherical" or "p", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" (or "c") or "spherical" (or "p") to specify the
        coordinate type of each `GeneralizedContractionShell` instance.
        Default value is "spherical".

    Returns
    -------
    basis : tuple of GeneralizedContractionShell
        Contractions for each atom.
        Contractions are ordered in the same order as in the values of `basis_dict`.

    Raises
    ------
    TypeError
        If `atoms` is not a list or tuple of strings.
        If `coords` is not a two-dimensional `numpy` array with 3 columns.
        If `tol` is not a float.
        If `ovr` is not boolean
    ValueError
        If the length of atoms is not equal to the number of rows of `coords`.

    """
    if not (isinstance(atoms, (list, tuple)) and all(isinstance(i, str) for i in atoms)):
        raise TypeError("Atoms must be provided as a list or tuple.")
    if not (isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3):
        raise TypeError(
            "Coordinates must be provided as a two-dimensional `numpy` array with three columns."
        )

    if len(atoms) != coords.shape[0]:
        raise ValueError("Number of atoms must be equal to the number of rows in the coordinates.")

    basis = []
    # expected number of coordinates
    num_coord_types = sum([len(basis_dict[i]) for i in atoms])

    # check and assign coord_types
    if isinstance(coord_types, str):
        if coord_types not in ["c", "cartesian", "p", "spherical"]:
            raise ValueError(
                f"If coord_types is a string, it must be either 'spherical'/'p' or 'cartesian'/'c'."
                f"got {coord_types}"
            )
        coord_types = [coord_types] * num_coord_types

    if len(coord_types) != num_coord_types:
        raise ValueError(
            f"If coord_types is a list, it must be the same length as the total number of contractions."
            f"got {len(coord_types)}"
        )

    # make shells
    for icenter, (atom, coord) in enumerate(zip(atoms, coords)):
        for angmom, exps, coeffs in basis_dict[atom]:
            basis.append(
                GeneralizedContractionShell(
                    angmom,
                    coord,
                    coeffs,
                    exps,
                    coord_types.pop(0),
                    icenter=icenter,
                )
            )
    return tuple(basis)
