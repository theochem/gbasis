"""Parsers for reading basis set files."""
import re

import numpy as np


def parse_nwchem(nwchem_basis):
    """Parse nwchem basis set file.

    Parameters
    ----------
    nwchem_basis : str
        Contents of the nwchem basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater tha "k", an error will be raised.

    """
    # pylint: disable=R0914
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
