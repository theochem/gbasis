r"""
Python C-API bindings for ``libcint`` GTO integrals library.

"""

from ctypes import CDLL, cdll, c_int, c_double, c_void_p

from ctypes.util import find_library

from operator import attrgetter

from numpy.ctypeslib import load_library, ndpointer

import numpy as np


__all__ = [
    "LIBCINT",
    "CBasis",
]


ELEMENTS = [
    "\0", "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
r"""
List of the elements.

This list has a placeholder element (the null character) at index zero
so that the index of each (real) element matches its atomic number.

"""


class _LibCInt:
    r"""
    ``libcint`` shared library helper class for generating C function bindings.

    """

    _libcint: CDLL = cdll.LoadLibrary(find_library('cint'))
    r"""
    ``libcint`` shared object library.

    """

    def __new__(cls):
        r"""
        Singleton class constructor.

        Returns
        -------
        instance : _LibCInt
            Singleton instance.

        """
        if not hasattr(cls, "_instance"):
            cls._instance = super(_LibCInt, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        r"""
        Singleton class inializer.

        """
        self._cache = dict()

    def __getattr__(self, attr):
        r"""
        Helper for returning function pointers from ``libcint`` with proper signatures.

        Matches the function to its signature based on the pattern of its name;
        possible because ``libcint`` function names and signatures are systematic.

        Parameters
        ----------
        attr : str
            Name of C function.

        Returns
        -------
        f : callable
            C function.

        """
        try:

            # Retrieve previously-cached function
            cfunc = self._cache[attr]

        except KeyError:

            # Make the bound C function
            cfunc = getattr(self._libcint, attr)

            if attr == 'CINTlen_cart':
                cfunc.argtypes = [c_int]
                cfunc.restype = c_int

            elif attr == 'CINTlen_spinor':
                cfunc.argtypes = [
                    c_int,
                    ndpointer(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS',)),
                    ]
                cfunc.restype = c_int

            elif attr == 'CINTgto_norm':
                cfunc.argtypes = [c_int, c_double]
                cfunc.restype = c_double

            elif attr.startswith('CINTcgto') or attr.startswith('CINTtot'):
                cfunc.argtypes = [
                    ndpointer(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS',)),
                    c_int,
                    ]
                cfunc.restype = c_int

            elif attr.startswith('CINTshells'):
                cfunc.argtypes = [
                    ndpointer(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS', 'WRITEABLE')),
                    ndpointer(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS',)),
                    c_int,
                    ]

            elif attr.startswith('cint1e') and not attr.endswith('optimizer'):
                cfunc.argtypes = [
                    # buf
                    ndpointer(dtype=c_double, ndim=1, flags=('C_CONTIGUOUS', 'WRITEABLE')),
                    # shls
                    ndpointer(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS',)),
                    # atm
                    ndpointer(dtype=c_int, ndim=2, flags=('C_CONTIGUOUS',)),
                    # natm
                    c_int,
                    # bas
                    ndpointer(dtype=c_int, ndim=2, flags=('C_CONTIGUOUS',)),
                    # nbas
                    c_int,
                    # env
                    ndpointer(dtype=c_double, ndim=1, flags=('C_CONTIGUOUS',)),
                    # opt (not used; put ``None`` as this argument)
                    c_void_p,
                    ]
                cfunc.restype = c_int

            elif attr.startswith('cint2e') and not attr.endswith('optimizer'):
                cfunc.argtypes = [
                    # buf
                    ndpointer(dtype=c_double, ndim=1, flags=('C_CONTIGUOUS', 'WRITEABLE')),
                    # shls
                    ndpointer(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS',)),
                    # atm
                    ndpointer(dtype=c_int, ndim=2, flags=('C_CONTIGUOUS',)),
                    # natm
                    c_int,
                    # bas
                    ndpointer(dtype=c_int, ndim=2, flags=('C_CONTIGUOUS',)),
                    # nbas
                    c_int,
                    # env
                    ndpointer(dtype=c_double, ndim=1, flags=('C_CONTIGUOUS',)),
                    # opt (not used; put ``None`` as this argument)
                    c_void_p,
                    ]
                cfunc.restype = c_int

            else:
                raise NotImplementedError('there is no ``gbasis`` API for this function')

            # Cache the C function
            self._cache[attr] = cfunc

        # Return the C function
        return cfunc

    def __getitem__(self, item):
        r"""
        Helper for returning function pointers from ``libcint`` with proper signatures.

        This is the same as `__getattr__` and exists only for convenience.

        Parameters
        ----------
        item : str
            Name of C function.

        Returns
        -------
        f : callable
            C function.

        """
        return self.__getattr__(item)


class CBasis:
    r"""
    ``libcint`` basis class.

    """
    def __init__(self, basis, atnums, atcoords, coord_type="spherical"):
        r"""
        Initialize a ``CBasis`` instance.

        Parameters
        ----------
        basis : List of GeneralizedContractionShell
            Shells of generalized contractions.
        atnums : List of str
            Element corresponding to each atomic center.
        atcoords : List of length-3 array-like of floats
            X, Y, and Z coordinates for each atomic center.
        coord_type : ('spherical'|'cartesian')
            Type of coordinates.

        """
        # Set coord type
        _coord_type = coord_type.lower()
        if _coord_type == "spherical":
            num_angmom = attrgetter("num_sph")
        elif _coord_type == "cartesian":
            num_angmom = lambda x: x.norm_cont.shape[1]#attrgetter("num_cart")
        else:
            raise ValueError("`coord_type` parameter must be 'spherical' or 'cartesian'; "
                             f"the provided value, '{coord_type}', is invalid")

        # Process `atnums`
        atnums = [ELEMENTS.index(elem) for elem in atnums]

        # Organize basis by atomic center
        atm_basis = {center: [] for center in set((shell.icenter for shell in basis))}
        for shell in basis:
            atm_basis[shell.icenter].append(shell)
        basis = list(atm_basis.items())

        # Organize basis by atomic center
        atm_basis = {center: [] for center in set((shell.icenter for shell in basis))}
        for shell in basis:
            atm_basis[shell.icenter].append(shell)
        basis = list(atm_basis.items())

        # Set up counts of atomic centers/shells/gbfs/exps/coeffs
        natm = len(basis)
        nshl = 0
        nbas = 0
        nexp = 0
        ncof = 0
        for _, contractions in basis:
            nshl += len(contractions)
            for shell in contractions:
                nbas += num_angmom(shell)
                nexp += shell.exps.size
                ncof += shell.coeffs.size

        # Allocate and fill C input arrays
        iatm = 0
        ibas = 0
        ioff = 20
        atm = np.zeros((natm, 6), dtype=c_int)
        bas = np.zeros((nbas, 8), dtype=c_int)
        env = np.zeros(20 + natm * 3 + nexp + ncof, dtype=c_double)
        offsets = []
        # Go to next atomic center's contractions
        for atnum, atcoord, (icenter, contractions) in zip(atnums, atcoords, basis):
            # Nuclear charge of `iatm` atom
            atm[iatm, 0] = atnum
            # `env` offset to save xyz coordinates
            atm[iatm, 1] = ioff
            # Save xyz coordinates; increment ioff
            env[ioff:ioff + 3] = atcoord
            ioff += 3
            # Go to next contracted GTO
            for shell in contractions:
                # Index of corresponding atom
                bas[ibas, 0] = iatm
                # Angular momentum
                bas[ibas, 1] = shell.angmom
                # Number of [primitive|contracted] GTOs in `ibas` basis function
                bas[ibas, 2:4] = shell.coeffs.shape
                # Kappa for spinor GTO; unused here
                bas[ibas, 4] = 0
                # `env` offset to save exponentss of primitive GTOs
                bas[ibas, 5] = ioff
                # Save exponents; increment ioff
                env[ioff:ioff + shell.exps.size] = shell.exps
                ioff += shell.exps.size
                # `env` offset to save column-major contraction coefficients,
                # i.e. a  (no. primitive-)by-(no. contracted) matrix
                bas[ibas, 6] = ioff
                # Save (normalized) coefficients; increment ioff
                env_tmp = env[ioff:ioff + shell.coeffs.size].reshape(*shell.coeffs.shape)
                for iexp, icoeffs, ienv in zip(shell.exps, shell.coeffs, env_tmp):
                    ienv[:] = icoeffs * LIBCINT.CINTgto_norm(shell.angmom, iexp)
                ioff += shell.coeffs.size
                # Increment contracted GTO
                ibas += 1
                # Save basis function offsets
                offsets.append(num_angmom(shell))
            # Increment atomic center
            iatm += 1

        # Save inputs to `libcint` functions
        self.natm = natm
        self.nshl = nshl
        self.nbas = nbas
        self.atm = atm
        self.bas = bas
        self.env = env

        # Save basis function offsets
        self.offsets = offsets
        self.max_off = max(offsets)


        # Make individual integral evaluation methods via `make_intNe` macros:

        # Kinetic energy integral
        self.kin = self.make_int1e(LIBCINT["cint1e_kin_cart" if _coord_type == "cartesian" else "cint1e_kin_sph"])
        # Nuclear-electron attraction integral
        self.nuc = self.make_int1e(LIBCINT["cint1e_nuc_cart" if _coord_type == "cartesian" else "cint1e_nuc_sph"])
        # Overlap integral
        self.olp = self.make_int1e(LIBCINT["cint1e_ovlp_cart" if _coord_type == "cartesian" else "cint1e_ovlp_sph"])
        # Electron repulsion integral
        self.eri = self.make_int2e(LIBCINT["cint2e_cart" if _coord_type == "cartesian" else "cint2e_sph"])

    def make_int1e(self, func):
        r"""
        Make an instance-bound 1-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func : callable
            ``libcint`` function.

        """
        # Make instance-bound integral method
        def int1e():
            # Make temporary arrays
            shls = np.zeros(2, dtype=c_int)
            buf = np.zeros(self.max_off ** 2, dtype=c_double)
            # Make output array
            out = np.zeros((self.nbas, self.nbas), dtype=c_double)
            # Evaluate the integral function over all shells
            ipos = 0
            for ishl in range(self.nshl):
                shls[0] = ishl
                p_off = self.offsets[ishl]
                jpos = 0
                for jshl in range(ishl + 1):
                    shls[1] = jshl
                    q_off = self.offsets[jshl]
                    # Call the C function to fill `buf`
                    func(buf, shls, self.atm, self.natm, self.bas, self.nbas, self.env, None)
                    # Fill `out` array
                    for p in range(p_off):
                        i_off = p + ipos
                        for q in range(q_off):
                            j_off = q + jpos
                            val = buf[p * q_off + q]
                            out[i_off, j_off] = val
                            out[j_off, i_off] = val
                    # Reset `buf`
                    buf[:] = 0
                    # Iterate `jpos`
                    jpos += q_off
                # Iterate `ipos`
                ipos += p_off
            # Return integrals in `out` array
            return out

        # Return instance-bound integral method
        return int1e

    def make_int2e(self, func):
        r"""
        Make an instance-bound 2-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func : callable
            ``libcint`` function.

        """
        # Make instance-bound integral method
        def int2e():
            # Make temporary arrays
            shls = np.zeros(4, dtype=c_int)
            buf = np.zeros(self.max_offset ** 4, dtype=c_double)
            # Make output array
            out = np.zeros((self.nbas, self.nbas, self.nbas, self.nbas), dtype=c_double)
            # Evaluate the integral function over all shells
            ipos = 0
            for ishl in range(self.nshl):
                shls[0] = ishl
                p_off = self.offsets[ishl]
                jpos = 0
                for jshl in range(ishl + 1):
                    ij = ((ishl + 1) * ishl) // 2 + jshl
                    shls[1] = jshl
                    q_off = self.offsets[jshl]
                    kpos = 0
                    for kshl in range(self.nshl):
                        shls[2] = kshl
                        r_off = self.offsets[kshl]
                        lpos = 0
                        for lshl in range(kshl + 1):
                            kl = ((kshl + 1) * kshl) // 2 + lshl
                            shls[3] = lshl
                            s_off = self.offsets[lshl]
                            if ij < kl:
                                lpos += s_off
                                continue
                            # Call the C function to fill `buf`
                            func(buf, shls, self.atm, self.natm, self.bas, self.nbas, self.env, None)
                            # Fill `out` array
                            for p in range(p_off):
                                i_off = p + ipos
                                for q in range(q_off):
                                    j_off = q + jpos
                                    for r in range(r_off):
                                        k_off = r + kpos
                                        for s in range(s_off):
                                            l_off = s + lpos
                                            val = buf[p * (q_off * r_off * s_off) +
                                                      q * (r_off * s_off) +
                                                      r * (s_off) +
                                                      s]
                                            out[i_off, j_off, k_off, l_off] = val
                                            out[i_off, j_off, l_off, k_off] = val
                                            out[j_off, i_off, k_off, l_off] = val
                                            out[j_off, i_off, l_off, k_off] = val
                                            out[k_off, l_off, i_off, j_off] = val
                                            out[k_off, l_off, j_off, i_off] = val
                                            out[l_off, k_off, i_off, j_off] = val
                                            out[l_off, k_off, j_off, i_off] = val
                            # Reset `buf`
                            buf[:] = 0
                            # Iterate `lpos`
                            lpos += s_off
                        # Iterate `kpos`
                        kpos += r_off
                    # Iterate `jpos`
                    jpos += q_off
                # Iterate `ipos`
                ipos += p_off
            # Return integrals in `out` array
            return out

        # Return instance-bound integral method
        return int2e


# Singleton LibCInt class instance

LIBCINT = _LibCInt()
r"""
LIBCINT C library handle and binding generator.

"""


# C fuctions used in CBasis class

cint1e_kin_cart = LIBCINT.cint1e_kin_cart
cint1e_nuc_cart = LIBCINT.cint1e_nuc_cart
cint1e_ovlp_cart = LIBCINT.cint1e_ovlp_cart
cint2e_cart = LIBCINT.cint2e_cart

cint1e_kin_sph = LIBCINT.cint1e_kin_sph
cint1e_nuc_sph = LIBCINT.cint1e_nuc_sph
cint1e_ovlp_sph = LIBCINT.cint1e_ovlp_sph
cint2e_sph = LIBCINT.cint2e_sph
