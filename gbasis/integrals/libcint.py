r"""
Python C-API bindings for ``libcint`` GTO integrals library.

"""

from ctypes import CDLL, cdll, c_int, c_double, c_void_p

from itertools import chain

from operator import attrgetter

from pathlib import Path

from numpy.ctypeslib import ndpointer

import numpy as np

from scipy.special import gamma


__all__ = [
    "LIBCINT",
    "CBasis",
]


ELEMENTS = (
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
)
r"""
Tuple of all 118 elements.

This tuple has a placeholder element (the null character) at index zero
so that the index of each (real) element matches its atomic number.

"""


class _LibCInt:
    r"""
    ``libcint`` shared library helper class for generating C function bindings.

    """

    _libcint: CDLL = cdll.LoadLibrary((Path(__file__).parent / "lib" / "libcint.so"))
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


# Singleton LibCInt class instance

LIBCINT = _LibCInt()
r"""
LIBCINT C library handle and binding generator.

"""


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
        coord_type = coord_type.lower()
        if coord_type == "spherical":
            num_angmom = attrgetter("num_sph")
            normalized_coeffs = normalized_coeffs_sph
        elif coord_type == "cartesian":
            num_angmom = attrgetter("num_cart")
            normalized_coeffs = normalized_coeffs_cart
        else:
            raise ValueError("`coord_type` parameter must be 'spherical' or 'cartesian'; "
                             f"the provided value, '{coord_type}', is invalid")

        # Process `atnums`
        atnums = [ELEMENTS.index(elem) for elem in atnums]

        # Get counts of atoms/shells/bfns/exps/coeffs
        natm = len(atnums)
        nbas = 0
        nbfn = 0
        nenv = 20 + 4 * natm
        mults = []
        for shell in basis:
            mults.extend([num_angmom(shell)] * shell.num_seg_cont)
            nbas += shell.num_seg_cont
            nbfn += num_angmom(shell) * shell.num_seg_cont
            nenv += shell.exps.size + shell.coeffs.size
        mults = np.asarray(mults, dtype=c_int)

        # Allocate and fill C input arrays
        ienv = 20
        atm = np.zeros((natm, 6), dtype=c_int)
        bas = np.zeros((nbas, 8), dtype=c_int)
        env = np.zeros((nenv,), dtype=c_double)

        # Fill `atm` array
        for atm_row, atnum, atcoord in zip(atm, atnums, atcoords):
            # Nuclear charge of i'th atom
            atm_row[0] = atnum
            # `env` offset to save xyz coordinates
            atm_row[1] = ienv
            # Save xyz coordinates; increment ienv
            env[ienv:ienv + 3] = atcoord
            ienv += 3
            # Nuclear model of i'th atm; unused here
            atm_row[2] = 0
            # `env` offset to save nuclear model zeta parameter; unused here
            atm_row[3] = ienv
            # Save zeta parameter; increment ienv
            env[ienv:ienv + 1] = 0
            ienv += 1
            # Reserved/unused in `libcint`
            atm_row[4:6] = 0

        # Fill `bas` array
        ibas = 0
        for shell in basis:
            # Get angular momentum of shell and # of primitive bfns
            nl = num_angmom(shell)
            nprim = shell.coeffs.shape[0]
            # Save exponents; increment ienv
            iexp = ienv
            ienv += shell.exps.size
            env[iexp:ienv] = shell.exps
            # Save coefficients; increment ienv
            icoef = ienv
            ienv += shell.coeffs.size
            env[icoef:ienv] = normalized_coeffs(shell).reshape(-1, order="F")
            # Unpack generalized contractions
            for iprim in range(icoef, icoef + shell.coeffs.size, nprim):
                # Basis function mult
                mults[ibas] = nl
                # Index of corresponding atom
                bas[ibas, 0] = shell.icenter
                # Angular momentum
                bas[ibas, 1] = shell.angmom
                # Number of primitive GTOs in shell
                bas[ibas, 2] = nprim
                # Number of contracted GTOs in shell
                bas[ibas, 3] = 1
                # Kappa for spinor GTO; unused here
                bas[ibas, 4] = 0
                # `env` offset to save exponents of primitive GTOs
                bas[ibas, 5] = iexp
                # `env` offset to save coefficients of segmented contractions
                bas[ibas, 6] = iprim
                # Reserved/unused in `libcint`
                bas[ibas, 7] = 0
                # Go to next basis function
                ibas += 1

        # Save coord type
        self.coord_type = coord_type

        # Save inputs to `libcint` functions
        self.natm = natm
        self.nbas = nbas
        self.nbfn = nbfn
        self.atm = atm
        self.bas = bas
        self.env = env

        # Save basis function mults
        self.mults = mults
        self.max_mult = max(mults)

        # Save integral functions
        if coord_type == "cartesian":
            self.kin = self.make_int1e(LIBCINT.cint1e_kin_cart)
            self.nuc = self.make_int1e(LIBCINT.cint1e_nuc_cart)
            self.olp = self.make_int1e(LIBCINT.cint1e_ovlp_cart)
            self.eri = self.make_int2e(LIBCINT.cint2e_cart)
        else:
            self.kin = self.make_int1e(LIBCINT.cint1e_kin_sph)
            self.nuc = self.make_int1e(LIBCINT.cint1e_nuc_sph)
            self.olp = self.make_int1e(LIBCINT.cint1e_ovlp_sph)
            self.eri = self.make_int2e(LIBCINT.cint2e_sph)

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
            buf = np.zeros(self.max_mult ** 2, dtype=c_double)
            # Make output array
            out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
            # Evaluate the integral function over all shells
            ipos = 0
            for ishl in range(self.nbas):
                shls[0] = ishl
                p_off = self.mults[ishl]
                jpos = 0
                for jshl in range(ishl + 1):
                    shls[1] = jshl
                    q_off = self.mults[jshl]
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
        def int2e(notation="physicist"):
            if notation == "physicist":
                physicist = True
            elif notation == "chemist":
                physicist = False
            else:
                raise ValueError("`notation` must be one of 'physicist' or 'chemist'")
            # Make temporary arrays
            shls = np.zeros(4, dtype=c_int)
            buf = np.zeros(self.max_mult ** 4, dtype=c_double)
            # Make output array
            out = np.zeros((self.nbfn, self.nbfn, self.nbfn, self.nbfn), dtype=c_double)
            # Evaluate the integral function over all shells
            ipos = 0
            for ishl in range(self.nbas):
                shls[0] = ishl
                p_off = self.mults[ishl]
                jpos = 0
                for jshl in range(ishl + 1):
                    ij = ((ishl + 1) * ishl) // 2 + jshl
                    shls[1] = jshl
                    q_off = self.mults[jshl]
                    kpos = 0
                    for kshl in range(self.nbas):
                        shls[2] = kshl
                        r_off = self.mults[kshl]
                        lpos = 0
                        for lshl in range(kshl + 1):
                            kl = ((kshl + 1) * kshl) // 2 + lshl
                            shls[3] = lshl
                            s_off = self.mults[lshl]
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
            return out.transpose(0, 2, 1, 3) if physicist else out

        # Return instance-bound integral method
        return int2e


def normalized_coeffs_sph(shell):
    r"""
    Normalize the spherical GeneralizedContractionShell coefficients.

    """
    l = shell.angmom
    n = (2 ** (2 * l + 3)) * gamma(l + 2) * ((2 * shell.exps) ** (l + 1.5))
    n /= gamma(2 * l + 3) * (np.pi ** 0.5)
    n **= 0.5
    c = shell.coeffs.copy()
    for ni, ci in zip(n, c):
        ci *= ni
    return c
