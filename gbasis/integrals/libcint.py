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


#
# Helper class for generating LibCInt function bindings


class _LibCInt:
    r"""
    ``libcint`` shared object library helper class.

    """

    _libcint: CDLL = cdll.LoadLibrary(find_library('cint'))
    r"""
    ``libcint`` shared object library.

    """

    def __new__(cls):
        r"""
        Singleton class pattern.

        """
        if not hasattr(cls, "_instance"):
            cls._instance = super(_LibCInt, cls).__new__(cls)
        return cls._instance

    def __getattr__(self, attr):
        r"""
        Helper for returning function pointers from ``libcint`` with proper signatures.

        """
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

        return cfunc


#
# C basis class; translates GBasis repr to C repr and binds integral methods


class CBasis:
    r"""
    ``libcint`` basis class.

    """

    def __init__(self, basis, coord_type="spherical"):
        r"""
        Initialize a ``CBasis`` instance.

        Parameters
        ----------
        basis : List of list of GeneralizedContractionShell
            Shells of generalized contractions by atomic center.
        coord_type : ('spherical'|'cartesian')
            Type of coordinates.

        """
        # Verify coordinate type
        _coord_type = coord_type.lower()
        if _coord_type == "spherical":
            num_angmom = attrgetter("num_sph")
        elif _coord_type == "cartesian":
            num_angmom = attrgetter("num_cart")
        else:
            raise ValueError("`coord_type` parameter must be 'spherical' or 'cartesian'; "
                             f"the provided value, '{coord_type}', is invalid")

        # Organize basis by atomic center
        atm_basis = {center: [] for center in set((shell.icenter for shell in basis))}
        for shell in basis:
            atm_basis[shell.icenter].append(shell)
        basis = list(atm_basis.values())

        # Set up counts of atomic centers/shells/gbfs/exps/coeffs
        natm = len(basis)
        nshl = 0
        nbas = 0
        nexp = 0
        ncof = 0
        for contractions in basis:
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
        # Go to next atomic center's contractions
        for contractions in basis:
            # Nuclear charge of `iatm` atom
            self.atm[iatm, 0] = np.round(contractions[0].charge).astype(int)
            # `env` offset to save xyz coordinates
            self.atm[iatm, 1] = ioff
            # Save xyz coordinates; increment ioff
            self.env[ioff:ioff + 3] = contractions[0].coord
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
                env[ioff:ioff + shells.exps.size] = shells.exps
                ioff += shells.exps.size
                # `env` offset to save column-major contraction coefficients,
                # i.e. a  (no. primitive-)by-(no. contracted) matrix
                bas[ibas, 6] = ioff
                # Save coefficients; increment ioff
                env[ioff:ioff + shells.coeffs.size] = shells.coeffs.T
                ioff += shells.coeffs.size
                # Increment contracted GTO
                ibas += 1
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
        self.offsets = list(map(num_angmom, self.bas[:, 1]))
        self.max_off = max(self.offsets)

        # Make individual integral evaluation methods via `make_intNe` macro:

        # Kinetic energy integral
        self.kin = self.make_int1e(cint1e_kin_cart if _coord_type == "cart" else cint1e_kin_sph)
        # Nuclear-electron attraction integral
        self.nuc = self.make_int1e(cint1e_nuc_cart if _coord_type == "cart" else cint1e_nuc_sph)
        # Overlap integral
        self.olp = self.make_int1e(cint1e_ovlp_cart if _coord_type == "cart" else cint1e_ovlp_sph)
        # Electrone repulsion integral
        self.eri = self.make_int2e(cint2e_cart if _coord_type == "cart" else cint2e_sph)

    def make_int1e(self, func, coord_type="spherical", doc=None):
        r"""
        Make an instance-bound 1-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func : callable
            ``libcint`` function.
        coord_type : ('spherical'|'cartesian')
            Type of coordinates.

        """
        # Verify coordinate type
        _coord_type = coord_type.lower()
        if _coord_type == "spherical":
            num_angmom = attrgetter("num_sph")
        elif _coord_type == "cartesian":
            num_angmom = attrgetter("num_cart")
        else:
            raise ValueError("`coord_type` parameter must be 'spherical' or 'cartesian'; "
                             f"the provided value, '{coord_type}', is invalid")

        # Make instance-bound integral method
        def int1e(self):
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
                for jshl in range(i + 1):
                    shls[1] = jshl
                    q_off = self.offsets[jshl]
                    # Call the C function to fill `buf`
                    func(buf, shls, self.atm, self.natm, self.nbas, self.bas, self.env, None)
                    # Fill `out` array
                    for p in range(p_off):
                        i_off = p + ipos
                        for q in range(q_off):
                            j_off = q + jpos
                            val = self.buf[p * q_off + q]
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

    def make_int2e(self, func, coord_type="spherical"):
        r"""
        Make an instance-bound 2-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func : callable
            ``libcint`` function.

        coord_type : ('spherical'|'cartesian')
            Type of coordinates.

        """
        # Verify coordinate type
        _coord_type = coord_type.lower()
        if _coord_type == "spherical":
            num_angmom = attrgetter("num_sph")
        elif _coord_type == "cartesian":
            num_angmom = attrgetter("num_cart")
        else:
            raise ValueError("`coord_type` parameter must be 'spherical' or 'cartesian'; "
                             f"the provided value, '{coord_type}', is invalid")

        # Make instance-bound integral method
        def int2e(self):
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
                for jshl in range(i + 1):
                    ij = ((ishl + 1) * ishl) // 2 + jshl
                    shls[1] = jshl
                    q_off = self.offsets[jshl]
                    kpos = 0
                    for kshl in range(self.nshl):
                        shls[2] = kshl
                        r_off = self.offsets[kshl]
                        lpos = 0
                        for lshl in range(k + 1):
                            kl = ((kshl + 1) * kshl) // 2 + lshl
                            shls[3] = lshl
                            s_off = self.offsets[lshl]
                            if ij < kl:
                                lpos += s_off
                                continue
                            # Call the C function to fill `buf`
                            func(buf, shls, self.atm, self.natm, self.nbas, self.bas, self.env, None)
                            # Fill `out` array
                            for p in range(p_off):
                                i_off = p + ipos
                                for q in range(q_off):
                                    j_off = q + jpos
                                    for r in range(r_off):
                                        k_off = r + kpos
                                        for s in range(s_off):
                                            l_off = s + lpos
                                            val = self.buf[p * (q_off * r_off * s_off) +
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


#
# LibCInt bindings


# Singleton helper class instance

LIBCINT = _LibCInt()
r"""
LIBCINT C library handle and binding generator.

"""


# Define utility functions

CINTlen_cart = LIBCINT.CINTlen_cart
CINTlen_spinor = LIBCINT.CINTlen_spinor

CINTcgtos_cart = LIBCINT.CINTcgtos_cart
CINTcgtos_spheric = LIBCINT.CINTcgtos_spheric
CINTcgtos_spinor = LIBCINT.CINTcgtos_spinor
CINTcgto_cart = LIBCINT.CINTcgto_cart
CINTcgto_spheric = LIBCINT.CINTcgto_spheric
CINTcgto_spinor = LIBCINT.CINTcgto_spinor

CINTtot_pgto_spheric = LIBCINT.CINTtot_pgto_spheric
CINTtot_pgto_spinor = LIBCINT.CINTtot_pgto_spinor

CINTtot_cgto_cart = LIBCINT.CINTtot_cgto_cart
CINTtot_cgto_spheric = LIBCINT.CINTtot_cgto_spheric
CINTtot_cgto_spinor = LIBCINT.CINTtot_cgto_spinor

CINTshells_cart_offset = LIBCINT.CINTshells_cart_offset
CINTshells_spheric_offset = LIBCINT.CINTshells_spheric_offset
CINTshells_spinor_offset = LIBCINT.CINTshells_spinor_offset


# Define integral functions

cint1e_kin_cart = LIBCINT.cint1e_kin_cart
cint1e_kin_sph = LIBCINT.cint1e_kin_sph

cint1e_nuc_cart = LIBCINT.cint1e_nuc_cart
cint1e_nuc_sph = LIBCINT.cint1e_nuc_sph

cint1e_ovlp_cart = LIBCINT.cint1e_ovlp_cart
cint1e_ovlp_sph = LIBCINT.cint1e_ovlp_sph

cint2e_cart = LIBCINT.cint2e_cart
cint2e_sph = LIBCINT.cint2e_sph
