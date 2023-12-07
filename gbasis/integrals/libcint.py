r"""
Python C-API bindings for ``libcint`` GTO integrals library.

"""

from ctypes import CDLL, cdll, c_int, c_double, c_void_p

from itertools import chain

from operator import attrgetter

from pathlib import Path

import re

import numpy as np

from scipy.special import factorial

from gbasis.utils import factorial2


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


INTEGRAL_REGEX = re.compile(r'^(?!.*optimizer$)int[12]e.+')
r"""
Regex for matching ``libcint`` integral functions.

"""


def ndptr(enable_null=False, **kwargs):
    r"""
    Wrapped ``numpy.ctypeslib.ndpointer`` that accepts null pointers.

    Null pointers are passed via ``None`` in Python.

    """
    def from_param(cls, obj):

        return obj if obj is None else base.from_param(obj)

    base = np.ctypeslib.ndpointer(**kwargs)

    if not enable_null:
        return base

    return type(base.__name__, (base,), {'from_param': classmethod(from_param)})


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
            # Check that the attr matches the regex
            if INTEGRAL_REGEX.match(attr):
                # Make the bound C function
                cfunc = getattr(self._libcint, attr)
                cfunc.argtypes = (
                    # out
                    ndptr(enable_null=True, dtype=c_double, ndim=1, flags=('C_CONTIGUOUS', 'WRITEABLE')),
                    # dims
                    ndptr(enable_null=True, dtype=c_int, ndim=1, flags=('C_CONTIGUOUS')),
                    # shls
                    ndptr(dtype=c_int, ndim=1, flags=('C_CONTIGUOUS',)),
                    # atm
                    ndptr(dtype=c_int, ndim=2, flags=('C_CONTIGUOUS',)),
                    # natm
                    c_int,
                    # bas
                    ndptr(dtype=c_int, ndim=2, flags=('C_CONTIGUOUS',)),
                    # nbas
                    c_int,
                    # env
                    ndptr(dtype=c_double, ndim=1, flags=('C_CONTIGUOUS',)),
                    # opt
                    ndptr(enable_null=True, dtype=c_void_p, ndim=1, flags=('C_CONTIGUOUS',)),
                    # cache
                    ndptr(enable_null=True, dtype=c_double, ndim=1, flags=('C_CONTIGUOUS',)),
                )
                cfunc.restype = c_int

            else:
                raise ValueError(f'there is no ``gbasis`` API for the function {attr}')

            # Cache the C function
            self._cache[attr] = cfunc

        # Return the C function
        return cfunc

    def __getitem__(self, item):
        r"""
        Helper for returning function pointers from ``libcint`` with proper signatures.

        This is the same as ``__getattr__`` and exists only for convenience.

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
            raise ValueError("``coord_type`` parameter must be 'spherical' or 'cartesian'; "
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
            # Integrals
            self.olp = self.make_int1e(LIBCINT.int1e_ovlp_cart, comp=(1,))
            self.kin = self.make_int1e(LIBCINT.int1e_kin_cart, comp=(1,))
            self.nuc = self.make_int1e(LIBCINT.int1e_nuc_cart, comp=(1,))
            self.eri = self.make_int2e(LIBCINT.int2e_cart, comp=(1,))
            # Gradients
            self.d_olp = self.make_int1e(LIBCINT.int1e_ipovlp_cart, comp=(self.natm, 3))
            self.d_nuc = self.make_int1e(LIBCINT.int1e_ipnuc_cart, comp=(self.natm, 3))
            self.d_kin = self.make_int1e(LIBCINT.int1e_ipkin_cart, comp=(self.natm, 3))
            # Hessians
            self.d2_olp = self.make_int1e(LIBCINT.int1e_ipipovlp_cart, comp=(self.natm, 3, self.natm, 3))
            self.d2_nuc = self.make_int1e(LIBCINT.int1e_ipipnuc_cart, comp=(self.natm, 3, self.natm, 3))
            self.d2_kin = self.make_int1e(LIBCINT.int1e_ipipkin_cart, comp=(self.natm, 3, self.natm, 3))
        else:
            # Integrals
            self.olp = self.make_int1e(LIBCINT.int1e_ovlp_sph, comp=(1,))
            self.kin = self.make_int1e(LIBCINT.int1e_kin_sph, comp=(1,))
            self.nuc = self.make_int1e(LIBCINT.int1e_nuc_sph, comp=(1,))
            self.eri = self.make_int2e(LIBCINT.int2e_sph, comp=(1,))
            # Gradients
            self.d_olp = self.make_int1e(LIBCINT.int1e_ipovlp_sph, comp=(self.natm, 3))
            self.d_nuc = self.make_int1e(LIBCINT.int1e_ipnuc_sph, comp=(self.natm, 3))
            self.d_kin = self.make_int1e(LIBCINT.int1e_ipkin_sph, comp=(self.natm, 3))
            # Hessians
            self.d2_olp = self.make_int1e(LIBCINT.int1e_ipipovlp_sph, comp=(self.natm, 3, self.natm, 3))
            self.d2_nuc = self.make_int1e(LIBCINT.int1e_ipipnuc_sph, comp=(self.natm, 3, self.natm, 3))
            self.d2_kin = self.make_int1e(LIBCINT.int1e_ipipkin_sph, comp=(self.natm, 3, self.natm, 3))

    def make_int1e(self, func, comp=(1,)):
        r"""
        Make an instance-bound 1-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func : callable
            ``libcint`` function.
        comp : tuple, default=(1,)
            Shape of components in each integral element.
            E.g., for normal integrals, ``comp=(1,)``, while for nuclear gradients,
            ``comp=(Natm, 3)``, and for nuclear Hessians, ``comp=(Natm, Natm, 3, 3)``, etc.

        """
        # Handle multi-component integral values
        prod_comp = np.prod(comp)
        comp_is_1 = prod_comp == 1
        if comp_is_1:
            comp = (1,)
        out_shape = (self.nbfn, self.nbfn, prod_comp)
        buf_shape = prod_comp * self.max_mult ** 2

        # Make instance-bound integral method
        def int1e():
            # Make output array
            out = np.zeros(out_shape, dtype=c_double)
            # Make temporary arrays
            buf = np.zeros(buf_shape, dtype=c_double)
            shls = np.zeros(2, dtype=c_int)
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
                    func(buf, None, shls, self.atm, self.natm, self.bas, self.nbas, self.env, None, None)
                    # Fill `out` array
                    for p in range(p_off):
                        i_off = p + ipos
                        for q in range(q_off):
                            j_off = q + jpos
                            buf_off = prod_comp * (q * p_off + p)
                            val = buf[buf_off:buf_off + prod_comp]
                            out[i_off, j_off] = val
                            out[j_off, i_off] = val
                    # Reset `buf`
                    buf[:] = 0
                    # Iterate `jpos`
                    jpos += q_off
                # Iterate `ipos`
                ipos += p_off
            # Return integrals in `out` array
            return out.squeeze(axis=2) if comp_is_1 else out.reshape(self.nbfn, self.nbfn, *comp)

        # Return instance-bound integral method
        return int1e

    def make_int2e(self, func, comp=(1,)):
        r"""
        Make an instance-bound 2-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func : callable
            ``libcint`` function.
        comp : tuple, default=(1,)
            Shape of components in each integral element.
            E.g., for normal integrals, ``comp=(1,)``, while for nuclear gradients,
            ``comp=(Natm, 3)``, and for nuclear Hessians, ``comp=(Natm, Natm, 3, 3)``, etc.

        """
        # Handle multi-component integral values
        prod_comp = np.prod(comp)
        comp_is_1 = prod_comp == 1
        if comp_is_1:
            comp = (1,)
        out_shape = (self.nbfn, self.nbfn, self.nbfn, self.nbfn, prod_comp)
        buf_shape = prod_comp * self.max_mult ** 4

        # Make instance-bound integral method
        def int2e(notation="physicist"):
            # Handle ``notation`` argument
            if notation == "physicist":
                physicist = True
            elif notation == "chemist":
                physicist = False
            else:
                raise ValueError("``notation`` must be one of 'physicist' or 'chemist'")
            # Make output array
            out = np.zeros(out_shape, dtype=c_double)
            # Make temporary arrays
            buf = np.zeros(buf_shape, dtype=c_double)
            shls = np.zeros(4, dtype=c_int)
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
                            func(buf, None, shls, self.atm, self.natm, self.bas, self.nbas, self.env, None, None)
                            # Fill `out` array
                            for p in range(p_off):
                                i_off = p + ipos
                                for q in range(q_off):
                                    j_off = q + jpos
                                    for r in range(r_off):
                                        k_off = r + kpos
                                        for s in range(s_off):
                                            l_off = s + lpos
                                            buf_off = prod_comp * (s * (r_off * q_off * p_off) +
                                                                   r * (q_off * p_off) +
                                                                   q * (p_off) +
                                                                   p)
                                            val = buf[buf_off:buf_off + prod_comp]
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
            if physicist:
                out = out.transpose(0, 2, 1, 3, 4)
            if comp_is_1:
                return out.squeeze(axis=4)
            else:
                return out.reshape(self.nbfn, self.nbfn, self.nbfn, self.nbfn, *comp)

        # Return instance-bound integral method
        return int2e

    def pntchrg(self, point_coords, point_charges):
        r"""Point charge integral."""
        # Make output array
        ncharge = len(point_charges)
        out = np.zeros((self.nbfn, self.nbfn, ncharge), dtype=c_double)
        # Make temporary arrays
        buf = np.zeros(self.max_mult ** 2, dtype=c_double)
        shls = np.zeros(2, dtype=c_int)
        # Evaluate the integral function over all shells
        func = LIBCINT["int1e_rinv_cart" if self.coord_type == "cartesian" else "int1e_rinv_sph"]
        for icharge, (coord, charge) in enumerate(zip(point_coords, point_charges)):
            # Set R_O of 1/|r - R_O|
            self.env[4:7] = coord
            ipos = 0
            for ishl in range(self.nbas):
                shls[0] = ishl
                p_off = self.mults[ishl]
                jpos = 0
                for jshl in range(ishl + 1):
                    shls[1] = jshl
                    q_off = self.mults[jshl]
                    # Call the C function to fill `buf`
                    func(buf, None, shls, self.atm, self.natm, self.bas, self.nbas, self.env, None, None)
                    # Fill `out` array
                    for p in range(p_off):
                        i_off = p + ipos
                        for q in range(q_off):
                            j_off = q + jpos
                            val = buf[q * p_off + p] * -charge
                            out[i_off, j_off, icharge] = val
                            if i_off != j_off:
                                out[j_off, i_off, icharge] = val
                    # Reset `buf`
                    buf[:] = 0
                    # Iterate `jpos`
                    jpos += q_off
                # Iterate `ipos`
                ipos += p_off
        # Return integrals in `out` array
        return out


INV_SQRT_PI = 0.56418958354775628694807945156077


def normalized_coeffs_sph(shell):
    r"""
    Normalize the spherical GeneralizedContractionShell coefficients.

    """
    l = shell.angmom
    c = shell.coeffs.copy()
    n = (INV_SQRT_PI * (2 ** (3 * l + 4.5))) * (factorial(l + 1) / factorial(2 * l + 2))
    n *= np.power(shell.exps, l + 1.5)
    n **= 0.5
    for ni, ci in zip(n, c):
        ci *= ni
    return c


PI_TO_THREE_HALVES = 5.5683279968317078452848179821188


def normalized_coeffs_cart(shell):
    r"""
    Normalize the cartesian GeneralizedContractionShell coefficients.

    """
    l = shell.angmom
    c = shell.coeffs.copy()
    d = PI_TO_THREE_HALVES * factorial2(2 * l - 1)
    n = np.power(2 * shell.exps, l + 1.5) / d
    n **= 0.5
    for ni, ci in zip(n, c):
        ci *= ni
    n = d * 2 ** -l * sum(
        ci * cj * (ei + ej) ** (-l - 1.5)
        for ei, ci in zip(shell.exps, c)
        for ej, cj in zip(shell.exps, c)
    )
    n **= -0.5
    for ni, ci in zip(n, c):
        ci *= ni
    return c
