r"""
Python C-API bindings for ``libcint`` GTO integrals library.

"""

from contextlib import contextmanager

from ctypes import CDLL, POINTER, Structure, cdll, byref, c_int, c_double, c_void_p

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
    "\0",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
r"""
Tuple of all 118 elements.

This tuple has a placeholder element (the null character) at index zero
so that the index of each (real) element matches its atomic number.

"""


INTEGRAL_REGEX = re.compile(r"^(?!.*optimizer$)int[12]e.+")
r"""
Regex for matching ``libcint`` integral functions.

"""


OPTIMIZER_REGEX = re.compile(r"^(?=.*optimizer$)int[12]e.+")
r"""
Regex for matching ``libcint`` optimizer functions.

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

    return type(base.__name__, (base,), {"from_param": classmethod(from_param)})


class PairData(Structure):
    r"""``libcint`` ``PairData`` class."""
    _fields_ = [
        ("rij", c_double * 3),
        ("eij", c_double),
        ("cceij", c_double),
    ]


class CINTOpt(Structure):
    r"""``libcint`` ``CINTOpt`` class."""
    _fields_ = [
        ("index_xyz_array", POINTER(POINTER(c_int))),
        ("non0ctr", POINTER(POINTER(c_int))),
        ("sortedidx", POINTER(POINTER(c_int))),
        ("nbas", c_int),
        ("log_max_coeff", POINTER(POINTER(c_double))),
        ("pairdata", POINTER(POINTER(PairData))),
    ]


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
        Singleton class initializer.

        """
        # Make the bound C functions we'll always need here
        cfunc = self._libcint.CINTdel_optimizer
        cfunc.argtypes = (POINTER(POINTER(CINTOpt)),)

        # Set up the cache
        self._cache = {
            "CINTdel_optimizer": cfunc,
        }

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
                    ndptr(dtype=c_double, ndim=1, flags=("C_CONTIGUOUS", "WRITEABLE")),
                    # dims
                    ndptr(enable_null=True, dtype=c_int, ndim=1, flags=("C_CONTIGUOUS")),
                    # shls
                    ndptr(dtype=c_int, ndim=1, flags=("C_CONTIGUOUS",)),
                    # atm
                    ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                    # natm
                    c_int,
                    # bas
                    ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                    # nbas
                    c_int,
                    # env
                    ndptr(dtype=c_double, ndim=1, flags=("C_CONTIGUOUS",)),
                    # opt
                    POINTER(CINTOpt),
                    # cache
                    ndptr(enable_null=True, dtype=c_double, ndim=1, flags=("C_CONTIGUOUS",)),
                )
                cfunc.restype = c_int

            elif OPTIMIZER_REGEX.match(attr):
                # Make the bound C function
                cfunc = getattr(self._libcint, attr)
                cfunc.argtypes = (
                    # opt
                    POINTER(POINTER(CINTOpt)),
                    # atm
                    ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                    # natm
                    c_int,
                    # bas
                    ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                    # nbas
                    c_int,
                    # env
                    ndptr(dtype=c_double, ndim=1, flags=("C_CONTIGUOUS",)),
                )

            else:
                raise ValueError(f"there is no ``gbasis`` API for the function {attr}")

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

    Attributes
    ----------
    coord_type : ("spherical" | "cartesian")
        Coordinate type of ``libcint`` basis.
    natm : int
        Number of atoms.
    nbas : int
        Number of shells.
    nbfn : int
        Number of basis functions.
    atm : np.ndarray(Natm, 6, dtype=float)
        Buffer of atom information for ``libcint``.
    bas : np.ndarray(Nbas, 8, dtype=float)
        Buffer of basis shell information for ``libcint``.
    env : np.ndarray(Nenv, dtype=float)
        Buffer of numerical atom/basis shell data for ``libcint``.
    atnums : np.ndarray(Natm, dtype=int)
        Array of atomic numbers.
    atcoords : np.ndarray(Natm, 3, dtype=float)
        Array of atomic coordinates.

    Methods
    -------
    make_int1e(self, func_name, components=tuple(), constant=None, is_complex=False, origin=False, inv_origin=False)
        Make an instance-bound 1-electron integral method from a ``libcint`` function.
    make_int2e(self, func_name, components=tuple(), constant=None, is_complex=False, origin=False, inv_origin=False)
        Make an instance-bound 2-electron integral method from a ``libcint`` function.
    overlap(self)
        Compute the overlap integrals.
    kinetic_energy(self)
        Compute the kinetic energy integrals.
    nuclear_attraction(self)
        Compute the nuclear attraction integrals.
    electron_repulsion(self)
        Compute the electron repulsion integrals.
    r_inv(self, origin=None)
        Compute the :math:`1/\left|\mathbf{r} - \mathbf{R}_\text{inv}\right|` integrals.
    momentum(self, origin=None)
        Compute the momentum integrals.
    angular_momentum(self, origin=None)
        Compute the angular momentum integrals.
    point_charge(self, point_coords, point_charges)
        Compute the point charge integrals.
    moment(self, orders, origin=None)
        Compute the moment integrals.

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
        elif coord_type == "cartesian":
            num_angmom = attrgetter("num_cart")
        else:
            raise ValueError(
                "``coord_type`` parameter must be 'spherical' or 'cartesian'; "
                f"the provided value, '{coord_type}', is invalid"
            )

        # Process `atnums`
        atnums = [ELEMENTS.index(elem) for elem in atnums]

        # Get counts of atoms/shells/bfns/exps/coeffs
        natm = len(atnums)
        nbas = 0
        nbfn = 0
        nenv = 20 + 4 * natm
        offs = []
        atm_offs = np.zeros(natm + 1, dtype=int)
        for shell in basis:
            offs.extend([num_angmom(shell)] * shell.num_seg_cont)
            atm_offs[shell.icenter + 1] += num_angmom(shell) * shell.num_seg_cont
            nbas += shell.num_seg_cont
            nbfn += num_angmom(shell) * shell.num_seg_cont
            nenv += shell.exps.size + shell.coeffs.size
        offs = np.asarray(offs, dtype=c_int)
        atm_offs = np.cumsum(atm_offs)

        # Get permutation vector for ordering convention
        permutations = []
        for shell in basis:
            if hasattr(shell, "permutation_libcint"):
                permutation = shell.permutation_libcint()
            else:
                permutation = list(range(num_angmom(shell)))
            for _ in range(shell.num_seg_cont):
                perm_off = len(permutations)
                permutations.extend(p + perm_off for p in permutation)

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
            env[ienv : ienv + 3] = atcoord
            ienv += 3
            # Nuclear model of i'th atm; unused here
            atm_row[2] = 0
            # `env` offset to save nuclear model zeta parameter; unused here
            atm_row[3] = ienv
            # Save zeta parameter; increment ienv
            env[ienv : ienv + 1] = 0
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
                # Basis function offset
                offs[ibas] = nl
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

        # Save atom coordinates and atom shell offsets
        self.atnums = atnums.copy()
        self.atcoords = atcoords.copy()
        self._atm_offs = atm_offs

        # Save basis function offsets and ordering permutation
        self._offs = offs
        self._max_off = max(offs)
        self._permutations = permutations

        # Set inverse sqrt of overlap integral (temporarily, for __init__)
        self._ovlp_minhalf = np.ones(nbfn)

        # Integrals
        self._ovlp = self.make_int1e("int1e_ovlp")
        self._kin = self.make_int1e("int1e_kin")
        self._nuc = self.make_int1e("int1e_nuc")
        self._eri = self.make_int2e("int2e")
        self._rinv = self.make_int1e("int1e_rinv", inv_origin=True)
        self._mom = self.make_int1e(
            "int1e_p", components=(3,), constant=-1j, is_complex=True, origin=True
        )
        self._amom = self.make_int1e(
            "int1e_rxp", components=(3,), constant=-1j, is_complex=True, origin=True
        )
        self._d_ovlp = self.make_int1e("int1e_ipovlp", components=(3,))
        self._d_kin = self.make_int1e("int1e_ipkin", components=(3,))
        self._d_nuc = self.make_int1e("int1e_ipnuc", components=(3,))
        self._d_eri = self.make_int2e("int2e_ip1", components=(3,))
        self._d_rinv = self.make_int1e("int1e_iprinv", components=(3,), inv_origin=True)
        self._moments = {}
        for nx in range(5):
            for ny in range(5):
                for nz in range(5):
                    if 0 < nx + ny + nz < 5:
                        self._moments[(nx, ny, nz)] = self.make_int1e(
                            "int1e_" + nx * "x" + ny * "y" + nz * "z",
                            origin=True,
                        )

        # Set proper value for inverse sqrt of overlap integral
        # for cartesian basis sets
        if coord_type == "cartesian":
            self._ovlp_minhalf = 1 / np.sqrt(np.diag(self._ovlp()))

    @contextmanager
    def optimizer(self, opt_func):
        r"""
        Create an optimizer in a memory-safe manner.

        Parameters
        ----------
        opt_init_func : callable
            A ``libcint`` optimizer C function.

        Yields
        ------
        opt : pointer(pointer(CINTOpt))
            An initialized optimizer pointer.

        """
        # Initialize optimizer
        opt = POINTER(CINTOpt)()
        opt_func(byref(opt), self.atm, self.natm, self.bas, self.nbas, self.env)
        # Return optimizer for use in calling function
        yield opt
        # Free optimizer from memory (always called)
        LIBCINT.CINTdel_optimizer(byref(opt))

    def make_int1e(
        self,
        func_name,
        components=tuple(),
        constant=None,
        is_complex=False,
        origin=False,
        inv_origin=False,
    ):
        r"""
        Make an instance-bound 1-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func_name : str
            ``libcint`` function name.
        components : tuple, default=()
            Shape of components in each integral element.
            E.g., for normal integrals, ``comp=()``, while for nuclear gradients,
            ``components=(Natm, 3)``, and for nuclear Hessians, ``components=(Natm, Natm, 3, 3)``, etc.
        constant : (float | complex), default=1.
            A constant by which to multiply the whole integral array.
        is_complex : bool, default=False
            Whether the components in each integral element are complex. Not required if only
            multiplying by a complex constant using the ``constant`` keyword argument..
        origin : bool, default=False
            Whether you must specify an origin ``R`` for the integral computation.
        inv_origin : bool, default=False
            Whether you must specify an origin ``1 / |r - R|`` for the integral computation.

        """
        # Get C functions
        func = LIBCINT[func_name + ("_cart" if self.coord_type == "cartesian" else "_sph")]
        opt_func = LIBCINT[func_name + "_optimizer"]

        # Handle multi-component integral values
        n_components = len(components)
        if n_components == 0:
            components = (1,)
            no_comp = True
        else:
            no_comp = False
        if is_complex:
            components += (2,)
        prod_comp = np.prod(components, dtype=int)
        out_shape = (self.nbfn, self.nbfn) + components
        buf_shape = prod_comp * self._max_off**2

        # Handle [inv_]origin argument (prevent shadowing)
        has_origin_arg = bool(origin)
        has_inv_origin_arg = bool(inv_origin)

        # Make einsum string for normalization
        norm_einsum = (
            f"a,b,ab{'cdefghijklmnopqrstuvwxyz'[:n_components]}->"
            + f"ab{'cdefghijklmnopqrstuvwxyz'[:n_components]}"
        )

        # Make instance-bound integral method
        def int1e(notation="physicist", transform=None, origin=None, inv_origin=None):
            # Handle ``notation`` argument
            if notation not in ("physicist", "chemist"):
                raise ValueError("``notation`` must be one of 'physicist' or 'chemist'")

            # Handle origin argument
            if has_origin_arg:
                if origin is None:
                    raise ValueError("``origin`` must be specified")
                else:
                    self.env[1:4] = origin
            elif origin is not None:
                raise ValueError("``origin`` must not be specified")

            # Handle inv_origin argument
            if has_inv_origin_arg:
                if inv_origin is None:
                    raise ValueError("``inv_origin`` must be specified")
                else:
                    self.env[4:7] = inv_origin
            elif inv_origin is not None:
                raise ValueError("``inv_origin`` must not be specified")

            # Make output array
            out = np.zeros(out_shape, dtype=c_double, order="F")

            # Make temporary arrays
            buf = np.zeros(buf_shape, dtype=c_double)
            shls = np.zeros(2, dtype=c_int)

            # Evaluate the integral function over all shells
            with self.optimizer(opt_func) as opt:
                ipos = 0
                for ishl in range(self.nbas):
                    shls[0] = ishl
                    p_off = self._offs[ishl]
                    jpos = 0
                    for jshl in range(ishl + 1):
                        shls[1] = jshl
                        q_off = self._offs[jshl]
                        # Call the C function to fill `buf`
                        func(
                            buf,
                            None,
                            shls,
                            self.atm,
                            self.natm,
                            self.bas,
                            self.nbas,
                            self.env,
                            opt,
                            None,
                        )
                        # Fill `out` array
                        buf_array = buf[: p_off * q_off * prod_comp].reshape(
                            p_off, q_off, *components, order="F"
                        )
                        for p in range(p_off):
                            i_off = p + ipos
                            for q in range(q_off):
                                j_off = q + jpos
                                out[i_off, j_off] = buf_array[p, q]
                                out[j_off, i_off] = buf_array[p, q]
                        # Reset `buf`
                        buf[:] = 0
                        # Iterate `jpos`
                        jpos += q_off
                    # Iterate `ipos`
                    ipos += p_off

            # Cast `out` to complex if `is_complex` is set
            if is_complex:
                out = out.reshape(*out.shape[:-2], -1).view(np.complex128)

            # Remove useless axis in `out` if no `components` was given
            if no_comp:
                out = out.squeeze(axis=-1)

            # Multiply by constant
            if constant is not None:
                out *= constant

            # Apply permutation
            out = out[self._permutations, :][:, self._permutations]

            # Normalize integrals
            if self.coord_type == "cartesian":
                out = np.einsum(norm_einsum, self._ovlp_minhalf, self._ovlp_minhalf, out)

            # Apply transformation
            if transform is not None:
                out = np.tensordot(transform, out, (1, 0))
                out = np.tensordot(transform, out, (1, 1))
                out = np.swapaxes(out, 0, 1)

            return out

        # Return instance-bound integral method
        return int1e

    def make_int2e(
        self,
        func_name,
        components=tuple(),
        constant=None,
        is_complex=False,
        origin=False,
        inv_origin=False,
    ):
        r"""
        Make an instance-bound 2-electron integral method from a ``libcint`` function.

        Parameters
        ----------
        func_name : str
            ``libcint`` function name.
        components : tuple, default=()
            Shape of components in each integral element.
            E.g., for normal integrals, ``components=(1,)``, while for nuclear gradients,
            ``components=(Natm, 3)``, and for nuclear Hessians, ``components=(Natm, Natm, 3, 3)``, etc.
        constant : (float | complex), default=1.
            A constant by which to multiply the whole integral array.
        is_complex : bool, default=False
            Whether the components in each integral element are complex. Not required if only
            multiplying by a complex constant using the ``constant`` keyword argument..
        origin : bool, default=False
            Whether you must specify an origin ``R`` for the integral computation.
        inv_origin : bool, default=False
            Whether you must specify an origin ``1 / |r - R|`` for the integral computation.

        """
        # Get C functions
        func = LIBCINT[func_name + ("_cart" if self.coord_type == "cartesian" else "_sph")]
        opt_func = LIBCINT[func_name + "_optimizer"]

        # Handle multi-component integral values
        n_components = len(components)
        if n_components == 0:
            components = (1,)
            no_comp = True
        else:
            no_comp = False
        if is_complex:
            components += (2,)
        prod_comp = np.prod(components, dtype=int)
        out_shape = (self.nbfn, self.nbfn, self.nbfn, self.nbfn) + components
        buf_shape = prod_comp * self._max_off**4

        # Handle [inv_]origin argument (prevent shadowing)
        has_origin_arg = bool(origin)
        has_inv_origin_arg = bool(inv_origin)

        # Make einsum string for normalization
        norm_einsum = (
            f"a,b,c,d,abcd{'efghijklmnopqrstuvwxyz'[:n_components]}->"
            + f"abcd{'efghijklmnopqrstuvwxyz'[:n_components]}"
        )

        # Make instance-bound integral method
        def int2e(notation="physicist", transform=None, origin=None, inv_origin=None):
            # Handle ``notation`` argument
            if notation == "physicist":
                physicist = True
            elif notation == "chemist":
                physicist = False
            else:
                raise ValueError("``notation`` must be one of 'physicist' or 'chemist'")

            # Handle origin argument
            if has_origin_arg:
                if origin is None:
                    raise ValueError("``origin`` must be specified")
                else:
                    self.env[1:4] = origin
            elif origin is not None:
                raise ValueError("``origin`` must not be specified")

            # Handle inv_origin argument
            if has_inv_origin_arg:
                if inv_origin is None:
                    raise ValueError("``inv_origin`` must be specified")
                else:
                    self.env[4:7] = inv_origin
            elif inv_origin is not None:
                raise ValueError("``inv_origin`` must not be specified")

            # Make output array
            out = np.zeros(out_shape, dtype=c_double, order="F")

            # Make temporary arrays
            buf = np.zeros(buf_shape, dtype=c_double)
            shls = np.zeros(4, dtype=c_int)

            # Evaluate the integral function over all shells
            with self.optimizer(opt_func) as opt:
                ipos = 0
                for ishl in range(self.nbas):
                    shls[0] = ishl
                    p_off = self._offs[ishl]
                    jpos = 0
                    for jshl in range(ishl + 1):
                        ij = ((ishl + 1) * ishl) // 2 + jshl
                        shls[1] = jshl
                        q_off = self._offs[jshl]
                        kpos = 0
                        for kshl in range(self.nbas):
                            shls[2] = kshl
                            r_off = self._offs[kshl]
                            lpos = 0
                            for lshl in range(kshl + 1):
                                kl = ((kshl + 1) * kshl) // 2 + lshl
                                shls[3] = lshl
                                s_off = self._offs[lshl]
                                if ij < kl:
                                    lpos += s_off
                                    continue
                                # Call the C function to fill `buf`
                                func(
                                    buf,
                                    None,
                                    shls,
                                    self.atm,
                                    self.natm,
                                    self.bas,
                                    self.nbas,
                                    self.env,
                                    opt,
                                    None,
                                )
                                # Fill `out` array
                                buf_array = buf[
                                    : p_off * q_off * r_off * s_off * prod_comp
                                ].reshape(p_off, q_off, r_off, s_off, *components, order="F")
                                for p in range(p_off):
                                    i_off = p + ipos
                                    for q in range(q_off):
                                        j_off = q + jpos
                                        for r in range(r_off):
                                            k_off = r + kpos
                                            for s in range(s_off):
                                                l_off = s + lpos
                                                out[i_off, j_off, k_off, l_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[i_off, j_off, l_off, k_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[j_off, i_off, k_off, l_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[j_off, i_off, l_off, k_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[k_off, l_off, i_off, j_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[k_off, l_off, j_off, i_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[l_off, k_off, i_off, j_off] = buf_array[
                                                    p, q, r, s
                                                ]
                                                out[l_off, k_off, j_off, i_off] = buf_array[
                                                    p, q, r, s
                                                ]
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

            # Cast `out` to complex if `is_complex` is set
            if is_complex:
                out = out.reshape(*out.shape[:-2], out.shape[-2] * 2).view(np.complex128)

            # Remove useless axis in `out` if no `components` was given
            if no_comp:
                out = out.squeeze(axis=-1)

            # Multiply by constant
            if constant is not None:
                out *= constant

            # Apply permutation
            out = out[self._permutations]
            out = out[:, self._permutations]
            out = out[:, :, self._permutations]
            out = out[:, :, :, self._permutations]

            # Normalize integrals
            if self.coord_type == "cartesian":
                out = np.einsum(
                    norm_einsum,
                    self._ovlp_minhalf,
                    self._ovlp_minhalf,
                    self._ovlp_minhalf,
                    self._ovlp_minhalf,
                    out,
                )

            # Transpose integrals in `out` array to proper notation
            if physicist:
                out = out.transpose(0, 2, 1, 3)

            # Apply transformation
            if transform is not None:
                out = np.tensordot(transform, out, (1, 0))
                out = np.tensordot(transform, out, (1, 1))
                out = np.tensordot(transform, out, (1, 2))
                out = np.tensordot(transform, out, (1, 3))
                out = np.swapaxes(np.swapaxes(out, 0, 3), 1, 2)

            return out

        # Return instance-bound integral method
        return int2e

    def overlap_integral(self, notation="physicist", transform=None):
        r"""
        Compute the overlap integrals.

        Parameters
        ----------
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Integral array.

        """
        return self._ovlp(notation=notation, transform=transform)

    def kinetic_energy_integral(self, notation="physicist", transform=None):
        r"""
        Compute the kinetic energy integrals.

        Parameters
        ----------
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Integral array.

        """
        return self._kin(notation=notation, transform=transform)

    def nuclear_attraction_integral(self, notation="physicist", transform=None):
        r"""
        Compute the nuclear attraction integrals.

        Parameters
        ----------
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Integral array.

        """
        return self._nuc(notation=notation, transform=transform)

    def electron_repulsion_integral(self, notation="physicist", transform=None):
        r"""
        Compute the electron repulsion integrals.

        Parameters
        ----------
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, Nbasis, Nbasis, dtype=float)
            Integral array.

        """
        return self._eri(notation=notation, transform=transform)

    def r_inv_integral(self, origin=None, notation="physicist", transform=None):
        r"""
        Compute the :math:`1/\left|\mathbf{r} - \mathbf{R}_\text{inv}\right|` integrals.

        Parameters
        ----------
        origin : np.ndarray(3, dtype=float), default=[0, 0, 0]
            Origin about which to evaluate integrals.
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Integral array.

        """
        return self._rinv(inv_origin=origin, notation=notation, transform=transform)

    def momentum_integral(self, origin=None, notation="physicist", transform=None):
        r"""
        Compute the momentum integrals.

        Parameters
        ----------
        origin : np.ndarray(3, dtype=float), default=[0, 0, 0]
            Origin about which to evaluate integrals.
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, 3, dtype=complex)
            Integral array.

        """
        return self._mom(origin=origin, notation=notation, transform=transform)

    def angular_momentum_integral(self, origin=None, notation="physicist", transform=None):
        r"""
        Compute the angular momentum integrals.

        Parameters
        ----------
        origin : np.ndarray(3, dtype=float), default=[0, 0, 0]
            Origin about which to evaluate integrals.
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, 3, dtype=complex)
            Integral array.

        """
        raise NotImplementedError("Angular momentum integral doesn't work; see Issue #149")
        # return self._amom(origin=origin, notation=notation, transform=transform)

    def point_charge_integral(
        self, point_coords, point_charges, notation="physicist", transform=None
    ):
        r"""
        Compute the point charge integrals.

        Parameters
        ----------
        point_coords : np.ndarray(N, 3, dtype=float)
            Coordinates of point charges.
        point_charges : np.ndarray(N, dtype=float)
            Charges of point charges.
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, N, dtype=float)
            Integral array.

        """
        # Make output array
        out = np.zeros((self.nbfn, self.nbfn, len(point_charges)), dtype=c_double, order="F")
        # Compute 1/|r - r_{inv}| for each charge
        for icharge, (coord, charge) in enumerate(zip(point_coords, point_charges)):
            val = self._rinv(inv_origin=coord, notation=notation, transform=transform)
            val *= -charge
            out[:, :, icharge] = val
        # Return integrals in `out` array
        return out

    def moment_integral(self, orders, origin=None, notation="physicist", transform=None):
        r"""
        Compute the moment integrals.

        Parameters
        ----------
        orders : np.ndarray(N, 3, dtype=int)
            Moment orders :math:`\left[x, y, z\right\]` to evaluate.
        origin : np.ndarray(3, dtype=float), default=[0, 0, 0]
            Origin about which to evaluate integrals.
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont)
            Transformation matrix from the basis set in the given coordinate system (e.g. AO) to linear
            combinations of contractions (e.g. MO).
            Transformation is applied to the left, i.e. the sum is over the index 1 of `transform`
            and index 0 of the array for contractions.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, N, dtype=float)
            Integral array.

        Notes
        -----
        This function is tied to the Libcint functions generated at compile-time.
        They were generated up to 4th order for any one X, Y, or Z, and up to 4th order
        for any combination of X, Y, or Z (still up to 4th order for any one component).

        """
        # Make output array
        out = np.zeros((self.nbfn, self.nbfn, len(orders)), dtype=np.float64)
        # Compute moment integral for each {X,Y,Z} order
        try:
            for i, order in enumerate(orders):
                if sum(order) == 0:
                    out[:, :, i] = self._ovlp(notation=notation, transform=transform)
                else:
                    out[:, :, i] = self._moments[tuple(order)](
                        origin=origin, notation=notation, transform=transform
                    )
        except KeyError:
            raise ValueError(
                "Invalid order; can use up to order 4 for any XYZ component,"
                "and up to 4th order total using combinations of XYZ components"
            )
        # Return integrals in `out` array
        return out


def normalized_coeffs(shell):
    r"""
    Normalize the GeneralizedContractionShell coefficients.

    Parameters
    ----------
    shell : GeneralizedContractionShell

    Returns
    -------
    coeffs : np.ndarray(K, M, dtype=float)
        Normalized contraction coefficients.

    Notes
    -----
    Adapted from `https://github.com/pyscf/pyscf/blob/master/pyscf/gto/mole.py`.

    """

    def gaussian_int(l, a):
        return 0.5 * factorial(0.5 * l - 0.5) * a ** (-0.5 * l - 0.5)

    def gto_norm(l, a):
        return 1 / np.sqrt(gaussian_int(2 * l + 2, 2 * a))

    # Normalize radial part of GTO
    cs = np.einsum("km,k->km", shell.coeffs, gto_norm(shell.angmom, shell.exps))
    # Normalize contractions
    es = gaussian_int(2 * shell.angmom + 2, shell.exps[:, np.newaxis] + shell.exps[np.newaxis, :])
    ss = 1 / np.sqrt(np.einsum("km,kl,lm->m", cs, es, cs))
    return np.einsum("km,m->km", cs, ss)
