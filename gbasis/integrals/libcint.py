r"""
Python C-API bindings for ``libcint`` GTO integrals library.

"""

from gbasis.integrals.lib import libcint_bindings


from ctypes import c_int, c_double

from operator import attrgetter




import numpy as np

from scipy.special import factorial



__all__ = [
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
        # Save coord type suffix for C bindings
        self._ct = "sph" if coord_type == "spherical" else "cart"

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
        self._permutations = permutations
        

        # Compute overlap-based normalization for cartesian coordinates
        if coord_type == "cartesian":
            raw_S = np.zeros((nbfn, nbfn), dtype=c_double, order="F")
            libcint_bindings.overlap_integral_array_cart(
                raw_S, natm, atm, nbas, bas, env, offs, nbfn
            )
            raw_S = raw_S[permutations, :][:, permutations]
            self._ovlp_minhalf = 1.0 / np.sqrt(np.diag(raw_S))
        else:
            self._ovlp_minhalf = None
    
    
    def overlap(self, transform=None):
        r"""
        Compute the overlap integrals.

        The overlap integral measures the degree to which two basis functions
        :math:`\phi_i` and :math:`\phi_j` occupy the same region of space,
        and is defined as:

        .. math::
            S_{ij} = \langle \phi_i | \phi_j \rangle

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Overlap integral array.

        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"overlap_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian (skip if _ovlp_minhalf not yet set)
        if getattr(self, '_ovlp_minhalf', None) is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def kinetic_energy(self, transform=None):
        r"""
        Compute the kinetic energy integrals.

        The kinetic energy integral represents the expectation value of the
        kinetic energy operator between basis functions :math:`\phi_i` and
        :math:`\phi_j`, and is defined as:

        .. math::
            T_{ij} = \langle \phi_i | -\frac{1}{2}\nabla^2 | \phi_j \rangle

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Kinetic energy integral array.

        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"kinetic_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )

        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def nuclear_attraction(self, transform=None):
        r"""
        Compute the nuclear attraction integrals.

        The nuclear attraction integral represents the electrostatic attraction
        between electrons and nuclei. For each pair of basis functions
        :math:`\phi_i` and :math:`\phi_j`, it is defined as:

        .. math::
            V_{ij} = \langle \phi_i | \sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} | \phi_j \rangle

        where :math:`Z_A` is the nuclear charge and :math:`\mathbf{R}_A` is
        the position of atom :math:`A`.
        
        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Nuclear attraction integral array.

        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"nuclear_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out
        

    
    def rinv(self, inv_origin=None, transform=None):
        r"""
        Compute the :math:`1/\left|\mathbf{r} - \mathbf{R}_\text{inv}\right|` integrals.

        The :math:`1/r` integral represents the electrostatic potential due to
        a unit point charge at a given origin. For each pair of basis functions
        :math:`\phi_i` and :math:`\phi_j`, it is defined as:

        .. math::
            V_{ij} = \langle \phi_i | \frac{1}{|\mathbf{r} - \mathbf{R}_\text{inv}|} | \phi_j \rangle
        
        Parameters
        ----------
        inv_origin : np.ndarray(3, dtype=float), optional
            Origin for 1/|r - R| operator.
            Default is [0, 0, 0].
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Returns
            -------
            out : np.ndarray(Nbasis, Nbasis, dtype=float)
                1/r integral array.
        
        """
        if inv_origin is None:
            inv_origin = np.zeros(3)
        self.env[4:7] = inv_origin

        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"rinv_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def momentum(self, origin=None, transform=None):
        r"""
        Compute the momentum integrals.

        The momentum integral represents the expectation value of the momentum
        operator between basis functions :math:`\phi_i` and :math:`\phi_j`,
        and is defined as:

        .. math::
            p_{ij} = \langle \phi_i | -i\nabla | \phi_j \rangle

        Parameters
        ----------
        origin : np.ndarray(3, dtype=float), default=[0, 0, 0]
            Origin about which to evaluate integrals.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, 3, dtype=complex)
            Momentum integral array.
        """
        if origin is None:
            origin = np.zeros(3)
        self.env[1:4] = origin
        out_real = np.zeros((self.nbfn, self.nbfn, 3), dtype=np.float64)
        getattr(libcint_bindings, f"momentum_integral_array_{self._ct}")(
            out_real,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out_real = out_real[self._permutations, :][:, self._permutations]
        # Momentum is purely imaginary: p = -i * (real buffer)
        out = -1j * out_real
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def dipole(self, transform=None):
        r"""
        Compute the dipole moment integrals.

        The dipole moment integral represents the expectation value of the
        position operator between basis functions :math:`\phi_i` and
        :math:`\phi_j`, and is defined as:

        .. math::
            \mu_{ij} = \langle \phi_i | \mathbf{r} | \phi_j \rangle

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Dipole moment integral array.

        Notes
        -----
        Returns the first component (x) of the dipole integral from
        ``int1e_r_sph``. The full 3-component dipole integral is
        available via ``moment_integral()``.

        """
        out = np.zeros((self.nbfn, self.nbfn, 3), dtype=np.float64)
        getattr(libcint_bindings, f"dipole_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # x-component only for dipole()
        out = out[:, :, 0]
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out
        

    def quadrupole(self, transform=None):
        r"""
        Compute the quadrupole moment integrals.

        The quadrupole moment integral represents the expectation value of the
        second-order position operator between basis functions :math:`\phi_i`
        and :math:`\phi_j`, and is defined as:

        .. math::
            Q_{ij} = \langle \phi_i | \mathbf{r}\mathbf{r} | \phi_j \rangle

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Quadrupole moment integral array.

        Notes
        -----
        Returns the first component (xx) of the quadrupole integral from
        ``int1e_rr_sph``. The full 9-component quadrupole integral is
        available via ``moment_integral()``.

        """
        out = np.zeros((self.nbfn, self.nbfn, 9), dtype=np.float64)
        getattr(libcint_bindings, f"quadrupole_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # xx-component only for quadrupole()
        out = out[:, :, 0]
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out


    def octupole(self, transform=None):
        r"""
        Compute the octupole moment integrals.

        The octupole moment integral represents the expectation value of the
        third-order position operator between basis functions :math:`\phi_i`
        and :math:`\phi_j`, and is defined as:

        .. math::
            O_{ij} = \langle \phi_i | \mathbf{r}\mathbf{r}\mathbf{r} | \phi_j \rangle

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Octupole moment integral array.

        Notes
        -----
        Returns the first component (xxx) of the octupole integral from
        ``int1e_rrr_sph``. The full 27-component octupole integral is
        available via ``moment_integral()``.

        """
        out = np.zeros((self.nbfn, self.nbfn, 27), dtype=np.float64)
        getattr(libcint_bindings, f"octupole_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # xxx-component only for octupole()
        out = out[:, :, 0]
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def gradient_kinetic(self, transform=None):
        r"""
        Compute the gradient of kinetic energy integrals (i∇ kinetic).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Gradient kinetic integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"ipkin_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def gradient_nuclear(self, transform=None):
        r"""
        Compute the gradient of nuclear attraction integrals (i∇ nuclear).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.


        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Gradient nuclear integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"ipnuc_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def gradient_rinv(self, inv_origin=None, transform=None):
        r"""
        Compute the gradient of 1/r integrals (i∇ rinv).

        Parameters
        ----------
        inv_origin : np.ndarray(3, dtype=float), optional
            Origin for 1/|r - R| operator. Default is [0, 0, 0].
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Gradient 1/r integral array.
        """
        if inv_origin is None:
            inv_origin = np.zeros(3)
        self.env[4:7] = inv_origin
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"iprinv_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out) 
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def ia01p(self, transform=None):
        r"""
        Compute the GIAO paramagnetic shielding integrals (ia01p).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.


        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO ia01p integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"ia01p_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def ircxp(self, transform=None):
        r"""
        Compute the GIAO angular momentum integrals (ircxp).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
            
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO ircxp integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"ircxp_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out


    def iking(self, transform=None):
        r"""
        Compute the GIAO kinetic energy integrals (igkin).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO igkin integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"igkin_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def iovlpg(self, transform=None):
        r"""
        Compute the GIAO overlap gradient integrals (igovlp).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO igovlp integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"igovlp_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def inucg(self, transform=None):
        r"""
        Compute the GIAO nuclear attraction integrals (ignuc).

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO ignuc integral array.
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
        getattr(libcint_bindings, f"ignuc_integral_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations, :][:, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, out)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def point_charge(self, point_coords, point_charges,transform=None):
        r"""
        Compute the point charge integrals.

        The point charge integral represents the electrostatic potential due to
        a set of point charges at given coordinates. For each pair of basis
        functions :math:`\phi_i` and :math:`\phi_j`, it is defined as:

        .. math::
            V_{ij}^{(n)} = -q_n \langle \phi_i | \frac{1}{|\mathbf{r} - \mathbf{R}_n|} | \phi_j \rangle

        Parameters
        ----------
        point_coords : np.ndarray(N, 3, dtype=float)
            Coordinates of point charges.
        point_charges : np.ndarray(N, dtype=float)
            Charges of point charges.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, N, dtype=float)
            Point charge integral array.

        """
        out = np.zeros((self.nbfn, self.nbfn, len(point_charges)), dtype=c_double, order="F")
        for icharge, (coord, charge) in enumerate(zip(point_coords, point_charges)):
            # Set inv_origin in env for this charge
            self.env[4:7] = coord
            val = np.zeros((self.nbfn, self.nbfn), dtype=c_double, order="F")
            getattr(libcint_bindings, f"rinv_integral_array_{self._ct}")(
                val,
                self.natm,
                self.atm,
                self.nbas,
                self.bas,
                self.env,
                self._offs,
                self.nbfn,
            )
            val *= -charge
            val = val[self._permutations, :][:, self._permutations]
            # Normalize cartesian
            if self._ovlp_minhalf is not None:
                val = np.einsum("a,b,ab->ab", self._ovlp_minhalf, self._ovlp_minhalf, val)
            out[:, :, icharge] = val

            # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def moment(self, orders, origin=None, transform=None):
        r"""
        Compute the moment integrals.

        Parameters
        ----------
        orders : np.ndarray(N, 3, dtype=int)
            Moment orders [x, y, z] to evaluate.
        origin : np.ndarray(3, dtype=float), default=[0, 0, 0]
            Origin about which to evaluate integrals.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, N, dtype=float)
            Moment integral array.
        """
        if origin is None:
            origin = np.zeros(3)
        self.env[1:4] = origin

        # Pre-fetch full multi-component buffers
        dip = np.zeros((self.nbfn, self.nbfn, 3), dtype=np.float64)
        getattr(libcint_bindings, f"dipole_integral_array_{self._ct}")(
            dip, self.natm, self.atm, self.nbas, self.bas, self.env, self._offs, self.nbfn)
        dip = dip[self._permutations, :][:, self._permutations]

        quad = np.zeros((self.nbfn, self.nbfn, 9), dtype=np.float64)
        getattr(libcint_bindings, f"quadrupole_integral_array_{self._ct}")(
            quad, self.natm, self.atm, self.nbas, self.bas, self.env, self._offs, self.nbfn)
        quad = quad[self._permutations, :][:, self._permutations]

        oct_ = np.zeros((self.nbfn, self.nbfn, 27), dtype=np.float64)
        getattr(libcint_bindings, f"octupole_integral_array_{self._ct}")(
            oct_, self.natm, self.atm, self.nbas, self.bas, self.env, self._offs, self.nbfn)
        oct_ = oct_[self._permutations, :][:, self._permutations]

        # Cartesian normalization
        if self._ovlp_minhalf is not None:
            s = self._ovlp_minhalf
            dip  = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, dip)
            quad = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, quad)
            oct_ = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, oct_)

        out = np.zeros((self.nbfn, self.nbfn, len(orders)), dtype=np.float64)
        for i, order in enumerate(orders):
            ox, oy, oz = int(order[0]), int(order[1]), int(order[2])
            total = ox + oy + oz
            if total == 0:
                out[:, :, i] = self.overlap()
            elif total == 1:
                # libcint int1e_r layout: x=0, y=1, z=2
                c = 0 * ox + 1 * oy + 2 * oz
                out[:, :, i] = dip[:, :, c]
            elif total == 2:
                # libcint int1e_rr layout: xx=0,xy=1,xz=2,yx=3,yy=4,yz=5,zx=6,zy=7,zz=8
                comp_map = {(2,0,0):0, (1,1,0):1, (1,0,1):2,
                            (0,2,0):4, (0,1,1):5, (0,0,2):8}
                out[:, :, i] = quad[:, :, comp_map[(ox,oy,oz)]]
            elif total == 3:
                # libcint int1e_rrr layout: xxx=0,xxy=1,xxz=2,xyx=3,xyy=4,xyz=5,
                #                           xzx=6,xzy=7,xzz=8,yxx=9,...,zzz=26
                comp_map = {
                    (3,0,0):0,  (2,1,0):1,  (2,0,1):2,
                    (1,2,0):4,  (1,1,1):5,  (1,0,2):8,
                    (0,3,0):13, (0,2,1):14, (0,1,2):17, (0,0,3):26
                }
                out[:, :, i] = oct_[:, :, comp_map[(ox,oy,oz)]]
            else:
                raise NotImplementedError(
                    f"moment() for order {order} (sum={total}) not yet implemented")

        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.swapaxes(out, 0, 1)
        return out

    def electron_repulsion(self, notation="physicist", transform=None):
        r"""
        Compute the electron repulsion integrals.

        The two-electron repulsion integral between basis functions
        :math:`\phi_i`, :math:`\phi_j`, :math:`\phi_k`, and :math:`\phi_l`
        is defined as:

        .. math::
            g_{ijkl} = \langle \phi_i \phi_j | \frac{1}{r_{12}} | \phi_k \phi_l \rangle

        Parameters
        ----------
        notation : ("physicist" | "chemist"), default="physicist"
            Axis order convention.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, Nbasis, Nbasis, dtype=float)
            Electron repulsion integral array.

        """
        if notation not in ("physicist", "chemist"):
            raise ValueError("``notation`` must be one of 'physicist' or 'chemist'")

        out = np.zeros((self.nbfn, self.nbfn, self.nbfn, self.nbfn), dtype=c_double)
        if self._ct == "cart":
            libcint_bindings.eri_array_cart(
                out, self.natm, self.atm, self.nbas, self.bas, self.env, self._offs, self.nbfn,
            )
        else:
            libcint_bindings.eri_array(
                out, self.natm, self.atm, self.nbas, self.bas, self.env, self._offs, self.nbfn,
            )

        # Apply permutation
        out = out[self._permutations]
        out = out[:, self._permutations]
        out = out[:, :, self._permutations]
        out = out[:, :, :, self._permutations]
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            s = self._ovlp_minhalf
            out = np.einsum("a,b,c,d,abcd->abcd", self._ovlp_minhalf, self._ovlp_minhalf, self._ovlp_minhalf, self._ovlp_minhalf, out)        # Apply notation
        if notation == "chemist":
            out = out.transpose(0, 2, 1, 3)
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.tensordot(transform, out, (1, 2))
            out = np.tensordot(transform, out, (1, 3))
            out = np.swapaxes(np.swapaxes(out, 0, 3), 1, 2)
        return out

    def three_center_two_electron(self, transform=None):
        r"""
        Compute the 3-center 2-electron integrals.

        The 3-center 2-electron integral between basis functions
        :math:`\phi_i`, :math:`\phi_j`, and auxiliary basis function
        :math:`\phi_k` is defined as:

        .. math::
            (ij|k) = \langle \phi_i \phi_j | \frac{1}{r_{12}} | \phi_k \rangle

        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, Nbasis, dtype=float)
            3-center 2-electron integral array.

        """
        out = np.zeros((self.nbfn, self.nbfn, self.nbfn), dtype=c_double)
        getattr(libcint_bindings, f"int3c2e_array_{self._ct}")(
            out,
            self.natm,
            self.atm,
            self.nbas,
            self.bas,
            self.env,
            self._offs,
            self.nbfn,
        )
        # Apply permutation
        out = out[self._permutations]
        out = out[:, self._permutations]
        out = out[:, :, self._permutations]
        # Apply transformation
        if transform is not None:
            out = np.tensordot(transform, out, (1, 0))
            out = np.tensordot(transform, out, (1, 1))
            out = np.tensordot(transform, out, (1, 2))
            out = np.moveaxis(out, [0, 1, 2], [2, 0, 1])
        return out

    
    
    
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
