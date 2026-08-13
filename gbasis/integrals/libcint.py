r"""
Python C-API bindings for ``libcint`` GTO integrals library.

This module provides the :class:`CBasis` class, which wraps the ``libcint``
C library to compute one- and two-electron integrals over Gaussian-type
orbitals (GTOs) for molecular systems. Integrals are computed using
C shell-loop implementations for efficiency, achieving significant speedups
over pure-Python implementations.
"""

import os
import sys

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
    gradient_kinetic(transform=None)
        Compute the gradient of kinetic energy integrals.
    gradient_nuclear(transform=None)
        Compute the gradient of nuclear attraction integrals.
    gradient_rinv(inv_origin=None, transform=None)
        Compute the gradient of 1/r integrals.
    ia01p(transform=None)
        Compute the GIAO paramagnetic shielding integrals.
    ircxp(transform=None)
        Compute the GIAO angular momentum integrals.
    iking(transform=None)
        Compute the GIAO kinetic energy integrals.
    iovlpg(transform=None)
        Compute the GIAO overlap gradient integrals.
    inucg(transform=None)
        Compute the GIAO nuclear attraction integrals.
    three_center_two_electron(transform=None)
        Compute the 3-center 2-electron integrals.
    

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

        Raises
        ------
        ValueError
            If ``coord_type`` is not 'spherical' or 'cartesian'.
 

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
            raw_S = np.zeros((nbfn, nbfn), dtype=c_double, order="C")
            libcint_bindings.overlap_integral_array_cart(
                np.ascontiguousarray(raw_S), natm, atm, nbas, bas, env, offs, nbfn
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
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            :math:`1/r` integral array.
        
        """
        if inv_origin is None:
            inv_origin = np.zeros(3)
        self.env[4:7] = inv_origin

        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
            Momentum integral array with three Cartesian components.
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
        # Normalize cartesian
        if self._ovlp_minhalf is not None:
            out_real = np.einsum("a,b,abc->abc", self._ovlp_minhalf, self._ovlp_minhalf, out_real)

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
            Dipole moment integral array(x-component only).

        Notes
        -----
        Returns the first component (x) of the dipole integral from
        ``int1e_r``. The full 3-component dipole integral is
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
            Quadrupole moment integral array(xx-component only).

        Notes
        -----
        Returns the first component (xx) of the quadrupole integral from
        ``int1e_rr``. The full 9-component quadrupole integral is
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
            Octupole moment integral array(xxx-component only).

        Notes
        -----
        Returns the first component (xxx) of the octupole integral from
        ``int1e_rrr``. The full 27-component octupole integral is
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

        This integral is a building block for computing nuclear coordinate
        gradients of the kinetic energy contribution.

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
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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

        This integral is a building block for computing nuclear coordinate
        gradients of the nuclear attraction contribution.

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
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the gradient of :math:`1/r` integrals (:math:`i\nabla\, 1/r`).
 
        This integral is a building block for computing nuclear coordinate
        gradients of the electron-nucleus attraction.

        Parameters
        ----------
        inv_origin : np.ndarray(3, dtype=float), optional
            Origin for 1/|r - R| operator. 
            Default is [0, 0, 0].
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.

        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            Gradient: math:`1/r` integral array.
        """
        if inv_origin is None:
            inv_origin = np.zeros(3)
        self.env[4:7] = inv_origin
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the GIAO paramagnetic shielding integrals (``int1e_ia01p``).
 
        This integral is a building block for NMR paramagnetic shielding
        tensor calculations using gauge-including atomic orbitals (GIAOs).
 
        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO paramagnetic shielding integral array.
 
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the GIAO angular momentum integrals (``int1e_ircxp``).
 
        This integral is a building block for NMR shielding tensor and
        magnetic susceptibility calculations using GIAOs.
 
        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO angular momentum integral array.
 
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the GIAO kinetic energy integrals (``int1e_igkin``).
 
        This integral is a building block for NMR shielding tensor
        calculations using gauge-including atomic orbitals (GIAOs).
 
        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO kinetic energy integral array.
 
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the GIAO overlap gradient integrals (``int1e_igovlp``).
 
        This integral is a building block for NMR shielding tensor
        calculations using gauge-including atomic orbitals (GIAOs).
 
        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO overlap gradient integral array.
 
        """

        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the GIAO nuclear attraction integrals (``int1e_ignuc``).
 
        This integral is a building block for NMR shielding tensor
        calculations using gauge-including atomic orbitals (GIAOs).
 
        Parameters
        ----------
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, dtype=float)
            GIAO nuclear attraction integral array.
 
        """
        out = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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

    def point_charge(self, point_coords, point_charges, transform=None):

        r"""
        Compute the point charge integrals.
 
        The point charge integral represents the electrostatic potential due to
        a set of point charges at given coordinates. For each pair of basis
        functions :math:`\phi_i` and :math:`\phi_j` and point charge :math:`q_n`
        at position :math:`\mathbf{R}_n`, it is defined as:
 
        .. math::
            V_{ij}^{(n)} = -q_n \langle \phi_i | \frac{1}{|\mathbf{r} - \mathbf{R}_n|} | \phi_j \rangle
 
        Parameters
        ----------
        point_coords : np.ndarray(N, 3, dtype=float)
            Coordinates of point charges in Bohr.
        point_charges : np.ndarray(N, dtype=float)
            Magnitude of each point charge.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, N, dtype=float)
            Point charge integral array, one matrix per point charge.
 
        """
        out = np.zeros((self.nbfn, self.nbfn, len(point_charges)), dtype=c_double)
        for icharge, (coord, charge) in enumerate(zip(point_coords, point_charges)):
            # Set inv_origin in env for this charge
            self.env[4:7] = coord
            val = np.zeros((self.nbfn, self.nbfn), dtype=c_double)
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
        Compute the moment integrals up to third order.
 
        For each pair of basis functions :math:`\phi_i` and :math:`\phi_j`
        and a given moment order :math:`(n_x, n_y, n_z)`, the integral is:
 
        .. math::
            M_{ij} = \langle \phi_i | (x - X_0)^{n_x} (y - Y_0)^{n_y} (z - Z_0)^{n_z} | \phi_j \rangle
 
        where :math:`\mathbf{R}_0 = (X_0, Y_0, Z_0)` is the origin.
 
        Parameters
        ----------
        orders : np.ndarray(N, 3, dtype=int)
            Moment orders :math:`[n_x, n_y, n_z]` to evaluate. Each row
            specifies one moment; the sum of each row must be 0, 1, 2, or 3.
        origin : np.ndarray(3, dtype=float), optional
            Origin about which to evaluate integrals. Default is ``[0, 0, 0]``.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, N, dtype=float)
            Moment integral array, one matrix per requested order.
 
        Raises
        ------
        NotImplementedError
            If any order has a sum greater than 3.
 
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
        Compute the electron repulsion integrals (ERIs).
 
        The two-electron repulsion integral between basis functions
        :math:`\phi_i`, :math:`\phi_j`, :math:`\phi_k`, and :math:`\phi_l`
        is defined as:
 
        .. math::
            g_{ijkl} = \langle \phi_i \phi_j | \frac{1}{r_{12}} | \phi_k \phi_l \rangle
 
        Parameters
        ----------
        notation : ("physicist" | "chemist"), optional
            Index ordering convention. Default is ``"physicist"``.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, Nbasis, Nbasis, dtype=float)
            Electron repulsion integral array.
 
        Raises
        ------
        ValueError
            If ``notation`` is not ``'physicist'`` or ``'chemist'``.
 
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
 
        .. note::
            This method is not yet implemented. See GitHub Issue #149.
 
        Parameters
        ----------
        origin : np.ndarray(3, dtype=float), optional
            Origin about which to evaluate integrals. Default is ``[0, 0, 0]``.
        notation : ("physicist" | "chemist"), optional
            Index ordering convention. Default is ``"physicist"``.
        transform : np.ndarray(K, K_cont), optional
            Transformation matrix from AO to MO basis.
            Default is no transformation.
 
        Returns
        -------
        out : np.ndarray(Nbasis, Nbasis, 3, dtype=complex)
            Angular momentum integral array.
 
        Raises
        ------
        NotImplementedError
            Always raised; angular momentum integrals are not yet implemented.
 
        """
        raise NotImplementedError("Angular momentum integral doesn't work; see Issue #149")
        # return self._amom(origin=origin, notation=notation, transform=transform)

    
    

def normalized_coeffs(shell):
    r"""
    Normalize the contraction coefficients of a ``GeneralizedContractionShell``.
 
    Parameters
    ----------
    shell : GeneralizedContractionShell
        Shell whose contraction coefficients are to be normalized.
 
    Returns
    -------
    coeffs : np.ndarray(Nprim, Ncont, dtype=float)
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
