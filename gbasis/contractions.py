"""Data classes for contracted Gaussians."""


class ContractedCartesianGaussians:
    r"""Data class for contracted Cartesian Gaussians of the same angular momentum.

    .. math::

        \phi_{\vec{a}, A} (\mathbf{r}) &=
        \sum_i d_i (x - X_A)^{a_x} (y - Y_A)^{a_y} (z - Z_A)^{a_z}
        \exp{-\alpha_i |\vec{r} - \vec{R}_A|^2}\\
        &= \sum_i d_i g_{i} (\vec{r} | \vec{a}, \vec{R}_A)

    where :math:`\vec{r} = (x, y, z)`, :math:`\vec{R}_A = (X_A, Y_A, Z_A)`,
    :math:`\vec{a} = (a_x, a_y, a_z)`, and :math:`g_i` is a Gaussian primitive.

    Since the integrals involving these contractions are computed using recursive relations that
    modify the :math:`\vec{a}`, we group the primitives that share the same properties (i.e.
    :math:`\vec{R}_A` and :math:`\alpha_i`) except for the :math:`\vec{a}` in the hopes of
    vectorizing and storing repeating elements.

    Attributes
    ----------
    angmom : int
        Angular momentum of the contractions.
        .. math::

            \sum_i \vec{a} = a_x + a_y + a_z

    coord : np.ndarray(3,)
        Coordinate of the center of the Gaussian primitives.
    charge : float
        Charge at the center of the Gaussian primitives.
    coeffs : np.ndarray(K,)
        Contraction coefficients, :math:`\{d_i\}`, of the primitives.
    exponents : np.ndarray(L,)
        Exponents of the primitives, :math:`\{\alpha_i\}`.

    """
