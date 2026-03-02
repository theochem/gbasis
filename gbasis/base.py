"""Base class for arrays that depend on one or more contracted Gaussians."""

import abc

from gbasis.contractions import GeneralizedContractionShell


class BaseGaussianRelatedArray(abc.ABC):
    """Base class for constructing arrays associated with contracted Gaussians.

    Attributes
    ----------
    _axes_contractions : tuple of tuple of GeneralizedContractionShell
        Contractions that are associated with each index of the array.
        Each tuple of `GeneralizedContractionShell` corresponds to an index of the array.

    Methods
    -------
    __init__(self, *axes_contractions)
        Initialize.
    construct_array_contraction(self, *contraction, **kwargs) : np.ndarray
        Return the array associated with a `GeneralizedContractionShell` instance.
    construct_array_cartesian(self, **kwargs) : np.ndarray
        Return the array associated with Cartesian Gaussians.
    construct_array_spherical(self, **kwargs) : np.ndarray
        Return the array associated with spherical Gaussians (atomic orbitals).
    construct_array_lincomb(self, *transform, coord_types, **kwargs) : np.ndarray
        Return the array associated with linear combinations of contractions in the given coordinate
        system.

    """

    def __init__(self, *contractions):
        """Initialize.

        Parameters
        ----------
        contractions : list/tuple of GeneralizedContractionShell
            Contractions that are associated with each index of the array.
            First contractions is associated with the first index, etc.

        Raises
        ------
        TypeError
            If `axes_contractions` is not given as a list or tuple of `GeneralizedContractionShell`.
        ValueError
            If `axes_contractions` is an empty list or tuple.

        """
        for each_contractions in contractions:
            if not isinstance(each_contractions, (list, tuple)):
                raise TypeError(
                    "Contractions must be given as a list or tuple of `GeneralizedContractionShell`"
                    " instance"
                )
            if not each_contractions:
                raise ValueError("At least one `GeneralizedContractionShell` must be given.")
            for contraction in each_contractions:
                if not isinstance(contraction, GeneralizedContractionShell):
                    raise TypeError(
                        "Given contractions must be instances of the `GeneralizedContractionShell` "
                        "class."
                    )
        self._axes_contractions = tuple(
            tuple(each_contractions) for each_contractions in contractions
        )

    @abc.abstractmethod
    def construct_array_contraction(self, *contractions, **kwargs):
        """Return the array associated with a contracted Cartesian Gaussian.

        Parameters
        ----------
        contractions : GeneralizedContractionShell
            Contracted Cartesian Gaussians (of the same shell) that will be used to construct an
            array.
            Note that multiple instances may be needed to construct the array.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array_contraction : np.ndarray
            Array associated with the given instance(s) of `GeneralizedContractionShell`.

        Notes
        -----
        The next level of classes will be divided by number of indices associated with
        `GeneralizedContractionShell`. Then this method's parameters will likely be different from
        it's children. This means that using this method's API blindly (by simply copying and
        pasting) will not be suitable for all its children. Here, we allow arbitrary number of
        `contractions` to indicate that its children may have different number of `contractions` in
        its parameters. However, in actual implementations, the number of `contractions` will likely
        be fixed, in which case, an arbitrary number of `contractions` should not be accepted.

        """

    @abc.abstractmethod
    def construct_array_cartesian(self, **kwargs):
        """Return the array associated with the given set of contracted Cartesian Gaussians.

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray
            Array associated with the given set of contracted Cartesian Gaussians.

        """

    @abc.abstractmethod
    def construct_array_spherical(self, **kwargs):
        """Return the array associated with spherical Gaussians (atomic orbitals).

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.

        """

    @abc.abstractmethod
    def construct_array_mix(self, coord_types, **kwargs):
        """Return the array associated with a set of Gaussians of the given coordinate systems.

        Parameters
        ----------
        coord_types : list/tuple of str
            Types of the coordinate system for each `GeneralizedContractionShell`.
            Each entry must be one of "cartesian" or "spherical".
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.

        """

    @abc.abstractmethod
    def construct_array_lincomb(self, *transform, coord_type, **kwargs):
        """Return the array associated with linear combinations of contractions.

        Parameters
        ----------
        transform : np.ndarray
            Transformation matrix that will be used for linearly combining the spherical
            contractions.
            Note that multiple instances may be needed to construct the array.
        coord_type : list/tuple of str
            Types of the coordinate system for each GeneralizedContractionShell.
            Each entry must be one of "cartesian" or "spherical". If multiple
            instances of GeneralizedContractionShell are given but only one string
            ("cartesian" or "spherical") is provided in the list/tuple, all of the
            contractions will be treated according to that string.
        kwargs : dict
            Other keyword arguments that will be used to construct the array.

        Returns
        -------
        array : np.ndarray
            Array associated with the atomic orbitals associated with the given set of contracted
            Cartesian Gaussians.

        Notes
        -----
        The next level of classes will be divided by number of indices associated with
        `GeneralizedContractionShell`. Then this method's parameters will likely be different from
        it's children. This means that using this method's API blindly (by simply copying and
        pasting) will not be suitable for all its children. Here, we allow arbitrary number of
        `transform` to indicate that its children may have different number of `transform` in
        its parameters. However, in actual implementations, the number of `transform` will likely
        be fixed, in which case, an arbitrary number of `transform` should not be accepted.

        """
