"""Utility functions for running tests."""


def skip_init(class_obj):
    """Return instance of the given class without initialization.

    Parameters
    ----------
    class_obj : type
        Class.

    Returns
    -------
    instance : class_obj
        Instance of the given class without intialization.

    """

    class NoInitClass(class_obj):
        """Class {} without the __init__."""

        def __init__(self):
            """Null initialization."""
            pass

    NoInitClass.__name__ = "NoInit{}".format(class_obj.__name__)
    NoInitClass.__doc__ = NoInitClass.__doc__.format(class_obj.__name__)
    return NoInitClass()
