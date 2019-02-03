# -*- coding: utf-8 -*-
# GBasis: Python library for Gaussian basis function evaluation & integrals.
#
# Copyright (C) 2019 HORTON-ChemTools Dev Team <horton.chemtools@gmail.com>.
#
# This file is part of GBasis.
#
# GBasis is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GBasis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Input and Output Module for Gaussian Basis Sets."""


from typing import List


__all__ = [
    'angmoms_str_to_num', 'angmoms_num_to_str',
]


def angmoms_str_to_num(angmoms: str, pure: bool = False) -> List[int]:
    """Convert angular momentum string of letters into a list of angular momentum quantum numbers.

    Parameters
    ----------
    angmoms
        Angular momentum string of letters of basis functions.
    pure
        Type of basis functions.

    """
    if pure:
        d = {'s': 0, 'p': 1, 'd': -2, 'f': -3, 'g': -4, 'h': -5, 'i': -6}
    else:
        d = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    return [d[angmom] for angmom in angmoms.lower()]


def angmoms_num_to_str(angmoms: List[int]) -> str:
    """Convert an angular momentum quantum number to a letter symbol.

    Parameters
    ----------
    angmoms
        Sequence of Angular momentum quantum numbers.

    """
    d = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i'}
    return ''.join([d[abs(angmom)] for angmom in angmoms])
