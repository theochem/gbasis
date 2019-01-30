# -*- coding: utf-8 -*-
# GBASIS: Python library for Gaussian basis evaluation & integrals.
#
# Copyright (C) 2019 The GBASIS Development Team
#
# This file is part of GBASIS.
#
# GBASIS is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GBASIS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Test gbasis.io"""


from numpy.testing import assert_equal

from .. io import angmoms_str_to_num, angmoms_num_to_str


def test_angmom_str_to_num_cart():
    assert_equal(angmoms_str_to_num('s'), [0])
    assert_equal(angmoms_str_to_num('ss'), [0, 0])
    assert_equal(angmoms_str_to_num('SS'), [0, 0])
    assert_equal(angmoms_str_to_num('Ss'), [0, 0])
    assert_equal(angmoms_str_to_num('sS'), [0, 0])
    assert_equal(angmoms_str_to_num('sp'), [0, 1])
    assert_equal(angmoms_str_to_num('SP'), [0, 1])
    assert_equal(angmoms_str_to_num('sP'), [0, 1])
    assert_equal(angmoms_str_to_num('Sp'), [0, 1])
    assert_equal(angmoms_str_to_num('SDD'), [0, 2, 2])
    assert_equal(angmoms_str_to_num('Pdf'), [1, 2, 3])
    assert_equal(angmoms_str_to_num('SpDf'), [0, 1, 2, 3])
    assert_equal(angmoms_str_to_num('IHDFG'), [6, 5, 2, 3, 4])


def test_angmom_str_to_num_prue():
    assert_equal(angmoms_str_to_num('s', True), [0])
    assert_equal(angmoms_str_to_num('ss', True), [0, 0])
    assert_equal(angmoms_str_to_num('SS', True), [0, 0])
    assert_equal(angmoms_str_to_num('Ss', True), [0, 0])
    assert_equal(angmoms_str_to_num('sS', True), [0, 0])
    assert_equal(angmoms_str_to_num('sp', True), [0, 1])
    assert_equal(angmoms_str_to_num('SP', True), [0, 1])
    assert_equal(angmoms_str_to_num('sP', True), [0, 1])
    assert_equal(angmoms_str_to_num('Sp', True), [0, 1])
    assert_equal(angmoms_str_to_num('SDD', True), [0, -2, -2])
    assert_equal(angmoms_str_to_num('Pdf', True), [1, -2, -3])
    assert_equal(angmoms_str_to_num('SpDf', True), [0, 1, -2, -3])
    assert_equal(angmoms_str_to_num('IHDFG', True), [-6, -5, -2, -3, -4])


def test_angmom_num_to_str():
    assert_equal(angmoms_num_to_str([0]), 's')
    assert_equal(angmoms_num_to_str([0, 1]), 'sp')
    assert_equal(angmoms_num_to_str([2]), 'd')
    assert_equal(angmoms_num_to_str([-2]), 'd')
    assert_equal(angmoms_num_to_str([1, -2, 3]), 'pdf')
    assert_equal(angmoms_num_to_str([1, -2, -3]), 'pdf')
    assert_equal(angmoms_num_to_str([0, -2, 4]), 'sdg')
    assert_equal(angmoms_num_to_str([2, 3, 4, 5]), 'dfgh')
    assert_equal(angmoms_num_to_str([-2, 4, 1, 0, -3]), 'dgpsf')
