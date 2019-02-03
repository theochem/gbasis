#!/usr/bin/env python
# -*- coding: utf-8 -*-
# {{cookiecutter.description}} 
# Copyright (C) {% now 'local', '%Y' %} {{cookiecutter.author}} <{{cookiecutter.email}}>
#
# This file is part of {{cookiecutter.project_name}}.
#
# {{cookiecutter.project_name}} is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# {{cookiecutter.project_name}} is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""GBasis Setup Script.

If you are not familiar with setup.py, just use pip instead:

    pip install gbasis --user --upgrade

Alternatively, you can install from source with

    ./setup.py install --user
"""


from setuptools import setup


def get_version():
    """Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable defining the
    version string with single quotes.

    """
    try:
        with open('gbasis/version.py', 'r') as f:
            return f.read().split('=')[-1].replace('\'', '').strip()
    except IOError:
        return "0.0.1"

def readme():
    """Load README.rst for display on PyPI."""
    with open('README.rst') as f:
        return f.read()


setup(
    name='gbasis',
    version=get_version(),
    description='Python library for Gaussian basis function evaluation & integrals.',
    long_description=readme(),
    author='HORTON-ChemTools Dev Team',
    author_email='horton.chemtools@gmail.com',
    url='https://github.com/theochem/gbasis',
    packages=['gbasis'],
    zip_safe=False,
)
