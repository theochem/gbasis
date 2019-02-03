.. image:: https://travis-ci.org/theochem/gbasis.svg?branch=master
    :target: https://travis-ci.org/theochem/gbasis
.. image:: https://anaconda.org/theochem/gbasis/badges/version.svg
    :target: https://anaconda.org/theochem/gbasis
.. image:: https://codecov.io/gh/theochem/gbasis/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/theochem/gbasis

GBasis
======
Python library for Gaussian basis function evaluation & integrals.


Installation
============

GBasis can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install gbasis

Alternatively, you can install GBasis in your home directory:

.. code:: bash

    pip install gbasis --user

Lastly, you can also install GBasis with conda. (See https://www.continuum.io/downloads)

.. code:: bash

    conda install -c theochem gbasis


Testing
=======

The tests can be executed as follows:

.. code:: bash

    nosetests -v gbasis


Background and usage
====================

*Put some more details here*


Release history
===============

- **2019-03-02** 0.0.0

  Initial Release


How to make a release (Github, PyPI and anaconda.org)
=====================================================

Before you do this, make sure everything is OK. The PyPI releases cannot be undone. If you
delete a file from PyPI (because of a mistake), you cannot upload the fixed file with the
same filename! See https://github.com/pypa/packaging-problems/issues/74

1. Update the release history.
2. Commit the final changes to master and push to github.
3. Wait for the CI tests to pass. Check if the README looks ok, etc. If needed, fix things
   and repeat step 2.
4. Make a git version tag: ``git tag <some_new_version>`` Follow the semantic versioning
   guidelines: http://semver.org
5. Push the tag to github: ``git push origin master --tags``
