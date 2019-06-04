"""Setup for gbasis.

See:
https://packaging.python.org/en/latest/distributing.html
Templated from:
https://github.com/pypa/sampleproject
"""

from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gbasis",
    version="0.0.0",
    description="A module for evaluating, differentiating, and integrating Gaussian functions.",
    long_description=long_description,
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    long_description_content_type="text/markdown",
    url="https://github.com/theochem/gbasis",
    # This should be your name or the name of the organization which owns the
    # project.
    # author='',
    # This should be a valid email address corresponding to the author listed
    # above.
    # author_email='',
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        # How mature is this project? Common values are
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        # Pick your license as you wish
        "License :: OSI Approved :: GNU Version 3",
        # Specify the Python versions you support here.
        # These classifiers are *not* checked by 'pip install'. See instead
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="gaussian primitive contraction orbital integral differentiation evaluation density ",
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.5",
    install_requires=["numpy>=1.10", "scipy>=1.0"],
    extras_require={
        "dev": [
            "tox",
            "flake8",
            "flake8-pydocstyle",
            "flake8-import-order",
            "pep8-naming",
            "pylint",
            "bandit",
            "pytest",
            "pytest-cov",
        ],
        "test": ["tox", "pytest", "pytest-cov"],
        "iodata": ["tox", "iodata>0.1.7"],
        "pyscf": ["tox", "pyscf>=1.6.1"],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={},
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
    # List additional URLs that are relevant to your project as a dict.
    project_urls={
        "Bug Reports": "https://github.com/theochem/gbasis/issues",
        "Organization": "https://github.com/theochem/",
        "Source": "https://github.com/theochem/gbasis/",
    },
)
