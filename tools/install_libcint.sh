#!/usr/bin/env bash

# Notice for user
echo ''
echo '    NOTE: If this script has been run previously, and it failed, please ensure'
echo '          the `build/libcint` directory is removed before running it again.'
echo ''
echo '    See the GBasis documentation for help:'
echo ''
echo '          https://theochem.github.io/gbasis/PLACEHOLDER/'
echo ''

# Terminate script if any command fails
set -e

# Get GBasis base directory name unambiguously
gbasis_dir=$(cd "$(dirname "$(dirname "${0}")")" && pwd)

# Cleanup function; runs on failure or exit
cleanup() {
    rm -rf "${gbasis_dir}/build/libcint"
}
trap cleanup EXIT

# Set Libcint CMake options
lc_cmake_opts=
lc_cmake_opts+=' -DWITH_FORTRAN=0'
lc_cmake_opts+=' -DWITH_CINT2_INTERFACE=0'
lc_cmake_opts+=' -DWITH_RANGE_COULOMB=1'
lc_cmake_opts+=' -DWITH_POLYNOMIAL_FIT=1'
lc_cmake_opts+=' -DWITH_F12=1'
lc_cmake_opts+=' -DPYPZPX=1'

# Ensure build directory exists and enter it
mkdir -p "${gbasis_dir}/build"
cd "${gbasis_dir}/build"

# Clone Libcint Git repo and enter it
git clone http://github.com/sunqm/libcint.git
cd libcint

# Setup Libcint build
mkdir build
cd build
cmake "-DCMAKE_INSTALL_PREFIX=${gbasis_dir}/gbasis/integrals" ${lc_cmake_opts} ..

# Compile and install Libcint
cmake --build .
cmake --install .
