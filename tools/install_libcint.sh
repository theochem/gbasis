#!/usr/bin/env bash

# Notice for user
echo ''
echo '    NOTE: If this script has been run previously, and it failed, please'
echo '          remove the `build/libcint` directory before running it again.'
echo ''
echo '    See the GBasis documentation for help:'
echo ''
echo '          https://theochem.github.io/gbasis/PLACEHOLDER/'
echo ''

# Terminate script if any command fails
set -e

# Get GBasis base directory name unambiguously
gbasis_dir=$(cd "$(dirname $(dirname ${0}))" && pwd)

# Ensure build directory exists and enter it
mkdir -p ${gbasis_dir}/build
cd ${gbasis_dir}/build

# Clone Libcint Git repo and enter it
git clone http://github.com/sunqm/libcint.git
cd libcint

# Setup Libcint build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${gbasis_dir}/gbasis/integrals ..

# Compile and install Libcint
cmake --build .
cmake --install .
