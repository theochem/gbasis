#!/usr/bin/env bash

# Libcint version to patch and install
LIBCINT_VERSION=v6.1.0

if [ -z ${USE_LIBCINT:-} ]; then
    cint_name=qcint
else
    cint_name=libcint
fi

# Notice for user
echo ""
echo "    NOTE: If this script has been run previously, and it failed, please ensure"
echo "          the \`build/${cint_name}/build\` directory is removed before running it again."
echo ""
echo "    See the GBasis documentation for help:"
echo ""
echo "          https://theochem.github.io/gbasis/PLACEHOLDER/"
echo ""

# Check that a Common Lisp program is installed
if command -v sbcl > /dev/null; then
    lisp_program='sbcl --script'
elif command -v clisp > /dev/null; then
    lisp_program='clisp'
else
    echo "ERROR: This script requires a Common Lisp interpreter."
    echo "       Please install either sbcl or clisp and try again."
    echo ""
    exit 1
fi

# Terminate script if any command fails
set -e

# Get GBasis base directory name unambiguously
gbasis_dir=$(cd "$(dirname "$(dirname "${0}")")" && pwd)

# Cleanup function; runs on failure or exit
cleanup() {
    rm -rf "${gbasis_dir}/build/${cint_name}/build"
}
trap cleanup EXIT

# Set Libcint CMake options
lc_cmake_opts=
lc_cmake_opts+=' -DWITH_FORTRAN=0'
lc_cmake_opts+=' -DWITH_CINT2_INTERFACE=0'
lc_cmake_opts+=' -DWITH_RANGE_COULOMB=1'

# Ensure build directory exists and enter it
mkdir -p "${gbasis_dir}/build"
cd "${gbasis_dir}/build"

# Clone Libcint Git repo and enter it
if [ ! -e "${gbasis_dir}/build/${cint_name}" ]; then
    git clone --branch ${LIBCINT_VERSION} --depth 1 https://github.com/sunqm/${cint_name}.git
fi
cd "${cint_name}"
git checkout .

# Auto-generate integrals
cd scripts
python ../../../tools/auto_intor_modify.py auto_intor.cl
${lisp_program} auto_intor.cl
mv *.c ../src/autocode
cd ..

# Setup Libcint build
mkdir build
cd build
cmake "-DCMAKE_INSTALL_PREFIX=${gbasis_dir}/gbasis/integrals" ${lc_cmake_opts} ..

# Compile and install Libcint
make
make install
