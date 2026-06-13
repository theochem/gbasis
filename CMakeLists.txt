# CMakeLists.txt — Build C extension for GBasis libcint integration  (Issue: #229)

# Usage:
#   Default (all platforms):  pip install gbasis
#   qcint (x86+AVX2 only):   pip install gbasis -C cmake.args="-DUSE_QCINT=ON"


cmake_minimum_required(VERSION 3.17)
project(gbasis_libcint C)

# Python and NumPy are required to build the C extension module

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Python COMPONENTS NumPy REQUIRED)

# Get NumPy include path for C extension (numpy/arrayobject.h)

execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c"
            "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ── Platform Detection — x86 vs ARM ──────────────────────
# Detect platform: use qcint on x86+AVX2, libcint on ARM/non-AVX
# qcint is an optimized version of libcint for x86 with SIMD support
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2" HAS_AVX2)

if(HAS_AVX2 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(USE_QCINT ON)
    message(STATUS "Architecture: x86 + AVX2 detected → use qcint")
else()
    set(USE_QCINT OFF)
    message(STATUS "Architecture: ARM/non-AVX detected → use libcint")
endif()


# Download libcint or qcint based on platform detection
# Both are pinned to v6.1.2 for stability
# qcint: optimized for x86+AVX2
# libcint: generic, works on all platforms

include(FetchContent)

if(USE_QCINT)
    FetchContent_Declare(
        libcint
        GIT_REPOSITORY https://github.com/sunqm/qcint.git
        GIT_TAG        v6.1.2
    )
else()
    FetchContent_Declare(
        libcint
        GIT_REPOSITORY https://github.com/sunqm/libcint.git
        GIT_TAG        v6.1.2
    )
endif()

# Compile libcint — Fortran disabled, range-separated Coulomb enabled

set(WITH_FORTRAN OFF CACHE BOOL "" FORCE)
# Enable range-separated Coulomb integrals
set(WITH_RANGE_COULOMB ON CACHE BOOL "" FORCE)

# Redirect libcint's install to build directory to avoid permission issues
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/local" CACHE PATH "" FORCE)
set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libcint)

# Build Python C extension module from libcint_wrap.c
python_add_library(libcint_bindings MODULE
    gbasis/integrals/src/libcint_wrap.c
    WITH_SOABI
)

# Include NumPy and libcint headers
target_include_directories(libcint_bindings PRIVATE
    ${NumPy_INCLUDE_DIRS}
    ${libcint_SOURCE_DIR}/include
)

# Link C extension with libcint
target_link_libraries(libcint_bindings PRIVATE cint)

# Prevent libcint from installing to system directories
set(CINT_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/cint_install CACHE PATH "" FORCE)

# Set RPATH so libcint.dylib is found at runtime
set_target_properties(libcint_bindings PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install only the Python extension into the wheel
install(TARGETS libcint_bindings
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/gbasis/integrals/lib
)

# Install libcint shared library alongside the extension
install(FILES $<TARGET_FILE:cint>
    DESTINATION ${SKBUILD_PLATLIB_DIR}/gbasis/integrals/lib
)