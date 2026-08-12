
# Require (a relatively recent) version of CMake.
cmake_minimum_required(VERSION 3.18)

# Force MinGW compiler on Windows before project() declaration
if(WIN32)
    set(CMAKE_C_COMPILER "gcc" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "" FORCE)
endif()

# Declare project.
project(gbasis LANGUAGES C)

# Import necessary Python modules.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Use qcint on x86 systems with (at least) SSE3, and libcint on non-x86/non-SSE3 systems.
# qcint is an optimized version of libcint for x86 systems with SIMD support;
# both are pinned to v6.1.2 for stability.
include(CheckCCompilerFlag)
set(GBASIS_CINT_REPO "https://github.com/sunqm/libcint.git")
set(GBASIS_CINT_VERSION "v6.1.2")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(i386)|(i686)")
    check_c_compiler_flag("-msse3" GBASIS_HAS_SSE3)
    if(GBASIS_HAS_SSE3)
        message(STATUS "Architecture: x86 with (at least) SSE3 -> use qcint")
        set(GBASIS_CINT_REPO "https://github.com/sunqm/qcint.git")
    else()
        message(STATUS "Architecture: x86 without SSE3 -> use libcint")
    endif()
else()
    message(STATUS "Architecture: not x86 -> use libcint")
endif()

# Download libcint or qcint based on platform.
include(FetchContent)
FetchContent_Declare(cint
    GIT_REPOSITORY ${GBASIS_CINT_REPO}
    GIT_TAG ${GBASIS_CINT_VERSION}
    GIT_SHALLOW
    EXCLUDE_FROM_ALL)

# Configure libcint/qcint.
# On Windows, build shared library so libcint_bindings.pyd can load it at runtime.
# Static everywhere — no runtime DLL issues
set(ENABLE_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WITH_FORTRAN OFF)
set(WITH_RANGE_COULOMB ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)  

# Import libcint/qcint CMake project.
FetchContent_MakeAvailable(cint)

# Declare C extension module.
set(GBASIS_EXTENSION_SOURCE "gbasis/integrals/src/libcint_wrap.c")
python_add_library(gbasis_ext MODULE WITH_SOABI ${GBASIS_EXTENSION_SOURCE})

# Set C extension module properties.
set_target_properties(gbasis_ext PROPERTIES
    # Set target library output name.
    OUTPUT_NAME libcint_bindings
    # Set RPATH so libcint.dylib is found at runtime.
    INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
    BUILD_WITH_INSTALL_RPATH TRUE)

# Expose Python and NumPy include directories to C extension module.
target_include_directories(gbasis_ext PRIVATE
    ${cint_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

# Link C extension module with libcint/qcint.
target_link_libraries(gbasis_ext PRIVATE cint)

# On Windows/MinGW, statically link runtime to avoid DLL dependency issues.
if(WIN32 AND MINGW)
    target_link_options(gbasis_ext PRIVATE
        -static-libgcc
        -static-libstdc++
        -Wl,-Bstatic
        -lpthread
        -Wl,-Bdynamic
    )
endif()

# Install C extension module.
set(GBASIS_WHEEL_DIRECTORY "${SKBUILD_PLATLIB_DIR}/gbasis/integrals/lib")
install(TARGETS gbasis_ext LIBRARY DESTINATION ${GBASIS_WHEEL_DIRECTORY})



# Install libcint shared library alongside the extension module.
install(FILES $<TARGET_FILE:cint>
    DESTINATION ${GBASIS_WHEEL_DIRECTORY})


