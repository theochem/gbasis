/*
 * libcint_wrap.c — Python/C API bindings for libcint GTO integral library.
 *
 * This file wraps libcint's spherical (sph) integral functions using the
 * Python/C API. It exposes the following integrals to Python as the
 * `libcint_bindings` extension module:
 *
 * 1-electron integrals (spherical):
 *   overlap_sph          — int1e_ovlp  — overlap matrix S
 *   kinetic_sph          — int1e_kin   — kinetic energy matrix T
 *   nuclear_sph          — int1e_nuc   — nuclear attraction matrix V
 *   momentum_sph         — int1e_ipovlp — momentum integral p
 *   angular_momentum_sph — int1e_cg_irxp — angular momentum L
 *   rinv_sph             — int1e_rinv  — 1/r operator
 *   dipole_sph           — int1e_r     — dipole moment (order 1)
 *   quadrupole_sph       — int1e_rr    — quadrupole moment (order 2)
 *   octupole_sph         — int1e_rrr   — octupole moment (order 3)
 *
 * 2-electron integrals (spherical):
 *   electron_repulsion_sph — int2e — electron repulsion integrals (ERI)
 *
 * All wrapper functions accept pointer arguments as Python integers
 * (via PyLong_AsVoidPtr) — matching the calling convention used by
 * gbasis/integrals/libcint.py (ctypes-based high-level interface).
 *
 * The DEFINE_INTEGRAL_INT1e macro generates all 1-electron wrappers
 * from a single pattern — following the mentor's design in cint.h.
 *
 * References:
 *   - libcint paper: Qiming Sun, J. Comp. Chem., 2015, 36, 1664
 *   - libcint v6: Qiming Sun, J. Chem. Phys., 2024
 *   - GBasis Issue #229: https://github.com/theochem/gbasis/issues/229
 *   - Python/C API: https://docs.python.org/3/c-api/
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>


/* Forward declarations for libcint spherical integral functions.
 * Signature: (out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
 * - out: output buffer
 * - dims: dimensions of output
 * - shls: shell indices
 * - atm: atom info array
 * - natm: number of atoms
 * - bas: basis shell info array
 * - nbas: number of shells
 * - env: numerical data (coords, exponents, coeffs)
 * - opt: optimizer (NULL = no optimization)
 * - cache: work buffer (NULL = auto-allocate)
 */

/* Forward declarations — 1-electron integrals */
extern int int1e_ovlp_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_kin_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_nuc_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_ipovlp_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_cg_irxp_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_rinv_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_r_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_rr_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
extern int int1e_rrr_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);

/* Forward declaration — 2-electron integral */
extern int int2e_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);

/*
 * DEFINE_INTEGRAL_INT1e(func_name, libcint_func)
 * Generates a Python/C API wrapper for a 1-electron libcint integral.
 * Accepts NumPy arrays directly — uses PyArray_GETPTR1 (NumPy C-API).
 */

/* Macro for 1-electron wrappers */
#define DEFINE_INTEGRAL_INT1e(func_name, libcint_func)                        \
static PyObject *                                                              \
func_name(PyObject *self, PyObject *args)                                      \
{                                                                              \
    PyArrayObject *out_arr, *dims_arr, *shls_arr, *atm_arr, *bas_arr, *env_arr; \
    int natm, nbas;                                                            \
    if (!PyArg_ParseTuple(args, "O!O!O!O!iO!iO!",                             \
                          &PyArray_Type, &out_arr,                             \
                          &PyArray_Type, &dims_arr,                            \
                          &PyArray_Type, &shls_arr,                            \
                          &PyArray_Type, &atm_arr, &natm,                      \
                          &PyArray_Type, &bas_arr, &nbas,                      \
                          &PyArray_Type, &env_arr))                            \
        return NULL;                                                           \
    double *out  = (double *)PyArray_GETPTR1(out_arr,  0);                     \
    int    *dims = (int *)   PyArray_GETPTR1(dims_arr, 0);                     \
    int    *shls = (int *)   PyArray_GETPTR1(shls_arr, 0);                     \
    int    *atm  = (int *)   PyArray_GETPTR2(atm_arr,  0, 0);                  \
    int    *bas  = (int *)   PyArray_GETPTR2(bas_arr,  0, 0);                  \
    double *env  = (double *)PyArray_GETPTR1(env_arr,  0);                     \
    int result = libcint_func(out, dims, shls, atm, natm,                      \
                              bas, nbas, env, NULL, NULL);                     \
    return PyLong_FromLong(result);                                            \
}

DEFINE_INTEGRAL_INT1e(overlap_sph,         int1e_ovlp_sph)
DEFINE_INTEGRAL_INT1e(kinetic_sph,         int1e_kin_sph)
DEFINE_INTEGRAL_INT1e(nuclear_sph,         int1e_nuc_sph)
DEFINE_INTEGRAL_INT1e(momentum_sph,        int1e_ipovlp_sph)
DEFINE_INTEGRAL_INT1e(angular_momentum_sph, int1e_cg_irxp_sph)
DEFINE_INTEGRAL_INT1e(rinv_sph,            int1e_rinv_sph)
DEFINE_INTEGRAL_INT1e(dipole_sph,          int1e_r_sph)
DEFINE_INTEGRAL_INT1e(quadrupole_sph,      int1e_rr_sph)
DEFINE_INTEGRAL_INT1e(octupole_sph,        int1e_rrr_sph)

/* 2-electron wrapper */

/*
 * electron_repulsion_sph — 2-electron ERI wrapper.
 * Uses PyArray_GETPTR1/2 for NumPy array access.
 * Same signature as 1-electron but uses int2e_sph (4-center integral).
 */

static PyObject *
electron_repulsion_sph(PyObject *self, PyObject *args)
{
    PyArrayObject *out_arr, *dims_arr, *shls_arr, *atm_arr, *bas_arr, *env_arr;
    int natm, nbas;
    if (!PyArg_ParseTuple(args, "O!O!O!O!iO!iO!",
                          &PyArray_Type, &out_arr,
                          &PyArray_Type, &dims_arr,
                          &PyArray_Type, &shls_arr,
                          &PyArray_Type, &atm_arr, &natm,
                          &PyArray_Type, &bas_arr, &nbas,
                          &PyArray_Type, &env_arr))
        return NULL;
    double *out  = (double *)PyArray_GETPTR1(out_arr,  0);
    int    *dims = (int *)   PyArray_GETPTR1(dims_arr, 0);
    int    *shls = (int *)   PyArray_GETPTR1(shls_arr, 0);
    int    *atm  = (int *)   PyArray_GETPTR2(atm_arr,  0, 0);
    int    *bas  = (int *)   PyArray_GETPTR2(bas_arr,  0, 0);
    double *env  = (double *)PyArray_GETPTR1(env_arr,  0);
    int result = int2e_sph(out, dims, shls, atm, natm,
                           bas, nbas, env, NULL, NULL);
    return PyLong_FromLong(result);
}

static PyMethodDef LibcintMethods[] = {
    {"overlap_sph",          overlap_sph,          METH_VARARGS, "Overlap integral"},
    {"kinetic_sph",          kinetic_sph,          METH_VARARGS, "Kinetic energy integral"},
    {"nuclear_sph",          nuclear_sph,          METH_VARARGS, "Nuclear attraction integral"},
    {"momentum_sph",         momentum_sph,         METH_VARARGS, "Momentum integral"},
    {"angular_momentum_sph", angular_momentum_sph, METH_VARARGS, "Angular momentum integral"},
    {"rinv_sph",             rinv_sph,             METH_VARARGS, "1/r integral"},
    {"dipole_sph",           dipole_sph,           METH_VARARGS, "Dipole moment integral"},
    {"quadrupole_sph",       quadrupole_sph,       METH_VARARGS, "Quadrupole moment integral"},
    {"octupole_sph",         octupole_sph,         METH_VARARGS, "Octupole moment integral"},
    {"electron_repulsion_sph", electron_repulsion_sph, METH_VARARGS, "Electron repulsion integral"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libcintmodule = {
    PyModuleDef_HEAD_INIT,
    "libcint_bindings",
    NULL,
    -1,
    LibcintMethods
};

PyMODINIT_FUNC
PyInit_libcint_bindings(void)
{
    import_array();
    return PyModule_Create(&libcintmodule);
}
