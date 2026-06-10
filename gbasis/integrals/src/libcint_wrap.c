#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

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

/* Macro for 1-electron wrappers */
#define DEFINE_INTEGRAL_INT1e(func_name, libcint_func)                        \
static PyObject *                                                              \
func_name(PyObject *self, PyObject *args)                                      \
{                                                                              \
    PyObject *out_obj, *dims_obj, *shls_obj, *atm_obj, *bas_obj, *env_obj;    \
    int natm, nbas;                                                            \
    if (!PyArg_ParseTuple(args, "OOOOiOiO",                                   \
                          &out_obj, &dims_obj, &shls_obj,                      \
                          &atm_obj, &natm, &bas_obj, &nbas, &env_obj))         \
        return NULL;                                                           \
    double *out   = (double *)PyLong_AsVoidPtr(out_obj);                       \
    int    *dims  = (int *)PyLong_AsVoidPtr(dims_obj);                         \
    int    *shls  = (int *)PyLong_AsVoidPtr(shls_obj);                         \
    int    *atm   = (int *)PyLong_AsVoidPtr(atm_obj);                          \
    int    *bas   = (int *)PyLong_AsVoidPtr(bas_obj);                          \
    double *env   = (double *)PyLong_AsVoidPtr(env_obj);                       \
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
static PyObject *
electron_repulsion_sph(PyObject *self, PyObject *args)
{
    PyObject *out_obj, *dims_obj, *shls_obj, *atm_obj, *bas_obj, *env_obj;
    int natm, nbas;
    if (!PyArg_ParseTuple(args, "OOOOiOiO",
                          &out_obj, &dims_obj, &shls_obj,
                          &atm_obj, &natm, &bas_obj, &nbas, &env_obj))
        return NULL;
    double *out   = (double *)PyLong_AsVoidPtr(out_obj);
    int    *dims  = (int *)PyLong_AsVoidPtr(dims_obj);
    int    *shls  = (int *)PyLong_AsVoidPtr(shls_obj);
    int    *atm   = (int *)PyLong_AsVoidPtr(atm_obj);
    int    *bas   = (int *)PyLong_AsVoidPtr(bas_obj);
    double *env   = (double *)PyLong_AsVoidPtr(env_obj);
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
    return PyModule_Create(&libcintmodule);
}