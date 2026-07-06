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
/* CINTOpt forward declaration */
typedef struct CINTOpt CINTOpt;
extern void CINTdel_optimizer(CINTOpt **);

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

/* Optimizer forward declarations */
extern void int1e_ovlp_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_kin_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_nuc_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_ipovlp_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_cg_irxp_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_rinv_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_r_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_rr_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void int1e_rrr_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void cint2e_sph_optimizer(CINTOpt **, int *, int, int *, int, double *);
extern void CINTall_1e_optimizer(CINTOpt **, int *, int, int *, int, double *);
/* Forward declaration — 2-electron integral */
extern int int2e_sph(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas, double *env, void *opt, double *cache);
/* Optimizer macro using token pasting */
#define MAKE_OPTIMIZER(func) c##func##_optimizer

/*
 * DEFINE_INT1E_ARRAY_FN(func_name, libcint_func)
 * Generates a Python/C API wrapper for a 1-electron libcint integral.
 * Accepts NumPy arrays directly — uses PyArray_GETPTR1 (NumPy C-API).
 */

/* Macro for 1-electron wrappers */
#define DEFINE_INT1E_ARRAY_FN(func_name, libcint_func)                        \
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

DEFINE_INT1E_ARRAY_FN(overlap_sph,         int1e_ovlp_sph)
DEFINE_INT1E_ARRAY_FN(kinetic_sph,         int1e_kin_sph)
DEFINE_INT1E_ARRAY_FN(nuclear_sph,         int1e_nuc_sph)
DEFINE_INT1E_ARRAY_FN(momentum_sph,        int1e_ipovlp_sph)
DEFINE_INT1E_ARRAY_FN(angular_momentum_sph, int1e_cg_irxp_sph)
DEFINE_INT1E_ARRAY_FN(rinv_sph,            int1e_rinv_sph)
DEFINE_INT1E_ARRAY_FN(dipole_sph,          int1e_r_sph)
DEFINE_INT1E_ARRAY_FN(quadrupole_sph,      int1e_rr_sph)
DEFINE_INT1E_ARRAY_FN(octupole_sph,        int1e_rrr_sph)

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

/* Macro for integrals WITH optimizer support */
#define DEFINE_INT1E_LOOP_FN_OPT(func_name, libcint_func,opt_func)                         \
static PyObject *                                                                   \
func_name(PyObject *self, PyObject *args)                                           \
{                                                                                   \
    PyArrayObject *out_arr, *atm_arr, *bas_arr, *env_arr, *offs_arr;               \
    int natm, nbas, nbfn;                                                           \
    if (!PyArg_ParseTuple(args, "O!iO!iO!O!O!i",                                   \
                          &PyArray_Type, &out_arr, &natm,                           \
                          &PyArray_Type, &atm_arr, &nbas,                           \
                          &PyArray_Type, &bas_arr,                                  \
                          &PyArray_Type, &env_arr,                                  \
                          &PyArray_Type, &offs_arr,                                 \
                          &nbfn))                                                   \
        return NULL;                                                                \
    double *out  = (double *)PyArray_DATA(out_arr);                                \
    int    *atm  = (int *)   PyArray_GETPTR2(atm_arr, 0, 0);                       \
    int    *bas  = (int *)   PyArray_GETPTR2(bas_arr, 0, 0);                       \
    double *env  = (double *)PyArray_GETPTR1(env_arr, 0);                          \
    int    *offs = (int *)   PyArray_GETPTR1(offs_arr, 0);                         \
    int max_off = 0;                                                                \
    for (int i = 0; i < nbas; i++) {                                               \
        if (offs[i] > max_off) max_off = offs[i];                                  \
    }                                                                               \
    size_t buf_size = (size_t)max_off * max_off;                                   \
    double *buf = calloc(buf_size, sizeof(double));                                \
    if (!buf) { PyErr_NoMemory(); return NULL; }                                   \
    CINTOpt *opt = NULL;                                                            \
    opt_func##_optimizer(&opt, atm, natm, bas, nbas, env);      \
    int shls[2];                                                                    \
    int ipos = 0;                                                                   \
    for (int ishl = 0; ishl < nbas; ishl++) {                                      \
        shls[0] = ishl;                                                             \
        int p_off = offs[ishl];                                                     \
        int jpos = 0;                                                               \
        for (int jshl = 0; jshl <= ishl; jshl++) {                                 \
            shls[1] = jshl;                                                         \
            int q_off = offs[jshl];                                                 \
            libcint_func(buf, NULL, shls, atm, natm, bas, nbas, env, opt, NULL);   \
            for (int p = 0; p < p_off; p++) {                                      \
                for (int q = 0; q < q_off; q++) {                                  \
                    double val = buf[p + q * p_off];                               \
                    out[(ipos+p) * nbfn + (jpos+q)] = val;                         \
                    out[(jpos+q) * nbfn + (ipos+p)] = val;                         \
                }                                                                   \
            }                                                                       \
            memset(buf, 0, buf_size * sizeof(double));                              \
            jpos += q_off;                                                          \
        }                                                                           \
        ipos += p_off;                                                              \
    }                                                                               \
    CINTdel_optimizer(&opt);                                                        \
    free(buf);                                                                      \
    Py_RETURN_NONE;                                                                 \
}

/*
 * DEFINE_INT1E_LOOP_FN(func_name, libcint_func)
 * Generates a C shell-loop wrapper for a 1-electron libcint integral that
 * returns the FULL integral array over all shells (not just one shell pair).
 * Loops over shells I, J; calls libcint_func per shell pair; fills the
 * symmetric output matrix using PyArray_GETPTR (NumPy C-API).
 */
#define DEFINE_INT1E_LOOP_FN(func_name, libcint_func, opt_func)                    \
static PyObject *                                                                   \
func_name(PyObject *self, PyObject *args)                                           \
{                                                                                   \
    PyArrayObject *out_arr, *atm_arr, *bas_arr, *env_arr, *offs_arr;               \
    int natm, nbas, nbfn;                                                           \
    if (!PyArg_ParseTuple(args, "O!iO!iO!O!O!i",                                   \
                          &PyArray_Type, &out_arr, &natm,                           \
                          &PyArray_Type, &atm_arr, &nbas,                           \
                          &PyArray_Type, &bas_arr,                                  \
                          &PyArray_Type, &env_arr,                                  \
                          &PyArray_Type, &offs_arr,                                 \
                          &nbfn))                                                   \
        return NULL;                                                                \
    double *out  = (double *)PyArray_DATA(out_arr);                                \
    int    *atm  = (int *)   PyArray_GETPTR2(atm_arr, 0, 0);                       \
    int    *bas  = (int *)   PyArray_GETPTR2(bas_arr, 0, 0);                       \
    double *env  = (double *)PyArray_GETPTR1(env_arr, 0);                          \
    int    *offs = (int *)   PyArray_GETPTR1(offs_arr, 0);                         \
    int shls[2];                                                                    \
    int max_off = 0;                                                                \
    for (int i = 0; i < nbas; i++) {                                               \
        if (offs[i] > max_off) max_off = offs[i];                                  \
    }                                                                               \
    size_t buf_size = (size_t)max_off * max_off * 27;                                   \
    double *buf = calloc(buf_size, sizeof(double));                                \
    if (!buf) { PyErr_NoMemory(); return NULL; }                                   \
    CINTOpt *opt = NULL;                                                            \
    opt_func##_optimizer(&opt, atm, natm, bas, nbas, env);                         \
    int ipos = 0;                                                                   \
    for (int ishl = 0; ishl < nbas; ishl++) {                                      \
        shls[0] = ishl;                                                             \
        int p_off = offs[ishl];                                                     \
        int jpos = 0;                                                               \
        for (int jshl = 0; jshl <= ishl; jshl++) {                                 \
            shls[1] = jshl;                                                         \
            int q_off = offs[jshl];                                                 \
            libcint_func(buf, NULL, shls, atm, natm, bas, nbas, env, opt, NULL);   \
            for (int p = 0; p < p_off; p++) {                                      \
                for (int q = 0; q < q_off; q++) {                                  \
                    double val = buf[p + q * p_off];                               \
                    out[(ipos+p) * nbfn + (jpos+q)] = val;                         \
                    out[(jpos+q) * nbfn + (ipos+p)] = val;                         \
                }                                                                   \
            }                                                                       \
            memset(buf, 0, buf_size * sizeof(double));                              \
            jpos += q_off;                                                          \
        }                                                                           \
        ipos += p_off;                                                              \
    }                                                                               \
    CINTdel_optimizer(&opt);                                                        \
    free(buf);                                                                      \
    Py_RETURN_NONE;                                                                 \
}

/* Generate shell-loop wrappers for all 1-electron integrals using the macro */
DEFINE_INT1E_LOOP_FN_OPT(overlap_integral_array, int1e_ovlp_sph, int1e_ovlp)
DEFINE_INT1E_LOOP_FN_OPT(kinetic_integral_array, int1e_kin_sph,  int1e_kin)
DEFINE_INT1E_LOOP_FN_OPT(nuclear_integral_array, int1e_nuc_sph,  int1e_nuc)
DEFINE_INT1E_LOOP_FN(momentum_integral_array,   int1e_ipovlp_sph, int1e_ipovlp)
DEFINE_INT1E_LOOP_FN(rinv_integral_array,       int1e_rinv_sph,   int1e_rinv)
DEFINE_INT1E_LOOP_FN(dipole_integral_array,     int1e_r_sph,      int1e_r)
DEFINE_INT1E_LOOP_FN(quadrupole_integral_array, int1e_rr_sph,     int1e_rr)
DEFINE_INT1E_LOOP_FN(octupole_integral_array,   int1e_rrr_sph,    int1e_rrr)

/* eri_array — array-based loop in C for 2-electron ERI (4 shells: I,J,K,L) */
static PyObject *
eri_array(PyObject *self, PyObject *args)
{
    PyArrayObject *out_arr, *atm_arr, *bas_arr, *env_arr, *offs_arr;
    int natm, nbas, nbfn;

    if (!PyArg_ParseTuple(args, "O!iO!iO!O!O!i",
                          &PyArray_Type, &out_arr,
                          &natm,
                          &PyArray_Type, &atm_arr,
                          &nbas,
                          &PyArray_Type, &bas_arr,
                          &PyArray_Type, &env_arr,
                          &PyArray_Type, &offs_arr,
                          &nbfn))
        return NULL;

    double *out  = (double *)PyArray_DATA(out_arr);
    int    *atm  = (int *)   PyArray_GETPTR2(atm_arr, 0, 0);
    int    *bas  = (int *)   PyArray_GETPTR2(bas_arr, 0, 0);
    double *env  = (double *)PyArray_GETPTR1(env_arr, 0);
    int    *offs = (int *)   PyArray_GETPTR1(offs_arr, 0);

    int shls[4];
    int max_off = 0;
    for (int i = 0; i < nbas; i++) {
        if (offs[i] > max_off) max_off = offs[i];
    }
    size_t buf_size = (size_t)max_off * max_off * max_off * max_off;
    double *buf = calloc(buf_size, sizeof(double));
    if (!buf) { PyErr_NoMemory(); return NULL; }
    CINTOpt *opt = NULL;
    cint2e_sph_optimizer(&opt, atm, natm, bas, nbas, env);
    int ipos = 0;
    for (int ishl = 0; ishl < nbas; ishl++) {
        shls[0] = ishl;
        int p_off = offs[ishl];
        int jpos = 0;
        for (int jshl = 0; jshl <= ishl; jshl++) {
            int ij = ((ishl + 1) * ishl) / 2 + jshl;
            shls[1] = jshl;
            int q_off = offs[jshl];
            int kpos = 0;
            for (int kshl = 0; kshl < nbas; kshl++) {
                shls[2] = kshl;
                int r_off = offs[kshl];
                int lpos = 0;
                for (int lshl = 0; lshl <= kshl; lshl++) {
                    int kl = ((kshl + 1) * kshl) / 2 + lshl;
                    if (ij < kl) {
                        lpos += offs[lshl];
                        continue;
                    }
                    shls[3] = lshl;
                    int s_off = offs[lshl];
                    int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, NULL);
                    for (int p = 0; p < p_off; p++) {
                        for (int q = 0; q < q_off; q++) {
                            for (int r = 0; r < r_off; r++) {
                                for (int s = 0; s < s_off; s++) {
                                    double val = buf[p + p_off*(q + q_off*(r + r_off*s))];
                                    int i = ipos+p, j = jpos+q, k = kpos+r, l = lpos+s;
                                    out[i*nbfn*nbfn*nbfn + k*nbfn*nbfn + j*nbfn + l] = val;
                                    out[i*nbfn*nbfn*nbfn + l*nbfn*nbfn + j*nbfn + k] = val;
                                    out[j*nbfn*nbfn*nbfn + k*nbfn*nbfn + i*nbfn + l] = val;
                                    out[j*nbfn*nbfn*nbfn + l*nbfn*nbfn + i*nbfn + k] = val;
                                    out[k*nbfn*nbfn*nbfn + i*nbfn*nbfn + l*nbfn + j] = val;
                                    out[k*nbfn*nbfn*nbfn + j*nbfn*nbfn + l*nbfn + i] = val;
                                    out[l*nbfn*nbfn*nbfn + i*nbfn*nbfn + k*nbfn + j] = val;
                                    out[l*nbfn*nbfn*nbfn + j*nbfn*nbfn + k*nbfn + i] = val;                                }
                            }
                        }
                    }
                    memset(buf, 0, buf_size * sizeof(double));
                    lpos += s_off;
                }
                kpos += r_off;
            }
            jpos += q_off;
        }
        ipos += p_off;
    }
    CINTdel_optimizer(&opt);
    free(buf);
    Py_RETURN_NONE;
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
    {"overlap_integral_array", overlap_integral_array, METH_VARARGS, "Overlap integral array in C"},
    {"kinetic_integral_array", kinetic_integral_array, METH_VARARGS, "Kinetic integral array in C"},
    {"nuclear_integral_array", nuclear_integral_array, METH_VARARGS, "Nuclear attraction integral array in C"},
    {"momentum_integral_array", momentum_integral_array, METH_VARARGS, "Momentum integral array in C"},
    {"rinv_integral_array", rinv_integral_array, METH_VARARGS, "1/r integral array in C"},
    {"dipole_integral_array", dipole_integral_array, METH_VARARGS, "Dipole integral array in C"},
    {"quadrupole_integral_array", quadrupole_integral_array, METH_VARARGS, "Quadrupole integral array in C"},
    {"octupole_integral_array", octupole_integral_array, METH_VARARGS, "Octupole integral array in C"},
    {"eri_array", eri_array, METH_VARARGS, "ERI 2-electron array in C"},
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
