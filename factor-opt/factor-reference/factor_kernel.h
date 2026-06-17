// Shared kernel interface implemented by both factor-reference/ and
// factor-optimized/. run_factor.c is compiled twice (once linked against each
// implementation) -- or links one and selects by flag -- and only ever sees
// these two entry points.

#ifndef FACTOR_OPT_FACTOR_KERNEL_H_
#define FACTOR_OPT_FACTOR_KERNEL_H_

#include <mujoco/mjtnum.h>

#ifdef __cplusplus
extern "C" {
#endif

// In-place sparse LDL^T factorization of CSR lower-triangular SPD matrix `mat`
// (overwritten with the factor); `diaginv` receives the inverse diagonal.
// CSR convention matches MuJoCo's reduced inertia: row k is
// [rowadr[k], rowadr[k]+rownnz[k]); diagonal is the last entry of the row.
void factor_kernel(mjtNum* mat, mjtNum* diaginv, int nv, const int* rownnz,
                   const int* rowadr, const int* colind);

// In-place sparse triangular solve x = inv(L'*D*L) * x for a single vector,
// using the factorization produced by factor_kernel.
void solve_kernel(mjtNum* x, const mjtNum* qLD, const mjtNum* diaginv, int nv,
                  const int* rownnz, const int* rowadr, const int* colind);

// Human-readable tag identifying which implementation is linked.
const char* factor_kernel_name(void);

#ifdef __cplusplus
}
#endif

#endif  // FACTOR_OPT_FACTOR_KERNEL_H_
