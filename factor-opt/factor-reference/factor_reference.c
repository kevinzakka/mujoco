// factor-reference/ : the ANCHOR. A thin shim that calls the library's
// mj_factorI / mj_solveLD directly. This file is NOT edited by the
// optimization loop.

#include "factor_kernel.h"
#include "src/engine/engine_core_smooth.h"

void factor_kernel(mjtNum* mat, mjtNum* diaginv, int nv, const int* rownnz,
                   const int* rowadr, const int* colind) {
  mj_factorI(mat, diaginv, nv, rownnz, rowadr, colind, NULL);
}

void solve_kernel(mjtNum* x, const mjtNum* qLD, const mjtNum* diaginv, int nv,
                  const int* rownnz, const int* rowadr, const int* colind) {
  mj_solveLD(x, qLD, diaginv, nv, 1, rownnz, rowadr, colind, NULL);
}

const char* factor_kernel_name(void) { return "reference (library mj_factorI/mj_solveLD)"; }
