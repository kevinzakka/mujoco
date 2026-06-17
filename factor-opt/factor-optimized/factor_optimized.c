// factor-optimized/ : the ONLY thing the optimization loop edits.
//
// ITERATION-0 (PULLED, UNMODIFIED): factorI_opt / solveLD_opt below are a
// faithful transcription of MuJoCo's mj_factorI / mj_solveLD
// (src/engine/engine_core_smooth.c), with the small mju_dotSparse helper
// (src/engine/engine_util_sparse.h, scalar path -- arm64 has no AVX) inlined
// here because it is a static-inline in a header and not an exported symbol.
//
// They are UNMODIFIED at this checkpoint, so the harness should report ~0
// solve-output difference vs the reference wrapper (which calls the library
// kernel). This is the "real copy, still identical" checkpoint: from here, the
// optimization loop edits THESE functions. (mju_addToScl / mju_scl are still
// called from libmujoco; inline them here too when an optimization needs to
// change how the factor's inner updates run.)
//
// Note: only the single-vector (n == 1) solve path is kept -- the harness only
// ever solves one RHS at a time. The multi-vector branches of the library
// mj_solveLD are dead code for this kernel and were dropped.

#include <mujoco/mujoco.h>   // mjtNum, NULL
#include "factor_kernel.h"

// Inlined, vectorizable copies of mju_addToScl / mju_scl. The library versions
// are scalar (mjUSEAVX is undefined on arm64) AND reached via a cross-TU call,
// so they neither inline nor use NEON. These contiguous loops auto-vectorize
// under -O3 -march=native. restrict tells the optimizer res and vec don't alias
// (true in factorI: res is row i, vec is row k, distinct rows).
static inline void addToScl_opt(mjtNum* restrict res, const mjtNum* restrict vec,
                                mjtNum scl, int n) {
  for (int i = 0; i < n; i++) res[i] += vec[i]*scl;
}
static inline void scl_opt(mjtNum* restrict res, const mjtNum* restrict vec,
                           mjtNum scl, int n) {
  for (int i = 0; i < n; i++) res[i] = vec[i]*scl;
}

// editable copy of mju_dotSparse (scalar 4-way unrolled; matches the library's
// non-AVX summation order ((res0+res2)+(res1+res3)) for bit-faithful output).
static mjtNum dotSparse_opt(const mjtNum* restrict vec1,
                            const mjtNum* restrict vec2, int nnz1,
                            const int* restrict ind1) {
  int i = 0;
  mjtNum res = 0;
  int n_4 = nnz1 - 4;
  mjtNum res0 = 0, res1 = 0, res2 = 0, res3 = 0;
  for (; i <= n_4; i += 4) {
    res0 += vec1[i+0] * vec2[ind1[i+0]];
    res1 += vec1[i+1] * vec2[ind1[i+1]];
    res2 += vec1[i+2] * vec2[ind1[i+2]];
    res3 += vec1[i+3] * vec2[ind1[i+3]];
  }
  res = (res0 + res2) + (res1 + res3);
  for (; i < nnz1; i++) {
    res += vec1[i] * vec2[ind1[i]];
  }
  return res;
}

// editable copy of mj_factorI: in-place sparse LDL^T of CSR lower-triangular M.
static void factorI_opt(mjtNum* restrict mat, mjtNum* restrict diaginv, int nv,
                        const int* restrict rownnz, const int* restrict rowadr,
                        const int* restrict colind, const int* index) {
  // General path: non-NULL elimination order (not used by this kernel's
  // call site, but kept for faithfulness).
  if (index || !diaginv) {
    for (int j = nv - 1; j >= 0; j--) {
      int k = index ? index[j] : j;
      int start = rowadr[k];
      int diag = rownnz[k] - 1;
      int end = start + diag;
      mjtNum invD = 1 / mat[end];
      if (diaginv) diaginv[k] = invD;
      for (int adr = end - 1; adr >= start; adr--) {
        int i = colind[adr];
        addToScl_opt(mat + rowadr[i], mat + start, -mat[adr] * invD, rownnz[i]);
      }
      scl_opt(mat + start, mat + start, invD, diag);
    }
    return;
  }

  // Fast path: identity order, diaginv present (this kernel's only call).
  for (int k = nv - 1; k >= 0; k--) {
    int start = rowadr[k];
    int diag = rownnz[k] - 1;
    int end = start + diag;
    mjtNum invD = 1 / mat[end];
    diaginv[k] = invD;

    // update triangle above row k
    for (int adr = end - 1; adr >= start; adr--) {
      int i = colind[adr];
      addToScl_opt(mat + rowadr[i], mat + start, -mat[adr] * invD, rownnz[i]);
    }

    // normalize row k
    scl_opt(mat + start, mat + start, invD, diag);
  }
}

// editable copy of mj_solveLD (single vector): x = inv(L'*D*L) * x.
static void solveLD_opt(mjtNum* restrict x, const mjtNum* restrict qLD,
                        const mjtNum* restrict qLDiagInv, int nv,
                        const int* restrict rownnz, const int* restrict rowadr,
                        const int* restrict colind, const int* index) {
  // index is always NULL on this kernel's call path (solve_kernel passes NULL);
  // the elimination order is identity. Specialize that path so no per-row
  // branch/indirection remains. (If a non-NULL index is ever passed, fall back
  // to the original general loop.)
  if (index) {
    for (int k = nv - 1; k >= 0; k--) {
      int i = index[k];
      if (rownnz[i] == 1) continue;
      mjtNum x_i;
      if ((x_i = x[i])) {
        int start = rowadr[i];
        int end = start + rownnz[i] - 1;
        for (int adr = start; adr < end; adr++) x[colind[adr]] -= qLD[adr]*x_i;
      }
    }
    for (int k = 0; k < nv; k++) { int i = index[k]; x[i] *= qLDiagInv[i]; }
    for (int k = 0; k < nv; k++) {
      int i = index[k];
      if (rownnz[i] == 1) continue;
      int d = rownnz[i] - 1;
      if (d > 0) x[i] -= dotSparse_opt(qLD + rowadr[i], x, d, colind + rowadr[i]);
    }
    return;
  }

  // x <- L^-T x  (identity order)
  for (int i = nv - 1; i >= 0; i--) {
    int nnz = rownnz[i];
    if (nnz == 1) continue;
    mjtNum x_i = x[i];
    if (x_i) {
      int start = rowadr[i];
      int end = start + nnz - 1;
      for (int adr = start; adr < end; adr++) {
        x[colind[adr]] -= qLD[adr] * x_i;
      }
    }
  }

  // x <- D^-1 x
  for (int i = 0; i < nv; i++) {
    x[i] *= qLDiagInv[i];
  }

  // x <- L^-1 x
  for (int i = 0; i < nv; i++) {
    int d = rownnz[i] - 1;
    if (d > 0) {
      int adr = rowadr[i];
      x[i] -= dotSparse_opt(qLD + adr, x, d, colind + adr);
    }
  }
}

void factor_kernel(mjtNum* mat, mjtNum* diaginv, int nv, const int* rownnz,
                   const int* rowadr, const int* colind) {
  factorI_opt(mat, diaginv, nv, rownnz, rowadr, colind, NULL);
}

void solve_kernel(mjtNum* x, const mjtNum* qLD, const mjtNum* diaginv, int nv,
                  const int* rownnz, const int* rowadr, const int* colind) {
  solveLD_opt(x, qLD, diaginv, nv, rownnz, rowadr, colind, NULL);
}

const char* factor_kernel_name(void) {
  return "optimized (pulled editable copy of mj_factorI/mj_solveLD)";
}
