// certify_factor <corpus.bin> <results.res>
//
// The INDEPENDENT certifier. For every snapshot/RHS it:
//   - densifies M (symmetric) from the corpus CSR,
//   - factors it with the DENSE Cholesky mju_cholFactor (shares no code with
//     the sparse mj_factorI/mj_solveLD path),
//   - solves M x_dense = b with mju_cholSolve,
//   - recomputes the kernel-under-test residual r_kernel = ||M x_k - b||_inf /
//     (||M||_inf ||x_k||_inf + ||b||_inf) from the x stored in the results
//     file, via an independent dense matvec, and
//   - certifies r_kernel <= max(8 * r_dense, 1e-11), where r_dense is the
//     dense oracle's own residual on the same snapshot (its intrinsic band).
//
// Also re-asserts every M is SPD (dense Cholesky full rank, positive pivots).
// Prints a fixed parseable summary and exits non-zero if any snapshot fails.

#include <math.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_solve.h"  // mju_cholFactor / mju_cholSolve
#include "corpus_format.h"

#define RESID_FLOOR 1e-11
#define RESID_FACT  8.0

static int has_bad(double v) { return isnan(v) || isinf(v); }

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <corpus.bin> <results.res>\n", argv[0]);
    return 2;
  }
  FILE* cf = fopen(argv[1], "rb");
  FILE* rf = fopen(argv[2], "rb");
  if (!cf) { fprintf(stderr, "FATAL: cannot open %s\n", argv[1]); return 2; }
  if (!rf) { fprintf(stderr, "FATAL: cannot open %s\n", argv[2]); return 2; }

  CorpusHeader ch;
  rd(&ch, sizeof(ch), 1, cf, "corpus header");
  if (ch.magic != CORPUS_MAGIC || ch.version != CORPUS_VERSION)
    die("bad corpus magic/version");

  ResultHeader rh;
  rd(&rh, sizeof(rh), 1, rf, "result header");
  if (rh.magic != RESULT_MAGIC || rh.version != RESULT_VERSION)
    die("bad results magic/version");

  if (ch.nsnap != rh.nsnap) {
    fprintf(stderr, "FATAL: corpus/results snapshot mismatch: %d vs %d\n",
            ch.nsnap, rh.nsnap);
    return 2;
  }
  int nsnap = ch.nsnap;

  double max_kernel_resid = 0.0;
  double max_dense_resid = 0.0;
  long n_over = 0, total_rhs = 0;
  int n_spd_fail = 0;

  for (int s = 0; s < nsnap; s++) {
    SnapHeader sh;
    rd(&sh, sizeof(sh), 1, cf, "snap header");
    int nv = sh.nv, nC = sh.nC, K = sh.K;

    int32_t* rownnz = malloc(nv * sizeof(int32_t));
    int32_t* rowadr = malloc(nv * sizeof(int32_t));
    int32_t* colind = malloc(nC * sizeof(int32_t));
    double*  Mval   = malloc(nC * sizeof(double));
    double*  rhs    = malloc((size_t)K * nv * sizeof(double));
    if (!rownnz || !rowadr || !colind || !Mval || !rhs) die("oom");
    rd(rownnz, sizeof(int32_t), nv, cf, "rownnz");
    rd(rowadr, sizeof(int32_t), nv, cf, "rowadr");
    rd(colind, sizeof(int32_t), nC, cf, "colind");
    rd(Mval, sizeof(double), nC, cf, "Mval");
    rd(rhs, sizeof(double), (size_t)K * nv, cf, "rhs");

    // densify symmetric M
    double* dense = calloc((size_t)nv*nv, sizeof(double));
    if (!dense) die("oom");
    for (int r = 0; r < nv; r++) {
      int adr = rowadr[r], nnz = rownnz[r];
      for (int e = 0; e < nnz; e++) {
        int c = colind[adr + e];
        double v = Mval[adr + e];
        dense[(size_t)r*nv + c] = v;
        dense[(size_t)c*nv + r] = v;
      }
    }
    double Minf = 0.0;
    for (int r = 0; r < nv; r++) {
      double row = 0.0;
      for (int c = 0; c < nv; c++) row += fabs(dense[(size_t)r*nv + c]);
      if (row > Minf) Minf = row;
    }

    // dense Cholesky (independent oracle)
    double* chol = malloc((size_t)nv*nv*sizeof(double));
    if (!chol) die("oom");
    memcpy(chol, dense, (size_t)nv*nv*sizeof(double));
    int rank = mju_cholFactor(chol, nv, 0.0);
    int spd_ok = (rank == nv);
    for (int i = 0; spd_ok && i < nv; i++)
      if (!(chol[(size_t)i*nv + i] > 0.0)) spd_ok = 0;
    if (!spd_ok) {
      fprintf(stderr, "CERTIFY FAIL: snapshot %d M not SPD (rank=%d/%d)\n",
              s, rank, nv);
      n_spd_fail++;
    }

    ResSnapHeader rsh;
    rd(&rsh, sizeof(rsh), 1, rf, "res snap header");
    if (rsh.nv != nv || rsh.K != K) {
      fprintf(stderr, "FATAL: results snapshot %d dims mismatch corpus\n", s);
      return 2;
    }

    double* xk = malloc(nv * sizeof(double));     // kernel-under-test x
    double* xd = malloc(nv * sizeof(double));     // dense oracle x
    if (!xk || !xd) die("oom");

    for (int k = 0; k < K; k++) {
      const double* b = rhs + (size_t)k*nv;

      ResEntry e;
      rd(&e, sizeof(e), 1, rf, "res entry");
      rd(xk, sizeof(double), nv, rf, "res x");
      total_rhs++;

      if (has_bad(e.residual)) {
        fprintf(stderr, "FATAL: NaN/Inf residual snapshot %d rhs %d\n", s, k);
        return 2;
      }

      // norms
      double xkinf = 0.0, binf = 0.0;
      for (int i = 0; i < nv; i++) {
        double xi = fabs(xk[i]); if (xi > xkinf) xkinf = xi;
        double bi = fabs(b[i]);  if (bi > binf) binf = bi;
      }

      // kernel residual via independent dense matvec
      double rinf = 0.0;
      for (int r = 0; r < nv; r++) {
        double acc = 0.0;
        const double* row = dense + (size_t)r*nv;
        for (int c = 0; c < nv; c++) acc += row[c] * xk[c];
        double res = fabs(acc - b[r]);
        if (res > rinf) rinf = res;
      }
      double denomk = Minf * xkinf + binf;
      double r_kernel = (denomk > 0.0) ? rinf / denomk : rinf;

      // dense oracle solve + its own residual
      mju_cholSolve(xd, chol, b, nv);
      double xdinf = 0.0, rdinf = 0.0;
      for (int i = 0; i < nv; i++) {
        double xi = fabs(xd[i]); if (xi > xdinf) xdinf = xi;
      }
      for (int r = 0; r < nv; r++) {
        double acc = 0.0;
        const double* row = dense + (size_t)r*nv;
        for (int c = 0; c < nv; c++) acc += row[c] * xd[c];
        double res = fabs(acc - b[r]);
        if (res > rdinf) rdinf = res;
      }
      double denomd = Minf * xdinf + binf;
      double r_dense = (denomd > 0.0) ? rdinf / denomd : rdinf;

      if (r_kernel > max_kernel_resid) max_kernel_resid = r_kernel;
      if (r_dense  > max_dense_resid)  max_dense_resid  = r_dense;

      double band = RESID_FACT * r_dense;
      if (band < RESID_FLOOR) band = RESID_FLOOR;
      if (r_kernel > band || !spd_ok) {
        n_over++;
        fprintf(stderr,
                "CERTIFY OVER snapshot %d rhs %d: r_kernel=%.3e "
                "r_dense=%.3e band=%.3e\n", s, k, r_kernel, r_dense, band);
      }
    }

    free(xk); free(xd); free(chol); free(dense);
    free(rownnz); free(rowadr); free(colind); free(Mval); free(rhs);
  }

  fclose(cf);
  fclose(rf);

  printf("CERTIFY snapshots=%d rhs=%ld max_kernel_resid=%.17g "
         "max_dense_resid=%.17g over_band=%ld spd_fail=%d "
         "resid_floor=%.1e resid_factor=%g\n",
         nsnap, total_rhs, max_kernel_resid, max_dense_resid, n_over,
         n_spd_fail, RESID_FLOOR, RESID_FACT);

  if (n_over == 0 && n_spd_fail == 0) {
    printf("CERTIFY VERDICT=PASS\n");
    return 0;
  }
  printf("CERTIFY VERDICT=FAIL over_band=%ld spd_fail=%d\n", n_over, n_spd_fail);
  return 1;
}
