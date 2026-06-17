// compare_factor <ref-results> <opt-results>
//
// THE CORE of the harness. Reads two results files produced by run_factor and
// applies the two-part match criterion from build-factor-harness.md:
//
//   1. Residual band (independent certification, already computed per-entry by
//      run_factor via a dense matvec that shares no code with the sparse path):
//        r_opt <= max(8 * r_ref, 1e-11)
//   2. Solve-output agreement:
//        ||x_opt - x_ref||_inf / (||x_ref||_inf + 1e-300) <= 1e-6
//
// Loud + non-zero on any malformed / length-mismatched / NaN/Inf input.
// This is NOT a byte diff: later optimized kernels are not bit-identical.

#include <math.h>
#include "corpus_format.h"

#define TOL_SOLVE   1e-6
#define RESID_FLOOR 1e-11
#define RESID_FACT  8.0

static int has_bad(double v) { return isnan(v) || isinf(v); }

static void read_result_header(FILE* f, const char* path, ResultHeader* h) {
  rd(h, sizeof(*h), 1, f, "result header");
  if (h->magic != RESULT_MAGIC || h->version != RESULT_VERSION) {
    fprintf(stderr, "FATAL: %s is not a valid results file (magic/version)\n",
            path);
    exit(2);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <ref-results> <opt-results>\n", argv[0]);
    return 2;
  }
  FILE* fr = fopen(argv[1], "rb");
  FILE* fo = fopen(argv[2], "rb");
  if (!fr) { fprintf(stderr, "FATAL: cannot open %s\n", argv[1]); return 2; }
  if (!fo) { fprintf(stderr, "FATAL: cannot open %s\n", argv[2]); return 2; }

  ResultHeader hr, ho;
  read_result_header(fr, argv[1], &hr);
  read_result_header(fo, argv[2], &ho);

  if (hr.nsnap != ho.nsnap) {
    fprintf(stderr, "FATAL: snapshot count mismatch: ref=%d opt=%d\n",
            hr.nsnap, ho.nsnap);
    return 2;
  }

  int nsnap = hr.nsnap;
  long total_rhs = 0;
  double max_solve_diff = 0.0;     // max relative ||x_opt-x_ref||/||x_ref||
  double max_opt_resid = 0.0;      // max optimized residual
  double resid_at_max = 0.0;       // ref residual paired with that opt residual
  double max_ref_resid = 0.0;
  long n_over = 0;

  double* xr = NULL;
  double* xo = NULL;
  int cap = 0;

  for (int s = 0; s < nsnap; s++) {
    ResSnapHeader sr, so;
    rd(&sr, sizeof(sr), 1, fr, "ref snap header");
    rd(&so, sizeof(so), 1, fo, "opt snap header");

    if (sr.model_id != so.model_id || sr.config_id != so.config_id ||
        sr.nv != so.nv || sr.K != so.K) {
      fprintf(stderr,
              "FATAL: snapshot %d mismatch (out-of-order or differing dims): "
              "ref(model=%d cfg=%d nv=%d K=%d) opt(model=%d cfg=%d nv=%d K=%d)\n",
              s, sr.model_id, sr.config_id, sr.nv, sr.K,
              so.model_id, so.config_id, so.nv, so.K);
      return 2;
    }

    int nv = sr.nv, K = sr.K;
    if (nv > cap) {
      cap = nv;
      xr = (double*)realloc(xr, cap * sizeof(double));
      xo = (double*)realloc(xo, cap * sizeof(double));
      if (!xr || !xo) die("oom");
    }

    for (int k = 0; k < K; k++) {
      ResEntry er, eo;
      rd(&er, sizeof(er), 1, fr, "ref entry");
      rd(&eo, sizeof(eo), 1, fo, "opt entry");
      rd(xr, sizeof(double), nv, fr, "ref x");
      rd(xo, sizeof(double), nv, fo, "opt x");

      // NaN / Inf are hard failures.
      if (has_bad(er.residual) || has_bad(eo.residual) ||
          has_bad(er.xnorm) || has_bad(eo.xnorm)) {
        fprintf(stderr,
                "FATAL: NaN/Inf in result snapshot %d rhs %d "
                "(ref_resid=%g opt_resid=%g)\n", s, k, er.residual, eo.residual);
        return 2;
      }
      for (int i = 0; i < nv; i++) {
        if (has_bad(xr[i]) || has_bad(xo[i])) {
          fprintf(stderr, "FATAL: NaN/Inf in x at snapshot %d rhs %d idx %d\n",
                  s, k, i);
          return 2;
        }
      }

      total_rhs++;

      // (2) solve-output agreement
      double dmax = 0.0, rnorm = 0.0;
      for (int i = 0; i < nv; i++) {
        double d = fabs(xo[i] - xr[i]);
        if (d > dmax) dmax = d;
        double a = fabs(xr[i]);
        if (a > rnorm) rnorm = a;
      }
      double rel = dmax / (rnorm + 1e-300);
      if (rel > max_solve_diff) max_solve_diff = rel;

      // (1) residual band against this snapshot's reference residual
      double band = RESID_FACT * er.residual;
      if (band < RESID_FLOOR) band = RESID_FLOOR;
      if (eo.residual > max_opt_resid) {
        max_opt_resid = eo.residual;
        resid_at_max = er.residual;
      }
      if (er.residual > max_ref_resid) max_ref_resid = er.residual;

      int over = 0;
      if (rel > TOL_SOLVE) over = 1;
      if (eo.residual > band) over = 1;
      if (over) {
        n_over++;
        fprintf(stderr,
                "OVER-TOL snapshot %d rhs %d: rel_solve_diff=%.3e (tol %.1e) "
                "opt_resid=%.3e ref_resid=%.3e band=%.3e\n",
                s, k, rel, TOL_SOLVE, eo.residual, er.residual, band);
      }
    }
  }

  // Ensure both files are fully consumed (no trailing extra data).
  char extra;
  if (fread(&extra, 1, 1, fr) != 0) {
    fprintf(stderr, "FATAL: trailing data in ref results file\n");
    return 2;
  }
  if (fread(&extra, 1, 1, fo) != 0) {
    fprintf(stderr, "FATAL: trailing data in opt results file\n");
    return 2;
  }
  fclose(fr);
  fclose(fo);
  free(xr);
  free(xo);

  // fixed parseable summary
  printf("COMPARE snapshots=%d rhs=%ld max_solve_diff=%.17g "
         "max_opt_resid=%.17g ref_resid_at_max=%.17g max_ref_resid=%.17g "
         "over_tol=%ld solve_tol=%.1e resid_floor=%.1e resid_factor=%g\n",
         nsnap, total_rhs, max_solve_diff, max_opt_resid, resid_at_max,
         max_ref_resid, n_over, TOL_SOLVE, RESID_FLOOR, RESID_FACT);

  if (n_over == 0) {
    printf("COMPARE VERDICT=PASS\n");
    return 0;
  }
  printf("COMPARE VERDICT=FAIL over_tol=%ld\n", n_over);
  return 1;
}
