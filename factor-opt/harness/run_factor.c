// run_factor [--reference|--optimized] <corpus.bin> <out.res>
//            [--qld-dump <out.qld>] [--repeat N] [--time]
//
// Runs ONE kernel (selected at link time -- this TU is compiled twice, once
// linked with factor-reference/ and once with factor-optimized/) over the
// corpus:
//   for each snapshot: factor M (in place on a copy), then for each RHS b
//   solve M x = b, compute the INDEPENDENT residual via a dense matvec
//   (shares no code with the sparse factor/solve path), and record
//   (xhash, ||x||_inf, residual, x) to the results file.
//
//   --repeat N --time : run the factor+solve over the whole corpus N times
//     back-to-back and print the MEDIAN total kernel-only time (monotonic
//     clock). Model load / corpus I/O / densification are excluded from the
//     timed region. Used by measure-factor.sh.
//
//   --qld-dump <file> : also dump the raw factor bytes (qLD + qLDiagInv) of
//     every snapshot, for the byte-identical determinism check.

#include <math.h>
#include <time.h>
#include "../factor-reference/factor_kernel.h"
#include "corpus_format.h"

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

// One snapshot's static data + scratch.
typedef struct {
  int nv, nC, K;
  int32_t *rownnz, *rowadr, *colind;
  double *Mval;       // pristine CSR values (nC)
  double *rhs;        // K*nv
  // scratch
  double *qLD;        // nC  (factor lives here)
  double *qLDiagInv;  // nv
  double *x;          // nv
  double *dense;      // nv*nv  symmetric M for independent residual matvec
  double *Mx;         // nv
  double Minf;        // ||M||_inf (precomputed, not timed)
} Snap;

// One full pass of the factor+solve work over the whole corpus -- the timed
// unit. Factoring is in-place/destructive, so each pass re-copies pristine M.
static void corpus_pass(Snap* snaps, int nsnap) {
  for (int s = 0; s < nsnap; s++) {
    Snap* sp = &snaps[s];
    int nv = sp->nv, nC = sp->nC, K = sp->K;
    memcpy(sp->qLD, sp->Mval, nC * sizeof(double));
    factor_kernel(sp->qLD, sp->qLDiagInv, nv, sp->rownnz, sp->rowadr,
                  sp->colind);
    for (int k = 0; k < K; k++) {
      memcpy(sp->x, sp->rhs + (size_t)k*nv, nv * sizeof(double));
      solve_kernel(sp->x, sp->qLD, sp->qLDiagInv, nv, sp->rownnz, sp->rowadr,
                   sp->colind);
    }
  }
}

int main(int argc, char** argv) {
  const char* corpus_path = NULL;
  const char* out_path = NULL;
  const char* qld_path = NULL;
  int repeat = 1;
  int do_time = 0;
  int inner = 0;                // passes per timed sample (0 = auto-calibrate)
  double min_sample_s = 0.025;  // auto-calibration target: >= 25 ms per sample

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--reference") || !strcmp(argv[i], "--optimized")) {
      // kernel is selected at link time; flag is accepted for clarity
    } else if (!strcmp(argv[i], "--qld-dump") && i+1 < argc) {
      qld_path = argv[++i];
    } else if (!strcmp(argv[i], "--repeat") && i+1 < argc) {
      repeat = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--inner") && i+1 < argc) {
      inner = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--min-sample-ms") && i+1 < argc) {
      min_sample_s = atof(argv[++i]) * 1e-3;
    } else if (!strcmp(argv[i], "--time")) {
      do_time = 1;
    } else if (!corpus_path) {
      corpus_path = argv[i];
    } else if (!out_path) {
      out_path = argv[i];
    } else {
      fprintf(stderr, "FATAL: unexpected arg %s\n", argv[i]);
      return 2;
    }
  }
  if (!corpus_path) { fprintf(stderr, "FATAL: no corpus path\n"); return 2; }

  FILE* cf = fopen(corpus_path, "rb");
  if (!cf) { fprintf(stderr, "FATAL: cannot open %s\n", corpus_path); return 2; }
  CorpusHeader ch;
  rd(&ch, sizeof(ch), 1, cf, "corpus header");
  if (ch.magic != CORPUS_MAGIC || ch.version != CORPUS_VERSION) {
    die("bad corpus magic/version");
  }
  int nsnap = ch.nsnap;

  // ---- load all snapshots into memory (NOT timed) ----
  Snap* snaps = (Snap*)calloc(nsnap, sizeof(Snap));
  if (!snaps) die("oom");
  for (int s = 0; s < nsnap; s++) {
    SnapHeader sh;
    rd(&sh, sizeof(sh), 1, cf, "snap header");
    Snap* sp = &snaps[s];
    sp->nv = sh.nv; sp->nC = sh.nC; sp->K = sh.K;
    int nv = sp->nv, nC = sp->nC, K = sp->K;
    sp->rownnz = malloc(nv * sizeof(int32_t));
    sp->rowadr = malloc(nv * sizeof(int32_t));
    sp->colind = malloc(nC * sizeof(int32_t));
    sp->Mval   = malloc(nC * sizeof(double));
    sp->rhs    = malloc((size_t)K * nv * sizeof(double));
    sp->qLD    = malloc(nC * sizeof(double));
    sp->qLDiagInv = malloc(nv * sizeof(double));
    sp->x      = malloc(nv * sizeof(double));
    sp->dense  = malloc((size_t)nv * nv * sizeof(double));
    sp->Mx     = malloc(nv * sizeof(double));
    if (!sp->rownnz || !sp->rowadr || !sp->colind || !sp->Mval || !sp->rhs ||
        !sp->qLD || !sp->qLDiagInv || !sp->x || !sp->dense || !sp->Mx) die("oom");
    rd(sp->rownnz, sizeof(int32_t), nv, cf, "rownnz");
    rd(sp->rowadr, sizeof(int32_t), nv, cf, "rowadr");
    rd(sp->colind, sizeof(int32_t), nC, cf, "colind");
    rd(sp->Mval, sizeof(double), nC, cf, "Mval");
    rd(sp->rhs, sizeof(double), (size_t)K * nv, cf, "rhs");

    // densify symmetric M for the independent residual matvec + ||M||_inf
    memset(sp->dense, 0, (size_t)nv*nv*sizeof(double));
    for (int r = 0; r < nv; r++) {
      int adr = sp->rowadr[r], nnz = sp->rownnz[r];
      for (int e = 0; e < nnz; e++) {
        int c = sp->colind[adr + e];
        double v = sp->Mval[adr + e];
        sp->dense[(size_t)r*nv + c] = v;
        sp->dense[(size_t)c*nv + r] = v;
      }
    }
    double Minf = 0.0;
    for (int r = 0; r < nv; r++) {
      double row = 0.0;
      for (int c = 0; c < nv; c++) row += fabs(sp->dense[(size_t)r*nv + c]);
      if (row > Minf) Minf = row;
    }
    sp->Minf = Minf;
  }
  fclose(cf);

  // store final results per (snapshot, rhs): entry + x vector
  // (we keep only the LAST repeat's outputs; identical across repeats)
  ResEntry** entries = malloc(nsnap * sizeof(ResEntry*));
  double**   xstore  = malloc(nsnap * sizeof(double*));
  if (!entries || !xstore) die("oom");
  for (int s = 0; s < nsnap; s++) {
    entries[s] = malloc(snaps[s].K * sizeof(ResEntry));
    xstore[s]  = malloc((size_t)snaps[s].K * snaps[s].nv * sizeof(double));
    if (!entries[s] || !xstore[s]) die("oom");
  }

  fprintf(stderr, "kernel: %s ; snapshots=%d repeat=%d\n",
          factor_kernel_name(), nsnap, repeat);

  double* times = NULL;
  if (do_time) {
    // Warm caches / let CPU frequency ramp (not measured).
    double tw = now_sec();
    while (now_sec() - tw < 0.05) corpus_pass(snaps, nsnap);

    // Auto-calibrate the inner repeat count so each timed sample lasts at least
    // min_sample_s. One corpus pass is only tens of microseconds, so a single
    // pass per sample is swamped by clock granularity and OS jitter; running
    // many passes inside one timed region and dividing back out gives a stable
    // per-pass number. --inner N overrides the calibration.
    if (inner <= 0) {
      inner = 1;
      for (;;) {
        double tc = now_sec();
        for (int it = 0; it < inner; it++) corpus_pass(snaps, nsnap);
        double el = now_sec() - tc;
        if (el >= min_sample_s || inner >= (1 << 24)) break;
        double grow = (el > 0.0) ? (min_sample_s / el) * 1.3 : 2.0;
        int next = (int)(inner * grow) + 1;
        inner = (next > inner) ? next : inner * 2;
      }
    }

    times = malloc(repeat * sizeof(double));
    if (!times) die("oom");
    for (int rep = 0; rep < repeat; rep++) {
      double t0 = now_sec();
      // ===================== TIMED REGION (factor + solve only) ==========
      for (int it = 0; it < inner; it++) corpus_pass(snaps, nsnap);
      // ===================================================================
      times[rep] = (now_sec() - t0) / inner;   // per-corpus-pass time
    }
  }

  // recompute outputs + residuals ONCE (outside the pure-time loop) so the
  // results file and residuals are produced exactly as in the timed kernel.
  for (int s = 0; s < nsnap; s++) {
    Snap* sp = &snaps[s];
    int nv = sp->nv, nC = sp->nC, K = sp->K;
    memcpy(sp->qLD, sp->Mval, nC * sizeof(double));
    factor_kernel(sp->qLD, sp->qLDiagInv, nv, sp->rownnz, sp->rowadr,
                  sp->colind);
    for (int k = 0; k < K; k++) {
      const double* b = sp->rhs + (size_t)k*nv;
      memcpy(sp->x, b, nv * sizeof(double));
      solve_kernel(sp->x, sp->qLD, sp->qLDiagInv, nv, sp->rownnz,
                   sp->rowadr, sp->colind);

      // independent residual via dense matvec (no shared code with solve)
      double xinf = 0.0, binf = 0.0, rinf = 0.0;
      for (int i = 0; i < nv; i++) {
        double xi = fabs(sp->x[i]); if (xi > xinf) xinf = xi;
        double bi = fabs(b[i]);     if (bi > binf) binf = bi;
      }
      for (int r = 0; r < nv; r++) {
        double acc = 0.0;
        const double* row = sp->dense + (size_t)r*nv;
        for (int c = 0; c < nv; c++) acc += row[c] * sp->x[c];
        double res = fabs(acc - b[r]);
        if (res > rinf) rinf = res;
      }
      double denom = sp->Minf * xinf + binf;
      double residual = (denom > 0.0) ? rinf / denom : rinf;

      ResEntry e;
      e.xhash = fnv1a64(sp->x, nv * sizeof(double));
      e.xnorm = xinf;
      e.residual = residual;
      entries[s][k] = e;
      memcpy(xstore[s] + (size_t)k*nv, sp->x, nv * sizeof(double));
    }
  }

  // ---- write results file ----
  if (out_path) {
    FILE* of = fopen(out_path, "wb");
    if (!of) { fprintf(stderr, "FATAL: cannot create %s\n", out_path); return 2; }
    ResultHeader rh = {RESULT_MAGIC, RESULT_VERSION, nsnap, 0};
    wr(&rh, sizeof(rh), 1, of, "result header");
    for (int s = 0; s < nsnap; s++) {
      Snap* sp = &snaps[s];
      ResSnapHeader rsh = {0, 0, sp->nv, sp->K};
      // model/config ids are not needed for compare ordering, but keep stable
      rsh.model_id = s;   // monotonic snapshot index; both runs share order
      rsh.config_id = 0;
      wr(&rsh, sizeof(rsh), 1, of, "res snap header");
      for (int k = 0; k < sp->K; k++) {
        wr(&entries[s][k], sizeof(ResEntry), 1, of, "res entry");
        wr(xstore[s] + (size_t)k*sp->nv, sizeof(double), sp->nv, of, "res x");
      }
    }
    fclose(of);
  }

  // ---- dump raw factor bytes for determinism check ----
  if (qld_path) {
    FILE* qf = fopen(qld_path, "wb");
    if (!qf) { fprintf(stderr, "FATAL: cannot create %s\n", qld_path); return 2; }
    for (int s = 0; s < nsnap; s++) {
      Snap* sp = &snaps[s];
      int nv = sp->nv, nC = sp->nC;
      memcpy(sp->qLD, sp->Mval, nC * sizeof(double));
      factor_kernel(sp->qLD, sp->qLDiagInv, nv, sp->rownnz, sp->rowadr,
                    sp->colind);
      wr(sp->qLD, sizeof(double), nC, qf, "qLD dump");
      wr(sp->qLDiagInv, sizeof(double), nv, qf, "qLDiagInv dump");
    }
    fclose(qf);
  }

  // ---- timing ----
  if (do_time) {
    qsort(times, repeat, sizeof(double), cmp_double);
    double median = (repeat & 1) ? times[repeat/2]
                                 : 0.5*(times[repeat/2-1] + times[repeat/2]);
    printf("TIME kernel=\"%s\" repeat=%d inner=%d median_s=%.9g min_s=%.9g max_s=%.9g\n",
           factor_kernel_name(), repeat, inner, median, times[0], times[repeat-1]);
    fprintf(stderr, "samples (per-pass s):");
    for (int i = 0; i < repeat; i++) fprintf(stderr, " %.9g", times[i]);
    fprintf(stderr, "\n");
    free(times);
  }

  return 0;
}
