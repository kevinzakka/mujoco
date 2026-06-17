// dump_corpus <models.txt> <out.bin>
//
// Model zoo -> corpus.bin. For each model and each seeded config:
//   load model, perturb qpos with a seeded PRNG, mj_forward, gather M in CSR,
//   assert M is SPD (all pivots > 0 via a scratch dense Cholesky), and emit
//   K seeded RHS vectors. The corpus is checksummed separately (sha256) by the
//   prove driver; it is read-only thereafter.
//
// Model load, mj_forward, and I/O are intentionally NOT part of any timed
// region -- timing happens in run_factor over the already-materialized corpus.

#include <math.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_misc.h"   // mju_gather
#include "src/engine/engine_util_solve.h"  // mju_cholFactor (SPD assert only)
#include "corpus_format.h"

// number of seeded configs and RHS vectors per model
#define N_CONFIG 3
#define N_RHS    4

// deterministic, platform-independent PRNG
static uint64_t sm_state;
static void seed_prng(uint64_t s) { sm_state = s; }
static uint64_t next_u64(void) {
  uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
// uniform double in [-1, 1)
static double next_unit(void) {
  return (double)(next_u64() >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <models.txt> <out.bin>\n", argv[0]);
    return 2;
  }
  FILE* mf = fopen(argv[1], "r");
  if (!mf) { fprintf(stderr, "FATAL: cannot open %s\n", argv[1]); return 2; }

  // collect model paths (ignore blank lines and #-comments)
  char (*paths)[1024] = NULL;
  int nmodel = 0, mcap = 0;
  char line[1024];
  while (fgets(line, sizeof(line), mf)) {
    char* p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
    // strip trailing whitespace
    size_t len = strlen(p);
    while (len && (p[len-1] == '\n' || p[len-1] == '\r' ||
                   p[len-1] == ' ' || p[len-1] == '\t')) p[--len] = '\0';
    if (!len) continue;
    if (nmodel == mcap) {
      mcap = mcap ? mcap*2 : 8;
      paths = (char(*)[1024])realloc(paths, mcap * sizeof(*paths));
      if (!paths) die("oom");
    }
    strncpy(paths[nmodel], p, 1023);
    paths[nmodel][1023] = '\0';
    nmodel++;
  }
  fclose(mf);
  if (!nmodel) die("no models in zoo");

  FILE* out = fopen(argv[2], "wb");
  if (!out) { fprintf(stderr, "FATAL: cannot create %s\n", argv[2]); return 2; }

  CorpusHeader ch = {CORPUS_MAGIC, CORPUS_VERSION, nmodel * N_CONFIG, 0};
  wr(&ch, sizeof(ch), 1, out, "corpus header");

  int nsnap_written = 0;
  for (int mi = 0; mi < nmodel; mi++) {
    char err[1024] = {0};
    mjModel* m = mj_loadXML(paths[mi], NULL, err, sizeof(err));
    if (!m) {
      fprintf(stderr, "FATAL: load failed for %s: %s\n", paths[mi], err);
      return 2;
    }
    mjData* d = mj_makeData(m);
    if (!d) die("mj_makeData failed");

    fprintf(stderr, "model %d %s : nv=%d nC=%d nM=%d\n",
            mi, paths[mi], (int)m->nv, (int)m->nC, (int)m->nM);

    for (int ci = 0; ci < N_CONFIG; ci++) {
      // seeded config: start from qpos0 and add small bounded noise.
      // seed mixes model index and config index for independence.
      seed_prng(0xC0FFEEull ^ ((uint64_t)(mi+1) << 32) ^ (uint64_t)(ci+1));
      mj_resetData(m, d);
      for (int i = 0; i < m->nq; i++) {
        d->qpos[i] = m->qpos0[i] + 0.05 * next_unit();
      }
      mj_forward(m, d);

      int nv = m->nv;
      int nC = (int)m->nC;

      // gather M in CSR (values, size nC)
      mj_markStack(d);
      mjtNum* Mval = mj_stackAllocNum(d, nC);
      mju_gather(Mval, d->qM, m->mapM2M, nC);

      // ---- SPD assert: densify (lower tri, mirror) and dense-Cholesky ----
      mjtNum* dense = mj_stackAllocNum(d, (size_t)nv * nv);
      memset(dense, 0, (size_t)nv * nv * sizeof(mjtNum));
      for (int r = 0; r < nv; r++) {
        int adr = m->M_rowadr[r];
        int nnz = m->M_rownnz[r];
        for (int e = 0; e < nnz; e++) {
          int c = m->M_colind[adr + e];
          mjtNum v = Mval[adr + e];
          dense[(size_t)r*nv + c] = v;
          dense[(size_t)c*nv + r] = v;
        }
      }
      mjtNum* chol = mj_stackAllocNum(d, (size_t)nv * nv);
      memcpy(chol, dense, (size_t)nv*nv*sizeof(mjtNum));
      int rank = mju_cholFactor(chol, nv, 0.0);
      if (rank != nv) {
        fprintf(stderr, "FATAL: M not SPD for %s config %d (rank=%d/%d)\n",
                paths[mi], ci, rank, nv);
        return 2;
      }
      // also assert all pivots strictly positive (diagonal of cholesky)
      for (int i = 0; i < nv; i++) {
        if (!(chol[(size_t)i*nv + i] > 0.0)) {
          fprintf(stderr, "FATAL: non-positive pivot %d for %s config %d\n",
                  i, paths[mi], ci);
          return 2;
        }
      }

      // ---- write snapshot ----
      SnapHeader sh = {mi, ci, nv, nC, N_RHS, 0};
      wr(&sh, sizeof(sh), 1, out, "snap header");
      wr(m->M_rownnz, sizeof(int32_t), nv, out, "rownnz");
      wr(m->M_rowadr, sizeof(int32_t), nv, out, "rowadr");
      wr(m->M_colind, sizeof(int32_t), nC, out, "colind");
      wr(Mval, sizeof(double), nC, out, "Mval");

      // K seeded RHS vectors (separate seed stream)
      seed_prng(0xB16B00B5ull ^ ((uint64_t)(mi+1) << 32) ^ (uint64_t)(ci+1));
      double* rhs = (double*)malloc((size_t)N_RHS * nv * sizeof(double));
      if (!rhs) die("oom");
      for (int k = 0; k < N_RHS; k++) {
        for (int i = 0; i < nv; i++) rhs[(size_t)k*nv + i] = next_unit();
      }
      wr(rhs, sizeof(double), (size_t)N_RHS * nv, out, "rhs");
      free(rhs);

      mj_freeStack(d);
      nsnap_written++;
    }

    mj_deleteData(d);
    mj_deleteModel(m);
  }

  fclose(out);
  free(paths);

  if (nsnap_written != nmodel * N_CONFIG) die("snapshot count mismatch");
  fprintf(stderr, "wrote %d snapshots (%d models x %d configs, %d RHS each)\n",
          nsnap_written, nmodel, N_CONFIG, N_RHS);
  return 0;
}
