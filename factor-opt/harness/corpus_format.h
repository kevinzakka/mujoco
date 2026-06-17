// Corpus + results binary format for the factorization harness.
// Self-describing: every block carries explicit counts and sizes.
// All deliverables live under factor-opt/; no MuJoCo source is modified.

#ifndef FACTOR_OPT_CORPUS_FORMAT_H_
#define FACTOR_OPT_CORPUS_FORMAT_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// corpus.bin layout
// ---------------------------------------------------------------------------
//   CorpusHeader
//   repeated nsnap times:
//     SnapHeader                          (model_id, config_id, nv, nC, K)
//     int32  rownnz[nv]
//     int32  rowadr[nv]
//     int32  colind[nC]
//     double Mval[nC]                     CSR values of lower-tri M
//     double rhs[K*nv]                    K seeded RHS vectors, row-major
//
// The CSR pattern uses MuJoCo's reduced-inertia convention: row k occupies
// [rowadr[k], rowadr[k]+rownnz[k]); the diagonal is the LAST entry of the row.

#define CORPUS_MAGIC   0x46414354u   // "FACT"
#define CORPUS_VERSION 1u
#define RESULT_MAGIC   0x46524553u   // "FRES"
#define RESULT_VERSION 1u

typedef struct {
  uint32_t magic;
  uint32_t version;
  int32_t  nsnap;     // number of snapshots
  int32_t  pad;
} CorpusHeader;

typedef struct {
  int32_t model_id;
  int32_t config_id;
  int32_t nv;
  int32_t nC;
  int32_t K;          // number of RHS vectors
  int32_t pad;
} SnapHeader;

// ---------------------------------------------------------------------------
// results file layout (one per kernel run)
// ---------------------------------------------------------------------------
//   ResultHeader
//   repeated nsnap times:
//     ResSnapHeader                       (model_id, config_id, nv, K)
//     repeated K times:
//       ResEntry                          (xhash, xnorm, residual)
//       double x[nv]                       the solve output itself

typedef struct {
  uint32_t magic;
  uint32_t version;
  int32_t  nsnap;
  int32_t  reserved;
} ResultHeader;

typedef struct {
  int32_t model_id;
  int32_t config_id;
  int32_t nv;
  int32_t K;
} ResSnapHeader;

typedef struct {
  uint64_t xhash;     // FNV-1a hash of the x bytes
  double   xnorm;     // ||x||_inf
  double   residual;  // ||M x - b||_inf / (||M||_inf*||x||_inf + ||b||_inf)
} ResEntry;

// ---------------------------------------------------------------------------
// small helpers (header-only, no MuJoCo dependency)
// ---------------------------------------------------------------------------

static inline uint64_t fnv1a64(const void* data, size_t n) {
  const unsigned char* p = (const unsigned char*)data;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static inline void die(const char* msg) {
  fprintf(stderr, "FATAL: %s\n", msg);
  exit(2);
}

static inline void rd(void* p, size_t sz, size_t n, FILE* f, const char* what) {
  if (fread(p, sz, n, f) != n) {
    fprintf(stderr, "FATAL: short read on %s\n", what);
    exit(2);
  }
}

static inline void wr(const void* p, size_t sz, size_t n, FILE* f,
                      const char* what) {
  if (fwrite(p, sz, n, f) != n) {
    fprintf(stderr, "FATAL: short write on %s\n", what);
    exit(2);
  }
}

#endif  // FACTOR_OPT_CORPUS_FORMAT_H_
