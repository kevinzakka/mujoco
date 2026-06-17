# Optimization log — MuJoCo inertia factor+solve kernel

One entry per hypothesis. Each records: the change, before/after **per-corpus-pass
median** (from `measure-factor.sh`, which auto-calibrates the inner repeat so each
timed sample is many passes — stable to ~±0.6%), the speedup vs the reference
baseline, every correctness gate, the keep/revert decision, and the cost.

**Ruler:** `prove-factor-harness.sh` gates each change — comparator (solve-output
within 1e-6 rel + residual band), independent dense-Cholesky certifier,
determinism (byte-identical factor), corpus sha256 unchanged. A change that fails
any gate is reverted regardless of speed. Realistic expectation: this is
double-precision sparse LDLᵀ already reasonably tuned, so the honest target is a
small single-digit factor on the large models, not a rewrite.

**Revert discipline (this branch):** the MuJoCo repo is on an active unrelated
feature branch, so we do NOT `git commit` to snapshot kept states. Instead the
last green kernel is copied to `factor_optimized.last-good.c`; revert = copy it
back.

---

## Iteration 0 — baseline (pulled, unmodified copy)

**State.** `factor-optimized/` is a faithful, editable transcription of the
shipping kernel: `factorI_opt` (= `mj_factorI`), `solveLD_opt` (= `mj_solveLD`,
single-vector path), and `dotSparse_opt` (= the scalar `mju_dotSparse`, inlined
because it is a header `static inline`, not an exported symbol). `mju_addToScl` /
`mju_scl` are still called from libmujoco. No optimization applied yet — this is
the proven-identical starting point.

**Corpus.** 18 snapshots / 72 RHS from 6 models, `nv ∈ {2, 8, 27, 30, 627, 1153}`
(`arm26`, `car`, `humanoid`, `balloons`, `humanoid100`, `floppy`). sha256
`905674e2…81fa1`, read-only.

**Measurement (per-corpus-pass median, N=9, inner=516 ≈ 33 ms/sample).**

| | median s/pass | spread |
|---|---|---|
| reference (`build/run_factor_ref`) | ≈ 6.8e-05 | — |
| optimized (`build/run_factor_opt`) | ≈ 6.4e-05 | ~±0.6% run-to-run |

**Speedup vs reference: ~1.0× (noise).** Identical algorithm; deviation from
1.00× is measurement noise. This is the calibration reading, not a result.

**Gates (all green).**
- comparator: `max_solve_diff = 0`, `over_tol = 0` → **PASS** (bit-identical to
  reference on arm64, as expected: no AVX, same summation order, same library
  BLAS on the factor side).
- certifier: kernel residual `3.30e-17` vs independent dense-Cholesky `7.64e-17`,
  `spd_fail = 0` → **PASS**.
- determinism: two factor runs byte-identical (qLD sha256 `e9f956b1…c3f18`).
- corpus sha256 unchanged.

**Verification:** `run_factor_opt` links no `mj_factorI` from libmujoco
(`nm -u | grep mj_factorI` → 0) — confirming the kernel is our own editable copy,
not a wrapper.

**Cost:** harness build + pull + measure-protocol hardening (prior turns).

**Decision:** KEEP as the baseline. `cp factor_optimized.c
factor_optimized.last-good.c`. Profiling and the first real candidate follow in
iteration 1.

---

## Iteration 1 — `-O3 -march=native` + inline `mju_addToScl`/`mju_scl` (NEON)

**Hypothesis.** On arm64 `mjUSEAVX` is undefined, so the library
`mju_addToScl`/`mju_scl` called from `factorI_opt`'s hot inner loop are scalar
AND reached through a cross-TU function call (no inline, no NEON). Replacing them
with `static inline restrict` copies and compiling the kernel TU at `-O3
-march=native` lets clang inline + auto-vectorize the contiguous `res[i] +=
vec[i]*scl` / `res[i] = vec[i]*scl` loops with NEON (2 doubles/lane, FMA).
Element-wise loops (no reduction) → bitwise-identical output preserved.

**Change.** (a) Added `addToScl_opt`/`scl_opt` (`static inline`, `restrict`) and
called them in `factorI_opt` instead of the libmujoco symbols. (b) Makefile: new
`OPTFLAGS := -O3 -march=native …`, split `factor_optimized.o` into its own rule
built with OPTFLAGS; `run_factor.c` stays at harness `CFLAGS`. The reference
target and harness are untouched.

**Measurement (full corpus, measure-factor.sh N=9, two runs):**

| | median s/pass |
|---|---|
| reference | 6.69e-05 … 6.79e-05 |
| optimized (iter 1) | 5.16e-05 … 5.19e-05 |

**Speedup vs reference: ~1.29–1.32× (≈1.30×).** Per-model (single-model corpora):
floppy 2.99e-5→1.91e-5 (**1.56×**), humanoid100 2.34e-5→1.82e-5 (**1.29×**),
balloons 1.07×, humanoid 1.04×. Gains concentrate in the big models where factor
fill (dense inner updates) dominates — exactly the Amdahl target.

**Gates (all green, prove-harness exit 0).**
- comparator: `max_solve_diff = 0`, `over_tol = 0` → **PASS** (vectorizing
  element-wise loops keeps bits exact).
- certifier: kernel residual `3.30e-17` vs dense-Cholesky `7.64e-17`, `spd_fail=0`
  → **PASS**.
- determinism: two factor dumps byte-identical, qLD sha256 `e9f956b1…c3f18` —
  **unchanged from baseline** (bit-for-bit same factor as scalar path).
- corpus sha256 `905674e2…81fa1` unchanged.

**Cost:** ~1 iteration; tokens: n/a — not run under nagent.

**Decision:** **KEEP.** Re-snapshotted `factor_optimized.last-good.c` and
`Makefile.last-good`. This is the primary lever; remaining candidates target the
solve sub-passes (≈60% of time), which iter 1 leaves scalar.

---

## Iteration 2 — `restrict` solve pointers + specialize index==NULL path

**Hypothesis.** Solve is now ~60–85% of kernel time (factor was vectorized in
iter 1). In `solveLD_opt`, (a) `qLD`/`colind`/`rownnz`/`rowadr` were not
`restrict`, so the compiler must assume the scatter `x[colind[adr]] -=
qLD[adr]*x_i` may alias `qLD`; (b) every row re-evaluated `index ? index[k] : k`
though `index` is always NULL here. Marking the read-only pointers `restrict` and
splitting out the identity-order (index==NULL) path removes per-row indirection
and aliasing barriers without touching arithmetic order (so output stays
bit-exact). Also folded the redundant `rownnz[i]==1` guard in the L⁻¹ pass into
the existing `d>0` test (same behavior, one fewer branch).

**Measurement (full corpus, N=9, three runs):**

| | median s/pass |
|---|---|
| reference | 6.75e-05 … 6.86e-05 |
| optimized iter 1 | 5.16e-05 … 5.19e-05 |
| optimized iter 2 | 5.07e-05 … 5.12e-05 |

**Speedup vs reference: ~1.33× (was ~1.30×).** ~1.5% improvement over iter 1 —
small, near the ±0.6% noise floor but consistently on the faster side across
runs, and it is a pure simplification (dead-branch removal) that strengthens the
optimizer's contract. The dominant solve passes (indexed scatter / gather) cannot
be vectorized without reordering FP reductions, which the prove harness forbids
(`max_solve_diff` must be exactly 0), so this is near the solve ceiling under the
bit-exact constraint.

**Gates (all green, prove-harness exit 0).**
- comparator `max_solve_diff=0`, `over_tol=0` → PASS.
- certifier kernel resid `3.30e-17` vs dense `7.64e-17`, `spd_fail=0` → PASS.
- determinism byte-identical, qLD sha256 `e9f956b1…c3f18` unchanged.
- corpus sha256 unchanged.

**Cost:** ~1 iteration; tokens: n/a — not run under nagent.

**Decision:** **KEEP** (pure simplification + small consistent gain). Re-snapshotted
`factor_optimized.last-good.c`.

---

## Iteration 3 — `restrict` factor pointers + specialize index==NULL/diaginv path

**Hypothesis.** Mirror iter 2 on `factorI_opt`: mark `mat`/`diaginv`/CSR pointers
`restrict` and split the identity-order, diaginv-present fast path so the per-row
`index ? index[j] : j` and `if (diaginv)` branches vanish. Faithful general
fallback (non-NULL index, or NULL diaginv) is retained.

**Measurement (full corpus, N=9, three runs):**

| | median s/pass |
|---|---|
| reference | 6.68e-05 … 6.89e-05 |
| optimized iter 2 | 5.07e-05 … 5.12e-05 |
| optimized iter 3 | 5.08e-05 … 5.25e-05 |

**Speedup vs reference: ~1.31–1.33× — no measurable change vs iter 2.** clang had
already hoisted these loop-invariant branches; the factor inner loop was already
NEON-vectorized in iter 1 and is only ~15–40% of kernel time now. This is a pure
simplification with **no regression**, kept for clarity/faithfulness, not for a
speed gain. Honest verdict: factor is at its practical ceiling under the bit-exact
constraint.

**Gates (all green, prove-harness exit 0).**
- comparator `max_solve_diff=0`, `over_tol=0` → PASS.
- certifier kernel resid `3.30e-17`, `over_band=0`, `spd_fail=0` → PASS.
- determinism byte-identical (qLD sha256 unchanged).
- corpus sha256 unchanged.

**Cost:** ~1 iteration; tokens: n/a — not run under nagent.

**Decision:** **KEEP** (pure simplification, no regression). Re-snapshotted.

### Candidate-list status after iter 3
1. ~~`-O3 -march=native` + inline addToScl/scl~~ — DONE iter 1 (the big win, 1.30×).
2. Vectorize solve L⁻¹ dot product — **REJECTED:** the prove harness requires
   `max_solve_diff == 0` exactly, so any change to the `(r0+r2)+(r1+r3)` reduction
   order (which NEON re-association would force) fails the gate. Cannot vectorize
   the gather without reordering. Bit-exact ceiling.
3. Tighten solve L⁻ᵀ scatter — partially DONE iter 2 (restrict + branch removal,
   ~1.5%). Full vectorization blocked: indexed scatter, no NEON scatter HW, and
   reordering would risk the bit-exact gate anyway.
4. restrict/const + pointer hoisting — DONE iters 2 & 3.
5. Block/supernodal dense diagonal — **REJECTED:** large rewrite; the dense
   diagonal already auto-vectorizes via inlined addToScl; and any blocked
   reduction reorders FP → breaks `max_solve_diff==0`. Payoff uncertain, cost high,
   gate-hostile.

Candidate list is exhausted under the bit-exact (`max_solve_diff==0`) constraint.

---

## REFRAME (iterations 4+) — the real contract is tolerance, not bit-exactness

The comparator (`compare_factor`) PASSES iff `over_tol == 0`, i.e. **‖x_opt −
x_ref‖∞ / ‖x_ref‖∞ ≤ 1e-6** (the solve-output band) AND the residual band
`r_opt ≤ max(8·r_ref, 1e-11)`. It is **not** a byte diff — see the file header:
"later optimized kernels are not bit-identical." The iter-2/iter-3 reasoning that
"`max_solve_diff` must be exactly 0 → cannot vectorize the solve" was a
self-imposed constraint the comparator never required. A solve change that makes
`max_solve_diff` ~1e-16 (a few ULPs from reassociated FP reductions) is correct
and PASSing, provided `over_tol == 0`, the certifier residual stays in band, and
the factor stays deterministic run-to-run. Iterations 4-6 below re-evaluate the
solve under this real contract.

Caveat recorded for the record: `prove-factor-harness.sh` carries an *extra*,
harness-side gate (line ~77) that hard-fails if `max_solve_diff != 0` — a holdover
from the iteration-0 identity-baseline self-check. We are not permitted to edit
the harness, so any **kept** change must keep `max_solve_diff == 0` to leave
prove-harness green. Reorder-based solve changes (iter 6) are therefore evaluated
directly via `compare_factor`/`certify_factor` (the true contract) and reverted if
they give no speed benefit — which, as measured below, they do not.

---

## Iteration 4 — `-ffp-contract=fast` on the optimized object only

**Hypothesis.** Let clang contract `a*b + c` into a single FMA in the solve's dot
products / AXPY updates, scoped to `factor_optimized.o` only (not harness/ref).
Expected to either speed up the reductions or, at worst, be a no-op; may perturb
bits ~1 ULP.

**Change.** Makefile OPTFLAGS: add `-ffp-contract=fast` (optimized object only).

**Gates.** Build clean. `compare_factor`: **`max_solve_diff = 0`**, `over_tol = 0`
→ PASS (the contraction produced bit-identical output here — the scalar 4-way dot
already separates the `*` and `+=`, and clang did not fuse across the independent
accumulators). `certify_factor`: kernel resid `3.30e-17`, `over_band=0`,
`spd_fail=0` → PASS.

**Measurement (full corpus, N=9, 3 runs).** speedup `1.281x / 1.294x / 1.359x`
(median ≈ 1.30x) vs baseline ≈ 1.32x — **no change, within the ±0.6% noise floor.**

**Decision:** **REVERT** (measured no-op; nothing to keep — flag adds no value and
no bits changed). Makefile restored.

---

## Iteration 5 — hand-written NEON intrinsics for `dotSparse_opt`

**Hypothesis.** The L⁻¹ pass `x[i] -= Σ qLD[adr]·x[colind[adr]]` is a gather-dot.
Replace the scalar 4-way tree with two 128-bit `float64x2_t` FMA accumulators
(4 doubles in flight), loading `qLD`/`vec1` contiguously via `vld1q_f64` and
building the gathered `vec2` lane-pair from scalar loads `vec2[ind1[i]]`, then
`vaddvq_f64`. This reassociates the reduction, so a ~1e-16 solve diff is expected
and acceptable under the 1e-6 contract.

**Change.** `dotSparse_opt`: `#if __ARM_NEON` NEON path (2× `vfmaq_f64`), scalar
fallback retained. `#include <arm_neon.h>`.

**Gates.** Build clean. `compare_factor`: `max_solve_diff = 0` (the gathered
lane-pairs are still built from the same scalar loads and the FMA happened to
match the scalar bits on this corpus — rows are short/sparse so most of the work
falls in the scalar tail where `d ≤ 4`), `over_tol = 0` → PASS. `certify_factor`:
resid `3.30e-17` → PASS.

**Measurement (full corpus, N=9, 3 runs).** speedup `1.270x / 1.345x / 1.307x`
(median ≈ 1.31x) — **no measurable gain.**

**Mechanism finding (the point of this experiment).** The dot product does **not**
speed up because it is **gather-bound, not arithmetic-bound**: `vec2[ind1[i]]`
issues a dependent scalar load per term to a scattered address, and arm64 has no
hardware gather. The two FMA accumulators sit idle waiting on those loads; widening
the arithmetic does nothing when the bottleneck is memory latency. The L rows are
also short and sparse, so the vectorized body rarely runs.

**Decision:** **REVERT** (no gain, added complexity). Kernel restored to last-good.

---

## Iteration 6 — aggressive scoped fast-math (`-fassociative-math` + friends)

**Hypothesis.** Give clang full freedom to reassociate and auto-vectorize the
solve reductions: `-fassociative-math -freciprocal-math -fno-signed-zeros
-fno-trapping-math -ffp-contract=fast`, scoped to `factor_optimized.o` only. This
is the strongest legal lever for letting the compiler NEON-ify the
reduction/AXPY. (Stops short of full `-ffast-math`, which assumes no NaN/Inf and
risks the SPD/residual gate.)

**Change.** Makefile OPTFLAGS (optimized object only) += the five flags above.

**Gates.** Build clean. `compare_factor`: **`max_solve_diff = 2.5679e-16`**,
`over_tol = 0` → **PASS** — this is the expected, correct behavior: reassociation
moved a few ULPs, comfortably inside the 1e-6 solve band. `certify_factor`: kernel
resid **unchanged at `3.30e-17`**, `over_band=0`, `spd_fail=0` → PASS (residual
quality untouched). Determinism: the optimized kernel is still deterministic
run-to-run (same flags → same code → same bytes); the ~1e-16 diff is vs the
*reference*, not run-to-run.

**Measurement (full corpus, N=9, 4 runs).** speedup `1.330x / 1.296x / 1.303x /
1.291x` (median ≈ 1.30x) vs baseline ≈ 1.32x — **no measurable gain.**

**Mechanism finding.** Even with full reassociation freedom (proven active by the
nonzero `max_solve_diff`), the solve does not get faster. This corroborates iter 5:
the solve's two dominant passes are **memory-bound on indexed gather/scatter**
(`vec2[ind1[i]]` and `x[colind[adr]] -= …`), not FP-reduction-latency-bound.
Reassociation removes a dependency chain that was never the critical path.

**Decision:** **REVERT.** It changes bits (max_solve_diff 2.57e-16) for zero speed
benefit; under "keep only if it measurably helps," there is nothing to keep.
Makefile restored. NOTE: even had it given a gain, the prove-harness line-77 gate
would have rejected it — but the dispositive reason is the measured no-gain.

---

## Candidate-list status after iter 6 — CLOSE-OUT (real tolerance contract)

The iter-3 entries that read "REJECTED because `max_solve_diff==0`" were rejected
for the **wrong reason** (a bit-exactness standard the comparator never imposes).
Re-evaluated under the actual 1e-6 / residual-band contract:

1. ~~`-O3 -march=native` + inline addToScl/scl~~ — DONE iter 1, the big win
   (≈1.30–1.33×). The factor inner update is the only arithmetic-bound hot loop
   and it is NEON-vectorized. **At its ceiling.**
2. Vectorize solve L⁻¹ gather-dot — **IMPLEMENTED & MEASURED (iter 5, NEON; iter 6,
   compiler reassoc). No gain — gather-bound, not arithmetic-bound; arm64 has no
   gather HW.** Reverted. (Supersedes the old "REJECTED: max_solve_diff must be 0"
   — the real reason is *memory-bound, measured zero speedup*, not the tolerance.)
3. Solve L⁻ᵀ scatter — partially tightened iter 2 (restrict + branch removal). Full
   vectorization is **scatter-bound** (`x[colind[adr]]` indexed RMW, no NEON
   scatter HW); the iter-6 fast-math pass covered it with the rest of the solve and
   showed no gain. **Memory-bound ceiling, measured.**
4. restrict/const + pointer hoisting — DONE iters 2 & 3.
5. Block/supernodal dense diagonal — still REJECTED, but now for the correct
   reason: the dense diagonal already auto-vectorizes via the inlined `addToScl`;
   a blocked rewrite is large and the solve (the remaining ~60%) is memory-bound,
   so the achievable upside is bounded by gather/scatter latency, not arithmetic.
   Cost high, measured upside ~0.

**Close-out verdict.** Both halves are at their practical single-thread ceiling on
this M1 Max. Factor: arithmetic-bound inner update is NEON-vectorized (iter 1).
Solve: memory-bound on indexed gather/scatter — proven by two independent
vectorization attempts (hand NEON, compiler reassoc) that changed the arithmetic
(iter 6: `max_solve_diff` 2.57e-16, comparator still PASS) yet moved the median by
0% beyond noise. The remaining ceiling is DRAM/cache latency on scattered
accesses, not instruction throughput; lifting it would require a different sparse
storage / reordering scheme (out of scope, and a different kernel). **Best kept
result stays iteration 3 at ≈1.33× with `max_solve_diff == 0` and prove-harness
green.**
