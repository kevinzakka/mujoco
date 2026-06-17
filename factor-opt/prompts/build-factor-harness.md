# MuJoCo Inertia Factorization — Test/Measurement Harness (Scaffold)

Instructions for an LLM agent. Build the **corpus, comparison, certification,
and measurement scaffold** that `optimize-factor.md`'s loop will use to prove an
optimized factorization kernel is correct and fast. In this task the "optimized"
kernel is a **verbatim, self-contained copy of the shipping kernel**, so the
scaffold must demonstrate the trivial **identity baseline** with real command
output: **no measurable speedup (~1.0×)**, **solve output bit-identical to the
reference (0.0 max difference)**, **residual identical to the reference on every
snapshot**, and **deterministic factorization** (two runs byte-identical). Never
report a step as complete without running it; never fabricate a measurement.

If the scaffold reports anything other than the identity baseline above when the
two kernels are byte-identical copies, **the scaffold is wrong — fix it before
finishing.**

## Why this exists

`optimize-factor.md` requires, every iteration, that you compare an optimized
factorization against the reference under a tolerance criterion, certify it
independently, measure a median speedup, and run a generalization check across a
model zoo. That machinery must exist and be **proven correct before any real
optimization is attempted** — otherwise a passing report proves nothing. This
task builds it and proves it against the one input for which the answer is known
exactly: a byte-identical copy of the shipping kernel.

The data-oriented point: the core of this scaffold is a **results-file
comparator** plus an **independent residual certifier** — pure data transforms
over the factor/solve outputs. When the two producing kernels are identical, the
comparator must yield PASS with zero difference. Build and verify those
transforms first; everything else is plumbing.

## The target kernel (read the real code first)

- `mj_factorM(m, d)` (`src/engine/engine_core_smooth.c`) → fills `d->qLD`
  (sparse LDLᵀ of the joint-space inertia `M`) and `d->qLDiagInv`. Its inner
  worker is `mj_factorI(mat, diaginv, nv, M_rownnz, M_rowadr, M_colind, …)`.
- `mj_solveLD(x, qLD, qLDiagInv, nv, n, …)` / `mj_solveM(m, d, x, y, n)` — the
  sparse triangular solve `x = M⁻¹ y` that consumes the factorization.
- The CSR sparsity (`m->M_rownnz`, `m->M_rowadr`, `m->M_colind`, sizes `m->nv`,
  `m->nC`, `m->nM`) describes `M`. See `test/benchmark/factorI_benchmark_test.cc`
  for how `M` is gathered (`mju_gather(M, d->qM, m->mapM2M, m->nC)`) and how
  `mj_factorI` is called — reproduce that call exactly.
- **Independent oracle (shares no code with the sparse path):**
  `mju_cholFactor` / `mju_cholSolve` (dense Cholesky, `mujoco.h`). Densify `M`
  into a full `nv×nv` matrix, factor it densely, solve with it — an answer
  derived from entirely separate code.

Read these before writing anything. Do not modify any MuJoCo source under
`src/`, `include/`, or `test/`. This scaffold lives entirely under `factor-opt/`
and links the prebuilt `libmujoco`.

## Fixed facts to reproduce (do not relax)

- **The corpus is the committed input.** A snapshot is `{model_id, config_id, M
  in CSR (values + M_rownnz/M_rowadr/M_colind), nv, nC, and K seeded random RHS
  vectors b}`. Generate it once from a fixed model zoo at seeded configurations,
  write it to a single binary file, and **checksum it (sha256)**. The corpus is
  read-only thereafter; the prove driver verifies its checksum before and after
  every run. `M` is SPD for valid configs — assert it (all pivots > 0).
- **Match criterion (both must hold), per snapshot, per RHS `b`:**
  1. **Residual certification (independent):** with `x` the solve output of the
     kernel under test, the relative residual
     `r = ‖M x − b‖∞ / (‖M‖∞·‖x‖∞ + ‖b‖∞)` must satisfy
     `r ≤ max(8 · r_ref, 1e-11)`, where `r_ref` is the reference kernel's
     residual on the same snapshot. `M x` is computed by a dense matvec that
     shares no code with either factorization. This certifies the output really
     solves `M x = b`, independent of the reference's *choice* of x.
  2. **Solve-output agreement:** `‖x_opt − x_ref‖∞ / (‖x_ref‖∞ + 1e-300) ≤
     1e-6` (LDLᵀ pivots are essentially unique for SPD `M`, so the band is
     tight). An absolute floor avoids division blow-up for near-zero solutions.
  A change that violates either, on any snapshot, is **rejected, however fast.**
  These default constants may be re-justified in `HARNESS-BASELINE.md` with
  evidence (e.g. measured reference residuals across the zoo) — but never
  loosened silently.
- **Determinism:** same `M` bytes in → byte-identical `qLD`/`qLDiagInv` out,
  every run.
- **Measurement protocol:** time the **factor + solve** kernel only (model load,
  `mj_forward`, corpus I/O, and densification are excluded). Run each kernel
  over the whole corpus N≥5 times back-to-back; report the **median** total
  kernel time. Speedup = reference median ÷ optimized median. Single runs are
  not evidence. (You may also wire a Google-Benchmark target mirroring
  `factorI_benchmark_test.cc`, but the median-of-N script is the protocol of
  record.)
- **No overfitting:** nothing keyed to a specific model or snapshot; the kernel
  must be a general transform of any valid CSR SPD `M`.

## Deliverables (create exactly these; start from a faithful copy)

    factor-optimized/              self-contained, EDITABLE copy of the shipping
                                   factor+solve kernel (mj_factorI + mj_solveLD
                                   and any private helpers they need), renamed
                                   with an _opt suffix and compiled into the
                                   harness. At start it is byte-for-byte the same
                                   algorithm as the library reference — verify
                                   identity. This is the only thing the
                                   optimization loop will edit.
    factor-reference/             a thin shim that calls the library's mj_factorI
                                   / mj_solveLD (the anchor). It is NOT edited.
    harness/dump_corpus.c         model-zoo → corpus.bin (+ sha256). Loads each
                                   model, applies seeded configs, mj_forward,
                                   gathers M in CSR, emits K seeded RHS vectors.
    harness/run_factor.c          runs ONE kernel (ref or opt, chosen by flag)
                                   over corpus.bin: factor each M, solve each b,
                                   compute the independent dense-matvec residual,
                                   write a results file (per snapshot/per b:
                                   x-vector hash+norm, residual, and x itself).
    harness/compare_factor.c      the comparator: reads two results files, applies
                                   the match criterion above, loud + non-zero on
                                   any malformed/length-mismatched input. THE CORE
                                   — build and verify this first.
    harness/certify_factor.c      the independent certifier: densify M, dense
                                   Cholesky (mju_cholFactor/Solve), residual check
                                   — shares no code with the sparse path. Run it
                                   against BOTH the reference and the optimized
                                   outputs as the ground-truth baseline.
    harness/measure-factor.sh     median-of-N protocol + speedup ratio.
    harness/models.txt            the committed model zoo (paths under model/ +
                                   plugin/, e.g. humanoid/humanoid100.xml,
                                   plugin/elasticity/coil.xml, humanoid/humanoid.xml,
                                   plus a few more spanning small/large nv and
                                   dense/sparse M). State the seeds used for
                                   configs and RHS.
    Makefile (or CMakeLists)      builds everything against the prebuilt
                                   libmujoco; -Wall -Wextra -Werror. Distinct
                                   binaries; do not touch any MuJoCo target.

### The comparator (build this first, it is the core)

`harness/compare_factor.c` → `compare_factor <ref-results> <opt-results>`:

- Reads both results files in a fixed parseable format.
- **Loud on malformed input:** different snapshot/RHS counts, a missing or
  out-of-order entry, a NaN/Inf → print a specific error and exit non-zero. Do
  not silently skip.
- Applies the two-part match criterion (residual band + solve-output agreement);
  tracks and prints, in fixed parseable form: snapshot count, RHS count, **max
  relative solve-output difference**, **max optimized residual** and the
  reference residual it is compared against, and the count of entries over
  tolerance.
- Exit 0 only when zero entries exceed either bound. This is the comparator
  `optimize-factor.md` reuses; treat it as the contract boundary. It is **not**
  a byte `diff`.

## Top-level proof driver

`factor-opt/prove-factor-harness.sh`: runs the whole scaffold end to end on the
identity copy and prints a FINAL SUMMARY. Default output is terse (summary +
verdict only; everything to a log); `--verbose` streams every step. This is what
gets wired into `optimize-factor.md` as `--hook-per-run`. Steps, each with real
output:

1. Clean build of `factor-reference/`, `factor-optimized/`, and `harness/`
   (`-Wall -Wextra -Werror`, zero warnings).
2. `dump_corpus` → `corpus.bin`; record + verify its sha256.
3. `run_factor --reference corpus.bin → ref.res`.
4. `run_factor --optimized corpus.bin → opt.res`.
5. **GATE** `compare_factor ref.res opt.res` → expect: 0 entries over tolerance,
   **max solve-output diff 0.0**, optimized residual identical to reference,
   PASS. (Because the kernels are identical here the files are in fact identical;
   the comparator still applies the tolerance criterion, since later optimized
   versions will not be identical.)
6. **GATE** `certify_factor` on both ref.res and opt.res → every solve residual
   within band against the independent dense-Cholesky ground truth; all `M` SPD.
7. Median-of-N for both kernels via `measure-factor.sh` → report both medians
   and the speedup. Expect **≈1.0×**; state explicitly that any deviation is
   pure measurement noise (identical code cannot beat itself) — do not dress
   noise up as speedup.
8. Generalization: confirm the corpus spans ≥4 models of differing `nv`; report
   per-model max residual and per-model speedup. (Here all ≈1.0×, 0.0 diff.)
9. **GATE** Determinism: run the optimized factor twice on the corpus; the two
   `qLD` dumps must be byte-identical.
10. Verify corpus sha256 unchanged. Print a final summary block: speedup ≈1.0×,
    max solve-output deviation 0.0, residuals within band, certifier PASS,
    determinism confirmed, corpus checksum unchanged. Exit 0 only if every GATE
    passed.

Save the summary, with the actual commands and output, to
`factor-opt/HARNESS-BASELINE.md`.

## Done means (identity baseline, with evidence)

- Builds clean: `-Wall -Wextra -Werror`, links only the prebuilt libmujoco.
- `compare_factor` on the corpus: 0 over tolerance, **max solve-output diff
  0.0**, optimized residual == reference residual, PASS.
- `certify_factor`: every residual within band vs the independent dense
  Cholesky; all `M` SPD.
- Speedup ≈1.0× reported as **no measurable gain**, with the N samples and
  medians shown; noise labeled as noise.
- Determinism: two optimized factor runs byte-identical.
- Corpus sha256 unchanged; no MuJoCo source touched.

## Hand-off to optimize-factor.md

State explicitly in `HARNESS-BASELINE.md` that the optimization loop reuses,
unchanged: `compare_factor` as the tolerance comparator, `certify_factor` as the
independent residual certifier, `measure-factor.sh` as the median-of-N protocol,
`corpus.bin` (checksummed) as the committed input, and `prove-factor-harness.sh`
as the per-turn proof. The optimization work then edits only `factor-optimized/`,
with this identity baseline as iteration-0 evidence that the measurement
pipeline itself is sound.

## Acceptance checklist

- [ ] `factor-optimized/` is a self-contained, editable copy of the shipping
      factor+solve kernel, byte-identical in behavior to the library reference
      at start; no MuJoCo source modified.
- [ ] `corpus.bin` generated from a committed model zoo at seeded configs, with
      K seeded RHS vectors, and checksummed; `M` asserted SPD.
- [ ] `compare_factor` implements the residual-band + solve-output criterion
      (not a `diff`), loud on malformed/mismatched input, exit code reflects
      PASS/FAIL.
- [ ] `certify_factor` certifies residuals against an independent dense Cholesky
      that shares no code with the sparse path.
- [ ] `measure-factor.sh` implements median-of-N and a speedup ratio over the
      factor+solve kernel only.
- [ ] `prove-factor-harness.sh` runs all steps and reports the identity
      baseline: ≈1.0×, 0.0 max solve-output deviation, residuals within band,
      certifier PASS, determinism, checksum unchanged.
- [ ] `HARNESS-BASELINE.md` records the proof with real command output and the
      hand-off note.
- [ ] No unmeasured performance claim anywhere; noise labeled as noise.
