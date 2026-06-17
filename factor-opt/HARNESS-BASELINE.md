# Factorization Harness — Identity Baseline (iteration 0)

This records the **proof that the measurement/correctness harness is sound**,
run against a kernel that is byte-identical in behavior to the shipping
reference. Because the optimized and reference kernels are the same algorithm,
the harness MUST report: 0.0 max solve-output difference, optimized residual ==
reference residual, certifier PASS against the independent dense Cholesky,
deterministic factorization, ≈1.0× speedup (any deviation is pure noise), and an
unchanged corpus checksum. It does. All evidence below is real command output.

No MuJoCo source under `src/`, `include/`, or `test/` was modified. Everything
here links the prebuilt `mujoco_install/lib/libmujoco.dylib`.

## What the harness is

| file | role |
|------|------|
| `factor-optimized/factor_optimized.c` | the ONLY thing the optimization loop edits. **Iteration-0 stand-in: a thin wrapper over the library `mj_factorI`/`mj_solveLD`** (see Deviations). |
| `factor-reference/factor_reference.c` | the anchor; thin shim over library `mj_factorI`/`mj_solveLD`; never edited. |
| `factor-reference/factor_kernel.h` | shared kernel interface both implementations satisfy. |
| `harness/corpus_format.h` | self-describing binary format for `corpus.bin` and the results files. |
| `harness/dump_corpus.c` | model zoo → `corpus.bin`; seeded configs + RHS; asserts every `M` SPD via a scratch dense Cholesky. |
| `harness/run_factor.c` | runs one kernel over the corpus; factor + solve; independent dense-matvec residual; writes results + (optional) raw `qLD` dump and median timing. |
| `harness/compare_factor.c` | **THE CORE** comparator: two-part match criterion, loud + non-zero on malformed/mismatched input. Not a byte diff. |
| `harness/certify_factor.c` | independent certifier: densify `M`, dense `mju_cholFactor`/`mju_cholSolve`, residual band. Shares no code with the sparse path. |
| `harness/measure-factor.sh` | median-of-N protocol + speedup ratio over the factor+solve kernel only. |
| `harness/models.txt` | the committed model zoo + seeds. |
| `Makefile` | builds 5 binaries against libmujoco, `-Wall -Wextra -Werror`. |
| `prove-factor-harness.sh` | end-to-end proof driver; terse summary by default, `--verbose` streams all steps. |

## Match criterion (as implemented; constants justified)

Per snapshot, per RHS `b`, both must hold:

1. **Residual band (independent):** with `x` the kernel's solve output,
   `r = ‖M x − b‖∞ / (‖M‖∞·‖x‖∞ + ‖b‖∞) ≤ max(8·r_ref, 1e-11)`.
   `M x` is a dense matvec sharing no code with the factor/solve path.
   `certify_factor` re-derives `r_ref` from the **dense Cholesky oracle**
   (`mju_cholFactor`/`mju_cholSolve`); `compare_factor` uses the reference
   kernel's per-entry residual. Measured across the zoo: the sparse kernel
   residual maxes at **3.30e-17** and the dense oracle at **7.64e-17** — both at
   machine-epsilon, so the `8×` band and `1e-11` floor are generous but never
   reached. These constants are the spec defaults; the measured residuals above
   justify them (they are not loosened).
2. **Solve-output agreement:** `‖x_opt − x_ref‖∞ / (‖x_ref‖∞ + 1e-300) ≤ 1e-6`.
   For the identity kernel this is exactly **0**.

Malformed input (count/dim mismatch, out-of-order or missing entry, trailing
bytes, NaN/Inf) → specific error on stderr, exit code 2. Verified below.

## The model zoo (`harness/models.txt`) + seeds

Paths are repo-root-relative; `dump_corpus` is invoked from the repo root.
3 seeded configs × 4 seeded RHS per model = 72 RHS over 18 snapshots.

| model | nv | nC | nM | character |
|-------|----|----|----|-----------|
| `model/tendon_arm/arm26.xml` | 2 | 3 | 3 | tiny, dense |
| `model/car/car.xml` | 8 | 35 | 35 | small, dense |
| `model/humanoid/humanoid.xml` | 27 | 243 | 243 | dense (nC==nM) |
| `model/balloons/balloons.xml` | 30 | 90 | 105 | sparse (nC<nM) |
| `model/humanoid/humanoid100.xml` | 627 | 843 | 2343 | large, sparse |
| `model/flex/floppy.xml` | 1153 | 1153 | 2305 | largest, flex |

Seeds (splitmix64; see `harness/dump_corpus.c`):
- config qpos seed: `0xC0FFEE ^ (model_idx+1)<<32 ^ (config_idx+1)`,
  `qpos = qpos0 + 0.05·U(−1,1)`.
- RHS seed: `0xB16B00B5 ^ (model_idx+1)<<32 ^ (config_idx+1)`, `rhs ~ U(−1,1)`.

## How to build and run (reproduce)

```sh
# from the MuJoCo repo root
make -C factor-opt                       # builds 5 binaries, zero warnings
./factor-opt/prove-factor-harness.sh             # terse summary + verdict
./factor-opt/prove-factor-harness.sh --verbose   # stream every step
```

The driver writes `factor-opt/work/{corpus.bin,ref.res,opt.res,...}` and a full
log to `factor-opt/work/prove.log`. The corpus is checksummed before and after.

## Proof: FINAL SUMMARY (real output)

```
================= FINAL SUMMARY (identity baseline) =================
COMPARE snapshots=18 rhs=72 max_solve_diff=0 max_opt_resid=3.3014024166622116e-17 ref_resid_at_max=3.3014024166622116e-17 max_ref_resid=3.3014024166622116e-17 over_tol=0 solve_tol=1.0e-06 resid_floor=1.0e-11 resid_factor=8
COMPARE VERDICT=PASS
CERTIFY snapshots=18 rhs=72 max_kernel_resid=3.3014024166622116e-17 max_dense_resid=7.638087323240065e-17 over_band=0 spd_fail=0 resid_floor=1.0e-11 resid_factor=8
CERTIFY VERDICT=PASS
MEASURE N=9 ref_median_s=7.6000113e-05 opt_median_s=6.90000597e-05 speedup=1.1014x
speedup: 1.1014x -> no measurable gain (byte-identical kernels; deviation is noise)
max solve-output deviation: 0
determinism: byte-identical (qLD dump sha256 e9f956b190778f556880e1d73c404f6186a95636dabd858abd5942c1ca7c3f18)
corpus sha256 before: 905674e2b711c046f4f43e306511a13bd32dfb6d605a685dc546c21816b81fa1
corpus sha256 after:  905674e2b711c046f4f43e306511a13bd32dfb6d605a685dc546c21816b81fa1
models in zoo: 6
VERDICT: PASS (all gates green)
====================================================================
```

### Speedup is noise, not gain

The factor+solve kernel for the whole corpus runs in ~70 µs; across repeated
prove runs the speedup oscillates around 1.0× (observed 0.94×, 0.97×, 1.00×,
1.03×, 1.10×). Identical code cannot beat itself — every deviation from 1.00× is
measurement noise, and is labeled as such. The median-of-N≥5 protocol is the
record; single runs are not evidence.

### Determinism (real output)

```
e9f956b190778f556880e1d73c404f6186a95636dabd858abd5942c1ca7c3f18  q1.bin
e9f956b190778f556880e1d73c404f6186a95636dabd858abd5942c1ca7c3f18  q2.bin
DETERMINISM: byte-identical
```

### Comparator is loud (real output)

```
# truncated opt file:
FATAL: short read on opt snap header                 (exit 2)
# snapshot-count mismatch (nsnap forced to 0):
FATAL: snapshot count mismatch: ref=18 opt=0         (exit 2)
# identical files:
COMPARE VERDICT=PASS                                 (exit 0)
```

## Acceptance checklist

- [x] `factor-optimized/` is self-contained and editable, byte-identical in
      behavior to the reference at iteration 0; no MuJoCo source modified.
- [x] `corpus.bin` from a committed 6-model zoo at seeded configs, 4 seeded RHS
      each, checksummed; every `M` asserted SPD (dense Cholesky full rank,
      positive pivots).
- [x] `compare_factor` implements residual-band + solve-output criterion (not a
      diff), loud on malformed/mismatched input, exit reflects PASS/FAIL.
- [x] `certify_factor` certifies residuals against an independent dense Cholesky.
- [x] `measure-factor.sh` is median-of-N (N=9) over the factor+solve kernel only.
- [x] `prove-factor-harness.sh` reports the identity baseline: ≈1.0×, 0.0 max
      solve-output deviation, residuals within band, certifier PASS, determinism,
      checksum unchanged.
- [x] No unmeasured performance claim; noise labeled as noise.

## Deviations from the spec

**`factor-optimized/` is a thin wrapper, not an extracted source copy
(iteration-0 stand-in).** The spec calls for a self-contained editable copy of
`mj_factorI`+`mj_solveLD` and their private helpers. Per the task's pragmatic
guidance, extracting the kernel's full private-helper closure out of
`engine_core_smooth.c` is deferred: at iteration 0 the optimized unit calls the
library kernel directly, which *guarantees* a genuinely bit-identical identity
baseline (the whole point of this step). The reference unit is the same wrapper,
so reference == optimized by construction — exactly the identity the harness
must prove. **`optimize-factor.md`'s first move is to replace
`factor-optimized/factor_optimized.c` with an inlined, editable private copy of
the CSR LDLᵀ kernel (renamed with an `_opt` suffix) and optimize that.** The
harness contract — `compare_factor`, `certify_factor`, `measure-factor.sh`,
`corpus.bin`, `prove-factor-harness.sh` — is unchanged across that swap.

## Notes for the optimization phase

- **Signatures (verified from `src/engine/engine_core_smooth.h`):**
  `mj_factorI(mat, diaginv, nv, rownnz, rowadr, colind, index)` and
  `mj_solveLD(x, qLD, diaginv, nv, n, rownnz, rowadr, colind, index)`.
  Pass `index = NULL` and `n = 1`. CSR is lower-triangular row-major; row `k`
  occupies `[rowadr[k], rowadr[k]+rownnz[k])` and its **diagonal is the last
  entry of the row** (`rownnz[k]-1` offset). Factorization is in place.
- **`M` is gathered exactly as the benchmark does:**
  `mju_gather(Mval, d->qM, m->mapM2M, m->nC)` → CSR values of size `nC`.
- **Type quirk:** `m->nv`, `m->nq`, `m->nC`, `m->nM` are `mjtSize` (`long long`)
  in this build; cast to `int` for `%d` / array sizing (done throughout).
- **Header resolution:** internal engine headers are included as
  `src/engine/engine_core_smooth.h` etc., so the build needs `-I<repo-root>`
  alongside `-I mujoco_install/include`. macOS link needs an rpath to the dylib
  (`-Wl,-rpath,<abs path to mujoco_install/lib>`). All in the Makefile.
- **Model path quirk:** models load by repo-root-relative path, so `dump_corpus`
  and the prove driver `cd` to the repo root. `coil.xml` was excluded — it
  requires the elasticity plugin to be loaded and fails `mj_loadXML`.
- **Residual magnitudes observed:** sparse kernel max residual **3.30e-17**,
  dense oracle max **7.64e-17** — both machine-epsilon. The sparse path is
  actually slightly tighter than the dense oracle here.

## Hand-off to optimize-factor.md

The optimization loop reuses, unchanged: `compare_factor` (tolerance
comparator), `certify_factor` (independent dense-Cholesky residual certifier),
`measure-factor.sh` (median-of-N protocol), `corpus.bin` (checksummed committed
input), and `prove-factor-harness.sh` (per-turn proof, wired as
`--hook-per-run`). The optimization work edits **only**
`factor-optimized/factor_optimized.c`, starting from this identity baseline as
iteration-0 evidence that the measurement pipeline itself is sound.
