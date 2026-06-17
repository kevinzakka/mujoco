# Optimized factor+solve kernel — host, profile, plan

## Host (measured)

- CPU: **Apple M1 Max** (arm64, clang). `uname -m` = arm64. No AVX/SSE; **NEON
  available** (`#include <arm_neon.h>` allowed, `-march=native` allowed).
- Performance cores: 8 (`hw.perflevel0.physicalcpu=8`). We are single-threaded.
- Caches (performance cluster): L1d **128 KiB** (`hw.perflevel0.l1dcachesize`),
  L1i 192 KiB, **L2 12 MiB** shared per 4-core cluster. `hw.cachelinesize=128`
  bytes.
- The whole working set per snapshot (largest, floppy: nC=1153 doubles ≈ 9 KiB
  qLD + nv index arrays) fits comfortably in L1/L2; this is **compute / latency
  bound, not capacity bound**. The dominant cost is the dependent-chain scalar
  inner loops, not cache misses.

## Reference profile (measured on this machine)

`measure-factor.sh` full corpus reference median ≈ **6.83e-05 s/pass**.

Per-model reference median (single-model corpora, `run_factor_ref --time`):

| model       | nv   | ref s/pass | share |
|-------------|------|-----------|-------|
| arm26       | 2    | 2.1e-7    | 0.3%  |
| car         | 8    | 1.4e-6    | 2.1%  |
| humanoid    | 27   | 7.8e-6    | 11.7% |
| balloons    | 30   | 3.8e-6    | 5.6%  |
| humanoid100 | 627  | 2.33e-5   | 34.8% |
| floppy      | 1153 | 2.96e-5   | 44.2% |

**The big two (floppy + humanoid100) ≈ 79% of cost; +humanoid ≈ 91%.**

Factor-vs-solve split (analysis-only `prof_split`, 4 RHS per factor as in the
corpus):

| model       | factor s | solve s (4 RHS) | solve share |
|-------------|----------|-----------------|-------------|
| humanoid    | 3.6e-6   | 5.0e-6          | 58%         |
| balloons    | 1.1e-6   | 2.6e-6          | 70%         |
| humanoid100 | 8.7e-6   | 13.7e-6         | 61%         |
| floppy      | 9.6e-6   | 15.6e-6         | 62%         |

**Solve ≈ 60% of kernel time, factor ≈ 40%.** Both are worth touching. Solve has
three sub-passes: x ← L⁻ᵀx (scatter `x[colind]-=qLD*x_i`), x ← D⁻¹x (trivial),
x ← L⁻¹x (gather dot product, `dotSparse_opt`).

## Why these loops are slow (no AVX on the shipping lib path)

On arm64 `mjUSEAVX` is undefined, so the library `mju_addToScl`/`mju_scl` called
by `factorI_opt` are **plain scalar loops reached by a cross-TU function call**
(no inlining, no NEON). The factor inner update is the hot loop. Likewise the
solve sub-passes are scalar. The build is `-O2` with no `-march=native`.

## Candidate list — ranked by Amdahl payoff (runtime-fraction × expected speedup)

1. **Build flags `-O3 -march=native` + inline `mju_addToScl`/`mju_scl` into the
   factor TU** so clang can auto-vectorize the inner column updates with NEON and
   fuse the FMA. Touches ~40% (all of factor) and enables vectorization on the
   single biggest contiguous loop. Highest payoff, lowest risk. **Pick first.**
2. **Vectorize the solve L⁻¹ dot product (`dotSparse_opt`)** — already 4-way
   unrolled scalar; NEON `vmlaq`/`vfmaq` with a gather. Touches the largest solve
   sub-pass. Constrained by the gather (indexed loads) — NEON has no native
   gather, so this may only help the contiguous-index case.
3. **Tighten the solve L⁻ᵀ scatter loop** (`x[colind[adr]] -= qLD[adr]*x_i`):
   hoist, unroll, help the compiler. Indexed scatter, hard to vectorize, but a
   tight unroll can still cut loop overhead. Medium payoff.
4. **`restrict` / `const` annotations and pointer hoisting** in factor and solve
   to free the optimizer from aliasing assumptions. Pure enabling transform,
   stacks with #1.
5. **Block / supernodal dense diagonal** — large structural change; high
   implementation cost, uncertain payoff at these nv. Deferred / likely rejected.

## Chosen TARGET

Estimate, stated as one: this is tuned double-precision sparse LDLᵀ. NEON gives
2 doubles/lane; realistic auto-vectorization plus call-inlining and `-O3` on the
~40% factor and partial gains on the dot-product-heavy solve put a plausible
ceiling around **1.3×–1.6×** on the dominant models. **TARGET = 1.3×** measured
median speedup on the committed corpus, with all gates green and generalization
passing. A modest, real, gate-clean speedup is the goal — not a rewrite.

ASSUMPTION: clang auto-vectorizes the inlined contiguous `+= vec[i]*scl` loop
under `-O3 -march=native` — affects candidate #1's expected payoff; verified by
measurement, not assumed.
