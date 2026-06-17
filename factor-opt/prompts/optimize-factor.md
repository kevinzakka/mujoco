# MuJoCo Inertia Factorization — Optimized Implementation

Instructions for an LLM agent. You must **implement, build, certify, measure,
and iterate** until the performance target is reached or you can prove no
further opportunity remains. "Done" means one of the two exit criteria at the
bottom is met **with real command output as evidence**. Never report a step as
complete without having run it; never fabricate a measurement.

## Context

The shipping kernel is the reference, and the full **test/certify/measure
harness** already exists and is proven. Before writing any code:

- Read `../README.md` and the target kernel in `src/engine/engine_core_smooth.c`
  (`mj_factorM`, `mj_factorI`, `mj_solveLD`, `mj_solveM`) and the CSR sparsity in
  `mjmodel.h` (`M_rownnz`, `M_rowadr`, `M_colind`, `nv`, `nC`, `nM`).
- Read `HARNESS-BASELINE.md` — the corpus/comparator/certifier/measure scaffold
  is **already built and proven** against an identity copy (iteration-0 evidence
  that the measurement pipeline is sound). You **reuse** it; you do not rebuild
  it.
- Read `prompts/build-factor-harness.md` for the exact contracts.

Reference baseline is whatever you **measure** on this machine with
`measure-factor.sh`, not a number quoted from anywhere.

## The harness already exists — use it, do not rebuild it

Your job is to edit `factor-optimized/` (the kernel) and nothing else in the
harness. Reuse these exactly as they are:

- `harness/run_factor --reference|--optimized corpus.bin → results` — runs one
  kernel over the committed corpus and writes the results file.
- `compare_factor <ref-results> <opt-results>` — the tolerance comparator:
  (1) residual band `r ≤ max(8·r_ref, 1e-11)` and (2) solve-output agreement
  `‖x_opt − x_ref‖∞/‖x_ref‖∞ ≤ 1e-6`, on every snapshot/RHS. Exits 0 only on
  PASS. **This is the matcher; never substitute a `diff`.**
- `certify_factor` — independent residual certification via dense Cholesky
  (`mju_cholFactor`/`mju_cholSolve`), sharing no code with the sparse path.
- `measure-factor.sh` — median-of-N over the factor+solve kernel only (model
  load, `mj_forward`, corpus I/O, densification excluded). **Use it for every
  number — do not invent a second protocol.**
- `corpus.bin` — the committed, checksummed input. Read-only. Generate alternate
  corpora for the generalization check with `dump_corpus` under a different seed,
  written elsewhere; **never overwrite `corpus.bin`.**
- `prove-factor-harness.sh` — runs all gates end to end; a convenient re-confirm.

## Goal

An optimized `factor + solve` kernel with a target of **<TARGET>×** improvement
in measured median kernel time over the reference on this machine, producing
results that **match the reference within tolerance** (residual within band;
solve output within `1e-6` relative; factorization deterministic).

Pick `<TARGET>` from the measured reference profile after Phase 1, and state the
reasoning: this is double-precision sparse LDLᵀ, already reasonably tuned, so the
honest target is likely a **small single-digit factor on the dominant model
class** (e.g. cache-friendlier CSR traversal, better blocking on dense diagonal
chunks, vectorizing the inner column updates, avoiding redundant `qLDiagInv`
work), not a 100× rewrite. An engineer's estimate, stated as one — not a proof
of a ceiling.

## Fixed constraints (do not relax any)

- **Correctness within tolerance, on the committed corpus AND ≥2 alternate-seed
  corpora.** Residual band + solve-output agreement as above; any snapshot
  outside the band rejects the change, however fast.
- **Numerical generality.** The kernel must factor any valid SPD `M` in the
  domain, not a specific sparsity pattern. The CSR structure varies by model;
  exploiting structure that holds for *all* SPD `M` (symmetry, positive pivots,
  the supernodal/elimination-tree structure implied by `M_rownnz/M_colind`) is
  encouraged. Hard-coding a single model's pattern is overfitting.
- **Determinism:** same `M` bytes in → byte-identical `qLD` out.
- **Double precision** (`mjtNum`), single-threaded, C (C11), `-Wall -Wextra
  -Werror`. SIMD within one thread (SSE/AVX/FMA, `-march=native`) is allowed and
  expected — inspect the host first (`/proc/cpuinfo`, cache sizes) and record
  what you found; do not assume a feature exists.
- No precomputed answers; nothing keyed to a specific model or snapshot.

## Phase 1 — Plan and baseline (before optimizing)

1. Write `factor-optimized/README.md`: host CPU features and cache sizes as
   measured; **where the cycles go** in the reference factor+solve across the
   zoo — profile the kernel (`perf record`/`perf stat` if available, else
   coarse timers around factor vs solve and around the inner loops). State which
   models dominate total kernel time and why (large `nv`, dense `M`,
   fill-in). A candidate list **ranked by expected payoff** (Amdahl: fraction of
   measured kernel time a change touches × expected speedup on it) — not by ease.
   Mark unverifiable facts `ASSUMPTION: <fact> — affects <decision>`.
2. Measure the reference baseline with `measure-factor.sh` and record it. Set
   `<TARGET>` from it.

## Phase 2 — Implement

`factor-optimized/` starts as a correct copy. Optimize from it. Never **commit
or carry forward** a broken state: every kept iteration ends with a clean build
and all gates green. Attempting a hard change that ends up broken is fine —
revert it. Do not let "keep the tree clean" talk you out of a high-payoff change.

## Phase 3 — Iterate (the core loop)

Repeat until an exit criterion is met. Each iteration:

1. **Pick** the highest-expected-payoff candidate (runtime-fraction × expected
   speedup), spanning two kinds of change: **(a) work removal** — skip redundant
   `qLDiagInv` reciprocals, exploit symmetry, cut passes over the CSR; and
   **(b) throughput / data layout** — block the dense diagonal supernodes,
   vectorize inner column updates (FMA), improve CSR traversal locality,
   align/pack the working set. A throughput change removes no operations, so an
   op-count view ranks it ≈0 — rank it by runtime-fraction × expected
   lane/pipeline speedup or you will never pick it. Implementation risk is **not**
   a selection criterion; a failed attempt costs ~one iteration and a logged
   negative result.
2. **Implement** that one change in `factor-optimized/`.
3. **Gate on correctness**, in order; a failure means fix or revert before
   measuring:
   - clean build (`-Wall -Wextra -Werror`);
   - `run_factor --optimized` then `compare_factor ref.res opt.res` → **PASS**
     (residual band + solve-output agreement), corpus checksum unchanged;
   - `certify_factor` on the optimized output → every residual within band vs
     the independent dense Cholesky;
   - two consecutive optimized factor runs byte-identical.
4. **Measure** with `measure-factor.sh` (median of N).
5. **Record** in `factor-optimized/OPTIMIZATION-LOG.md`: hypothesis, change,
   before/after medians, speedup so far, keep/revert decision, and the **cost**
   of the hypothesis (wall-clock + tokens), per the same accounting the
   `differentiable-collisions-optc` log used.
6. **Keep or revert — and make every kept state durable.** Keep only if every
   gate passed and it measurably helps, is a pure simplification, or is an
   enabling transform you will build on next. On keep, **commit immediately**
   (`git add factor-optimized/ … && git commit`) so revert is always safe.
   Otherwise `git checkout -- factor-optimized/`. Keep the log entry either way —
   a rejected hypothesis with its numbers is a result. Never stack two
   speculative changes into one measurement.
7. **Update** the candidate list.

## Phase 4 — Generalization check (mandatory before claiming any exit)

1. Generate ≥2 alternate corpora with `dump_corpus` under different seeds
   (different configs and RHS), written **outside** `corpus.bin`.
2. For each: `run_factor` both kernels, `compare_factor` must PASS, and report
   the measured speedup alongside the committed-corpus speedup.
3. If speedup collapses or residuals diverge on alternate data, the change is
   overfit or wrong: diagnose, fix or revert, return to Phase 3.

`prove-factor-harness.sh` runs all gates end to end — use it to re-confirm.

## Exit criteria (exactly one, with evidence)

- **TARGET REACHED**: median measured kernel time on the committed corpus is
  ≥`<TARGET>`× faster than the measured reference baseline, all Phase 3 gates
  pass, and Phase 4 generalization passes.
- **NO FURTHER OPPORTUNITIES**: the candidate list is empty — every candidate
  was implemented and kept, implemented/measured/reverted with numbers in the
  log, or rejected with a stated cost reason (e.g. "requires threads —
  prohibited", "breaks the residual band — measured divergence"). Report the
  best achieved speedup. "Ran out of ideas" without a log of tried-and-measured
  candidates is not this criterion.

## Final report

- Measured reference baseline and final optimized median, the speedup factor,
  per-model breakdown — on this machine, no projections.
- The optimization log: what was tried, what each change measured, what was kept.
- Generalization results on the alternate corpora.
- Verification evidence: build output, comparator PASS, certifier PASS, residual
  numbers, determinism, corpus checksum — actual commands and output.
- Anything not verified, stated explicitly, with why.

## Acceptance checklist

- [ ] `factor-optimized/README.md`: host inspection, reference profile, ranked
      candidates, labeled ASSUMPTIONs, chosen `<TARGET>` with reasoning.
- [ ] `measure-factor.sh` used for every number; no second protocol invented.
- [ ] Reference baseline measured on this machine.
- [ ] Builds clean (`-Wall -Wextra -Werror`), double precision, single-threaded.
- [ ] `compare_factor` PASS (residual band + solve-output) on the committed
      corpus **and** ≥2 alternate-seed corpora.
- [ ] `certify_factor` PASS against the independent dense Cholesky.
- [ ] Deterministic; `corpus.bin` never modified (checksum evidence).
- [ ] Only `factor-optimized/` changed; no harness piece weakened or forked.
- [ ] `OPTIMIZATION-LOG.md` has one entry per iteration with before/after
      measurements, keep/revert decisions, and per-hypothesis cost.
- [ ] Exit criterion explicitly identified with the evidence it demands.
- [ ] No unmeasured performance claim anywhere.
