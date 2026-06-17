# MuJoCo Inertia Factorization — reference and LLM-optimized

A case study in driving an LLM at a numerical-optimization problem **with a
harness it cannot talk its way around** — applied to MuJoCo's joint-space
inertia factorization and solve.

The target kernel is the sparse LDLᵀ factorization of the joint-space inertia
matrix `M` and the triangular solve that uses it:

- `mj_factorM(m, d)` — factorizes `M` into `d->qLD` (LDLᵀ) and `d->qLDiagInv`.
- `mj_factorI(qLD, qLDiagInv, nv, M_rownnz, M_rowadr, M_colind, …)` — the inner
  CSR kernel that does the work.
- `mj_solveLD(...)` / `mj_solveM(m, d, x, y, n)` — the sparse triangular solve
  `x = M⁻¹ y` that consumes the factorization.

`M` is symmetric positive-definite for any valid configuration, so the kernel is
in the hottest part of `mj_fwdAcceleration` on every step.

## Why factorization is the right first target

It has the property the collision study had to engineer for, **for free**:

1. **Self-certifying, chaos-free.** Correctness is a one-step algebraic identity,
   not a trajectory. Factor `M`, solve `M x = b` for a random `b`, check the
   residual `‖M x − b‖ / ‖b‖`. It must be ≈ machine-eps for *any* correct
   factorization. No golden trajectory, no sensitive-dependence, no butterfly
   effect — the thing the stateful parts of MuJoCo would force us to fight.
2. **An independent oracle already ships.** `mju_cholFactor` / `mju_cholSolve`
   is a *dense* Cholesky that shares no code with the sparse path. Densify `M`,
   factor it densely, and you have an answer derived from completely separate
   code — the "independent validator" role, built in.
3. **The two-implementations-compared pattern is already native.**
   `test/benchmark/factorI_benchmark_test.cc` already benchmarks
   `mj_factorI_legacy` vs `mj_factorI` on `humanoid100.xml` and `coil.xml`. We
   are extending an existing idiom, not inventing one.
4. **It proves the ruler before we trust it.** Because correctness here is
   exact algebra, the identity baseline (optimized == verbatim copy of
   reference) must report 0.0 difference and ≈1.0× — a clean iteration-0 proof
   that the measurement pipeline itself is sound, before any real optimization.

## Method — two documents, two phases

Mirrors the `differentiable-collisions-optc` repo this was modeled on.

1. **[`prompts/build-factor-harness.md`](prompts/build-factor-harness.md) → the harness.**
   Builds the snapshot corpus, the comparator, the dense-Cholesky certifier, the
   residual gate, the determinism check, and the median timing protocol — and
   **proves them against an identity copy** of the shipping kernel before any
   optimization exists. Output recorded in `HARNESS-BASELINE.md`.

2. **[`prompts/optimize-factor.md`](prompts/optimize-factor.md) → the optimized kernel.**
   The hillclimb loop. Edits only `factor-optimized/`, gated every iteration by
   the harness, logged in `OPTIMIZATION-LOG.md`.

The proof driver `prove-factor-harness.sh` is wired to run **once per agent
turn**, so each turn begins with the real, measured gate status injected into
the conversation — not the model's memory of it.

## The match contract — "faster" is not "bit-identical"

The optimized kernel is accepted only when, on every snapshot in the corpus:

- the **solve residual** `‖M x − b‖ / ‖b‖` is within the reference's residual
  band (the factorization is genuinely a factorization of `M`, certified
  independently of the reference code); and
- the **solve output** `x` agrees with the reference solve within a stated
  tolerance for the same `b` (LDLᵀ pivots are essentially unique for SPD `M`, so
  this band is tight); and
- the factorization is **deterministic** — same `M` bytes in → same `qLD` bytes
  out, every run.

`M` is never modified; the snapshot corpus is checksummed so the benchmark can't
be quietly edited to flatter a result.
