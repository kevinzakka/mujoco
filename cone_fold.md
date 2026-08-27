# Cone fold

**Problem.** In the elliptic Newton solver, `HessianCone` rebuilds `Lcone`
every iteration by copying `L` and applying `dim` sparse rank-1 updates per
cone-state contact. Profiling a 26-card house of cards being knocked over
(~90 contacts, nv=156, one island, 8.6 iterations/step) put
`mju_cholUpdateSparse` at 63% of total step time: on this problem one sparse
rank-1 update costs about a third of a full numeric factorization, so with
many sliding contacts the updates dwarf the factorization they avoid.

**Fix.** When `ncone > 24`, build `Lcone` with one factorization. Writing
`Hc = Lc Lc'`, the cone contribution `Jc' Hc Jc` equals `(Lc' Jc)'(Lc' Jc)`,
so replacing each cone contact's Jacobian rows with `Lc' Jc` (weight 1 in D)
lets the existing `J'DJ` + factorize pipeline produce the cone-inclusive
Hessian. Rows of one contact share a sparsity pattern, so J, H, and L
structures are unchanged; no symbolic work. One new static function in
`engine_solver.c`, sparse path only.

**Results.** 3000 recorded collapse states replayed identically per variant
(single step, warmstart restored), removing trajectory chaos; iteration
counts are identical, so this is pure cost. Apple M-series, us/step:

|            | stock | folded |
|------------|-------|--------|
| mean       | 1234  | 1085   |
| p95        | 2729  | 2084   |
| worst step | 7039  | 3453   |

The gain concentrates in the many-sliding-contact steps that spike during
pile-ups. Threshold 24 swept empirically (plateau 24-32, gone by 48); a
principled version could compare `sum(dim) * nnz(L)` to factorization cost.
