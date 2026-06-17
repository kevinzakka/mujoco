# Optimization progress viewer

A static, dependency-free web page (vanilla JS / inline SVG, no libraries) for
browsing the optimization history in `factor-optimized/OPTIMIZATION-LOG.md` — the
MuJoCo sparse-LDLᵀ **inertia factor + solve** kernel optimization.

Top-to-bottom the page reads as a single story: **it's ~1.33× faster, the
correctness is gate-proven (not claimed), and the win lands where the runtime
actually is.**

- **Hero stat tiles** — best speedup (`1.33×`, `+33%`), the `1.3×` target with a
  **REACHED** badge, **gates: ALL GREEN**, iterations kept, and the max certifier
  residual (`3.30e-17`, annotated ≈ machine precision / double-eps).
- **"How this works" strip** — four compact steps: pull MuJoCo's real
  `mj_factorI`/`mj_solveLD`; change one thing; gate it with an independent
  comparator + dense-Cholesky certifier + determinism check on a frozen corpus;
  keep only if measurably faster **and** all gates pass.
- **Trust panel** — the three gates for the best state with their real numbers:
  comparator `max_solve_diff = 0`, certifier kernel residual `3.30e-17` vs the
  independent dense-Cholesky `7.64e-17`, determinism qLD sha256. Each card states
  it is independent of the optimized code.
- **Per-model Amdahl panel** — a paired horizontal bar chart of each model's
  **share of reference runtime** (from `factor-optimized/README.md`) against its
  **measured speedup** (floppy ≈44% / 1.56×, humanoid100 ≈35% / 1.29×, …), making
  the Amdahl targeting visually obvious.
- **Progress line chart** — x = iteration index, y = speedup (×). A dashed blue
  guide marks the **1.3× target**, a faint grey line marks the **1.0× no-op
  baseline**, and an annotation marks where the target was first reached. The
  y-axis auto-scales to roughly `[0.9, max(1.6, data)]`. Toggle to **per-pass time
  (µs)** (lower is better). **Hover** to move the cursor line + tooltip; **click**
  a point or list row to pin its full markdown entry below (with a per-model
  mini-panel and gate chips).
- **Searchable, filterable iteration list** (All / Kept / Reverted / Baseline)
  plus a kept/reverted donut.
- **Honest limits** note — wins are small single-digit by nature (tuned
  double-precision sparse LDLᵀ), and the bit-exact `max_solve_diff == 0` gate is
  the ceiling that blocks vectorizing the solve reductions.
- Links to the raw log, the kernel profile + candidate list, the project README,
  and the driving prompt.

## Run

```bash
# regenerate data.json from the log (only needed when the log changes)
cd /Users/kevin/dev/mujoco/factor-opt/web
python3 build.py            # add --debug to print the extracted table

# serve from factor-opt/ (one level up) so the "explore the repo" links
# (../factor-optimized/…, ../README.md, ../prompts/…) resolve. The page is at /web/.
cd /Users/kevin/dev/mujoco/factor-opt
python3 -m http.server 8012
# open http://localhost:8012/web/
```

The page **must be served over `http://`** (not opened as `file://`) because it
`fetch`es `data.json`. Serving from `factor-opt/` (not `web/`) is what makes the
relative `../…` repo links resolve.

## How parsing works

`build.py` parses `../factor-optimized/OPTIMIZATION-LOG.md` into `data.json`, one
record per top-level `## Iteration N — <title>` entry (deeper `###` subsections
such as "Candidate-list status" stay in the current entry's body; the preamble
before `## Iteration 0` is kept only as an `about` blurb). For each entry it
extracts:

- `decision` — `**Decision:** KEEP…` → `kept`, `REVERT…` → `reverted`,
  Iteration 0 → `baseline`.
- `speedup` — from the `Speedup vs reference: …` line (handles `~`, `x`, `×`;
  ranges like `~1.31–1.33×` take the last number; an explicit `(≈1.30×)`
  representative wins; a `(was …)` / `(noise)` parenthetical is ignored).
- `ref_median`, `opt_median` — first scientific-notation float of the `reference`
  row and of the **last** `optimized` row in the measurement table;
  `opt_us = opt_median × 1e6`.
- `gates_green` + `certifier_residual` — "all green" / comparator + certifier
  PASS, plus the kernel residual (e.g. `3.30e-17`).
- `per_model` — best-effort per-model speedup list when present.
- `target_reached` — `speedup >= target_speedup` (per entry).
- `gate_detail` — the concrete trust-panel numbers from the entry's gate block:
  `comparator_max_solve_diff`, `comparator_over_tol`, `certifier_kernel_residual`,
  `certifier_dense_residual` (independent dense-Cholesky), `certifier_spd_fail`,
  `determinism_sha` (qLD sha256), `corpus_sha`. Keys are **omitted, not invented**
  when a value isn't in that entry.
- `markdown` — the raw entry body for the pinned view.

It also parses `factor-optimized/README.md` for the per-model **profile shares**
(the runtime-share table, identified by its integer `nv` column) into
`profile_shares: [{model, nv, ref_s, share}]`, and emits top-level
`best_speedup`, `target_reached`, `target_first_reached`, and a merged `trust`
object (the union of green entries' `gate_detail`) for the hero + trust panels.

There are no dates or token costs in this log, so those are `null`. The header
shows **gate status** where the reference viewer showed token cost.

`data.json` is `{source, profile_source, generated_by, target_speedup, target_reached,
best_speedup, target_first_reached, trust, profile_shares, about, entries:[…]}`.
Re-run `build.py` whenever the log changes; `python3 build.py --debug` prints the
extracted table for spot-checking.

## Keeping the log parser-friendly

The parser keys off the current log conventions. To avoid breaking it, keep:

- iteration headers as top-level `## Iteration N — <title>` (em-dash optional;
  N must be an integer; only `##` starts a new entry, deeper `###` do not);
- one `Speedup vs reference: …×` line per entry, with any non-representative
  aside in parens flagged by the word `was` or `noise`;
- a measurement table whose row labels start with `reference` / `optimized`, the
  median as the first scientific-notation float in the row, and **this** entry's
  optimized row listed **last**;
- gate status spelled with `PASS` / "all green" and the residual as
  `kernel residual \`<sci>\``.
