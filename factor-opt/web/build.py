#!/usr/bin/env python3
"""Parse factor-optimized/OPTIMIZATION-LOG.md into structured JSON for the browser.

One JSON record per browsable log entry. Entries start at a top-level
`## Iteration N — <title>` header. `## Iteration 0` is the baseline; everything
up to the next `## Iteration` header (including deeper `###` subsections such as
"Candidate-list status") is part of the current entry's body. The preamble before
`## Iteration 0` is skipped (kept as an "about" blurb only).

For each entry we pull: the kept/reverted decision, a representative speedup vs
the reference baseline, the reference/optimized per-pass medians (from the
measurement table), gate status (all-green + certifier residual), an optional
per-model speedup breakdown, and the raw markdown body so the page can render the
full text of every entry. This project has no dates or token costs — those are
intentionally left null.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.normpath(
    os.path.join(HERE, "..", "factor-optimized", "OPTIMIZATION-LOG.md")
)
PROFILE = os.path.normpath(
    os.path.join(HERE, "..", "factor-optimized", "README.md")
)

TARGET_SPEEDUP = 1.3

# A new browsable entry starts at a top-level "## Iteration N — title" header.
# Deeper headers (###, ####) belong to the current entry's body.
ENTRY_RX = re.compile(r"^##\s+Iteration\s+(\d+)\b\s*[—–-]?\s*(.*)$")


def split_entries(text):
    """Split the log into (preamble, [entries]) at top-level Iteration headers."""
    lines = text.split("\n")
    preamble = []
    entries = []
    cur = None
    for ln in lines:
        m = ENTRY_RX.match(ln)
        if m:
            if cur:
                entries.append(cur)
            n = int(m.group(1))
            cur = {
                "n": n,
                "id": "Iteration %d" % n,
                "title": m.group(2).strip(),
                "header": ln,
                "body_lines": [],
            }
        elif cur is not None:
            cur["body_lines"].append(ln)
        else:
            preamble.append(ln)
    if cur:
        entries.append(cur)
    return "\n".join(preamble).strip("\n"), entries


def flat(text):
    return re.sub(r"\s+", " ", text)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def find_decision(n, body):
    """Iteration 0 is the baseline; others parse **Decision:** KEEP / REVERT."""
    if n == 0:
        return "baseline"
    m = re.search(r"\*\*Decision:\*\*\s*\**\s*([A-Za-z]+)", body)
    if m:
        d = m.group(1).upper()
        if d.startswith("KEEP") or d.startswith("KEPT") or d.startswith("ADOPT"):
            return "kept"
        if d.startswith("REVERT") or d.startswith("REJECT"):
            return "reverted"
    f = flat(body).upper()
    if "DECISION" in f and ("REVERT" in f or "REJECT" in f):
        return "reverted"
    if "DECISION" in f and ("KEEP" in f or "KEPT" in f):
        return "kept"
    return "info"


def find_speedup(n, body):
    """Parse 'Speedup vs reference: ~1.33×' / '~1.0× (noise)' / range '~1.31–1.33×'.

    Handles ~, x and ×. For a range like '1.31-1.33' take the last (representative)
    number. Iteration 0 is the calibration ~1.0 reading.
    """
    # The canonical line: "**Speedup vs reference: ~1.33×**". It may carry a range
    # ("~1.31–1.33×"), and/or a trailing parenthetical that is EITHER an explicit
    # representative ("(≈1.30×)") OR a non-representative note ("(was ~1.30×)",
    # "(noise)") that must be ignored.
    m = re.search(r"Speedup vs reference:\s*([^\n*]+)", body)
    seg = m.group(1).strip() if m else None
    if seg is None:
        # fall back to any "Speedup ... N×" phrase in the body
        m = re.search(r"speedup[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*[x×]", body, re.I)
        if m:
            v = num(m.group(1))
            return round(v, 3) if v is not None else None
        return 1.0 if n == 0 else None

    # An explicit representative marked with ≈/~ inside parentheses wins, e.g.
    # "(≈1.30×)". A "(was …)" / "(noise)" parenthetical is NOT a representative.
    for pm in re.finditer(r"[(\[]([^)\]]*)[)\]]", seg):
        inner = pm.group(1)
        if re.search(r"\bwas\b|noise", inner, re.I):
            continue
        r = re.search(r"[≈~]\s*([0-9]+(?:\.[0-9]+)?)\s*[x×]", inner)
        if r:
            v = num(r.group(1))
            if v is not None:
                return round(v, 3)

    # Otherwise use the main range BEFORE any parenthetical, taking its last
    # number (the representative end of "~1.31–1.33×").
    main = re.split(r"[(\[]", seg, maxsplit=1)[0]
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?=[x×–\-—])", main)
    nums += re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*[x×]", main)
    if nums:
        v = num(nums[-1])
        if v is not None:
            return round(v, 3)
    return 1.0 if n == 0 else None


def find_medians(body):
    """Parse first float of the 'reference' row and the LAST 'optimized (iter N)'
    row from the measurement table.

    Rows look like:
        | reference | 6.69e-05 … 6.79e-05 |
        | optimized (iter 1) | 5.16e-05 … 5.19e-05 |
    Some tables list several optimized rows (prior iters for context); the row for
    THIS entry is the last one, so we take the last optimized row's first float.
    Iteration 0's table uses "≈ 6.8e-05" approximate values.
    """
    flt = r"[≈~]?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    ref = None
    opt = None
    for ln in body.split("\n"):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not cells:
            continue
        label = cells[0].lower()
        rest = " ".join(cells[1:])
        m = re.search(flt, rest)
        if not m:
            continue
        val = num(m.group(1))
        if val is None:
            continue
        if label.startswith("reference"):
            if ref is None:
                ref = val
        elif label.startswith("optimized"):
            opt = val  # keep overwriting -> last optimized row wins
    return ref, opt


def find_gates(body):
    """Detect 'all green' gate status and pull the certifier kernel residual."""
    f = flat(body)
    fl = f.lower()
    pass_count = len(re.findall(r"\bPASS\b", f))
    all_green = ("all green" in fl) or (
        ("comparator" in fl and "certifier" in fl and pass_count >= 2)
    )
    # certifier kernel residual, e.g. "kernel residual `3.30e-17`" /
    # "kernel resid `3.30e-17`"
    resid = None
    m = re.search(
        r"kernel\s+resid(?:ual)?\s*`?\s*([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+)",
        f,
    )
    if m:
        resid = m.group(1)
    return bool(all_green), resid


def find_gate_detail(body):
    """Pull the concrete trust-panel numbers from the **Gates** block of an entry.

    Returns a dict with whatever is present (keys omitted, not invented, when a
    value can't be parsed from this entry's body):
      comparator_max_solve_diff   e.g. "0"
      comparator_over_tol         e.g. "0"
      certifier_kernel_residual   e.g. "3.30e-17"
      certifier_dense_residual    e.g. "7.64e-17"   (independent dense-Cholesky)
      certifier_spd_fail          e.g. "0"
      determinism_sha             e.g. "e9f956b1…c3f18"  (qLD sha256, truncated)
      corpus_sha                  e.g. "905674e2…81fa1"
    """
    f = flat(body)
    out = {}

    m = re.search(r"max_solve_diff\s*[=:]\s*`?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", f)
    if m:
        out["comparator_max_solve_diff"] = m.group(1)
    m = re.search(r"over_tol\s*[=:]\s*`?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", f)
    if m:
        out["comparator_over_tol"] = m.group(1)

    m = re.search(
        r"kernel\s+resid(?:ual)?\s*`?\s*([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+)",
        f,
    )
    if m:
        out["certifier_kernel_residual"] = m.group(1)
    # "vs ... dense-Cholesky `7.64e-17`" or "vs dense `7.64e-17`"
    m = re.search(
        r"vs\s+(?:independent\s+)?dense(?:-Cholesky)?\s*`?\s*"
        r"([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+)",
        f,
    )
    if m:
        out["certifier_dense_residual"] = m.group(1)
    m = re.search(r"spd_fail\s*[=:]\s*`?\s*([0-9]+)", f)
    if m:
        out["certifier_spd_fail"] = m.group(1)

    # determinism: qLD sha256 `e9f956b1…c3f18`
    m = re.search(r"qLD\s+sha256\s*`?\s*([0-9a-f]+(?:[…\.]+[0-9a-f]+)?)", f, re.I)
    if m:
        out["determinism_sha"] = m.group(1)
    # corpus sha256 `905674e2…81fa1`
    m = re.search(r"corpus\s+sha256\s*`?\s*([0-9a-f]+(?:[…\.]+[0-9a-f]+)?)", f, re.I)
    if m:
        out["corpus_sha"] = m.group(1)
    return out


def parse_profile_shares(path):
    """Parse the per-model 'where the cycles go' share table from
    factor-optimized/README.md. Returns an ordered list of
    {model, nv, ref_s, share} dicts, plus the corpus headline.

    The table rows look like:
        | floppy      | 1153 | 2.96e-5   | 44.2% |
    We only emit a row when the share % is parseable; otherwise omit it.
    """
    shares = []
    try:
        with open(path, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return shares
    flt = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    for ln in txt.split("\n"):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        model = cells[0]
        if model.lower() in ("model", "") or set(model) <= set("-: "):
            continue
        # share is the last cell, of the form "44.2%"
        ms = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*%$", cells[-1])
        if not ms:
            continue
        # The runtime-share table is uniquely identified by an integer `nv`
        # column (cell 1). The factor-vs-solve "solve share" table has no nv
        # column, so this guard keeps us from mixing the two.
        nvm = re.match(r"^[0-9]+$", cells[1])
        if not nvm:
            continue
        refm = re.search(flt, cells[2])
        shares.append({
            "model": model,
            "nv": int(cells[1]),
            "ref_s": num(refm.group(1)) if refm else None,
            "share": float(ms.group(1)),
        })
    return shares


def find_per_model(body):
    """Best-effort per-model speedups, e.g. 'floppy 2.99e-5→1.91e-5 (**1.56×**)',
    'humanoid100 ... (**1.29×**)', 'balloons 1.07×', 'humanoid 1.04×'."""
    f = flat(body)
    # Longer names first so "humanoid100" wins over "humanoid".
    models = [
        "humanoid100", "humanoid", "arm26", "car", "balloons", "floppy",
    ]
    out = []
    seen = set()
    other = "|".join(re.escape(x) for x in models)
    for name in models:
        if name in seen:
            continue
        # "<name> ... (1.56×)" — allow the e-notation deltas before the ×, but do
        # not let the match cross into the NEXT model's name.
        pat = (r"\b" + re.escape(name) + r"(?![0-9])"
               r"(?:(?!" + other + r")[^\n×x]){0,80}?"
               r"\(?\**\s*([0-9]+(?:\.[0-9]+)?)\s*[x×]")
        m = re.search(pat, f, re.I)
        if not m:
            continue
        sp = num(m.group(1))
        if sp is not None and 0.5 < sp < 10:
            out.append({"model": name, "speedup": round(sp, 3)})
            seen.add(name)
    return out


def main():
    with open(LOG, encoding="utf-8") as fh:
        text = fh.read()
    about, raw_entries = split_entries(text)

    records = []
    for idx, e in enumerate(raw_entries):
        n = e["n"]
        body = "\n".join(e["body_lines"]).strip("\n")
        decision = find_decision(n, body)
        speedup = find_speedup(n, body)
        ref, opt = find_medians(body)
        gates_green, residual = find_gates(body)
        per_model = find_per_model(body)
        gate_detail = find_gate_detail(body)
        rec = {
            "index": idx,
            "id": e["id"],
            "title": e["title"],
            "decision": decision,
            "speedup": speedup,
            "target_reached": (speedup is not None
                               and speedup >= TARGET_SPEEDUP),
            "ref_median": ref,
            "opt_median": opt,
            "opt_us": round(opt * 1e6, 3) if opt is not None else None,
            "gates_green": gates_green,
            "certifier_residual": residual,
            "gate_detail": gate_detail,
            "per_model": per_model,
            "date": None,
            "tokens": None,
            "markdown": body,
        }
        records.append(rec)

    profile_shares = parse_profile_shares(PROFILE)

    # Best (highest) measured speedup and the iteration that first reached target.
    speeds = [r for r in records if r["speedup"] is not None]
    best_speedup = max((r["speedup"] for r in speeds), default=None)
    first_reached = next(
        (r["id"] for r in records if r.get("target_reached")), None
    )
    # Headline trust numbers: merge gate detail across all green entries so the
    # panel shows the full set (comparator + both certifier residuals +
    # determinism), preferring the most recent value for any given key. Every
    # number is still traceable to a log entry; iterations report a consistent
    # set, so a later entry that omits e.g. the dense residual does not erase it.
    trust = {}
    for r in records:
        if r["decision"] in ("kept", "baseline") and r["gate_detail"]:
            trust.update(r["gate_detail"])

    out = {
        "source": "factor-optimized/OPTIMIZATION-LOG.md",
        "profile_source": "factor-optimized/README.md",
        "generated_by": "web/build.py",
        "target_speedup": TARGET_SPEEDUP,
        "target_reached": best_speedup is not None
        and best_speedup >= TARGET_SPEEDUP,
        "best_speedup": best_speedup,
        "target_first_reached": first_reached,
        "trust": trust,
        "profile_shares": profile_shares,
        "about": about,
        "entries": records,
    }
    with open(os.path.join(HERE, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    if "--debug" in sys.argv:
        hdr = ("%3s %-12s %-9s %-7s %-6s %-8s %-9s %-6s %-10s | %s"
               % ("idx", "id", "decision", "speed", "tgt?", "opt_us",
                  "ref", "green", "resid", "title"))
        print(hdr)
        print("-" * len(hdr))
        for r in records:
            pm = ",".join("%s=%s" % (p["model"], p["speedup"])
                          for p in r["per_model"])
            print("%3d %-12s %-9s %-7s %-6s %-8s %-9s %-6s %-10s | %s"
                  % (r["index"], r["id"], str(r["decision"]),
                     str(r["speedup"]),
                     "yes" if r["target_reached"] else "no",
                     str(r["opt_us"]),
                     str(r["ref_median"]), str(r["gates_green"]),
                     str(r["certifier_residual"]), r["title"][:44]))
            if pm:
                print("    per-model: %s" % pm)
            if r["gate_detail"]:
                print("    gates: %s" % ", ".join(
                    "%s=%s" % (k, v) for k, v in r["gate_detail"].items()))
        print()
        print("target_speedup = %s, target_reached = %s (first at %s)"
              % (TARGET_SPEEDUP, out["target_reached"],
                 out["target_first_reached"]))
        print("best_speedup   = %s" % best_speedup)
        print("trust          = %s" % (trust or "(none parsed)"))
        print("profile_shares (%d models):" % len(profile_shares))
        for s in profile_shares:
            print("    %-12s nv=%-5s ref_s=%-9s share=%s%%"
                  % (s["model"], s["nv"], s["ref_s"], s["share"]))
    print("\nWrote data.json with %d entries" % len(records))


if __name__ == "__main__":
    main()
