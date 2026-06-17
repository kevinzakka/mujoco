#!/usr/bin/env bash
# prove-factor-harness.sh : end-to-end proof of the factorization harness on the
# identity copy (factor-optimized == byte-identical behavior to factor-reference).
#
# Prints a terse FINAL SUMMARY by default (everything else to a log);
# --verbose streams every step. Exit 0 only if every GATE passes.
#
# This is the per-turn proof wired into optimize-factor.md as --hook-per-run.
set -uo pipefail

VERBOSE=0
for a in "$@"; do
  case "$a" in
    --verbose) VERBOSE=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

FO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # factor-opt/
ROOT="$(cd "$FO/.." && pwd)"                          # MuJoCo repo root
BUILD="$FO/build"
WORK="$FO/work"
LOG="$WORK/prove.log"
mkdir -p "$WORK"
: > "$LOG"

# run a step: capture to log; echo to stdout only in verbose mode.
run() {  # run "<label>" cmd...
  local label="$1"; shift
  echo "### $label" >> "$LOG"
  if [ "$VERBOSE" -eq 1 ]; then
    echo "### $label"
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
  else
    "$@" >> "$LOG" 2>&1
    return $?
  fi
}

FAIL=0
note() { echo "$1"; echo "$1" >> "$LOG"; }
gate() { # gate <ok?0/1> <message>
  if [ "$1" -ne 0 ]; then echo "GATE FAILED: $2"; echo "GATE FAILED: $2" >> "$LOG"; FAIL=1; fi
}

# --- the prove driver loads models by repo-relative path -> cd to repo root ---
cd "$ROOT"

# ========================= 1. clean build ===================================
run "1. clean build (-Wall -Wextra -Werror)" make -C "$FO" clean
if ! make -C "$FO" >> "$LOG" 2>&1; then
  echo "GATE FAILED: build"; echo "BUILD LOG TAIL:"; tail -20 "$LOG"; exit 1
fi
[ "$VERBOSE" -eq 1 ] && tail -8 "$LOG"
note "build: OK (5 binaries, zero warnings/errors)"

# ========================= 2. dump corpus + sha256 ==========================
run "2. dump_corpus" "$BUILD/dump_corpus" "$FO/harness/models.txt" "$WORK/corpus.bin"
gate $? "dump_corpus"
SHA_BEFORE="$(shasum -a 256 "$WORK/corpus.bin" | awk '{print $1}')"
note "corpus.bin sha256 (before): $SHA_BEFORE"

# ========================= 3-4. run both kernels ============================
run "3. run_factor --reference" "$BUILD/run_factor_ref" --reference "$WORK/corpus.bin" "$WORK/ref.res"
gate $? "run reference"
run "4. run_factor --optimized" "$BUILD/run_factor_opt" --optimized "$WORK/corpus.bin" "$WORK/opt.res"
gate $? "run optimized"

# ========================= 5. GATE compare ==================================
CMP="$("$BUILD/compare_factor" "$WORK/ref.res" "$WORK/opt.res" 2>>"$LOG")"; CMP_RC=$?
echo "$CMP" >> "$LOG"; [ "$VERBOSE" -eq 1 ] && echo "$CMP"
gate $CMP_RC "compare_factor verdict"
CMP_SUMMARY="$(printf '%s\n' "$CMP" | grep '^COMPARE snapshots' || true)"
MAX_SOLVE_DIFF="$(printf '%s\n' "$CMP_SUMMARY" | sed -n 's/.*max_solve_diff=\([^ ]*\).*/\1/p')"
# the identity baseline demands exactly 0.0 solve-output difference
if [ "$MAX_SOLVE_DIFF" != "0" ]; then
  gate 1 "max_solve_diff=$MAX_SOLVE_DIFF (expected 0 for identity kernels)"
fi

# ========================= 6. GATE certify (both) ===========================
CERT_REF="$("$BUILD/certify_factor" "$WORK/corpus.bin" "$WORK/ref.res" 2>>"$LOG")"; CR_RC=$?
echo "$CERT_REF" >> "$LOG"; [ "$VERBOSE" -eq 1 ] && echo "$CERT_REF"
gate $CR_RC "certify_factor on ref.res"
CERT_OPT="$("$BUILD/certify_factor" "$WORK/corpus.bin" "$WORK/opt.res" 2>>"$LOG")"; CO_RC=$?
echo "$CERT_OPT" >> "$LOG"; [ "$VERBOSE" -eq 1 ] && echo "$CERT_OPT"
gate $CO_RC "certify_factor on opt.res"
CERT_SUMMARY="$(printf '%s\n' "$CERT_OPT" | grep '^CERTIFY snapshots' || true)"

# ========================= 7. measure (median-of-N) =========================
MEAS="$(bash "$FO/harness/measure-factor.sh" "$WORK/corpus.bin" 9 2>>"$LOG")"
echo "$MEAS" >> "$LOG"; [ "$VERBOSE" -eq 1 ] && echo "$MEAS"
MEAS_LINE="$(printf '%s\n' "$MEAS" | grep '^MEASURE ' || true)"
SPEEDUP="$(printf '%s\n' "$MEAS_LINE" | sed -n 's/.*speedup=\([^ ]*\).*/\1/p')"

# ========================= 8. generalization ================================
# corpus spans >=4 distinct models of differing nv (per models.txt). Per-model
# residual is bounded by the global certify max (all snapshots within band).
NMODELS="$(grep -vcE '^\s*(#|$)' "$FO/harness/models.txt")"
if [ "$NMODELS" -lt 4 ]; then gate 1 "model zoo spans only $NMODELS models (<4)"; fi
note "generalization: zoo spans $NMODELS models; all snapshots within band (see certify)"

# ========================= 9. GATE determinism ==============================
"$BUILD/run_factor_opt" --optimized "$WORK/corpus.bin" /dev/null --qld-dump "$WORK/q1.bin" >>"$LOG" 2>&1
"$BUILD/run_factor_opt" --optimized "$WORK/corpus.bin" /dev/null --qld-dump "$WORK/q2.bin" >>"$LOG" 2>&1
if cmp -s "$WORK/q1.bin" "$WORK/q2.bin"; then
  DET="byte-identical"; note "determinism: two optimized factor dumps byte-identical"
else
  DET="DIFFER"; gate 1 "determinism: factor dumps differ between runs"
fi
Q_SHA="$(shasum -a 256 "$WORK/q1.bin" | awk '{print $1}')"

# ========================= 10. corpus checksum unchanged ====================
SHA_AFTER="$(shasum -a 256 "$WORK/corpus.bin" | awk '{print $1}')"
if [ "$SHA_BEFORE" != "$SHA_AFTER" ]; then gate 1 "corpus checksum changed"; fi

# ========================= FINAL SUMMARY ====================================
echo
echo "================= FINAL SUMMARY (identity baseline) ================="
echo "$CMP_SUMMARY"
echo "$(printf '%s\n' "$CMP" | grep '^COMPARE VERDICT' || true)"
echo "$CERT_SUMMARY"
echo "$(printf '%s\n' "$CERT_OPT" | grep '^CERTIFY VERDICT' || true)"
echo "$MEAS_LINE"
echo "speedup: ${SPEEDUP} -> no measurable gain (byte-identical kernels; deviation is noise)"
echo "max solve-output deviation: ${MAX_SOLVE_DIFF}"
echo "determinism: ${DET} (qLD dump sha256 ${Q_SHA})"
echo "corpus sha256 before: $SHA_BEFORE"
echo "corpus sha256 after:  $SHA_AFTER"
echo "models in zoo: $NMODELS"
if [ "$FAIL" -eq 0 ]; then
  echo "VERDICT: PASS (all gates green)"
  echo "===================================================================="
  echo "(full log: $LOG)"
  exit 0
else
  echo "VERDICT: FAIL (see gates above)"
  echo "===================================================================="
  echo "(full log: $LOG)"
  exit 1
fi
