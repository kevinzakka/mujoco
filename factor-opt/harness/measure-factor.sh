#!/usr/bin/env bash
# measure-factor.sh : median-of-N protocol + speedup ratio.
#
# Times the factor+solve kernel ONLY (run_factor's timed region excludes model
# load, corpus I/O and densification). Runs each kernel N>=5 times back-to-back
# over the whole corpus and reports the median total kernel time, then the
# speedup = reference_median / optimized_median.
#
# usage: measure-factor.sh <corpus.bin> [N]
#   (run from the MuJoCo repo root; binaries live in factor-opt/build/)
set -euo pipefail

CORPUS="${1:?usage: measure-factor.sh <corpus.bin> [N]}"
N="${2:-9}"
if [ "$N" -lt 5 ]; then echo "N must be >= 5" >&2; exit 2; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # factor-opt/
REF="$HERE/build/run_factor_ref"
OPT="$HERE/build/run_factor_opt"

ref_line="$("$REF" --reference "$CORPUS" /dev/null --repeat "$N" --time 2>/dev/null)"
opt_line="$("$OPT" --optimized "$CORPUS" /dev/null --repeat "$N" --time 2>/dev/null)"

echo "$ref_line"
echo "$opt_line"

ref_med="$(printf '%s\n' "$ref_line" | sed -n 's/.*median_s=\([^ ]*\).*/\1/p')"
opt_med="$(printf '%s\n' "$opt_line" | sed -n 's/.*median_s=\([^ ]*\).*/\1/p')"

speedup="$(awk -v r="$ref_med" -v o="$opt_med" 'BEGIN{ if (o>0) printf "%.4f", r/o; else print "inf" }')"

echo "MEASURE N=$N ref_median_s=$ref_med opt_median_s=$opt_med speedup=${speedup}x"
echo "NOTE: median_s is per-corpus-pass (run_factor auto-calibrates inner repeats so each timed sample is many passes). While the optimized kernel is an unmodified pulled copy, any deviation from 1.00x is measurement noise."
