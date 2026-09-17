#!/usr/bin/env bash
# Run from repository root. Requires python_eckf/initialization_candidates.py to be v3.
set -euo pipefail
ROOT="$(pwd)"
MODULE="$ROOT/python_eckf/initialization_candidates.py"
DIAG="$ROOT/tests/diagnose_synchronized_eckf.py"
COMPARE="$ROOT/tests/compare_selector_full_song.py"
V2_ZIP="${1:?Pass path to previously downloaded selector_priority_v2.zip as the first argument}"
[[ -f "$MODULE" && -f "$DIAG" && -f "$COMPARE" && -f "$V2_ZIP" ]] || { echo 'Missing module/diagnostic/comparison script/v2 ZIP' >&2; exit 2; }
command -v unzip >/dev/null || { echo 'unzip required' >&2; exit 2; }
TMP="$(mktemp -d)"
cp -p "$MODULE" "$TMP/v3_original.py"
restore() {
  cp -p "$TMP/v3_original.py" "$MODULE"
  rm -rf "$TMP"
}
trap restore EXIT
# Verify V2 ZIP contains expected path before changing anything.
unzip -p "$V2_ZIP" python_eckf/initialization_candidates.py > "$TMP/v2.py"
[[ -s "$TMP/v2.py" ]] || { echo 'v2 module missing from ZIP' >&2; exit 2; }
python -m py_compile "$TMP/v2.py" "$TMP/v3_original.py" "$COMPARE"
mkdir -p "$ROOT/tests"
echo '=== V3 UNIT TESTS (original tests remain unchanged) ==='
for pattern in test_initialization_candidates.py test_selector_priority_regressions.py test_selector_family_v3.py; do
  python -m unittest discover -s tests -p "$pattern" -v
done
echo '=== FULL PREDESTINATI V2 BASELINE: TEMPORARILY SWITCH SELECTOR ==='
cp "$TMP/v2.py" "$MODULE"
python "$DIAG" --start 0 --end 67.5 --output tests/PREDESTINATI_full_v2.txt > "$TMP/v2_stdout.log"
echo '=== RESTORE V3 AND RUN IDENTICAL FULL PREDESTINATI ==='
cp "$TMP/v3_original.py" "$MODULE"
python "$DIAG" --start 0 --end 67.5 --output tests/PREDESTINATI_full_v3.txt > "$TMP/v3_stdout.log"
echo '=== READ-ONLY SIDE-BY-SIDE ANALYSIS ==='
python "$COMPARE" --v2 tests/PREDESTINATI_full_v2_frames.csv --v3 tests/PREDESTINATI_full_v3_frames.csv --v2-report tests/PREDESTINATI_full_v2.txt --v3-report tests/PREDESTINATI_full_v3.txt --output tests/PREDESTINATI_full_selector_comparison.txt > "$TMP/comparison_stdout.log"
sed -n '1,/=== EVERY VOICING STATUS CHANGE ===/p' "$TMP/comparison_stdout.log"
echo 'Original v3 module will be restored on exit (also on failures).'
