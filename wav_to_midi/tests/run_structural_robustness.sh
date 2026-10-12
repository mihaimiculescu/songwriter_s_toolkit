#!/usr/bin/env bash

set -euo pipefail

# ================================================================
# STRUCTURAL ROBUSTNESS TEST
#
# Runs the frozen pre-musical-interpretation pipeline on additional
# guinea pigs.
#
# IMPORTANT:
#   - same ECKF parameters as RATATA
#   - same validity logic
#   - same trajectory interpretation
#   - same smoothing
#   - same gesture features
#   - same object construction
#   - same Structural Splitting V3.1
#   - NO retuning per song
#   - NO musical interpretation
#   - NO MIDI generation
# ================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SONGS=(
    "Ochiitai"
    "PREDESTINATI"
)

echo "================================================================"
echo "STRUCTURAL ROBUSTNESS TEST"
echo "================================================================"
echo
echo "Repository:"
echo "  $ROOT"
echo
echo "Songs:"
printf '  %s\n' "${SONGS[@]}"
echo

# ----------------------------------------------------------------
# First make sure the frozen structural code is syntactically valid.
# ----------------------------------------------------------------

echo "Checking Python files..."

python -m py_compile \
    python_eckf/gesture_structural_splitter.py \
    tests/diagnose_ratata_structural_splitting.py

echo "OK"
echo

# ================================================================
# PROCESS EACH SONG
# ================================================================

for SONG in "${SONGS[@]}"; do

    WAV="tests/${SONG}.wav"
    CSV="${WAV}.eckf.csv"

    REPORT="tests/${SONG}_structural_splitting_v3_1.txt"

    echo "================================================================"
    echo "$SONG"
    echo "================================================================"

    if [[ ! -f "$WAV" ]]; then
        echo "ERROR: WAV not found:"
        echo "  $WAV"
        exit 1
    fi

    # ------------------------------------------------------------
    # STEP 1
    # Frozen ECKF + offline validity/correction pipeline
    # ------------------------------------------------------------

    echo
    echo "[1/2] ECKF + offline validity..."
    echo

    python -m python_eckf.cli \
        "$WAV" \
        --mode offline \
        --block-size 2048 \
        --c 7 \
        --wait 2 \
        --npeaks 3 \
        --nsemitones 2 \
        --vocal-floor-hz 60

    if [[ ! -f "$CSV" ]]; then
        echo
        echo "ERROR: expected CSV was not produced:"
        echo "  $CSV"
        exit 1
    fi

    # ------------------------------------------------------------
    # STEP 2
    # Frozen structural stack through V3.1
    # ------------------------------------------------------------

    echo
    echo "[2/2] Structural analysis through V3.1..."
    echo

    python tests/diagnose_ratata_structural_splitting.py \
        "$CSV" \
        > "$REPORT"

    echo "Report:"
    echo "  $REPORT"

    # Show the summary in the terminal as well.
    echo
    echo "Summary:"
    sed -n '1,19p' "$REPORT"

    echo
done

echo "================================================================"
echo "ROBUSTNESS RUN COMPLETE"
echo "================================================================"
echo
echo "Reports:"
for SONG in "${SONGS[@]}"; do
    echo "  tests/${SONG}_structural_splitting_v3_1.txt"
done
echo