# ECKF V2 — bounded adaptive offline lookahead

This ZIP replaces the `python_eckf/` package from `python_eckf_v2_silence_interface.zip` and retains its fixed silence calibration interface. It does not change MATLAB-mode lookahead.

## Operation

* Offline mode tries the current frame as before, including `choose_initialization` and `reconcile_initialization`.
* Only if that choice fails, it speculatively examines up to `--wait` following whole frames (default 2; 0 disables).
* Stops immediately at original-file end, a frame below the existing fixed energy-silence threshold, an unvoiced frame, a different nearest equal-tempered note, or a detected harmonic discontinuity.
* Future estimates must be from an acoustically admitted candidate; the candidate must match the anchor frame's measured ACF frequency to the same nearest note, each strictly within 49 cents of the note center.
* Re-estimates amplitude and phase ON THE ORIGINAL anchor frame at the future-supported frequency. Future amplitude and phase are never transplanted. The tracker then processes the original audio samples starting at the original `start`; speculative scans never write output or shift onset time.
* If no safe future result appears, preserves `INITIALIZATION_UNRESOLVED` instead of filling an estimate.
* Records an `OFFLINE_LOOKAHEAD` trace event and `VOICED_TRACKED_LOOKAHEAD` frame decision when successful.

## What this does not do

It does NOT look ahead to revise an already accepted current-frame pitch or rescue a current frame rejected as silence/unvoiced. It is not a full bidirectional smoothing algorithm or a verified pitch-accuracy improvement. It does not implement adaptive silence calibration, or add any filtering of the WAV. Frame comparisons are intentionally conservative to avoid crossing ornaments. Some weak but genuinely pitched onsets will remain unresolved.

## Install

Back up the existing `python_eckf/` folder, then unpack `python_eckf_v2_adaptive_lookahead.zip` at repo root to replace it.

## Run

`python -m python_eckf.cli tests/RATATA.wav --mode offline --wait 3 --csv tests/RATATA_v2_lookahead.csv`

The WAV path is illustrative: substitute its actual path. `--wait 0` disables lookahead for an A/B run. `--wait 3` allows *up to* three future frames, not a mandatory three-frame skip. Keep the existing `--silence-mode fixed` baseline until calibration is developed separately.

Copy `test_eckf_v2_adaptive_lookahead.py` into `tests/`, then run `python tests/test_eckf_v2_adaptive_lookahead.py` from the repo root. Existing `test_eckf_v2_silence.py` remains a separate regression test.

## Caveat

Tests verify control flow, silence vetoes, note boundaries and onset anchoring, not absolute pitch accuracy. Full four-WAV audio/ground-truth evaluation is still required before treating this as production-ready. 
