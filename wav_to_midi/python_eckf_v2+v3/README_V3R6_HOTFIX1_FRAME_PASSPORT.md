# V3R6 Hotfix 1 — frame-passport reconciliation

This hotfix repairs a V3R6 integration bug exposed by the first full CLI run.

## Symptom

`trajectory_interpreter.py` could raise:

`ValueError: valid=True on a non-VOICED_TRACKED frame`

immediately after V3R6 promoted rejected pitch ownership from strong primary
periodicity evidence.

## Cause

V3R6 correctly changed `offline_validity.valid` and `clean_f0_hz`, but did not
promote the corresponding frame-level acoustic passport to `VOICED_TRACKED`.
The trajectory layer intentionally enforces the invariant that authoritative
valid pitch can exist only on a tracked frame.

A second integration detail compounded this: `sampled_frame_state` had been
materialized before V3R6 reconciliation, so it remained stale even if the
underlying frame state was repaired.

## Fix

When `PRIMARY_EVIDENCE_RECOVERED_V3R6` promotes pitch ownership, the same frame
is now promoted to `VOICED_TRACKED`. `cli.py` then refreshes the sampled frame
passport before trajectory interpretation.

`LOW_ENERGY_SILENCE` is still never promotable. No thresholds, candidate
selection rules, V3R5 logic, or V13 renderer logic changed.
