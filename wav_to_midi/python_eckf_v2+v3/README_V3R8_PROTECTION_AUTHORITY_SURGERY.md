# V3R8 — Protection Authority Surgery

V3R8 keeps the V3R7 135-Hz rate-aware interpretation architecture and frozen
MRV13 renderer. It changes only the tracker-side primary reconciliation stage.

## Why

V3R7's candidate-authorizing trajectory logic worked well, but its protection
side had one sharp failure class: a real tracker drift could be split by an
arbitrary ~46-ms primary-frame boundary, causing each fragment to miss the
per-frame persistence duration gate. The local-trajectory veto could then
protect a smooth but acoustically false ECKF path.

Canonical observed examples included Damelo ~26.78-26.83, Afurisita ~15.556,
and RATATA ~10.665. Comomelo ~47.60 remains the protected positive control: its
real V-shaped expressive excursion is short enough to stay below the continuous
persistence gate.

## Changes

1. **Continuous physical-time persistence across primary-frame boundaries**
   - Generic non-integer drift disagreement is assembled on the interpretation
     trajectory itself.
   - A run may cross primary-frame boundaries while the primary candidate family
     remains locally compatible.
   - The default persistence threshold remains 18 ms.
   - The V3R7 20-ms primary-family-jump guard still wins at the leading edge of
     a newly changed primary family.

2. **Narrow embedded register-outlier certificate**
   - Repairs only an already-valid single register outlier when:
     - ACF >= 0.95,
     - CMNDF <= 0.05,
     - ACF and CMNDF agree on the same primary family,
     - the accepted sample is integer-related to that family, and
     - trusted valid pitch on BOTH sides (within 50 ms) agrees with the primary
       family within 1.5 semitones.
   - This is intended for the single-sample Bodies ~116.823 class exposed by
     the 135-Hz topology. It is not a general octave rewrite lane.

3. **Everything else stays put**
   - 135 Hz remains the default offline interpretation rate.
   - `--analysis-hz` remains available.
   - V3R6 rejected-pitch recovery is unchanged in concept.
   - HF2 isolated-false-valid certificate is retained.
   - V3R7 trajectory candidate authority and family-jump guard are retained.
   - MRV13 files are unchanged.

## New audit columns

`*.primary_reconciliation_v3r8.csv` adds:

- `continuous_persistence_count`
- `embedded_register_outlier_count`

New correction reasons include:

- `PRIMARY_TRACKER_DRIFT_REPLACED_V3R8_CONTINUOUS`
- `PRIMARY_EMBEDDED_REGISTER_OUTLIER_REPLACED_V3R8`

## Development rule

No song/time exceptions and no MIDI ground truth are used by V3R8. Tracker
changes remain judged against acoustic evidence.
