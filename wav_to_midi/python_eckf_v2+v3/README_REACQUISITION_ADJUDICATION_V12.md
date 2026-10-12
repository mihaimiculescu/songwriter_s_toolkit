# V12 — Reacquisition Adjudication Feedback

This release closes the gap between frame-level adjudication and the 100 Hz corrected trajectory.

## Scope

Only samples that are:

- `REACQUIRED_FUTURE_CONFIRMED`, or
- `RANGE_ZERO_VETO`

are eligible for feedback. Ordinary already-stable tracking is untouched.

## Rules

1. If hard range gating removed the tracker-selected owner, an existing resolved jury/HD winner for that frame may refill the trajectory instead of leaving a hole.
2. For reacquisition frames whose jury result is only a `provisional_tournament_champion`:
   - if the interval juror had usable previous/following evidence, the jury remains authoritative;
   - if the interval juror had no usable reference, the existing Harmonic Detective performs a reacquisition-only review over the same in-range, acoustically admissible contestants;
   - the review must satisfy the existing ordinary or high-confidence HD thresholds.
3. Any replacement is projected back onto the 100 Hz corrected-valid trajectory before trajectory/gesture interpretation is rebuilt.
4. No new octave detector, no new thresholds, and no GT-specific logic are introduced.

## Audit

A new sidecar is emitted:

`<csv>.reacquisition_feedback.csv`

It records the original owner, jury owner, reviewed owner, HD scores/margin, and number of 100 Hz samples replaced.

## Renderer defaults preserved

- fixed velocity 100
- `CC11_DB_COMPRESSION = 0.3`
- both CC11 dB-to-linear ratio formulas use the compression constant
