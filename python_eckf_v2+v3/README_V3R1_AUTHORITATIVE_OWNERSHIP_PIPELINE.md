# V3R1 — Authoritative ownership pipeline

V3R1 branches from V3R0 / revised bidirectional V19.

## Purpose

Reduce repeated pitch-ownership reinvention without changing the tracker,
range model, juror weights, HD thresholds, onset logic, or MIDI renderer.

## Reordered flow

After offline validity and the pre-rescue range gate:

1. build final V13/V22 range references;
2. build juror evidence and pair verdicts;
3. run the jury bench and Harmonic Detective;
4. build a persistent interval-authority ledger;
5. apply adjudicated reacquisition feedback;
6. build trajectory / expressive / gesture structure for the first time;
7. run V18 register re-adjudication only where needed;
8. rebuild the musical interpretation stack once only if V18 changed ownership;
9. structural collaboration remains last.

This removes the previous provisional trajectory/gesture build that was thrown
away whenever reacquisition feedback changed F0 ownership.

## Persistent interval authority

Only an EXISTING octave-pair interval-juror verdict is persistent.

For a candidate pair separated by exactly 12 MIDI semitones, a verdict is
recorded only when both already-computed interval components exist and differ.
The candidate with the larger component (closer to zero) is interval-preferred;
the other is interval-disfavoured for ordinary downstream octave/register
re-attribution.

If interval evidence was absent because of an interruption, or both components
were equal, there is no veto and downstream behavior remains unconstrained by
this mechanism.

This does NOT make the overall jury winner immutable.  It preserves only the
specific interval-juror opinion, avoiding the design error of freezing a bad
jury winner when the interval juror had no opinion.

## New audit

`<csv>.interval_authority.csv`

Columns identify the preferred and rejected octave candidate plus the original
interval components and provenance.

## Unchanged

- ECKF tracker and 2.0-semitone reset threshold
- bidirectional V19 24/40/64/96 ms tracker reacquisition
- preliminary and final range logic
- juror weights and HD thresholds
- 100 Hz offline interpretation rate
- target/gesture algorithms
- structural collaboration rules
- MIDI renderer, fixed velocity 100, CC11_DB_COMPRESSION=0.3
