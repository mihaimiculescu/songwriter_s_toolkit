# V3R5 — Moving reacquisition-onset rescue

V3R5 branches directly from V3R4.  It preserves the V3R3/V3R4 integer-family
variable-aperture architecture and keeps the V3R4 subharmonic-borrowing
certificate, with `BORROWING_MIN_HIGH_ACF = 0.77` as requested for this test
cycle.

## Problem addressed

A real vocal F0 can move appreciably during the existing 24/40/64/96-ms
reacquisition aperture.  A misleading current-frame seed may therefore be
roughly subharmonic-related to the true onset, yet the true family measured at
64/96 ms can drift outside the frozen seed's 50-cent integer-family gate.
V3R4 then reports `no_integer_family_contest` and keeps the bad seed.

Bodies around 114.939 s is the reference case:

- current-frame seed: ~215.12 Hz
- 64-ms measured family: ~416.04 Hz
- 96-ms measured family: ~408.33 Hz

The 215-Hz seed disappears from the aperture ladder while one coherent upper
family persists.  The upper family is moving, so V3R5 does **not** solve this
by widening the integer-family cents tolerance.

## V3R5 rescue certificate

The new path is considered only when the ordinary integer-family contest has
fewer than two families.  It may replace an existing seed only when ALL of the
following are true:

1. `previous_hz is None`: this is a true tracker reset/gap, not continuous
   tracking or a normal harmonic-change transition.
2. An original current-frame seed exists.
3. No measured candidate within 4 semitones of that original seed survives at
   ANY reached aperture (24/40/64/96 ms).  A seed that remains acoustically
   visible is not reinterpreted.
4. Both 64 ms and 96 ms are reachable without crossing the existing silence,
   unvoiced, harmonic-change, or end-of-audio boundary gates.
5. Each wide aperture contains exactly ONE surviving measured family.
6. The 64-ms and 96-ms measurements each independently satisfy:
   - ACF >= 0.80
   - CMNDF <= 0.20
7. The two wide measurements stay within the existing 4-semitone local-family
   free zone, allowing genuine onset pitch motion without treating unrelated
   notes as one family.

When 96 ms confirms persistence, V3R5 initializes from the 64-ms measurement,
not the 96-ms measurement, because 64 ms is temporally closer to the actual
reacquisition instant.

## What V3R5 deliberately does NOT change

- the 50-cent integer-family relation used by V3R3/V3R4
- the ECKF equations or process/measurement noise
- silence calibration or boundary detection
- harmonic-change thresholds
- range logic, jurors, trajectory interpretation, or MIDI rendering
- the existing 24/40/64/96-ms aperture ladder
- the V3R4 subharmonic-borrowing logic except for retaining the requested
  `BORROWING_MIN_HIGH_ACF = 0.77`

## Regression fixture: six repeated Bodies melisms

Using the V3R470/V3R4-0.77 twelve-patient outputs as the reference state, the
new rescue certificate is eligible at exactly one reacquisition in the full
12-song corpus: Bodies ~114.939 s.

Direct variable-aperture adjudication of the six repeated Bodies entrances:

- ~31.208 s: unchanged (existing profile confirms original)
- ~35.016 s: unchanged (existing integer-family profile replaces ~182 Hz with
  ~361 Hz)
- ~38.824 s: unchanged
- ~111.224 s: unchanged
- ~114.939 s: NEW rescue, ~215.12 Hz -> ~416.04 Hz
- ~118.840 s: unchanged

This is a structural certificate, not a timestamp/song-specific exception.
