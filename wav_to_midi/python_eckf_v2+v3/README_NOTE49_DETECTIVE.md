# ECKF V2 — strict <49-cent note grouping + Harmonic Detective V1

This build extends `python_eckf_v2_v22_range_gate_v1` without changing the ECKF,
validity, variable-aperture, gesture, or expressive lanes.

## 1. Historical V12 same-note grouping restored

For every candidate frame:

- nearest equal-tempered MIDI note is computed upstream;
- only candidates **strictly** inside `abs(cents_from_note) < 49.0` are allowed to
  share a same-note group;
- candidates at 49 cents or farther are **not discarded** and are **not silently
  assigned** to the nearest note: each remains an independent singleton contestant;
- one representative per true note group is selected by strongest ACF support,
  retaining deterministic tie breakers only for reproducible export;
- only representatives enter the inter-group jury tournament.

This prevents multiple near-identical F0 hypotheses for the same musical note from
receiving multiple effective votes.

## 2. Harmonic Detective enabled conditionally

The expert witness is now active after the four-juror tournament.

- Existing jury champions are immutable. The Detective never overrides them.
- The Detective is called only for genuine hung-jury outcomes.
- Hard range-gate / no-acoustic-support abstentions are not rescued.
- Only acoustically admitted `<49c` group representatives are eligible.
- If a live in-range outside-49c singleton exists, the Detective abstains rather
  than silently deleting that contestant.
- Two centered FFT/STFT windows are used: 64 ms and 96 ms.
- The historical V2 qualification rules are retained:
  - at least 2 valid windows;
  - support >= 0.50;
  - score >= 0.45;
  - best-minus-runner margin >= 0.08.

New files:

- `.harmonic_detective_candidates.csv`
- `.harmonic_detective.csv`
- `.adjudication_final.csv`

`adjudication_final.csv` records provenance as `jury`, `harmonic_detective`, or
`abstention`.

## Important range-calibration note

The V22 range **geometry** and `W_RANGE=0.5` remain active. However, the current V2
anchor source is still `corrected_valid_clean_f0_v2`, which is not a byte-for-byte
replacement for historical V22's V13 `diagnostic_settled && usable_as_next_reference`
reference contract. This is visible especially in RATATA and should be audited before
the final unresolved-case census.

`--matlab` remains untouched.
