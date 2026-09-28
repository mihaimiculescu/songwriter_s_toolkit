# ECKF V2 — Harmonic Detective Authority V2

This build starts from `python_eckf_v2_range_reference_restored_v1` and changes only the conditional Harmonic Detective policy.

## What stays unchanged

- restored V13/V22 range-reference evaluator
- V22 range geometry and `W_RANGE = 0.5`
- strict `<49 cents` same-note grouping
- jury winners remain sovereign: the Detective is never called when the jury already has a champion
- hard range rejection remains sovereign
- weak/insufficient harmonic evidence still abstains
- ECKF, silence, variable aperture, gesture, and expressive lanes are unchanged

## Change 1 — score >=49-cent singleton contestants too

The previous Detective scored only the `<49c` note-group representatives. If an admitted singleton contestant existed outside that grouping, it abstained with `incomplete_field_outside_49c_contestant`.

V2 scores every acoustically admissible contestant that survived the range gate. `<49c` candidates are still collapsed to one representative per note group upstream. `>=49c` candidates remain independent singleton contestants and are now audible to the Detective rather than forcing an automatic abstention.

## Change 2 — confidence-adaptive authority

The historical ordinary route remains:

- harmonic score >= 0.45
- harmonic support >= 0.50
- at least 2 valid windows
- winner margin >= 0.08

A second expert-confidence route is added:

- harmonic score >= 0.80
- harmonic support >= 0.90
- at least 2 valid windows
- winner margin >= 0.02

This does **not** globally lower the margin requirement. Weak close races still abstain. Only strong absolute harmonic evidence is allowed to break a tighter race.

Reasons exported in `.harmonic_detective.csv` distinguish:

- `harmonic_detective_tiebreak`
- `harmonic_detective_high_confidence_tiebreak`
- `harmonic_margin_insufficient`

## Next planned stage

Per-file silence/noise calibration will replace the fixed `-50` threshold only after this Detective V2 regression is inspected. The planned estimator will use sufficiently long silence regions in each file; it is intentionally not included in this build so rules are not changed simultaneously.
