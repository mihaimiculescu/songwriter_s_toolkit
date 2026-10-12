# ECKF V2 — Explicit interval-reference interruption barrier V1

This stage restores the historical V13/V22 interval-reference barrier.

## What changes

The interval juror no longer uses the initializer-selected candidate from the
immediately adjacent candidate-bearing frame as context.

Instead it uses the historical trusted V13 reference stream:

- `evidence_status == acoustic_period_supported_source_unverified`
- `diagnostic_settled == true`
- `usable_as_next_reference == true`

The previous and following references are searched independently within the
historical **50 ms** maximum gap.

A reference is **not allowed to cross** an explicit interruption timestamp.
Exactly as in historical V22, an interruption is defined by a V13 row with:

- `low_energy_flag == true`, **or**
- `decaying_flag == true`.

An unresolved F0 by itself is **not** treated as silence/interruption.

## Why

This prevents interval plausibility from leaking across a breath, low-energy
break, decay/reset, or other explicit interruption.  A candidate after a break
must establish itself from evidence on its own side of the break.

## What does NOT change

- V21 shifted temporal persistence V2
- direct WAV completion of missing tournament edges
- range gate / W_RANGE
- Harmonic Detective
- adaptive-silence V4
- ECKF / validity / target formation / expressive lanes
- MATLAB mode

## New juror-evidence audit columns

- `interval_previous_reference_time_s`
- `interval_previous_reference_hz`
- `interval_following_reference_time_s`
- `interval_following_reference_hz`
- `interval_previous_blocked_by_interruption`
- `interval_following_blocked_by_interruption`

These make every barrier action directly auditable.
