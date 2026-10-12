# V3R2 — Provisional-Context Interval Authority

V3R2 branches from V3R1 and changes only persistence of existing octave-pair
interval verdicts.  The cubic interval curve, jury weights, range gate,
variable-aperture reacquisition, onset logic, renderer, and musical structure
algorithms are unchanged.

## Problem

Immediately after an interruption a reacquisition frame can establish a
provisional owner while no interval reference exists.  On the next frame the
interval juror can then prefer continuation of that provisional owner and veto
the correct octave, creating a self-confirming loop:

    NO INTERVAL VERDICT
      -> provisional post-gap owner
      -> next-frame interval verdict relative to that provisional owner
      -> wrong owner becomes authoritative

A nine-patient counterfactual showed that discarding all first-post-gap verdicts
would be unsafe.  V3R2 therefore targets only the actual dependency above.

## Rule

A frame is a `provisional_post_gap_reference` only when:

* at least one candidate is `selected_by_initializer`; and
* no candidate in that frame has a usable previous or following interval
  component.

If a later octave-pair interval verdict explicitly uses that exact provisional
frame/F0 as its previous interval reference:

* the verdict is still recorded in `<csv>.interval_authority.csv`;
* `authoritative=0`;
* `context_status=provisional_post_gap_reference`;
* downstream register/reacquisition re-attribution may not use it as a hard
  veto.

All other V3R1 interval verdicts retain their existing authority semantics.

## Audit additions

`<csv>.interval_authority.csv` now includes:

* `authoritative`
* `context_status`
* `reference_time_s`
* `reference_hz`

## Unchanged

* hard range gate
* interval cubic penalty
* jury / HD scoring
* V19 bidirectional 24/40/64/96 ms reacquisition
* V18 register re-adjudication objective
* 2-semitone ECKF reset threshold
* MIDI renderer and articulation
* fixed velocity 100
* `CC11_DB_COMPRESSION = 0.3`
