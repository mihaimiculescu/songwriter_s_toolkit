# V3R3 — Integer-family variable-aperture reacquisition

V3R3 branches from V3R2 and changes only the upstream reacquisition-family contest.

## Problem
The initializer can prefer a long-period integer subharmonic because ACF/CMNDF
are slightly cleaner, even when a higher integer-related family has much stronger
measured fundamental and harmonic support.  This can produce either an in-range
low-register error or a below-range seed that is later discarded as if the onset
had no usable F0.

## Change
The existing bidirectional 24/40/64/96-ms variable-aperture adjudicator is
generalized from octave-only families to all already-measured integer-related
families with ratios x2..x5, using the existing 50-cent integer-relation tolerance.

No pitch is multiplied or divided to manufacture a candidate.  Only measured
period families already present in the waveform evidence participate.

The same bidirectional support-profile decision remains in force:
- short->long establishment,
- long->short persistence,
- total support breadth,
with a strict-majority requirement before replacing an existing seed.

Normal melodic drift within the existing <=4-semitone free zone is still treated
as the same local family center.

## Unchanged
- V3R2 provisional-context interval-authority semantics
- hard range gate
- silence/interruption boundaries
- 2-semitone ECKF harmonic-change/reset threshold
- jury / Harmonic Detective scoring
- target / gesture interpretation
- onset logic
- MIDI renderer and its project defaults

## Targeted regression checks
On Champagne excerpts, V3R3:
- reacquires the formerly missing first attack at ~292-294 Hz instead of ~97 Hz;
- reacquires the former ~56.38-s low island around ~342 Hz;
- reacquires the former ~71.06-s low island around ~342-345 Hz.

These are targeted checks, not a substitute for the full nine-patient robustness run.
