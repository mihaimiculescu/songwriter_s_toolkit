# Structural Collaboration V3

V3 keeps all v2 jury, Harmonic Detective, silence, interval-juror, and frame-span ownership behavior unchanged.

Only residual unresolved frames are touched. After stable-target and existing gesture ownership have had first authority, v3 examines the already-frozen corrected-valid 100 Hz trajectory mask and the maximal contiguous trusted run intersecting the physical frame. No new acoustic score, juror, confidence threshold, or duration threshold is introduced.

Topology determines the interpretation:

- run continues only to the left: `resolved_boundary_left` / left release
- run continues only to the right: `resolved_boundary_right` / right attack
- run continues through both sides: `resolved_continuous_motion`
- run is isolated from both sides: `resolved_short_target_candidate`
- no trusted points: `resolved_nonmusical`
- multiple disconnected trusted runs: `resolved_fragmented_voiced`

These are temporal/structural interpretations. They intentionally do not manufacture a whole-frame F0 winner.
