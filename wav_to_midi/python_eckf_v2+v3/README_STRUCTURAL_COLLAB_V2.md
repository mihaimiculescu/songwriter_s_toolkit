# Structural Collaboration V2

V2 is a residual post-adjudication collaboration pass. It introduces no new
juror and never overrides an existing jury or Harmonic Detective winner.

Changes from V1:

1. **Frame-span ownership bug fixed.** Structural ownership is computed from the
   full physical frame interval `[start, end)`, not by sampling one timestamp at
   the left edge. The trajectory region with the largest temporal overlap owns
   the frame; no overlap threshold is introduced.
2. **Residual temporal ownership.** If neither a stable target nor a covered
   transition owns an unresolved frame, V2 uses only the already-frozen trusted
   100 Hz trajectory mask inside that physical frame:
   - trusted prefix -> `resolved_boundary_left` / `left_release`
   - trusted suffix -> `resolved_boundary_right` / `right_attack`
   - no trusted points -> `resolved_nonmusical`
   - interior trusted island -> `resolved_short_target_candidate`
   - trusted material spanning both edges or fragmented -> remains unresolved

Stable target pitch remains the median in semitone/log-frequency space, which is
correct for equal-tempered/geometric frequency relationships.
