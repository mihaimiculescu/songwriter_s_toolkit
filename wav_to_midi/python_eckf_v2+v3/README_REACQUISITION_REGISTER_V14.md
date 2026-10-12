# V14 — Register-authoritative reacquisition islands

V14 replaces V13's frame-local island sweeps with a single global candidate-path
solution for every contiguous reacquisition/recovery island.

Principles:
- candidate states are only already-existing, in-range, acoustically admissible
  contestants from the existing jury evidence;
- no octave bucket, bend cap, song-specific pitch, GT target, or new weighted
  score is introduced;
- path objective is lexicographic:
  1. minimise the existing cubic interval-juror penalty;
  2. retain as many recovery frames as possible;
  3. minimise changes from existing adjudicated owners;
  4. use existing Harmonic-Detective score only as a tie-breaker;
- an immediate corrected-valid sample directly before/after the recovery island
  may act as a boundary anchor; this never bridges silence because the anchor
  must be directly adjacent to the island;
- a frame may be left invalid when every available candidate contradicts the
  coherent island path. No replacement F0 is invented.

This specifically prevents recovery islands from producing frame-local register
staircases such as 53 -> 58 -> 65 when the trusted continuation is already in
register 65.
