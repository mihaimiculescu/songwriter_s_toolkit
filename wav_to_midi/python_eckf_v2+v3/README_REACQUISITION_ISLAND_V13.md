# V13 reacquisition-island adjudication

V13 replaces V12's frame-local reacquisition feedback with island-level
consistency.  It introduces no song-specific values and no new musical
thresholds.

* A recovery island comprises contiguous 100-Hz samples whose first-pass state
  is `WAITING_FOR_REACQUISITION` / `REACQUIRED_FUTURE_CONFIRMED` or whose
  correction provenance is rescue / hard-range-veto / prior adjudication.
* V12's existing final-jury and reacquisition-only HD proposals are retained.
* A frame is reconsidered only when its current proposal incurs a non-zero
  existing cubic interval-juror penalty relative to another adjudicated frame
  inside the same recovery island.
* Only already-existing, range-admitted, acoustically admissible candidates may
  replace it.
* The replacement must strictly lower that same interval-juror penalty.
* HD harmonic score is a tie-breaker only.
* External silence/breath gaps are never bridged to fabricate interval evidence.
* V11 hard zero-support range gate remains authoritative.
