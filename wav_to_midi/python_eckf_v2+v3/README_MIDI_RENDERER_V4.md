# MIDI Renderer V4 — ownership-complete event construction

V4 fixes a renderer architecture hole discovered with Trandafiri: corrected-valid F0 could live inside trajectory regions conservatively labelled `UNRESOLVED`, while the renderer created primary events only from `STABLE_TARGET`/short-target structures. Entire valid pitched islands could therefore disappear from MIDI.

Changes:

- Adds `midi_event_builder.py`.
- After normal stable/short-target construction, it scans the complete `target_formation.csv` corrected-valid timeline.
- Any contiguous corrected-valid F0 sub-span not already owned by an existing MIDI event is recovered as `residual_valid_f0`.
- No new duration or pitch-confidence threshold is introduced; upstream validity remains authoritative.
- Existing transition, boundary, articulation, nonmusical-gap and pitch-bend logic still runs afterwards.
- Adds `<output>.mid.coverage.csv` and a loud `UNOWNED_VALID_F0` warning if any corrected-valid timeline cell remains unrepresented by the final MIDI event set.

This does not weaken jury, Harmonic Detective, target-formation, or structural-collaboration logic. It fixes only MIDI-side ownership completeness.
