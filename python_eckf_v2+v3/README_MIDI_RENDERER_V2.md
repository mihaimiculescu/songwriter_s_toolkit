# MIDI Renderer V2 — Musical articulation

Renderer V2 no longer assumes `STABLE_TARGET == one MIDI Note On`.

It preserves the existing pitch/gesture interpretation and adds an articulation
layer built only from already-computed evidence:

1. ECKF reset/onset landmarks (`*.onsets.csv`, newly exported by `cli.py`).
2. Renewed attacks inferred from valleys in the existing amplitude-expression
   shape. The prominence floor is song-adaptive (median positive local-valley
   prominence), not a hand-tuned dB constant.
3. Persistent internal target changes from existing target-formation
   `structural_pitch_st` and the already-used variable-aperture durations.
4. Frozen `TRANSITION` regions suppress articulation cuts and remain continuous
   pitch-bend gestures.

Nearby candidate articulation points are de-duplicated with the project's
existing `min_stable_threshold_ms`; no new minimum-note-duration constant is
introduced.

The 2.0-semitone harmonic-change/reset detector is untouched. Its landmarks are
used as one input to musical articulation, not redefined as final Note Ons.

MIDI invariants remain unchanged: Type 1, melody Track 1/channel 0, GM #65
Soprano Saxophone (Program Change payload 64), explicit pitch-bend sensitivity
RPN before every bend-bearing note, and Pitch Bend = 0 after Note Off.
