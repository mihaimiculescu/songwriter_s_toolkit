# MIDI Renderer V5

Targeted production refinement on top of V4.

## Change
Completely ownerless corrected-valid F0 islands are no longer promoted to MIDI notes merely because F0 exists. They must survive at least the target-former's existing `min_stable_threshold_ms` duration. This reuses an upstream musical timescale rather than adding a renderer-specific threshold.

This is intended to reject tiny pitched consonant fragments while retaining genuine ownerless note islands.

## Unchanged
- Jury / HD / structural collaboration
- 2.0-semitone ECKF reset policy
- Nonmusical-gap articulation
- Stable-target / legato articulation
- Transition ownership and pitch-bend behavior
- MIDI 1.0 Type 1, Track 1 melody, channel 0, GM #65 Soprano Saxophone
- Explicit RPN pitch-bend sensitivity and post-Note-Off bend reset
