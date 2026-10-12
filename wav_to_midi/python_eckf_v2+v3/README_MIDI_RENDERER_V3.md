# MIDI Renderer V3

V3 replaces V2's direct cue-to-split logic with a semantic articulation hierarchy.

## Musical articulation hierarchy

1. **Nonmusical interruption first**
   - A contiguous corrected-invalid run between trusted pitch islands is a hard
     musical boundary.
   - The renderer carves the interruption out of the MIDI note, so consonants,
     breaths, and genuinely non-pitched gaps are not rendered as saxophone.
   - The next trusted pitch island gets a new Note On.

2. **Legato note changes from stable target formation**
   - Continuous voicing does not imply one MIDI note.
   - A new stable pitch center established by the existing target-formation lane
     becomes a new Note On even with no consonant/silence break.

3. **Vibrato / glissando stay continuous**
   - Pitch movement alone cannot create a Note On.
   - If the existing target-formation layer does not establish a new stable
     center, the motion stays inside the note as Pitch Bend.

4. **ECKF resets and amplitude valleys are supporting evidence only**
   - They no longer create MIDI Note Ons by themselves.
   - The existing 2.0-semitone ECKF harmonic-change/reset threshold is unchanged.

## MIDI contract retained

- MIDI 1.0 SMF Type 1
- PPQ 960
- Track 0 meta, Track 1 melody
- Melody channel 0 (DAW channel 1)
- GM #65 Soprano Saxophone (Program Change payload 64)
- Explicit Pitch Bend Sensitivity RPN before every bend-bearing note
- Pitch Bend reset to center after Note Off
