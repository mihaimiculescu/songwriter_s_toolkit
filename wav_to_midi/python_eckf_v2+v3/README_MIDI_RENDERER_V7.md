# MIDI Renderer V7

V7 combines the final articulation refinement with MIDI dynamics, without changing ECKF, jury, Harmonic Detective, structural collaboration, or the 2.0-semitone reset detector.

## Articulation hierarchy

A new MIDI Note On may be created by:

1. an existing nonmusical/corrected-invalid interruption;
2. an existing persistent stable-target landing;
3. a renewed acoustic attack inside continuous voicing.

The renewed-attack detector uses only the existing amplitude-expression lane. A local amplitude valley must fall below its local baseline, be deeper than the lane's own local modulation RMS, then rebound through the local baseline more steeply than the preceding decay. Pitch variation alone never creates a Note On. Ordinary amplitude variation alone also does not; the asymmetric renewed-attack signature is required.

ECKF reset + amplitude support remains an additional same-pitch rearticulation cue. The existing 2.0-semitone ECKF reset policy is unchanged.

## Velocity

Dynamic velocity is now the default. For each rendered note, the strongest `shape_rms_dbfs` within the existing project stable-timescale after musical Note On is used as attack level. The strongest rendered attack in the file maps to velocity 127; quieter attacks preserve their measured linear RMS ratio.

`--velocity N` remains available as a fixed-velocity override.

## CC11 Expression

Amplitude modulation is serialized independently of Pitch Bend. Within a continuous musical span, repeated crossings of the existing local amplitude baseline identify an oscillatory amplitude span. At least two cycles / four crossings are required before a CC11 burst is emitted.

A CC11 burst may span several notes. CC11 is explicitly reset to 127 before and after the serviced span. Inside the burst, values preserve the measured relative dB amplitude with the local burst peak mapped to 127.

This does not alter pitch, pitch-bend range, note identity, or articulation decisions.

## MIDI invariants retained

- MIDI 1.0 SMF Type 1
- Track 0 global metadata
- Track 1 monophonic melody
- channel 0 / DAW channel 1
- GM #65 Soprano Saxophone (`program=64` wire payload)
- every bend-bearing note explicitly sets RPN 0,0 Pitch Bend Sensitivity
- Pitch Bend resets to center after Note Off
- CC11 resets to 127 before and after every serviced note/group span
