# MIDI Renderer V9

V9 changes MIDI dynamics semantics only; articulation/event construction remains V8/V7-compatible.

## Velocity
- MIDI Note On velocity is fixed to 100 by default.
- `--velocity` may still override the fixed value explicitly.
- Source amplitude is no longer encoded redundantly in both velocity and CC11.

## CC11 expression
- Track/channel initializes CC11 to 127 once at file start.
- Every Note On is preceded by a CC11 baseline derived from that note's measured attack amplitude.
- The loudest rendered attack maps to CC11=127.
- Quieter attacks preserve linear amplitude ratio using `10 ** (delta_dB / 20)`.
- Amplitude modulation is applied differentially around the current note baseline using the existing amplitude-residual lane.
- CC11 is NOT reset at Note Off.
- CC11 is NOT reset at the end of an amplitude-modulation span.
- The next Note On simply establishes the next note's CC11 baseline.
- A modulation span may continue across several notes; each new note re-anchors the same residual modulation to its own measured CC11 baseline.

## CC7
- CC7 is initialized once to 127 at file start.
- CC7 is not modulated afterward.

## Pitch Bend
- Pitch Bend is initialized to center (0) at file start unconditionally.
- Bend-bearing notes still set their required bend range explicitly.
- Every bend-bearing note receives PB=0 before use and PB=0 after its Note Off.

## Initial channel state
Track 1 starts with:
- GM Program #65 Soprano Saxophone (wire program 64)
- Pitch Bend = 0
- CC7 = 127
- CC11 = 127

