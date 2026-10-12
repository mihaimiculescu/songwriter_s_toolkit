# MIDI Renderer V8

V8 is a dynamics/state-semantics correction on top of MIDI Renderer V7.

## Changes

1. Deterministic MIDI-channel initialization at file start (Track 1 / melody channel):
   - Program Change = GM #65 Soprano Saxophone (wire payload 64)
   - Pitch Bend = 0 (center)
   - CC7 Volume = 127
   - CC11 Expression = 127

   Pitch Bend = 0 is emitted unconditionally, even when no later pitch-bend messages are needed.

2. Dynamic velocity remains based on physical relative attack amplitude:
   - source attack level is measured in dBFS from the existing amplitude lane;
   - dB difference is converted to a linear amplitude ratio with 10^(dB/20);
   - the strongest rendered attack maps to velocity 127;
   - quieter attacks preserve the measured linear RMS ratio.

3. CC11 expression spans:
   - admission remains conservative and independent of pitch modulation;
   - repeated amplitude-baseline crossings are required;
   - once admitted, CC11 follows the whole contiguous serviced span's measured shape RMS;
   - dB is converted to a linear amplitude ratio before mapping to 0..127;
   - the span peak maps to 127;
   - therefore a gentle attack may start below 127 and later rise above its attack CC11 value;
   - a CC11 span may cover multiple notes;
   - CC11 is reset to 127 before establishing the serviced span's measured start level and reset to 127 after the serviced span ends.

4. CC7 is not used as a modulation controller. It is initialized once to 127 and left unchanged.

5. Pitch Bend behavior from V7 is retained:
   - per-note bend-range RPN where needed;
   - PB=0 before bend-bearing note;
   - PB=0 after Note Off;
   - plus the new unconditional PB=0 at file start.

No tracker, jury, Harmonic Detective, target-formation, structural-collaboration, or articulation thresholds are changed in V8.
