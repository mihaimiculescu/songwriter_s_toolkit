# ECKF V2 — historical range-reference evaluator restored (V1)

This build keeps the active V2 pipeline, `<49 cents` grouping, `W_RANGE=0.5`,
and conditional Harmonic Detective, but replaces the provisional range anchor
population (`all corrected-valid clean F0`) with the historical V13-style
**unique locally supported period** contract that fed the mature V22 gate.

## Restored contract

Per candidate-bearing frame:

1. Use the median corrected-valid V2 F0 for that frame only as the
   observational pointer.
2. A measured candidate must lie within **35 cents** of that pointer.
3. A rival more than **70 cents** away makes the frame acoustically ambiguous
   when its ACF is within **0.07** of the selected candidate and its measured
   component strength is at least **0.5×** the selected candidate.
4. The historical interval+energy joint challenge is preserved: a transition
   penalty can veto use as a subsequent range reference only when the local
   waveform is both weak relative to its own ±300 ms neighbourhood
   (`relative_energy < 0.25`) and rapidly decaying (`slope < -4 dB`).
5. Only `acoustic_period_supported_source_unverified` rows become range
   references.
6. V22 range calibration itself remains unchanged:
   q02/q98 -> nearest MIDI anchors -> ±2 semitone full-confidence plateau ->
   2-semitone half-cosine shoulder -> hard zero.
7. `W_RANGE = 0.5` remains unchanged.

The within-frame component-strength ratio is the V2 equivalent of historical
V13's `fundamental_fraction` ratio.  Since both candidates are measured in the
same frame, the common file gain cancels.

## New audit export

`<output>.range_references.csv`

This exposes every candidate-bearing frame, whether it became a trusted range
reference, why it was rejected, local energy/decay evidence, and the transition
challenge if one existed.

No GroundTruth MIDI range is read by production code.  GroundTruth is only for
our regression validation.
