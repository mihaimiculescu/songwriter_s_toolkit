# V19 revised — bidirectional variable-aperture reacquisition

Baseline: V18 cross-region re-adjudication.

This revision keeps the V19 insertion point upstream of ECKF forth/back filtering but replaces the prototype greedy short-to-long stopping rule.

For octave/register ambiguity, every reachable aperture in the existing ladder is evaluated:

- 24 ms
- 40 ms
- 64 ms
- 96 ms

The same raw-waveform candidate evidence is interpreted three ways:

1. short -> long preference (local establishment),
2. long -> short preference (broad-context persistence),
3. total support breadth across the ladder.

A competing register family must win a strict majority of these reasons before it may replace an existing V18 seed. Ties preserve the V18 seed.

Register-family identity may drift within the interval juror's existing <=4-semitone free zone, so ordinary melodic movement across wider apertures is not mistaken for disappearance of a register family.

The adjudicator still fails closed at silence, unvoiced material, harmonic-change boundaries, or end-of-real-audio. It has no octave-up/down preference, uses no downstream trajectory/MIDI information, and does not add a bend-size or song-specific threshold.

The historical block lookahead remains the fallback when variable-aperture evidence does not establish a seed.

Existing frame audit columns remain:

- `va_reacquisition_aperture_ms`
- `va_reacquisition_reason`
- `va_reacquisition_original_hz`
- `va_reacquisition_selected_hz`

Typical revised reasons include:

- `bidirectional_profile_majority`
- `bidirectional_profile_confirms_original`
- `bidirectional_profile_no_majority`
- `bidirectional_unique_future_family`
