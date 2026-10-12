# HD dispute filter V2

Changes relative to `v2_hd_dispute_filter_v1`:

1. Removed the stale `same_note_representatives_unexpected` Harmonic Detective guard.
   Same-MIDI representatives are now adjudicated by the existing HD score/support/margin rules.
2. Raised the active offline-V2 silence fallback/default floor from `-27.7734` to `-13.7734`
   on the historical statistic `20*log10(sum(frame**2))`, corresponding to approximately
   `-40 dBFS RMS` for 2048-sample frames.
3. MATLAB compatibility mode remains at its historical `-50` default unless explicitly overridden.

No other adjudication thresholds or juror weights were changed.
