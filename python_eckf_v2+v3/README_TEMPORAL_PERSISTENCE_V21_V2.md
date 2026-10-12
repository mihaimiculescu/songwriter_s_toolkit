# ECKF V2 — V21 shifted temporal persistence V2

This revision keeps the exact historical V21 waveform probe geometry from V1:

- apertures: 24 / 40 / 64 ms
- shifts: -8 / 0 / +8 ms
- overlap-normalized lag ACF
- support threshold: ACF >= 0.72
- persistent: support in at least 6 paired shifted views
- clearly non-persistent: support in at most 2 paired shifted views
- fewer than 3 jointly testable views: V21 abstains

## Integration change from V1

V1 replaced the existing V2 temporal component with zero whenever the V21 probe
returned both-persistent, mixed/weak, or insufficient evidence. That discarded
useful pre-existing temporal information and increased abstentions.

V2 treats the shifted V21 measurement as a directional expert test:

- `lower_period_persistent_higher_not` -> V21 overrides toward LOW
- `higher_period_persistent_lower_not` -> V21 overrides toward HIGH
- `both_integer_periods_shift_persistent` -> keep existing V2 temporal component
- `temporal_support_mixed_or_weak` -> keep existing V2 temporal component
- `insufficient_paired_shift_measurements` -> keep existing V2 temporal component

No juror weights or other thresholds are changed. `W_TEMPORAL` remains 0.20.
The V21 pattern and shifted-window diagnostics remain exported in the juror pair
report even when the production temporal score falls back to the existing V2
component.
