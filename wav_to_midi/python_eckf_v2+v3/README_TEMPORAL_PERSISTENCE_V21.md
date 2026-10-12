# ECKF V2 — exact V21 shifted temporal persistence

This build is based on `python_eckf_v2_adaptive_silence_v4` and implements the first of the final three historical-shadow restorations.

## What changed

The temporal juror no longer uses only adjacent-frame note-group presence.  For every pairwise jury comparison it now re-measures the **original WAV** exactly in the historical V21 geometry:

- apertures: **24, 40, 64 ms**
- shifts: **-8, 0, +8 ms**
- overlap-normalized lag ACF
- candidate considered supported in a shifted view when **ACF >= 0.72**
- a view is counted only when *both* candidate periods are testable
- exact V21 persistence categories:
  - `both_integer_periods_shift_persistent`
  - `higher_period_persistent_lower_not`
  - `lower_period_persistent_higher_not`
  - `temporal_support_mixed_or_weak`
  - `insufficient_paired_shift_measurements`

The historical decision thresholds are preserved:

- persistent: at least **6** supported shifted views
- non-persistent opponent: at most **2** supported shifted views
- fewer than **3** paired measurements => insufficient

The temporal component remains weighted by the existing V2 `W_TEMPORAL = 0.20`.  No other juror weight or threshold was changed.

## Auditability

`<output>.juror_pairs.csv` now includes:

- `temporal_pattern`
- `temporal_paired_measurements`
- `temporal_low_supported_shift_count`
- `temporal_high_supported_shift_count`
- `temporal_low_acf_median`
- `temporal_high_acf_median`

This lets us compare exactly how the restored V21 persistence evidence changes jury outcomes and the final Detective/unresolved census.

## Deliberately not included yet

This build does **not** yet implement the next two agreed restorations:

1. direct WAV measurement for missing tournament edges;
2. explicit interruption barrier for interval references.

Those stay separate so each effect can be measured independently.
