# Hard Range Consistency Gate V11

V11 extends V10 range-aware rescue into a single final invariant for the offline trajectory:

> If the preliminary per-file V22 zero-support range is available, no corrected-valid F0 outside `[zero_low_midi, zero_high_midi]` may enter trajectory interpretation.

This applies regardless of validity provenance, including ordinary first-pass validity, future-confirmed reacquisition, forward/backward/ornament rescue, and continuity states. Raw ECKF F0 is not altered; out-of-range samples are simply marked invalid with `RANGE_ZERO_VETO` before `clean_f0_hz` and trajectory interpretation are built.

The preliminary range itself is unchanged from V10: it uses first-pass-valid observations, the existing V13 reference evaluator, and V22 q02/q98 + plateau/shoulder geometry. If preliminary calibration is unavailable, the hard gate is disabled and behavior falls back unchanged.

Renderer defaults carried forward from the project-local production renderer:
- velocity 100
- CC11_DB_COMPRESSION = 0.3
- the current dB->linear CC11 baseline and differential-modulation formulas
