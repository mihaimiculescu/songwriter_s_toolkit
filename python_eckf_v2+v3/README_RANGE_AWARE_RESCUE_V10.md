# Range-aware offline rescue V10

This release adds one consistency rule to the offline validity correction:

- build a preliminary per-file range calibration before rescue, using only first-pass-valid observations;
- use the existing V13 range-reference evaluator and unchanged V22 q02/q98 + plateau/shoulder geometry;
- if that calibration is unavailable (<40 anchors), rescue behavior is unchanged;
- if available, a coherent rejected subrun is not eligible for ORNAMENT/BACKWARD/FORWARD rescue when any constituent F0 lies outside the preliminary zero-support bounds;
- existing first-pass-valid samples are never invalidated by this pass;
- jury, Harmonic Detective, structural collaboration, MIDI articulation, and renderer logic are unchanged.

The CLI exports `<csv>.pre_rescue_range_calibration.csv` for audit and labels vetoed rows `RANGE_ZERO_VETO`.
