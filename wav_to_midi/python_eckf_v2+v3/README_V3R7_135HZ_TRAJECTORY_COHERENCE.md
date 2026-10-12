# V3R7 — 135 Hz rate-aware trajectory coherence

V3R7 is branched from V3R6 Hotfix 2.  It combines two orthogonal upstream changes while keeping MIDI Renderer V13 byte-for-byte frozen.

## 1. Offline interpretation grid is now rate-aware, default 135 Hz

The ECKF core is unchanged and still runs at full audio-sample resolution.
Only the offline interpretation trajectory changes.

A new `analysis_timing.py` owns:

- `DEFAULT_OFFLINE_ANALYSIS_HZ = 135.0`
- `resolve_offline_analysis_grid(sr, requested_hz)`
- the actual integer-hop-derived `actual_hz` and `dt_s`

CLI now accepts:

```bash
--analysis-hz 135
```

135 Hz is the default, so the flag may be omitted.  The actual per-file rate is always:

```text
actual_hz = sample_rate / round(sample_rate / requested_hz)
```

For example:

- 44.1 kHz -> 327 samples -> 134.862385 Hz -> 7.414966 ms
- 48 kHz -> 356 samples -> 134.831461 Hz -> 7.416667 ms

All active offline analysis config defaults now reference the shared 135-Hz default, but CLI passes the measured actual rate explicitly downstream.

There is deliberately **no global `DT = 0.01`**.  Grid-dependent code uses `analysis_hz`/derived timestamps.  Physical-time tolerances remain physical seconds.

## 2. Existing-valid drift replacement now uses local trajectory coherence

Rejected-pitch recovery from V3R6/HF2 is unchanged.
The isolated false-valid certificate from HF2 is unchanged.

The old HF2 universal rule:

```text
large disagreement must persist into an adjacent primary frame
```

was too blunt.  It protected real fast expressive motion, but also threw away acoustically correct single-frame repairs.

V3R7 retains persistent drift as one authority, but persistence is now measured in **physical disagreement duration**, not merely "one outlier exists in each adjacent frame".  The default minimum is 18 ms of contiguous disagreement in each participating primary frame.

For non-persistent disagreements, V3R7 adjudicates the immutable pre-V3R7 100/135/333-Hz trajectory around the disputed sample:

- future support for the existing ECKF path protects a real excursion;
- bilateral support for the primary family repairs an isolated tracker spike;
- current/old pitch on the left plus strong candidate/new pitch on the right can repair tracker lag into a newly established family;
- candidate support only on the left is insufficient, protecting phrase-end falls/releases.

A 20-ms guard also protects the leading edge of a primary frame whose primary candidate itself just jumped by >=3.5 semitones.  A frame-wide periodicity estimate is not sample-accurate evidence that the new family owned the first few trajectory samples of that frame.

All decisions use the immutable input snapshot.  One V3R7 correction cannot become evidence for another.

## 3. New audit fields

`*.primary_reconciliation_v3r7.csv` adds:

- `trajectory_candidate_support_count`
- `trajectory_current_support_count`
- `trajectory_inconclusive_count`
- `trajectory_jump_guard_count`

Replacement provenance distinguishes:

- `PRIMARY_TRACKER_DRIFT_REPLACED_V3R7_PERSISTENT`
- `PRIMARY_TRACKER_DRIFT_REPLACED_V3R7_TRAJECTORY`
- `PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R7`
- `PRIMARY_EVIDENCE_RECOVERED_V3R7`

## 4. 10-ms constant audit

No blind `0.01 -> 1/analysis_hz` replacement was performed.

Examples that remain physical time:

- local waveform/RMS hops expressed as 10 ms;
- reset/attack pre-roll or post-roll expressed in seconds;
- dimensionless thresholds whose value happens to be `0.01`.

The active V13 renderer is intentionally unchanged in this release.  It already derives normal pitch-grid `dt` from the input timestamps.  Its existing `min(0.01, pitch.dt)` expression is therefore explicitly grid-aware, while other `0.01` values are physical-time tolerances.  Renderer timing semantics will be changed only in a renderer-focused release.

The legacy warning text still says "corrected-valid 100-Hz points" because `render_midi.py` is byte-identical V13.  At 135 Hz interpret the count as corrected-valid **interpretation points**; duration is `count * pitch.dt` approximately.

## 5. Regression discipline

V13 is byte-for-byte frozen.  V3R7 must be judged against the WAV first.
A MIDI improvement alone is not acceptance evidence.

The intended tracker regression panel includes at least:

Keep/fix:
- ElTelefono ~38.93
- ElTelefono ~58.45
- ElTelefono ~150.x
- Damelo ~78.7-79.0
- Bodies ~116.83
- PREDESTINATI ~40.59
- Comomelo ~50.24

Protect expressive motion:
- Comomelo ~47.60
- Afurisita ~24.10
- MaInnec ~23.73
- Bodies ~19.69

Renderer-only debt remains renderer-only, including the bookmarked MaInnec ~25.813 competing-owner failure.
