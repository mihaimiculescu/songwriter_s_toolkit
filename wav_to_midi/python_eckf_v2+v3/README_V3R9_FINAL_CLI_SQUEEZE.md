# V3R9 — Final CLI Squeeze (135 Hz)

V3R9 is the final planned tracker/CLI revision before freezing the pitch layer and returning to renderer work.

It keeps the V3R7/V3R8 135-Hz rate-aware analysis architecture and leaves MIDI Renderer V13 byte-for-byte unchanged.

## 1. Persistent-drift hysteresis

V3R8 enters a generic tracker-drift run only when the disagreement with qualified primary evidence is at least 3.5 semitones and persists for at least 18 ms in continuous physical time.

V3R9 keeps that strict entry rule. Once such a run is already established, it may extend through directly adjacent shoulders down to 2.5 semitones while the qualified primary family stays continuous.

A 2.5-semitone shoulder cannot create a drift run by itself. This fixes threshold clipping at the edges of proven drift runs without weakening protection of short expressive excursions such as Comomelo ~47.60.

New audit reason:

- `PRIMARY_TRACKER_DRIFT_REPLACED_V3R9_HYSTERESIS`

The reconciliation sidecar adds `hysteresis_shoulder_count`.

## 2. Embedded low-register outlier certificate repaired

V3R8 attempted to repair a lone valid low-register octave-family glitch using trusted corrected-valid pitch on both sides. That could fail because some neighboring samples are only corrected later by register readjudication.

V3R9 therefore permits a second, independent certificate using overwhelming ACF/CMNDF-agreeing primary families in nearby frames on both sides.

Safety constraints remain deliberately narrow:

- own-frame ACF >= 0.95
- own-frame CMNDF <= 0.05
- ACF and CMNDF agree on the primary family
- accepted tracker sample is integer-related to the primary family
- disagreement is at least 3.5 semitones
- the primary candidate is HIGHER than the accepted tracker sample
- trusted corrected-valid context on both sides OR strong agreeing primary-family context on both sides supports that higher family

The upward-only restriction is intentional: corpus inspection showed that strong lower raw-primary families are often subharmonics and must not be allowed to pull a valid high tracker trajectory downward through this certificate.

New/retained audit reason:

- `PRIMARY_EMBEDDED_REGISTER_OUTLIER_REPLACED_V3R9`

## Frozen components

- Default offline interpretation rate remains 135 Hz (actual per-file rate derived from integer audio hop).
- Rejected-pitch recovery is unchanged.
- Isolated false-valid repair is unchanged.
- V3R7 trajectory candidate authority and primary-family jump guard are unchanged.
- MIDI Renderer V13 musical logic is unchanged.

