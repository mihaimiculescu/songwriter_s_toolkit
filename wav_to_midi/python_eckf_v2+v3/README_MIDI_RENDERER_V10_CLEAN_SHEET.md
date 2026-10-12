# MIDI Renderer V10 — Clean-Sheet Event Architecture

V10 is a replacement renderer, not another patch on the V2–V9 event-building stack.
The V3R5 tracker is frozen and remains the normal pitch authority.

## Parallel evidence lanes

V10 assembles musical events once from four independent lanes:

1. **V3R5 trusted pitch / ownership**
   - `corrected_valid`
   - expressive `shape_pitch_st`
   - target stability / structural pitch
2. **V3R5 supporting CLI evidence**
   - ECKF reset landmarks
   - persistent target-center changes
3. **V3R5 primary acoustic-periodicity evidence**
   - raw frame ACF / CMNDF period measurements
   - used only to service an otherwise orphaned acoustic attack
   - bounded by the tracker's calibrated vocal range
4. **Source WAV attack timing**
   - high-resolution spectral/energy novelty (~5 ms hop)
   - never estimates or overrides pitch

The CLI is therefore not a bystander: it supplies pitch authority, stable-center
structure, reset landmarks, vocal range, and the primary rescue lane.  The WAV
adds the one thing the pitch tracker was never designed to represent precisely:
the physical start of consonant/noisy/half-spoken attacks.

## Important semantic changes from V9

- `min_stable_threshold_ms` is no longer a universal MIDI note-duration veto.
  A short corrected-valid island may become a note when independent attack/reset
  evidence supports it.
- A Note On may begin before trustworthy periodic F0.  The first trusted pitch is
  held backward to the independently measured acoustic attack time.
- Same-pitch rearticulation may be created by independent WAV attack evidence,
  but only with supporting amplitude restart, ECKF reset, or unusually strong
  novelty.
- Primary frame periodicity can rescue an attack only when no corrected-valid
  owner exists.  It does not rewrite V3R5 tracking output.
- The legacy `midi_event_builder.py` and `midi_articulation.py` are not called by
  V10.

## MIDI contract retained

- SMF Type 1
- PPQ 960
- Melody track 1, channel 1 by default
- GM Program #65 Soprano Saxophone
- fixed Note-On velocity 100 by default
- CC11 amplitude expression, compression 0.3
- per-note pitch-bend range and PB reset after Note Off

## Audits

Every render emits:

- `.mid.events.csv` — final musical events
- `.mid.attacks.csv` — accepted/rejected attack evidence and ownership reason
- `.mid.coverage.csv` — corrected-valid F0 coverage audit

The attack audit is intentionally verbose so future missed-onset reports can be
traced to either attack detection, ownership, or pitch rescue without modifying
the tracker first.
