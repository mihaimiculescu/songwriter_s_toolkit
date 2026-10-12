# MIDI Renderer V6 — consolidated articulation

V6 keeps the V5 ownership-complete renderer and changes only musical articulation interpretation.

## Goals
- Collapse redundant Note Ons caused by multiple cues describing one attack.
- Preserve hard boundaries caused by corrected-valid interruptions.
- Preserve legato note changes when a genuinely new stable target center is established.
- Recover same-pitch rearticulation only when two independent existing lanes agree.
- Never split from pitch motion alone, so vibrato and glissando remain bends unless a new target lands.

## Rules
1. **Nonmusical interruption**: hard Note-Off / Note-On boundary.
2. **Persistent stable-target change**: new legato Note-On only when the target-former's own
   `stable_range_threshold_st` is exceeded and the new center persists for its existing
   `min_stable_threshold_ms`.
3. **Same-pitch rearticulation**: ECKF reset + amplitude valley/attack must agree within the
   existing project stable-target timescale. Neither signal can split by itself.
4. **Consolidation**: multiple proposals inside the existing stable-target timescale collapse
   to one musical articulation, strongest semantic cue winning.
5. The ECKF harmonic-change/reset threshold remains 2.0 semitones.
6. V5 ownerless-valid-F0 recovery and its existing minimum-stable-duration gate are unchanged.

No Trandafiri/GT-specific pitches, times, note counts, or thresholds are encoded.
