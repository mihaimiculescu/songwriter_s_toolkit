# V17 — Contradictory Register Re-adjudication

V17 is branched directly from the V14 fix1 production baseline.

It deliberately does **not** reuse V15 register propagation or V16
discontinuity-wall behavior.

## Rule

Inside an already-existing V14 trajectory region:

1. Adjacent corrected-valid samples are checked with the existing cubic
   interval-juror penalty.
2. A non-zero penalty is only a **trigger** that marks a connected
   contradictory ownership burst.
3. That burst is re-adjudicated from the already-existing, in-range,
   acoustically-admissible candidate field.
4. The winning path is selected lexicographically:
   - minimize existing interval-juror penalty, including immediate same-region
     left/right anchors when available;
   - maximize retained frames;
   - minimize disagreement with existing resolved jury/HD winners;
   - maximize existing Harmonic-Detective acoustic score.
5. The selected owner is applied only to the affected ECKF frames inside the
   same V14 trajectory region.
6. If no candidate can participate in a coherent path, the frame may abstain
   rather than force an interval-impossible register.

No octave bucket, bend cap, song-specific pitch/timing, GT MIDI, new musical
threshold, blind register propagation, or discontinuity wall is introduced.

## Audit

A new sidecar is written:

`<csv>.register_readjudication_v17.csv`

## Renderer

Renderer behavior is unchanged from the supplied V14 fix1 package:
- `DEFAULT_VELOCITY = 100`
- `CC11_DB_COMPRESSION = 0.3`
- existing CC11 mapping
- existing Pitch Bend range/reset behavior
