# V18 — Cross-region register re-adjudication

V18 keeps the V17/V14 lineage. It does not reuse V15 register propagation or V16 discontinuity walls.

## New rule

The existing cubic interval juror may trigger re-adjudication not only inside one trajectory region, but also at a boundary between two immediately adjacent corrected-valid voiced ownership regions.

When the observed A→B boundary has non-zero existing interval penalty, V18 re-adjudicates the two adjacent regions together from the already-existing in-range acoustically-admissible candidate field. The immediate trusted material outside the pair is used as left/right anchor when available.

The interval juror is only the trigger. It never picks the octave/register directly. No propagation, discontinuity wall, bend cap, GT, song-specific pitch, or new musical threshold is introduced. NULL/abstention remains available when no coherent candidate path exists.

## Audit

`<csv>.register_readjudication_v18.csv`

`peer_region_id` is populated for cross-region reviews and blank for the original V17 same-region contradiction bursts.
