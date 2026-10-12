# ECKF V3R0 — behavior-preserving ownership-evidence refactor

V3R0 is branched from the bidirectional variable-aperture V19 implementation.
It intentionally does **not** change musical decisions, thresholds, ordering,
range behavior, tracker behavior, validity/rescue behavior, jury/HD scoring,
register re-adjudication objectives, renderer articulation, or MIDI defaults.

## Purpose

The V2/V19 lineage accumulated several modules that separately asked similar
questions about pitch ownership.  V3R0 begins the cleanup by introducing one
shared access layer, `ownership_evidence.py`, for facts that must mean exactly
the same thing everywhere:

- range admission (`range_confidence > 0`);
- Harmonic-Detective acoustic admission (same V19 `acf_median >= 0.72` rule);
- MIDI/log-frequency conversion helpers;
- cents distance;
- the existing cubic interval-juror transition penalty;
- a read-only `OwnershipEvidenceSnapshot` provenance container for future
  explicit verdict propagation.

The existing public/private helper names used by older modules are retained as
compatibility wrappers where useful.  This is deliberate: V3R0 is a refactor,
not an adjudication redesign.

## Modules routed through the shared evidence layer

- `harmonic_detective.py`
- `reacquisition_adjudication.py`
- `register_readjudication_v18.py`

The tracker, V19 bidirectional variable aperture, validity/rescue, jury bench,
trajectory/gesture layers, structural collaboration, and MIDI renderer are
unchanged in behavior.

## Equivalence check

A 12-second Champagne WAV regression run was processed once with the source
bidirectional V19 tree and once with V3R0 using identical CLI arguments.
All 28 common generated CSV/sidecar outputs matched byte-for-byte.

This does not replace the full nine-patient regression test, but confirms that
the initial refactor plumbing itself does not alter the tested musical result.

## Next line

Future V3R releases may use the centralized evidence/provenance layer to:

1. persist interval/range/acoustic verdicts instead of silently recomputing or
   forgetting them;
2. make any downstream owner change an explicit re-adjudication with old and
   new evidence;
3. streamline/reorder ownership decisions only after V3R0 is validated against
   the full robustness set.
