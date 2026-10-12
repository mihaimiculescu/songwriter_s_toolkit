# ECKF V2 — direct missing tournament edges V1

Restores the historical V22 policy that a missing tournament edge must be
measured from the original WAV rather than inferred from other pair results.

In the current V2 bench every representative pair has a summary-based pair row,
so an edge is considered **missing** when that pair has no decisive winner and
the first-pass tournament is hung/incomplete. Only those unresolved admitted
edges are remeasured.

Direct measurement uses the restored V21 field:

- 24 / 40 / 64 ms apertures
- -8 / 0 / +8 ms shifts
- overlap-normalized lag ACF directly from the original waveform
- ACF support threshold 0.72
- at least 2 supported shifted measurements for direct acoustic admission

The fresh direct ACF medians replace only the ACF part of the spectral juror for
that missing edge. Existing CMNDF/harmonic/component summaries, interval, range,
and the directional-only V21 temporal logic remain unchanged.

Safety rules:

- existing decisive edges are never remeasured or overturned;
- only range-admitted unresolved pairs are eligible;
- a direct measurement that also abstains does not create an edge;
- the Harmonic Detective remains downstream and sees only the post-completion
  hung-jury cases.

Audit output:

`<output>.direct_edge_measurements.csv`

records every attempted direct completion and whether it actually filled an edge.
