# ECKF V2 — Juror Evidence Bridge V1

This build is the read-only bridge between the live pitch-candidate field and the historical adjudication bench.

It does not change ECKF tracking, validity, variable-aperture target formation, gesture structure, interpretation candidates, or accepted F0.

New export:

`<output>.juror_evidence.csv`

For each frame and nearest-note group it exports one representative plus the evidence needed by the later jurors:

- range confidence using the historical observed-range shape (±2 semitone plateau extension + 2 semitone half-cosine shoulder);
- interval plausibility components through the existing `vocal_transition_penalty` curve;
- spectral evidence: median ACF, CMNDF, harmonic count, and component amplitude;
- temporal persistence evidence: whether the same note group is present on the immediately adjacent candidate frames;
- initializer-selection provenance.

The stage intentionally does not combine jurors into a weighted verdict yet. This avoids silently inventing a new spectral/temporal scoring rule before the historical bench is reconnected.

The CLI label formerly printed as `threshold: 2.0 semitones` is renamed to `harmonic-change threshold: 2.0 semitones` for clarity. It is the harmonic-discontinuity detector threshold, not a note-classification tolerance and not the variable-aperture threshold.

`--matlab` remains untouched. The fixed -50 silence threshold remains untouched. Harmonic Detective is not enabled in this build.
