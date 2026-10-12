# ECKF V2 — Structural Collaboration V1

Built on `python_eckf_v2_hd_dispute_filter_v2`.

No new acoustic juror is added. No existing jury or Harmonic Detective winner is changed.
The collaboration pass runs only after the complete existing adjudication path still abstains.

Residual rules:

1. **STABLE_TARGET authority** — if the unresolved frame lies inside an already-frozen
   `STABLE_TARGET`, the target interpreter's frozen median pitch becomes the structural
   frame result. Raw instantaneous F0 representatives are no longer forced to fight over
   an identity already established by the target interpreter.

2. **Gesture absorption** — if the unresolved frame lies inside a frozen `TRANSITION`
   covered by an already-existing gesture object, the frame is marked `resolved_gesture`.
   It has no standalone winning F0 because the continuous pitch trajectory/gesture lane
   already owns that instant for later expressive rendering.

3. Everything else remains `unresolved` exactly as before.

Audit output:

- `*.structural_collaboration.csv`
- `*.adjudication_final.csv` now reflects the residual structural pass.

Existing interval-juror cubic behavior, W_INTERVAL=0.30, 2-cent HD dispute filter,
same-MIDI HD guard removal, and the -40 dB silence threshold are unchanged.
