# V3R4 — Subharmonic-borrowing certificate

V3R4 branches directly from V3R3 and changes only
`variable_aperture_reacquisition.py`.

## Problem

V3R3 correctly invokes the 24/40/64/96-ms integer-family contest at the two
remaining Champagne low-register reacquisitions, but the wrong low octave can
survive because its longer period has slightly cleaner ACF/CMNDF even while the
actual upper-family fundamental is overwhelmingly stronger.

A weak first harmonic is not, by itself, evidence that a low vocal F0 is false.
The useful distinction is whether the low family has spectral evidence that is
*exclusive to its own longer-period harmonic grid*.

For an integer-related pair

    high ~= N * low

harmonics `N, 2N, 3N, ...` on the low grid are shared with the higher family.
The remaining low-grid harmonics are exclusive to the lower period.

## V3R4 certificate

A lower family is treated as a borrowed subharmonic at the wide apertures only
when the SAME low/high pair satisfies ALL of the following at BOTH 64 ms and
96 ms:

- measured integer relation x2..x5 within the existing 50-cent tolerance;
- higher-family ACF >= 0.80;
- higher-family CMNDF <= 0.20;
- measured higher fundamental amplitude >= 8x the lower fundamental;
- RMS(low-exclusive harmonic grid) / RMS(shared harmonic grid) <= 0.12.

The harmonic-grid RMS measurement uses the same least-squares sinusoid
amplitude estimator already used by initialization candidate assessment and the
same 12-harmonic ceiling used by the existing harmonic-support code.

The certificate fails closed if either wide aperture is unavailable or does not
contain enough harmonic-grid evidence.

## How it enters the existing vote

V3R4 does NOT loosen `_dominates()` globally and does NOT directly force an
upper pitch.  When the certificate is present, only the certified low family is
treated as absent at 64 and 96 ms.  The existing V3R3 profile then decides as
before using:

- short->long establishment;
- long->short persistence;
- total support breadth;
- strict 2-of-3 majority.

Short 24/40-ms evidence is untouched.

## Targeted checks

Using the full V3R3 nine-patient outputs as the frame list, the conservative
certificate triggered only at:

- Champagne ~56.37805 s: ~169.6 Hz -> ~341.9 Hz;
- Champagne ~71.05306 s: ~172.3 Hz -> ~344.5 Hz.

No certificate fired in the checked 96-ms contests from Pedestal,
MaInnecArseneEN, Damelo, Comomelo, Afurisita, RATATA, ElTelefono, or Bodies.
The available older Trandafiri V3R3 frame export also produced no trigger.

This conservative version intentionally leaves RATATA ~84.8 s for separate
inspection rather than lowering the independent-upper-periodicity threshold to
capture it by tuning.

## Unchanged

Everything outside the variable-aperture family contest is unchanged from
V3R3, including:

- initialization-candidate global dominance semantics;
- V3R2 provisional-context interval-authority semantics;
- silence/interruption and harmonic-change boundaries;
- hard range gate;
- ECKF tracker equations;
- jury / Harmonic Detective logic;
- target/gesture interpretation;
- MIDI renderer and project defaults.
