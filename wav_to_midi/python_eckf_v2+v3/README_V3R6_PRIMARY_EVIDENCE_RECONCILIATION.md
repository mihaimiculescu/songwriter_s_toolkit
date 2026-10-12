# V3R6 — Primary-Evidence Reconciliation

V3R6 branches the tracker from **V3R5**. The MIDI renderer is **V13 unchanged**.

## Purpose

V10–V13 exposed cases where the tracker already possessed strong primary periodicity evidence but the final corrected-valid F0 either disappeared or remained contaminated by stale ECKF state. Those are tracker responsibilities and are repaired here before musical interpretation.

V3R6 does **not** consume MIDI, manual attack timestamps, wishlist ground truth, or renderer decisions.

## Pipeline position

The new stage runs:

1. after V3R5 range/jury/Harmonic-Detective/reacquisition feedback;
2. before trajectory interpretation / targets / gestures / expression.

Therefore the file-level vocal range is already independently established and cannot be defined by the new rescue itself.

## New module

`primary_reconciliation_v3r6.py`

It may perform only three narrow tracker-side actions:

### 1. Recover rejected tracker state from sustained primary periodicity

A rejected sample can become valid when a strong ACF/CMNDF primary family is:

- inside the independently calibrated vocal range;
- temporally coherent with another primary frame or trusted trajectory context;
- not inside authoritative low-energy silence;
- associated with a tracker/range failure such as `BELOW_VOCAL_FLOOR`, `RANGE_ZERO_VETO`, `LOCK_LOST_DISCONTINUITY`, or a rejected reacquisition island.

This is intended for cases such as tracker-state contamination where ECKF stays below the vocal floor while the waveform periodicity has already recovered the correct family.

### 2. Resolve an exceptionally strong ACF/CMNDF integer-family dispute

When ACF and CMNDF disagree by an integer family but the independently calibrated range rejects one family, a frame may be recovered only with very strong periodicity and trusted temporal support.

This is the class exposed by ElTelefono around 38.96 s.

### 3. Replace non-register ECKF drift

An already-valid ECKF sample may be replaced only when:

- strong primary periodicity is sustained across neighboring frames;
- disagreement is at least 3.5 semitones;
- the disagreement is **not** an octave/integer-family dispute.

Existing V3R3/V3R4/V3R5 register machinery remains authoritative for octave-family questions.

## Hard guards

- `LOW_ENERGY_SILENCE` is never promoted.
- No song/time exceptions.
- No MIDI or attack-GT input.
- No universal promotion of short islands.
- No weakening of the V3R5 borrowing certificate (`BORROWING_MIN_HIGH_ACF = 0.77`).
- No change to the 2.0-semitone tracker reset threshold.

## New audit sidecar

Every offline run additionally writes:

`<output>.primary_reconciliation_v3r6.csv`

with the primary candidate, candidate source, temporal support, number of promoted/replaced samples, and action/reason for every frame.

## Deliberate abstentions

V3R6 does not force cases where the primary acoustic evidence itself is weak. In particular, a real musical attack is not enough to manufacture F0. Those cases remain renderer territory or tracker abstentions until stronger pitch evidence exists.

## Renderer

The packaged `render_midi.py` and V10–V13 renderer modules are byte-for-byte the frozen V13 renderer lineage. V3R6 is intended to be evaluated first with V13 unchanged so that upstream gains and regressions remain measurable.
