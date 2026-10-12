# MIDI Renderer V11 — Attack-Ownership Interruption Barrier

V11 is built on the frozen V3R5 pitch tracker and the V10 clean-sheet renderer.
The V3R5 tracker and `BORROWING_MIN_HIGH_ACF = 0.77` are unchanged.

## Purpose

V10 correctly separated acoustic attack timing from pitch authority, but a wide
attack→pitch association aperture could let a short breath/noise object steal the
Note-On of a later real vocal attack. V11 adds a *forward acoustic ownership*
test: an earlier attack may own later pitch only while the waveform remains part
of the same acoustic object.

## What "significant energy collapse" means

It is **not** one fixed dB threshold. A hard interruption requires all of the
following:

1. the envelope reaches the recording's adaptive near-inactive region
   (`active_rms_floor_db + 3 dB`),
2. the trough is materially below the local acoustic material on both sides, or
   the candidate is a weak detached precursor followed by a much stronger
   object,
3. independent spectral novelty confirms a restart after the trough, and
4. the low-energy portion is long/deep enough to exceed the provisional
   short-closure allowance.

This is intended to distinguish:

- breath/noise → interruption → new note: **two acoustic objects**
- consonant/aspiration → vowel with no true interruption: **one musical onset**

## Provisional short-closure rule

Low-energy closures up to 25 ms are provisionally tolerated rather than treated
as hard ownership barriers. The 25–40 ms transition zone requires a very deep
two-sided collapse to become a hard barrier. This rule is intentionally exposed
in `*.attacks.csv` and should be reviewed against the next 12-patient run.

A weak detached precursor is not treated as a stop closure merely because its
near-floor trough is short: if that object dies and a much stronger acoustic
restart follows, it may still be rejected as a separate breath/noise object.

## Audit additions

`*.mid.attacks.csv` now includes:

- `continuity_status`
- `barrier_time_s`
- `barrier_valley_dbfs`
- `barrier_depth_left_db`
- `barrier_depth_right_db`
- `barrier_low_duration_s`
- `short_closure_tolerated`

Rejected attack candidates use reasons such as:

- `ownership_interruption_barrier`
- `primary_rescue_interruption_barrier`
- `structural_alignment_interruption_barrier`

## Development-set sanity checks before release

Using the submitted V3R5 twelve-patient sidecars and WAVs, V11:

- moves Afurisita ~77.718 from the V10 breath precursor (~77.635) to the real
  restart (~77.715),
- removes Champagne's spurious pre-41.97 primary-rescue note,
- moves the ElTelefono breath example around 17.28 toward the later voiced
  restart,
- preserves the already-good Bodies 20.878 / 24.725 / 91.365 timings,
- preserves Afurisita 78.644,
- preserves the already-good Champagne 58.172, ElTelefono 48.629 / 58.445,
  PREDESTINATI 52.412, and RATATA 27.913 / 48.708,
- improves RATATA ~25.276 in the local regression run.

The short-closure allowance fired in the development run and remains explicitly
provisional; inspect it in the next uploaded full archive before promotion.
