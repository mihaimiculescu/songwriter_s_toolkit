# MIDI Renderer V17 — corrected-valid residual ownership closure

## Scope

V17 is the first substantive renderer release after the V16.x repair line.
Frozen tracker/CLI baseline: **V3R9 at the measured ~135 Hz interpretation grid**.

V17 does exactly one job:

> after V16.1 has completed articulation, continuous-motion, owner-corridor,
> pitch-only handoff, and bounded V16-created gap repair, find any remaining
> corrected-valid V3R9 interpretation run that has no MIDI owner and give that
> stable run MIDI ownership.

V17 does **not** perform WAV-only pitch rescue, subharmonic adjudication,
transient-note cleanup, or generic short-note suppression. Those remain later
renderer work.

## Safety contract

V17 is intentionally layered on top of `midi_event_assembler_v16_1.py` rather
than copying or rewriting V16.1.

A residual run is promotable only when:

1. every point is already `corrected_valid` in frozen V3R9;
2. those points remain unowned after the complete V16.1 event assembly;
3. the contiguous residual duration is at least the existing upstream
   `min_stable_s` scale (80 ms on the twelve-patient corpus).

There is **no new MIDI-specific minimum note duration**. V17 simply reuses the
existing target-former stability scale. This also prevents the residual pass
from manufacturing one/two-frame micro-notes.

If V16.1 already accepted a physical articulation whose frozen pitch anchor
belongs to the residual run, V17 may reuse that existing attack and hold the
first trustworthy corrected-valid pitch backward to it. V17 performs no new
attack search.

## Twelve-patient preflight

V16.1 -> V17:

- notes: **3101 -> 3106**
- new events: **exactly 5**
- inherited V16.1 events altered: **0 / 3101**
- unowned corrected-valid points: **111 -> 0**

Only four songs change:

- Pedestal: +1
- MaInnecArseneEN: +1
- ElTelefono: +1
- RATATA: +2

The five recovered residuals are:

1. Pedestal `29.956463–30.208571` — 35 V3R9 points
2. MaInnecArseneEN `27.220340–27.383469` — 23 points
3. ElTelefono `138.437415–138.615374` — 25 points
4. RATATA `48.512417–48.594000` — 12 points
5. RATATA `76.643833–76.755083` — 16 points

Independent WAV/YIN medians agree with the recovered MIDI families:

- Pedestal ~64.92 -> MIDI 65
- MaInnec ~57.90 -> MIDI 58
- ElTelefono ~61.00 -> MIDI 61
- RATATA ~48.51 ~62.96 -> MIDI 63
- RATATA ~76.64 ~70.00 -> MIDI 70

MaInnec reuses the already-accepted physical attack at ~27.170 s while pitch
authority begins with the frozen corrected-valid run at ~27.220 s.

MIDI wire invariants pass for all twelve outputs: SMF Type 1, PPQ 960,
monophonic melody, GM #65 Soprano Saxophone, and PB=0 reset after each Note-Off.

## Explicitly deferred to V18

V18 should address acoustic/transient adjudication, including:

- ultra-short notes that appear on preceding-note tails/transients (for example
  PREDESTINATI ~00:00:53:22 F4 ~0.056 beats and ~00:00:35:25 F#4 ~0.043 beats);
- tracker-abstention/WAV-rescue cases such as Afurisita ~68.12;
- demonstrably wrong-family/subharmonic cases such as RATATA ~86.19;
- other transient-as-independent-note cases already bookmarked.

## Commands

V3R9 is frozen. Do **not** rerun the tracker merely to test V17 if the existing
`*_v3R9.csv` sidecars are present.

```bash
mrv=17

declare -A bpm=(
    [Afurisita]=125
    [ElTelefono]=88
    [Damelo]=103
    [Comomelo]=100
    [Champagne]=98
    [Bodies]=126
    [MaInnecArseneEN]=128
    [Pedestal]=114
    [RATATA]=117
    [PREDESTINATI]=108
    [Ochiitai]=130
    [Trandafiri]=120
)

songs=(
    Pedestal MaInnecArseneEN Bodies Damelo
    Comomelo Champagne ElTelefono Afurisita
    RATATA PREDESTINATI Ochiitai Trandafiri
)

for song in "${songs[@]}"; do
    python -m python_eckf.render_midi \
        "tests/${song}_v3R9.csv" \
        --wav "tests/${song}.wav" \
        --out "tests/${song}_v3R9_mrv${mrv}.mid" \
        --bpm "${bpm[$song]}" \
        --time-signature 4/4
done
```

Package the full regression return:

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```
