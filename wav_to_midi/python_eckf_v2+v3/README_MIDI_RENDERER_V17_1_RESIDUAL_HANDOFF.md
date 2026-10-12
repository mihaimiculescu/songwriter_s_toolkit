# MIDI Renderer V17.1 — residual corrected-valid ownership without artificial reattack

## Status

- Tracker/CLI baseline: **V3R9 frozen**.
- Renderer parent: **MRV16.1**.
- MRV17 principle retained: every sufficiently stable corrected-valid V3R9 F0 point must have MIDI ownership.
- MRV17.1 is a narrow repair of MRV17 topology. It does **not** add WAV-only pitch rescue, transient cleanup, subharmonic adjudication, or the V18 acoustic layer.

## Why V17.1 exists

MRV17 correctly closed the remaining 111 unowned corrected-valid interpretation points, but it always materialized each qualifying residual run as a new MIDI event. Two RATATA corridors showed why that was too strong:

- ~48.512 s: residual MIDI-63 material continues an immediately preceding MIDI-63 owner, with no independent physical reattack.
- ~76.644 s: residual MIDI-70 material continues an immediately preceding MIDI-70 owner, again with no independent physical reattack.

The correct result is ownership continuity, not Note-Off / Note-On.

## V17.1 rule

For each qualifying residual corrected-valid run:

1. Find the immediately preceding rendered event.
2. Permit ownership transfer only when:
   - the residual begins no more than **60 ms** after that event ends;
   - the preceding event and the residual quantize to the same MIDI family;
   - the preceding sounding pitch tail is within **0.75 semitone** of the residual median;
   - there is **no already-accepted physical reattack within ±20 ms** of the residual start.
3. If all conditions hold, extend the existing owner through the residual run and append the frozen V3R9 corrected-valid pitch samples to its PB trajectory.
4. Otherwise preserve MRV17 behavior and create the residual event exactly as before.

This is not a general event-merger. It operates only on the residual-valid ownership debt that MRV17 already identified.

## Twelve-patient regression

Compared with MRV17:

- total notes: **3106 -> 3104**
- corrected-valid unowned points: **0 -> 0**
- same-family residual handoffs: **2**
- separately promoted residual notes: **3**

Only RATATA loses Note-Ons:

- ~48.512 s: MIDI-63 residual transfers to the preceding MIDI-63 owner.
- ~76.644 s: MIDI-70 residual transfers to the preceding MIDI-70 owner.

The three musically distinct recovered notes remain separate:

- Pedestal ~29.956 s
- MaInnecArseneEN ~27.170 / residual pitch from ~27.220 s
- ElTelefono ~138.437 s

All twelve regression MIDIs pass:

- MIDI 1.0 SMF Type 1
- PPQ 960
- monophonic melody track / no overlaps
- GM #65 Soprano Saxophone (wire program 64)
- PB reset to 0 at every Note-Off

## Explicit non-goals / V18 bookmark

MRV17.1 does not address acoustic/transient adjudication. In particular V18 remains responsible for:

- Trandafiri first note borrowing ~300 ms of preceding breath/noise;
- ultra-short tail/transient notes, including PREDESTINATI examples around ~53:22 (F4 ~0.056 beat) and ~35:25 (F#4 ~0.043 beat);
- Afurisita ~68.12 acoustic rescue;
- RATATA ~86.19 false low/subharmonic family;
- transient-as-note cases such as PREDESTINATI ~47.74 and ElTelefono ~165.57;
- other WAV-side acoustic rescue / false-family cases.

## Commands

V3R9 is frozen, so do not rerun the tracker.

```bash
mrv="17.1"

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

Package the regression return:

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```
