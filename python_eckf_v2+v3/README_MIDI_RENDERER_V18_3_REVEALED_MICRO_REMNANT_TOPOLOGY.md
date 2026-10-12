# MIDI Renderer V18.3 — revealed micro-remnant topology cooperation

V18.3 is a strict child of V18.2 on the frozen V3R9 tracker/CLI baseline.
It adds **one** renderer-side cooperation rule and changes no V18.1/V18.2
acoustic thresholds, PB calibration, repetition recovery, subharmonic logic,
or corrected-valid pitch authority.

## Why V18.3 exists

V18.2 correctly identified unsupported weak starts and retracted four long
borrowed leads.  It deliberately refused to retract two additional Damelo
starts because doing so would expose about 20 ms of real corrected-valid pitch,
which would become new standalone micro-notes.

That was acoustically conservative but left 114–154 ms of known unsupported
silence/noise attached to those events.

V18.3 makes the two already-trusted subsystems cooperate:

1. run the exact V18.2 unsupported-start certificate;
2. retract to the first frozen corrected-valid target-family point;
3. if >=55 ms remains, use ordinary V18.2 timing repair unchanged;
4. if <55 ms remains, inspect the immediately following owner using the
   existing ownership-continuity lane;
5. only when the remnant and following owner are already classified as one
   continuous acoustic object, preserve the remnant pitch as a PB lead-in of
   that following owner and remove the redundant micro Note-On;
6. if continuity is not proven, retain V18.2 behavior unchanged.

There is no new pitch inference and no new acoustic threshold.

The 55 ms micro-remnant scale is exactly V18.2's existing
`UNSUPPORTED_BACKDATE_MIN_REMAINING_S`.
The 30 ms neighbor aperture is exactly V18's existing
`TRANSIENT_HEAD_MAX_GAP_TO_NEXT_S`.

## Canonical Damelo cases

### ~31.889 s

V18.2:

- Note-On ~31.889
- trustworthy corrected-valid pitch begins ~32.002993
- only ~20.748 ms of real MIDI-57-family pitch remains
- following owner begins at ~32.023741 and is acoustically `continuous`

V18.3:

- unsupported ~114 ms borrowed lead is removed;
- the ~20.7 ms real low pitch is preserved as PB lead-in;
- following owner now begins ~32.002993;
- no standalone ~20 ms Note-On is created.

### ~92.726 s

V18.2:

- Note-On ~92.725782
- trustworthy corrected-valid pitch begins ~92.879864
- only ~20.522 ms of real MIDI-55-family pitch remains
- immediately following MIDI-55 owner is acoustically `continuous`

V18.3:

- unsupported ~154 ms borrowed lead is removed;
- the real ~20.5 ms pitch is folded into the following MIDI-55 owner;
- no redundant micro Note-On remains.

## Twelve-patient regression

Actual MRV18.2 return versus V18.3 preflight:

| Metric | MRV18.2 | MRV18.3 |
|---|---:|---:|
| Total notes | 3038 | 3036 |
| events <40 ms | 177 | 176 |
| events <60 ms | 362 | 362 |
| median event duration | ~184.2 ms | ~184.3 ms |
| unowned corrected-valid V3R9 points | 0 | 0 |

Only Damelo changes note topology, by exactly two Note-Ons.  Every other song's
event table is identical to MRV18.2.

The four ordinary V18.2 retractions remain unchanged:

- Bodies ~89.393 -> ~89.543
- Damelo ~54.612 -> ~54.760
- Champagne ~37.297 -> ~37.431
- ElTelefono ~160.138 -> ~160.267

V18.3 additionally folds exactly two revealed micro-remnants, both in Damelo.

## MIDI wire regression

All twelve preflight MIDIs pass:

- Standard MIDI File Type 1
- PPQ 960
- GM #65 Soprano Saxophone (`program=64` on the wire)
- monophonic / no overlaps
- pitch wheel reset to zero after every Note-Off
- zero unowned corrected-valid V3R9 points

## Files changed from V18.2

Existing file modified:

- `render_midi.py`

New files:

- `midi_event_assembler_v18_3.py`
- `README_MIDI_RENDERER_V18_3_REVEALED_MICRO_REMNANT_TOPOLOGY.md`

Frozen tracker/CLI files and all inherited renderer modules remain unchanged.

## Exact twelve-song render command

No V3R9 tracker rerun is required.

```bash
mrv="18.3"

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

## Exact regression packaging command

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```
