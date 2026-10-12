# MIDI Renderer V16.1 — V16 Mechanical Repair

## Status

**V3R9 remains frozen.** V16.1 changes the MIDI renderer only.

V16 established two renderer principles that remain in force:

1. continuous voiced motion does not automatically imply a new Note-On;
2. a later trusted pitch owner may not borrow a physically earlier pitch-incompatible corridor.

The twelve-WAV V16 regression showed that those principles were musically correct, but exposed two mechanical side effects. V16.1 fixes only those side effects. It does **not** begin the next missing-F0/acoustic-rescue project.

### Release map after V16

- **V16.1**: repair V16's pitch/ownership mechanics (this release).
- **V17**: dedicated renderer coverage of remaining corrected-valid V3R9 pitch that is still unowned.
- **V18**: special WAV-side acoustic rescue / subharmonic fail-safes for cases where frozen V3R9 itself abstains or chooses a demonstrably wrong family.

---

## 1. Pitch authority is decoupled from articulation

V16 can suppress a strong primary-periodicity rearticulation because the WAV proves that the singer remains inside one physical vocal object. That suppression is correct for articulation, but it must not also throw away a useful pitch handoff.

V16.1 therefore permits a **pitch-only handoff**:

- the proposed Note-On remains suppressed;
- a later frozen V3R9 trusted island must exist;
- strict primary periodicity must independently agree with that future trusted family;
- the acoustic corridor to the future trusted owner must remain continuous;
- the primary bridge must reach to within the inherited 120 ms bridge aperture of the future trusted island;
- the future owner remains the pitch-family authority; primary periodicity only corroborates the handoff timing/shape.

The current articulation continues, while Pitch Bend authority changes inside it.

Audit reason:

`pitch_authority_handoff_without_note_v16_1`

Canonical regression: RATATA ~40.131 s. V16 correctly removed an unnecessary Note-On, but its remaining old-owner PB trajectory sagged roughly one semitone low. V16.1 keeps the Note-On suppressed and restores the ~MIDI-63 pitch family through PB only.

---

## 2. V16 corridor-retraction holes get a bounded continuity handoff

When V16 rejects a future owner's backward attack claim (`trusted_pitch_conflict_v16`), the rejection may expose a short interval between the end of the preceding trusted owner and the true start of the future owner.

V16.1 may extend the **preceding articulation** across that interval only when all of the following are true:

- the gap was created by that exact V16 trusted-pitch conflict mechanism;
- gap duration is no more than **140 ms**;
- V13 object-corridor continuity reports no interruption barrier;
- the gap remains audibly above the recording-adaptive floor;
- ACF and CMNDF both survive with moderate transition evidence;
- ACF and CMNDF pitch families agree within **150 cents**;
- the transition candidate remains inside the pitch bracket formed by the two adjacent frozen trusted owners, with a **2.5 semitone** safety margin;
- transition evidence must begin materially before the next trusted owner. A primary frame beginning only at the next owner cannot retroactively prove the preceding gap.

The relaxed primary lane cannot create a Note-On and cannot originate a new pitch family. It only shapes a short transition already bracketed by frozen V3R9 owners.

Audit reason:

`corridor_transition_handoff_v16_1`

Canonical regression: MaInnec ~50.807–50.852 s. V16 correctly stopped the later low owner from stealing the earlier high corridor, but left a voiced transition hole. V16.1 lets the preceding articulation continue through the surviving transition evidence until the low trusted owner begins.

---

## 3. Explicit non-goals

V16.1 does **not** solve the following; they remain deliberately staged:

- Pedestal ~29.956–30.209 corrected-valid ownership gap;
- MaInnec ~27.220–27.383 corrected-valid ownership gap;
- ElTelefono ~138.437–138.615 corrected-valid ownership gap;
- RATATA ~48.512–48.594 and ~76.644–76.755 corrected-valid ownership gaps;
- Afurisita ~68.12 tracker-abstention acoustic rescue;
- RATATA ~86.19 false low/subharmonic family;
- PREDESTINATI ~47.74 transient promoted to a musical note;
- ElTelefono ~165.57 transient/breath standalone note.

Those belong to V17/V18, not to a V16 repair.

---

## 4. Twelve-patient preflight

Frozen V3R9 + V16 versus frozen V3R9 + V16.1:

- Note count: **3101 -> 3101**
- corrected-valid unowned points: **111 -> 111**
- events <40 ms: **240 -> 239**
- events <60 ms: **434 -> 432**
- median event duration: **~183.9 ms -> ~184.1 ms**
- V16 anti-fragmentation topology is retained.
- V16.1 produces no new Note-On from either repair mechanism.

V16.1 continuity repairs fired only in bounded V16-created corridors. A separate pitch-only handoff fired at the RATATA ~40.131 regression.

MIDI-wire regression passed on all twelve patients:

- SMF Type 1
- PPQ 960
- monophonic channel 1
- GM #65 Soprano Saxophone
- no overlaps
- PB=0 reset at every Note-Off

---

## 5. Commands — frozen V3R9 -> MRV16.1

No tracker rerun is required.

```bash
mrv="16.1"

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

Package the twelve-patient return run:

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```

The `mid*` glob includes events, attacks, coverage, and rate-contract audits.
