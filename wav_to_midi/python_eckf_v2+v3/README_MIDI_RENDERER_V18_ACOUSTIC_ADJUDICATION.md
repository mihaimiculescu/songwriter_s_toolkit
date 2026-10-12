# MIDI Renderer V18 — Acoustic Adjudication and Transient Cleanup

V18 is a renderer-only release on top of the accepted MRV17.1 baseline.

**Frozen tracker contract:** V3R9 remains untouched. V18 does not modify CLI/tracker state, corrected-valid F0, stable targets, resets, reacquisition, or any prior adjudication layer.

V17.1 remains the normal pitch/ownership authority and its guarantee is preserved: **every corrected-valid V3R9 interpretation point remains owned by MIDI**.

V18 is the first deliberately acoustic/adjudicative renderer layer. It handles only cases where ordinary V17.1 ownership is insufficient or musically misleading.

## Independent V18 certificates

### 1. Breath/noise backdating trim

A very early WAV attack may not borrow a much later trusted pitch owner through a long breath/noise corridor.

V18 trims such starts only when:

- the trusted expression/pitch anchor is at least 70 ms later;
- a later already-accepted trusted/reset attack points to the same pitch anchor;
- no moderate primary periodicity proves that the earlier corridor was already voiced;
- the later attack or nearby ACF/CMNDF-agreeing primary frame is materially stronger than the original start.

Canonical fixes:

- Trandafiri first note: ~2.4809 s -> ~2.7503 s.
- Trandafiri breath around ~13.193 s: event now begins near ~13.375 s at the voiced arrival.

This is intentionally different from generic attack quantization. A quiet **voiced glide** is protected by the early-primary guard.

### 2. Ultra-short tail / detour cleanup

A WAV novelty peak does not automatically deserve a Note-On when it creates an extremely short internal detour inside one already-owned acoustic object.

V18 suppresses only events <= 35 ms when all of the following hold:

- event source is `wav_rearticulation`;
- no bilateral envelope restart was proven;
- it belongs to the same trusted owner as the preceding event;
- it is tightly bracketed by adjacent events;
- morphology is either a tiny pitch detour between nearly the same surrounding family, or a tiny tail fragment immediately before a new trusted owner.

Canonical fixes include:

- PREDESTINATI ~35.835 s: ~24 ms F#4 tail blip removed.
- PREDESTINATI ~53.75 s: ~31 ms F4 tail detour removed and absorbed into the surrounding owner.

This pass is deliberately not a blanket minimum-note-duration filter.

### 3. Short transient head -> pitch-bend lead-in

A weak/noisy early start with only <= 50 ms of trustworthy voiced pitch before the next connected note is treated as an onset transient, not an independent MIDI note.

When the following acoustic object is continuous, the transient pitch is preserved as the PB lead-in of the following note. If an independent object break is proven, the tiny transient is dropped rather than smeared into the next event.

Canonical fixes:

- PREDESTINATI ~47.74 s: short high transient no longer becomes an independent Note-On; it becomes the high PB lead-in of the following descent.
- ElTelefono ~165.57 s: false standalone high transient/breath note removed; sounding line begins at the trustworthy voiced arrival and continues into the following event.

### 4. Tracker-abstention acoustic rescue

V18 may create pitch where frozen V3R9 abstains **only** under a narrow certificate:

- strong high-resolution WAV attack;
- ACF >= 0.68;
- CMNDF <= 0.30;
- ACF/CMNDF frequencies agree within 100 cents;
- source is audibly above the recording-adaptive floor;
- no corrected-valid V3R9 pitch already occupies the gap.

Canonical fix:

- Afurisita ~68.12 s: a ~57 ms MIDI-59 event is recovered from the strong attack and agreeing ~249–251 Hz periodicity evidence.

### 5. Short accepted-owner acoustic backfill

A later trusted owner may begin slightly earlier inside a short tracker-abstention gap only when an already-accepted structural/trusted attack and moderate primary evidence agree with that same family.

This is bounded to <= 60 ms and does not alter corrected-valid ownership.

### 6. Extreme subharmonic octave-up adjudication

V18 may override a frozen low tracker family by +12 semitones only when the evidence is exceptional:

- event is at least 80 ms;
- octave-up spectral energy exceeds the low candidate by at least 30 dB median;
- >= 60% of sampled frames independently show strong octave dominance;
- the octave-up family improves local continuity with both neighboring events by at least 8 semitones total;
- corrected octave remains within 5 semitones of each neighbor.

Canonical fix:

- RATATA ~86.19 s: false MIDI-55 / ~196 Hz event corrected to MIDI 67 / ~392 Hz family. The WAV octave-up dominance is ~36 dB in the preflight.

## Twelve-patient preflight

Frozen V3R9 + MRV17.1 vs V18:

- Note events: **3104 -> 3033** (-71, -2.29%).
- Events < 40 ms: **238 -> 180**.
- Events < 60 ms: **432 -> 367**.
- Median event duration: ~184.08 ms -> ~185.37 ms.
- Unowned corrected-valid V3R9 points: **0 -> 0**.
- Maximum pitch-bend range in the corpus remains <= 12 semitones.

V18 action totals in the reference corpus:

- breath/noise starts trimmed: 14
- ultra-short tail/detour Note-Ons removed: 49
- transient-head Note-Ons collapsed: 23
- tracker-abstention acoustic rescues added: 1
- short owner backfills: 5
- extreme subharmonic octave corrections: 1

All 12 output files pass:

- MIDI 1.0 SMF Type 1
- PPQ = 960
- melody channel = 0 (DAW channel 1)
- GM Program payload 64 = GM #65 Soprano Saxophone
- monophonic note topology / no overlaps
- pitch-bend reset to zero at every Note-Off

## Rate contract

V18 preserves the MRV15+ timing contract. Interpretation-grid quantities derive from the measured `pitch.dt`; physical acoustic apertures remain expressed in seconds. There is no 100-Hz fallback or hidden 100-Hz frame assumption.

## Exact 12-song render command

V3R9 is frozen. Do **not** rerun the tracker merely for V18.

```bash
mrv=18

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

## Exact regression archive command

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```

The `mid*` glob intentionally includes the `.events.csv`, `.attacks.csv`, `.coverage.csv`, and `.rate_contract.csv` renderer audits.
