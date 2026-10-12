# MIDI Renderer V18.1 — calibration surgery

V18.1 is a renderer-only calibration branch on top of accepted MRV17.1 and candidate MRV18. Frozen V3R9 tracker/CLI state is not modified.

## Purpose

V18 introduced the correct acoustic/adjudicative mechanisms. V18.1 tunes the remaining counterexamples without globally loosening the renderer:

1. **weak/noisy attack backdating**
   - a long, very weak borrowed start may now move to the earliest near-anchor ACF/CMNDF-agreeing primary frame;
   - a later accepted bilateral WAV reattack of the same pitch family can also replace the weak start;
   - V17.1's zero-unowned-valid contract is a hard guard: trimming never moves past the first frozen corrected-valid point already owned by the event.

2. **weak short trusted-island heads**
   - <=55 ms, <=-45 dBFS at start, immediately followed by a materially stronger owner;
   - preserved as PB lead-in of the following owner rather than a standalone Note-On.

3. **terminal decay islands**
   - <=140 ms;
   - strong envelope collapse on the left, no meaningful restart on the right;
   - falling level through the event;
   - folded into the preceding event as PB tail rather than a fresh Note-On.

4. **asymmetric / ostinato rearticulation**
   - V16's 5 dB / 5 dB bilateral protection remains unchanged globally;
   - V18.1 adds a separate repetition certificate for same-pitch attacks with strong right restart;
   - either a highly regular >=3-attack sequence or an isolated bracketed repetition with sufficient left dip is required.

5. **decay-aware subharmonic adjudication**
   - keeps V18's strict normal octave certificate;
   - adds a separate short-decay certificate when 2*f0 overwhelmingly dominates at the head, the event rapidly decays, and both neighboring events strongly favor the octave-up family.

6. **fast Pitch-Bend residual calibration**
   - no global PB shrinkage;
   - a local least-squares pitch trend is estimated over +/-75 ms;
   - only alternating residual extrema runs are eligible;
   - adjacent opposite-sign extrema must be 30–130 ms apart (about 3.8–16.7 Hz full-cycle), with at least four extrema and residual excursions <=1.2 semitones;
   - only that residual is multiplied by **0.90**;
   - 20 ms edge taper keeps entry/exit continuous;
   - directed glissandi are the local trend and remain unchanged.

## PB what-if sweep

Automatic detector, 36 representative high-activity events, 2,623 comparable renderer/WAV-YIN samples:

| residual factor | median abs error vs WAV/YIN |
| --- | ---: |
| 1.00 | ~0.0877 st / 8.8 cents |
| 0.95 | ~0.0788 st / 7.9 cents |
| 0.90 | ~0.0723 st / 7.2 cents |
| 0.85 | ~0.0673 st / 6.7 cents |

Median raw renderer/WAV-YIN residual-width ratio was ~1.19. V18.1 deliberately chooses **0.90**, not 0.85, because YIN itself smooths modulation and the first calibration pass should preserve expression conservatively.

## Canonical regression outcomes

- Damelo ~37.57: false low MIDI 52 family -> octave-up MIDI 64 family via decay-aware subharmonic certificate.
- Damelo ~106.35: breath/noise borrowing removed; onset moves to ~106.583 (coverage guard / trusted-valid onset), near the real voiced arrival.
- Comomelo ~69.96 / 70.107 / 70.257: the three suppressed same-pitch ostinato reattacks are restored.
- Champagne ~21.428: suppressed same-pitch repeat restored without lowering the global V16 5/5 dB restart threshold.
- Champagne ~31.46: terminal decay no longer gets a new Note-On; it becomes PB tail of the preceding event.
- Champagne ~69.195 and ~69.799: weak short high transient islands no longer receive independent Note-Ons; they become PB lead-ins into the following owner.
- Champagne ~72.23: left unchanged. WAV evidence supports a real high-pitch excursion; V18.1 does not override the WAV merely because listening context feels unusual.
- MaInnec ~3.65: borrowed silence/noise trimmed to ~3.762.
- MaInnec ~9.69: borrowed silence/noise trimmed to ~9.845.
- RATATA ~9.46: early low-energy backdating trimmed to ~9.557; the high pitch family itself is retained because WAV evidence supports it.

## Twelve-patient preflight

| metric | MRV18 | MRV18.1 |
| --- | ---: | ---: |
| total notes | 3033 | 3038 |
| events <40 ms | 180 | 177 |
| events <60 ms | 367 | 362 |
| median event duration | ~185.4 ms | ~184.3 ms |
| unowned corrected-valid V3R9 points | 0 | 0 |

MRV18.1 restores a small number of real repeated attacks while removing additional transient/decay articulations. Net note count rises by only 5 across the full corpus and short-note counts still fall.

All 12 preflight MIDIs pass:
- MIDI Type 1
- PPQ 960
- monophonic melody / no overlaps
- GM #65 Soprano Saxophone (wire program 64)
- PB reset to zero after every Note-Off
- zero unowned corrected-valid V3R9 points

## Render commands

V3R9 is frozen. Do not rerun the tracker merely for MRV18.1 if the `_v3R9.csv*` set already exists.

```bash
mrv="18.1"

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

Package the regression:

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```

## Status

MRV18 should remain available as the rollback/candidate-final checkpoint. MRV18.1 is intentionally subject to listening-based second-pass calibration, especially for the 0.90 fast-PB residual factor and asymmetric repetition thresholds.
