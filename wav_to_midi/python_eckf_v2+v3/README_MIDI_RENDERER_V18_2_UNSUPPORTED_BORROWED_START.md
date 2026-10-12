# MIDI Renderer V18.2 — unsupported borrowed-start retraction

V18.2 is a renderer-only surgical child of accepted MRV18.1. Frozen V3R9 tracker/CLI state is not modified. Every V18.1 calibration — fast-PB residual factor 0.90, asymmetric repetition recovery, weak/transient folding, terminal-decay cleanup, breath trimming, acoustic rescue, and subharmonic adjudication — remains unchanged.

## Purpose

MRV18.1 still left a small residual class where a trusted-island owner borrowed a Note-On 90–150 ms backward into very weak material even though the borrowed corridor contained neither frozen corrected-valid target pitch nor defensible primary periodicity of that target family.

V18.2 adds exactly one fallback for that class.

An inherited trusted-island event is retracted only when **all** of the following are true:

1. trusted expression/pitch anchor is at least **90 ms** later than the Note-On;
2. WAV level at the borrowed Note-On is at or below **−48 dBFS**;
3. median WAV level across seven samples of the borrowed corridor is at or below **−40 dBFS**;
4. there is **no corrected-valid V3R9 point at all** before the trusted anchor;
5. there is no even-moderate target-family ACF/CMNDF evidence in the borrowed corridor:
   - ACF >= 0.45,
   - CMNDF <= 0.55,
   - ACF/CMNDF disagreement <= 150 cents,
   - primary family within 1.75 semitones of the event family;
6. the first frozen corrected-valid target-family point exists inside the event;
7. retracting to that point would leave at least **55 ms** of event duration.

The final 55-ms guard is deliberate. During preflight, two Damelo backdated events would otherwise have become ~20-ms standalone notes. The WAV confirms that brief pitch exists, but deciding whether those micro-fragments should be separate articulations is not V18.2's job. V18.2 therefore leaves those two cases unchanged rather than improving timing by worsening micro-fragmentation.

## Canonical V18.2 changes

Exactly four events change on the twelve-patient corpus:

- **Bodies ~89.393 -> ~89.543**
- **Damelo ~54.612 -> ~54.760**
- **Champagne ~37.297 -> ~37.431**
- **ElTelefono ~160.138 -> ~160.267**

Each new start is the first frozen corrected-valid target-family point. Note count, base pitch, end time, and all other inherited articulation decisions remain unchanged.

Two additional unsupported-start candidates are intentionally *not* changed:

- Damelo ~31.889: retraction would leave ~20.7 ms;
- Damelo ~92.726: retraction would leave ~20.5 ms.

These remain under the existing V18.1 transient/articulation policy instead of being exposed as new micro-notes.

## Twelve-patient regression

| metric | MRV18.1 | MRV18.2 |
| --- | ---: | ---: |
| total notes | 3038 | 3038 |
| events <40 ms | 177 | 177 |
| events <60 ms | 362 | 362 |
| median event duration | ~184.34 ms | ~184.21 ms |
| unowned corrected-valid V3R9 points | 0 | 0 |
| unsupported borrowed starts retracted | — | 4 |

All twelve MIDIs pass:

- MIDI Type 1
- PPQ 960
- monophonic melody / no overlaps
- GM #65 Soprano Saxophone (wire program 64)
- pitch-bend reset to zero after every Note-Off
- zero unowned corrected-valid V3R9 points

## Render commands

V3R9 is frozen. Do not rerun the tracker merely for MRV18.2 if the `_v3R9.csv*` set already exists.

```bash
mrv="18.2"

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

MRV18.1 remains the rollback candidate. MRV18.2 should be evaluated as a timing-only refinement: it does not change note count or pitch topology and deliberately refuses to create new micro-notes merely to remove unsupported borrowed starts.
