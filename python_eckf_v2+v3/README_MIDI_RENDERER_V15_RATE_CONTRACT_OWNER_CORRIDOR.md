# MIDI Renderer V15 — Rate Contract + Attack→Owner Corridor Compatibility

## Status

V3R9 is frozen.  V15 changes the MIDI renderer only.

The renderer retains all V13 object-corridor protections and all V14 nearer-
stable-target owner arbitration.  Redundant renderer-side protections remain by
design: future WAVs may contain failure modes that the frozen tracker does not.

V15 has two deliberately narrow purposes:

1. make the post-135-Hz renderer timing contract explicit and remove implicit
   dependence on the historical 100-Hz interpretation grid;
2. prevent a future trusted pitch owner from borrowing an ordinary/noisy attack
   across a long pitchless corridor when the actual voiced owner arrives much
   later and substantially louder.

It does **not** yet perform the separate continuous-motion / glissando
fragmentation cleanup.  Keeping that surgery separate makes regressions easier
to attribute.

---

## 1. Explicit renderer rate contract

Frozen V3R9 uses `--analysis-hz 135`, approximately 7.415 ms per interpretation
point.  100 Hz is no longer a supported design assumption.

V15 distinguishes two classes of timing values.

### Interpretation-grid quantities

Anything that means "one/multiple interpretation points" derives from
`PitchEvidence.dt`, measured from the supplied interpretation timeline.
Examples include trusted-island continuity, island end time, local ownership
radius terms and post-anchor grid grace.

The renderer now exposes:

- `interpretation_dt_s`
- `interpretation_hz`

in event/attack audits, and writes:

`<output.mid>.rate_contract.csv`

for every render.

A degenerate input with fewer than two timed interpretation points now raises an
error instead of silently falling back to the historical 10-ms / 100-Hz grid.

### Physical-time acoustic tolerances

Several historical `10 ms` values are intentionally **not** converted to
7.407 ms.  They describe physical acoustic/query tolerances, not tracker frame
counts.  V15 gives them explicit names, including:

- `RESET_POST_ATTACK_GRACE_S = 0.010`
- `PRIMARY_QUERY_PREROLL_S = 0.010`
- `PRIMARY_BRIDGE_POSTROLL_S = 0.020`

This is important: a literal 10 ms is no longer ambiguous about whether it
means "one old 100-Hz point" or "ten milliseconds of physical time".

The WAV attack lane remains a separate physical-time lane (~23 ms analysis,
5 ms hop) and does not inherit interpretation-grid cadence.

---

## 2. V15 attack→owner corridor compatibility

V10-V14 intentionally allow a physical attack to precede the first trustworthy
F0 frame.  That remains correct for consonants, plosives and half-spoken vocal
onsets.

One failure class remained: an ordinary/noisy attack could be borrowed far
backward across a long pitchless lead and then assigned the pitch of a much
later trusted owner.  The MIDI therefore started during breath/noise rather
than near the actual voiced arrival.

V15 adds a conservative certificate.  It revokes this renderer-created
backdating only when all of the following hold:

1. attack→trusted-owner lag is at least 140 ms;
2. owner-compatible primary periodicity does not appear early enough
   (`max(100 ms, 90% of the lag)`);
3. the attack is acoustically near the recording floor (within 18 dB of the
   adaptive active floor);
4. the trusted owner's level is at least 5 dB stronger than the candidate
   attack; and
5. the attack is below the recording's 70th-percentile attack-strength gate
   (with a minimum gate of 12).

If the certificate fires, V15 does **not** delete the trusted island and does
not invent a new pitch.  It only refuses the weak/noisy early attack.  Existing
reset / trusted-pitch timing may still move Note-On slightly before the first
corrected-valid point.

Audit reason:

`long_unpitched_weak_lead_v15`

### Why the broader trusted-pitch-conflict rule is deferred

A prototype also revoked backdating whenever sustained different trusted pitch
occupied the attack→owner corridor.  The geometry is real, but applying it
before continuous-motion / rearticulation cleanup exposed too many inherited
splits and increased note count.  V15 therefore leaves that broader certificate
non-authoritative for now.  V14's narrower stable-target competition rule
remains active and unchanged.

---

## 3. Twelve-patient frozen-V3R9 preflight

Against the 12 reference WAVs and frozen V3R9 sidecars:

- MRV14 notes: **3239**
- MRV15 notes: **3239**
- note-count delta: **0**
- base-pitch changes caused by V15 corridor rule: **0**
- event-end changes caused by V15 corridor rule: **0**
- Note-On starts retracted from noisy/pitchless leads: **10**

The two user-identified Afurisita breath cases move:

- ~38.354 s → ~38.592 s
- ~65.248 s → ~65.387 s

Other V15 corridor firings occur only where the same conservative geometry is
present.  No V15 corridor firing occurs in Pedestal, MaInnecArseneEN, Damelo,
Champagne, RATATA, PREDESTINATI, Ochiitai or Trandafiri on the frozen V3R9
corpus.

MIDI wire invariants pass across all twelve:

- SMF Type 1
- PPQ 960
- channel 1 / zero-based channel 0
- GM #65 Soprano Saxophone (`program=64` in mido)
- monophonic, no overlaps
- one zero Pitch-Bend reset at every Note-Off
- velocity 100 default
- CC11 compression 0.3

### Controlled cadence preflight

Using the historical controlled V3R6-HF2 artifacts where tracker logic and
MRV13 input semantics were held constant while interpretation cadence was run
at 100 / 135 / 333 Hz, V15 was additionally exercised on the historically
rate-sensitive songs ElTelefono, Afurisita, PREDESTINATI and Trandafiri.

The new V15 rate/corridor code does not scale event creation with point count.
The 135-Hz run remains the product baseline.  100 Hz is retained only as a
historical regression fixture; it is not a target configuration.

---

## 4. Scope intentionally deferred

V15 does not attempt to solve, in the same release:

- continuous glissando / melisma over-fragmentation;
- MaInnec ~50.75 different-owner backdating without the V14 stable-target
  certificate;
- the remaining 134 corrected-valid points not consumed by the renderer;
- Afurisita ~68.12 frozen-tracker-abstention rescue;
- RATATA ~86.19 subharmonic fail-safe;
- tiny transient / micro-note adjudication such as PREDESTINATI ~47.74 and
  ElTelefono ~165.57.

Those are distinct mechanisms and should remain separable in regression work.

---

## 5. Commands — frozen V3R9 → MRV15

### No tracker rerun is required

V3R9 is frozen.  If the existing `tests/<song>_v3R9.csv*` sidecars are present,
render directly from them.

```bash
mrv=15

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

### Optional V3R9 sanity rerun

Only use this when deliberately re-verifying the frozen tracker artifacts:

```bash
for song in Pedestal MaInnecArseneEN Bodies Damelo Comomelo Champagne ElTelefono Afurisita RATATA PREDESTINATI Ochiitai Trandafiri; do
    python -m python_eckf.cli "tests/${song}.wav" \
        --mode offline \
        --wait 3 \
        --analysis-hz 135 \
        --csv "tests/${song}_v3R9.csv"
done
```

### Package the twelve-patient regression

```bash
7z a "tests/eckf_v3R9_mrv${mrv}_twelve_patient_full.zip" \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9.csv* \
    tests/{Pedestal,MaInnecArseneEN,Bodies,Damelo,Comomelo,Champagne,ElTelefono,Afurisita,RATATA,PREDESTINATI,Ochiitai,Trandafiri}_v3R9_mrv${mrv}.mid*
```

The `mid*` glob intentionally includes `.events.csv`, `.attacks.csv`,
`.coverage.csv` and the new `.rate_contract.csv` audit.

### Quick cadence audit

```bash
for f in tests/*_v3R9_mrv15.mid.rate_contract.csv; do
    echo "===== $f ====="
    grep -E 'interpretation_(dt|hz)' "$f"
done
```

For frozen V3R9, `interpretation_hz` should report approximately 135 Hz.
