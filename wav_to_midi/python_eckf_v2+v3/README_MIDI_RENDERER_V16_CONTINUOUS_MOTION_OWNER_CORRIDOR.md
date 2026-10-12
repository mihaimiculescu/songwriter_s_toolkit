# MIDI Renderer V16 — Continuous Motion + Authoritative Owner Corridor

## Status

**V3R9 is frozen.** V16 changes the MIDI renderer only.

All V13 object-corridor protections, V14 nearer-stable-target arbitration, and
V15 rate/corridor fail-safes remain in place. Renderer-side redundancy remains
intentional for future unseen WAVs.

V16 has two paired purposes:

1. reduce false Note-On fragmentation inside one physically continuous vocal
   object (glissando / continuous pitch motion / detector echo);
2. make the previously advisory V15 trusted-pitch corridor contradiction
   authoritative, now that false internal rearticulations have their own
   protection layer.

This is **not generic smoothing**. Pitch authority remains frozen V3R9.

---

## 1. Physical reattack certificate

Spectral novelty alone is not sufficient to create another Note-On inside an
already voiced object. Pitch movement itself can generate enormous spectral
flux.

V16 therefore asks whether the waveform proves a bilateral envelope restart.
A physical reattack is protected when:

- there is audible energy before a local valley;
- there is audible energy after the valley;
- both sides rise at least **5.0 dB** above that valley;
- attack strength is at least 8; and
- the post-attack object is audibly above the adaptive recording floor.

A proven bilateral restart always outranks the V16 smoothing certificates.

---

## 2. Continuous-motion structural crossings

A CLI stable target remains authoritative **pitch evidence**, but a stable
pitch center is not automatically a new physical articulation.

A structural Note-On may be suppressed only when all of the following hold:

- the boundary lies inside one corrected-valid trusted island;
- no bilateral waveform reattack is proven;
- no ECKF reset is nearby;
- trusted pitch is moving in the same direction on both sides of the boundary;
- the absolute pitch slope on each side is at least **9.5 semitones/s**;
- the larger slope is no more than **2.5×** the smaller slope.

The slope test uses a **60 ms physical-time half-window** on each side. It is not
expressed in interpretation-point counts and therefore does not assume 100 Hz.

Audit reason:

`continuous_motion_stable_target_no_note_v16`

---

## 3. Continuous-motion WAV rearticulation cleanup

A free WAV rearticulation inside a trusted pitch owner is suppressed only when
there is no protected physical reattack and one of these narrow geometries is
present:

1. the same directed-motion crossing described above;
2. a detector echo within **25 ms** after an already-owned structural boundary,
   with local trusted pitch matching that boundary within 0.75 semitone; or
3. terminal-decay morphology: strong energy on the left, essentially no right
   restart, while pitch is moving through the candidate boundary.

Audit reasons:

- `continuous_motion_rearticulation_suppressed_v16`
- `continuous_motion_primary_rearticulation_suppressed_v16`

### Short-island fail-safe

V16 will not smooth away a late internal attack in a short trusted island when
that can orphan the island tail:

- short-island guard: **220 ms**
- late-attack threshold: **100 ms after island start**

This was added after preflight caught two newly unowned MaInnec valid-F0 points.

---

## 4. Trusted-pitch owner corridor becomes authoritative

V15 detected, but deliberately did not act on, this geometry:

> a future trusted owner borrows an earlier attack while a materially different
> corrected-valid pitch family already occupies the attack→owner corridor.

V16 now revokes that backdating when the conflicting trusted-pitch run lasts at
least the inherited 25 ms contradiction duration **and no bilateral physical
reattack is proven**.

The later trusted island is not deleted. It simply starts from its own defended
anchor/reset instead of stealing the earlier physical attack.

Audit reason:

`trusted_pitch_conflict_v16`

This fixes the MaInnec ~50.75 family without weakening V14's nearer-stable-owner
certificate.

---

## 5. Twelve-patient preflight

Frozen V3R9 + MRV15 versus frozen V3R9 + MRV16:

- notes: **3239 → 3101** (**−138 / −4.26%**)
- events shorter than 40 ms: **269 → 240**
- events shorter than 60 ms: **476 → 434**
- median event duration: **174.6 ms → 183.9 ms**
- maximum required Pitch Bend range: unchanged at **12 semitones**
- no new corrected-valid coverage holes
- uncovered corrected-valid points: **134 → 111**

The 23-point improvement is RATATA ~40.280–40.443: removing an inherited false
split lets the existing trusted ~310 Hz / MIDI-63 owner cover that old hole.

### Bookmark cases

**Pedestal ~97.55 glissando**

V15 rendered six articulation objects through the gliss. V16 renders three.
The motion is still not collapsed to one monolithic note; V16 deliberately
stops where the acoustic/CLI evidence is no longer strong enough to merge.

**MaInnec ~18.97**

The duplicate A4 WAV rearticulation after the already-owned structural boundary
is removed. The genuine preceding pitch transition remains.

**MaInnec ~49.908**

Terminal-decay false rearticulation is removed.

**MaInnec ~50.751**

The future low owner can no longer borrow the earlier ~MIDI-57 attack through a
conflicting trusted-pitch corridor. The high owner runs to ~50.807 and the low
owner begins near its own ~50.852 anchor.

### Deliberately preserved

Afurisita ~65.423 and ElTelefono ~108.335 have bilateral envelope-restart
morphology, so V16 does **not** smooth them away merely because pitch is
continuous nearby.

---

## 6. Still deferred

V16 does not attempt to solve every renderer problem at once. Still separate:

- MaInnec ~27.22–27.38 remaining valid-F0 coverage gap;
- ElTelefono ~48.66 early owner backdating where no different *trusted* pitch
  survives in the corridor;
- Afurisita ~68.12 frozen-tracker-abstention acoustic rescue;
- RATATA ~86.19 false low/subharmonic note;
- PREDESTINATI ~47.74 transient promoted to a musical note;
- ElTelefono ~165.57 transient / breath-start standalone note;
- remaining 111 corrected-valid points not consumed by the renderer.

---

## 7. Commands — frozen V3R9 → MRV16

### No tracker rerun is required

```bash
mrv=16

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

### Optional frozen-V3R9 sanity rerun

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

The `mid*` glob includes `.events.csv`, `.attacks.csv`, `.coverage.csv` and
`.rate_contract.csv`.

### Quick rate-contract audit

```bash
for f in tests/*_v3R9_mrv16.mid.rate_contract.csv; do
    echo "===== $f ====="
    grep -E 'interpretation_(dt|hz)' "$f"
done
```

Frozen V3R9 should report approximately 135 Hz (measured from actual sample-grid
spacing, not assumed exactly).
