# python_eckf — Frozen Production Slim Tree

Frozen functional baseline:

- Tracker / CLI: **V3R9**
- MIDI renderer: **MRV18.3**
- Interpretation cadence: `--analysis-hz 135` (actual cadence derived from `pitch.dt`)
- MIDI: Type 1, PPQ 960, channel 1, GM #65 Soprano Saxophone
- Default velocity: 100
- CC11 dB compression: 0.3
- Fast alternating pitch-modulation residual compression: 0.90

This directory is a **behavior-preserving production cleanup** of the frozen development tree.
No runtime Python source was rewritten. Only files that are not reachable from either public entry point were removed.

## Public entry points

Tracker:

```bash
python -m python_eckf.cli input.wav --mode offline --wait 3 --analysis-hz 135 --csv output.csv
```

Renderer:

```bash
python -m python_eckf.render_midi output.csv --wav input.wav --out output.mid --bpm 120 --time-signature 4/4
```

## What was intentionally removed

- historical per-version README files
- obsolete MIDI assemblers V10–V16 and V17 that are no longer imported
- obsolete MIDI attack evidence V10–V12
- obsolete `midi_articulation.py` / `midi_event_builder.py`
- superseded primary reconciliation V3R6–V3R8
- superseded register readjudication V17
- other modules not reachable from the frozen CLI or renderer

## Why some versioned filenames remain

Files such as `midi_pitch_evidence_v10.py`, `midi_attack_evidence_v13.py`,
`midi_event_assembler_v16_1.py`, and the V18.x assembler chain are still live dependencies of MRV18.3.
They are retained byte-for-byte in this cleanup. Renaming or flattening them would be a separate code refactor and is deliberately not part of the frozen production cleanup.

## Freeze rule

Do not change musical/tracker behavior in this tree. Any future development should branch from this frozen package.

## R4 Stage 2 performance layer

Stage 2 adds execution-only acceleration while retaining the frozen V3R9 +
MRV18.3 musical logic:

- `runtime_resources.py` detects the host CPU/RAM budget per run.
- `traffic_control.py` is the central concurrency governor.
- `frame_preflight.py` computes frozen per-frame silence/periodicity evidence
  once and caches it for the serial tracker.
- expensive `periodicity.assess_periodicity()` calls run unchanged across a
  bounded POSIX/Linux fork pool; the WAV is inherited copy-on-write.
- each worker is constrained to one native BLAS/OpenMP thread.
- RAM can reduce the worker count below the 75% logical-CPU budget.
- pool failure or unsupported `fork` degrades to the exact serial implementation.

Regression controls:

```bash
# normal Stage 2 path
python -m python_eckf.convert input.wav --out output.mid --bpm 120 --wait 3

# same cache, one worker; useful for proving cache semantics
python -m python_eckf.convert input.wav --out output.mid --bpm 120 --wait 3 \
  --frame-preflight serial

# bypass Stage 2 cache and use frozen V3R9 on-demand measurements
python -m python_eckf.convert input.wav --out output.mid --bpm 120 --wait 3 \
  --frame-preflight off
```

On the development regression fixture, Stage 2 parallel output (all generated
CSV sidecars and final MIDI) was byte-for-byte identical to Stage 1 while
reducing tracker+analysis wall time from 9.60 s to 7.15 s with only five logical
CPUs available to the test process. Real speedup depends on voiced-frame density
and available worker count.

## R4 Stage 3 — traffic police on the measured hotspots

Stage 3 is execution-only. It does not change V3R9 tracker/adjudication logic or
MRV18.3 rendering.

A real Ochiitai cProfile identified two dominant parallel-safe blocks:

* observational pitch-candidate construction (`_measured_period_candidates` and
  its per-frame acoustic measurements), and
* V21 juror pair temporal measurements (`temporal_persistence_v21._coherence`).

Stage 3 therefore adds two bounded fork-worker stages:

* `--candidate-parallel auto|serial|off`
* `--juror-parallel auto|serial|off`

`auto` uses the same host CPU/RAM traffic budget as Stage 2. Workers inherit the
immutable WAV copy-on-write on Linux and are internally limited to one native
BLAS/OpenMP thread. Any pool failure falls back to the exact serial path.

Candidate-field workers calculate independent per-frame acoustic evidence. The
only cross-frame field, the historical transition penalty from the previously
selected initialization, is reapplied serially after workers return, in the
original frame order.

Juror workers calculate independent original pair measurements in the exact
frozen pair order. The tournament remains serial. If a frame needs historical
direct missing-edge completion, those measurements run through the same pool;
the final tournament again remains serial.

For strict A/B regression use:

```bash
python -m python_eckf.convert INPUT.wav --out OUT.mid --bpm 130 \
  --candidate-parallel serial --juror-parallel serial
```

The default `auto` enables both Stage 3 worker stages.

## R4 Stage 4 — persistent traffic-police crew

Stage 4 keeps one RAM/CPU-capped POSIX fork crew alive across the parallel-safe
tracker/evidence stages instead of repeatedly constructing worker pools.  It
also precomputes pure harmonic-change and variable-aperture acoustic evidence;
the serial tracker remains the sole adjudicator.  Worker measurements never
change Kalman state, pitch ownership, or target decisions.

Execution controls:

```bash
--harmonic-preflight auto|serial|off
--aperture-preflight auto|serial|off
```

The persistent crew is capped by both the 75% logical-CPU budget and the
conservative per-worker RAM reserve.  Unsupported/failing parallel execution
falls back to the frozen serial calculation.

## R4 Stage 5 — keep the crew through interpretation

A Stage 4 Ochiitai profile showed that, after the tracker-side parallel work was
compressed, two downstream pure numerical loops had become material:

* per-point 24/40/64/96-ms trajectory evidence, and
* per-point expressive-amplitude modulation metrics.

Stage 5 keeps the existing persistent worker crew alive through those two
measurements.  Workers calculate disjoint output rows using the frozen NumPy
rules; the parent process reassembles rows in original timeline order before
any target formation, gesture, or structural decision is made.

Execution controls:

```bash
--interpretation-parallel auto|serial|off
--amplitude-parallel auto|serial|off
```

Stage 5 also contains two renderer execution-only cleanups:

* `PitchEvidence.primary_support()` uses a pre-parsed stable time index instead
  of reparsing/rescanning the complete frame CSV for every attack query.
* MRV18.1 fast-modulation processing reuses the already-computed local trend
  instead of computing the identical trend twice.

Neither cleanup changes a renderer threshold, ownership rule, pitch choice, or
MIDI event rule.

Regression fixtures require byte identity of all generated sidecars and MIDI
between Stage 4 and Stage 5.  `serial`/`off` switches remain available for
forensic A/B runs.


## R4 Stage 6 exact initialization preflight

Stage 6 leaves the recursive Numba Kalman kernel unchanged.  The profile shows
it is now the largest single hotspot, but a much faster scalar spelling changes
last-bit floating-point results and is therefore not enabled in the frozen
behavior path.

The exact Stage 6 optimization precomputes `_measured_period_candidates()` only
for frames predicted by the existing frozen initialization gate.  Workers run
the historical function unchanged; the serial tracker still performs all
transition-prior ranking and every pitch decision. Cache misses fall back to the
original on-demand path.

Execution control: `--initialization-preflight auto|serial|off`.
