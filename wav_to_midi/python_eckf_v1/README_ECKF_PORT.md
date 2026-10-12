# Python ECKF port — v1

This is the first Python implementation of the improved ECKF path from
`orchidas/Pitch-Tracking`.

It deliberately keeps pitch tracking separate from MIDI interpretation.

## Modes

### `matlab`

Compatibility-oriented translation of the checked-in MATLAB source.

It preserves important source quirks, including:

- the broken nominal "below 50 Hz" bin calculation;
- the source's `linspace(-fs/2, fs/2, nfft)` frequency coordinates;
- skipping the final complete padded frame;
- after onset look-ahead, rewinding the output/time cursor while still using
  the future stabilized audio frame for the first backtracked block.

### `offline`

Vocal-only corrected mode.

v1 deliberately changes three things:

1. frequencies below `vocal_floor_hz` are excluded using true FFT-bin
   frequencies;
2. the final complete/padded frame is processed;
3. after look-ahead initialization, filtering resumes on the actual
   backtracked audio and the initialized state is re-anchored to that sample.

Default vocal floor is **60 Hz**, slightly below C2 (~65.4 Hz), to provide
margin for unusually low male vocals.

## Install

From the repository root, with the `eckf-pitch` environment activated:

```bash
pip install -r requirements-eckf.txt
```

## Run on a mono WAV

```bash
python -m python_eckf.cli input.wav --mode offline
```

Repository-comment vocal preset:

```bash
python -m python_eckf.cli input.wav \
  --mode offline \
  --block-size 2048 \
  --c 7 \
  --wait 2 \
  --npeaks 3 \
  --nsemitones 2 \
  --vocal-floor-hz 60
```

Paper-style comparison with NP=4:

```bash
python -m python_eckf.cli input.wav \
  --mode offline \
  --block-size 2048 \
  --c 7 \
  --wait 2 \
  --npeaks 4 \
  --nsemitones 2 \
  --vocal-floor-hz 60
```

The program derives all timing from the actual WAV sample rate.  44.1 kHz and
48 kHz are both accepted; there is no hard-coded 48 kHz assumption.

## Synthetic validation

```bash
python tests/test_saw.py
```

This mirrors the repository's `test_saw.m` and uses an analytically known
415–465 Hz vibrato trajectory.

## Not implemented yet

- MIDI note segmentation
- ornament classification
- pitch bend / dynamic bend-range writer
- forward/backward offline smoothing
- octave-error repair
- whole-gesture classification

Those belong *after* raw ECKF validation.
