# ECKF V2 — Adaptive Silence V4

This package supersedes Adaptive Silence V3.

## Changes

1. **Digital mute remains defined at -145 dBFS RMS.**
   Frames at or below this RMS level are excluded from noise-floor calibration and break candidate silence runs.

2. **Offline V2 fallback/default silence threshold is now -27.7734** on the historical MATLAB statistic `20*log10(sum(frame**2))`.
   With the normal 2048-sample frame this is approximately **-47 dBFS RMS**.

3. **No-trustworthy-noise-floor detector added.**
   Long quiet regions are not automatically accepted. After trimming 100 ms from both ends, a candidate region must have at least 250 ms of interior and be approximately stationary:
   - P90-P10 RMS-level spread <= 6 dB
   - absolute linear level slope <= 6 dB/s

   This is intentionally simple. A decaying reverb tail tends to fail because it is directional and/or too wide in level. If no trustworthy region exists, the calibrator falls back to -27.7734 rather than learning a threshold from reverb.

4. The per-run `*.silence_calibration.csv` now reports whether a trustworthy noise floor was found, how many long runs passed or failed the stationarity check, and any fallback reason.

## MATLAB compatibility

`--mode matlab` remains frozen as agreed and therefore retains its historical default `-50` unless explicitly overridden. The new `-27.7734` default/fallback is for the active offline V2 path.

## Normal run

```bash
for song in Ochiitai PREDESTINATI RATATA Trandafiri; do
    python -m python_eckf.cli "tests/${song}.wav" \
        --mode offline \
        --wait 3 \
        --csv "tests/${song}_v2_adaptive_silence_v4.csv"
done
```

## Package results

```bash
7z a tests/eckf_v2_adaptive_silence_v4_four_song.zip \
    tests/Ochiitai_v2_adaptive_silence_v4.csv* \
    tests/PREDESTINATI_v2_adaptive_silence_v4.csv* \
    tests/RATATA_v2_adaptive_silence_v4.csv* \
    tests/Trandafiri_v2_adaptive_silence_v4.csv*
```
