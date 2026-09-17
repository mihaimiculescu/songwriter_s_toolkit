#!/usr/bin/env python3
"""Read-only WAV microstructure at unresolved islands 2, 3, 7 and 8.

Reports short-window RMS, positive energy change, spectral-flux novelty,
waveform periodicity, and STFT phase-advance residual. These are OBSERVATIONS:
none individually establishes a note boundary, consonant, or performed F0.
Ground-truth MIDI is never read. The tracker is never run or modified.

From repository root:
    python tests/diagnose_acoustic_microstructure.py

Outputs: tests/PREDESTINATI_acoustic_microstructure.txt and .csv
Dependencies: numpy, scipy, soundfile, existing python_eckf.periodicity.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_eckf.periodicity import assess_periodicity

# Measured unresolved intervals from the earlier read-only audit, NOT music labels.
ISLANDS = {
    2: (40.542041, 40.588481),
    3: (42.771156, 42.864036),
    7: (52.709297, 52.755737),
    8: (54.613333, 54.659773),
}
# Human annotations, used only as plot/report guideposts: not score or tuning inputs.
LANDMARKS = {7: ((52.716, 'observed K attack'), (53.055, 'observed vowel pitch change'))}


def fmt(v, digits=4):
    return f'{v:.{digits}f}' if v is not None and np.isfinite(v) else '--'


def periodicity(frame, sr):
    try:
        p = assess_periodicity(frame, sr)
        return dict(voiced=bool(p.voiced), acf=float(p.acf_peak),
                    cmndf=float(p.cmndf_minimum), acf_hz=float(p.acf_frequency_hz),
                    yin_hz=float(p.cmndf_frequency_hz), reason=str(p.reason))
    except (ValueError, FloatingPointError) as ex:
        return dict(voiced=None, acf=None, cmndf=None, acf_hz=None,
                    yin_hz=None, reason=f'unavailable:{type(ex).__name__}:{ex}')


def analyse(audio, sr, begin, end, hop, win):
    """STFT is window-centered; use exact original WAV samples (no resampling)."""
    margin = win / sr
    a = max(0, int(math.floor(begin * sr)) - win)
    b = min(len(audio), int(math.ceil(end * sr)) + win)
    segment = np.asarray(audio[a:b], dtype=np.float64)
    if len(segment) < win:
        raise ValueError('Insufficient WAV samples for STFT window')
    frequencies, centers, z = stft(segment, fs=sr, window='hann', nperseg=win,
                                    noverlap=win-hop, nfft=max(8192, 2**math.ceil(math.log2(win))),
                                    boundary=None, padded=False)
    centers = centers + a/sr
    mags = np.abs(z)
    norm = mags / np.maximum(np.linalg.norm(mags, axis=0, keepdims=True), 1e-15)
    flux = np.zeros(len(centers))
    flux[1:] = np.sqrt(np.sum(np.maximum(0, norm[:, 1:] - norm[:, :-1])**2, axis=0))
    energy = np.zeros(len(centers))
    rms = np.zeros(len(centers))
    observations = []
    # Frequency-bin phase residual only compares consecutive STFT frames at the
    # SAME bin. There is no pitch correction or candidate propagation.
    previous_bin = None
    for i, t in enumerate(centers):
        s = a + i*hop
        wave = audio[s:s+win]
        rms[i] = np.sqrt(np.mean(np.square(wave)))
        energy[i] = rms[i]**2
        p = periodicity(wave, sr)
        # Independent period estimate locates frequency band for phase probe.
        # This is NOT promoted to performed pitch and not copied to the tracker.
        f = p['acf_hz'] if p['voiced'] and p['acf_hz'] is not None else None
        if f is not None and 70 <= f <= 1000:
            eligible = (frequencies >= 0.88*f) & (frequencies <= 1.12*f)
            candidates = np.flatnonzero(eligible)
        else:
            candidates = np.array([], dtype=int)
        if len(candidates):
            peak_bin = int(candidates[np.argmax(mags[candidates, i])])
        else:
            peak_bin = None
        phase_residual = None
        phase_bin_hz = None
        phase_amp_ratio = None
        if i > 0 and previous_bin is not None:
            k = previous_bin
            # Both phases evaluated at previous frame's exact FFT frequency;
            # phase shift from hop is subtracted before wrapping.
            if mags[k, i-1] > 1e-12 and mags[k, i] > 1e-12:
                observed = np.angle(z[k, i] * np.conj(z[k, i-1]))
                expected = 2*math.pi*frequencies[k]*hop/sr
                phase_residual = float(np.angle(np.exp(1j*(observed-expected))))
                phase_bin_hz = float(frequencies[k])
                phase_amp_ratio = float(min(mags[k,i-1],mags[k,i]) /
                                        max(mags[k,i-1],mags[k,i]))
        previous_bin = peak_bin
        observations.append(dict(time_s=float(t), rms=float(rms[i]),
                                 flux=float(flux[i]), **p,
                                 probe_bin_hz=float(frequencies[peak_bin]) if peak_bin is not None else None,
                                 phase_residual_rad=phase_residual,
                                 phase_amplitude_ratio=phase_amp_ratio,
                                 phase_comparison_bin_hz=phase_bin_hz))
    for i, row in enumerate(observations):
        row['rms_dbfs'] = float(20*np.log10(max(rms[i], 1e-12)))
        row['energy_rise_db'] = (float(10*np.log10(max(energy[i],1e-24)/
                                           max(energy[i-1],1e-24))) if i else None)
        row['phase_residual_deg'] = (math.degrees(row['phase_residual_rad'])
                                     if row['phase_residual_rad'] is not None else None)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, default=ROOT/'tests'/'PREDESTINATI.wav')
    parser.add_argument('--out', type=Path, default=ROOT/'tests'/'PREDESTINATI_acoustic_microstructure.txt')
    parser.add_argument('--hop-ms', type=float, default=5.0)
    parser.add_argument('--window-ms', type=float, default=46.44,
                        help='Short window (default one old tracker block); timestamps centered.')
    parser.add_argument('--margin-ms', type=float, default=180.0)
    args = parser.parse_args()
    audio, sr = sf.read(args.wav, dtype='float64')
    if audio.ndim != 1:
        parser.error('Expected mono WAV; will not downmix automatically')
    hop = max(1, round(args.hop_ms*sr/1000))
    win = max(128, round(args.window_ms*sr/1000))
    if hop >= win or args.margin_ms < 0:
        parser.error('Require 0 < hop-ms < window-ms and nonnegative margin-ms')
    if win > len(audio):
        parser.error('WAV shorter than chosen analysis window')
    out = [
        '=== READ-ONLY ACOUSTIC MICROSTRUCTURE: ISLANDS 2, 3, 7, 8 ===',
        f'WAV={args.wav.resolve()} samples={len(audio)} fs={sr} mono=True',
        f'window={win} samples ({win/sr*1000:.2f} ms); hop={hop} samples ({hop/sr*1000:.2f} ms)',
        'STFT times are WINDOW CENTERS; energy/periodicity describe overlapping windows.',
        'Positive energy rise and spectral flux are descriptive; neither proves an onset.',
        'Phase residual: consecutive-window complex STFT at SAME previous frequency bin,',
        'minus expected bin rotation; wrapped to [-180,180] degrees. Strongly dependent',
        'on probe frequency, amplitude, modulation and window overlap. NOT a phase-lock verdict.',
        'NO MIDI READ. NO pitch filling, note creation, correction or segmentation.\n',
    ]
    records = []
    for island, (start, stop) in ISLANDS.items():
        # include user-annotated landmark at 53.055 for island 7 even beyond margin
        landmarks = LANDMARKS.get(island, ())
        lo = max(0.0, min([start] + [v for v,_ in landmarks])-args.margin_ms/1000)
        hi = min(len(audio)/sr, max([stop] + [v for v,_ in landmarks])+args.margin_ms/1000)
        rows = analyse(audio, sr, lo, hi, hop, win)
        for row in rows:
            t = row['time_s']
            row['island'] = island
            row['zone'] = 'INSIDE' if start <= t < stop else ('BEFORE' if t < start else 'AFTER')
            row['landmarks_near_20ms'] = ';'.join(label for value,label in landmarks if abs(t-value)<=.020)
            records.append(row)
        interior = [r for r in rows if start <= r['time_s'] < stop]
        if not interior:
            out.append(f'ISLAND {island}: no centered windows in interval!')
            continue
        peak_flux = sorted(interior, key=lambda r: r['flux'], reverse=True)[:3]
        out.extend([
            f'=== ISLAND {island:02d}: {start:.6f}–{stop:.6f} seconds ===',
            f'Analysis range {lo:.6f}–{hi:.6f}; centered windows inside={len(interior)}',
            'Observed local flux peaks inside unresolved interval: '+', '.join(
                f"{r['time_s']:.4f}s flux={r['flux']:.3f}" for r in peak_flux),
            'time_s     zone   RMS_dB  rise_dB  flux    ACF  CMNDF  ACF_Hz  YIN_Hz  phase_deg phase_ratio  landmark',
        ])
        for r in rows:
            out.append(f"{r['time_s']:8.4f} {r['zone']:<7s} {fmt(r['rms_dbfs'],1):>7s} "
                       f"{fmt(r['energy_rise_db'],1):>8s} {r['flux']:6.3f} "
                       f"{fmt(r['acf'],2):>5s} {fmt(r['cmndf'],2):>6s} "
                       f"{fmt(r['acf_hz'],1):>7s} {fmt(r['yin_hz'],1):>7s} "
                       f"{fmt(r['phase_residual_deg'],1):>9s} "
                       f"{fmt(r['phase_amplitude_ratio'],2):>11s}  {r['landmarks_near_20ms']}")
        out.append('Interpretation intentionally UNDETERMINED: no single feature proves note onset, offset, or rearticulation.\n')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(out)+'\n', encoding='utf-8')
    csv_path = args.out.with_suffix('.csv')
    if records:
        fields = ['island','zone','landmarks_near_20ms']+[key for key in records[0] if key not in ('island','zone','landmarks_near_20ms')]
        with csv_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    print(f'Islands: {len(ISLANDS)} | frames: {len(records)}\nTXT: {args.out}\nCSV: {csv_path}')

if __name__ == '__main__':
    main()
