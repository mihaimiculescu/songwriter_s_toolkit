#!/usr/bin/env python3
"""WAV-only, read-only, time-resolved acoustic diagnostics for three marked passages.

Run from songwriter_s_toolkit root:
    python tests/diagnose_piggy_reverb.py
    python tests/diagnose_piggy_reverb.py --case Trandafiri

Uses no MIDI, tracker, selector, note targets or artificial candidate multiplication.
ACF evidence and spectral peaks are descriptive, not ground-truth F0 decisions.
A mono mixture cannot establish a direct/reverberant energy ratio without a
known impulse response or separate reference signals.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

CASES = {
    'RATATA': (42.78, 43.20, (42.965, 43.008)),
    'Ochiitai': (42.05, 42.55, (42.214, 42.353)),
    'Trandafiri': (10.60, 11.06, (10.774, 10.821)),
}

def fnum(x, precision=3):
    return 'NA' if not np.isfinite(x) else f'{x:.{precision}f}'

def wave_metrics(x, fs):
    """Normalized overlapping ACF; explicit measured local maxima only."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x*x)))
    energy = float(np.dot(x, x))
    if energy < 1e-15:
        return rms, [], [], []
    n = len(x)
    minlag = max(2, int(fs / 900))
    maxlag = min(int(fs / 65), n - 2)
    if maxlag <= minlag:
        return rms, [], [], []
    # Unbiased overlap normalization suppresses the trivial long-lag bias.
    acf = np.correlate(x, x, mode='full')[n-1:]
    acf = acf / np.maximum(1, n - np.arange(n))
    acf /= acf[0]
    seg = acf[minlag:maxlag+1]
    peaks, _ = find_peaks(seg)
    lags = (peaks + minlag).tolist()
    if seg.size > 1 and seg[0] > seg[1]:
        lags.append(minlag)
    if seg.size > 1 and seg[-1] > seg[-2]:
        lags.append(maxlag)
    lags = sorted(set(lags), key=lambda lag: float(acf[lag]), reverse=True)
    # FFT peaks describe spectral spacing. Zero padding interpolates display;
    # it does NOT improve the actual resolution of a short analysis window.
    win = np.hanning(n)
    nfft = 1 << max(14, (4*n-1).bit_length())
    mag = abs(np.fft.rfft(x*win, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    inds, _ = find_peaks(mag)
    inds = [int(i) for i in inds if 65 <= freqs[i] <= min(4000, fs/2)]
    inds.sort(key=lambda i: mag[i], reverse=True)
    spectral = [(float(freqs[i]), float(mag[i])) for i in inds[:12]]
    candidates = [(int(lag), float(fs/lag), float(acf[lag])) for lag in lags[:8]]
    return rms, candidates, spectral, (freqs, mag)

def spectral_family(freqs, mag, hz, fs, win_size):
    """Observed peak energy near measured candidate harmonics, not a score/verdict."""
    if hz <= 0:
        return []
    # Bin radius cannot be narrower than finite-window resolution.
    width = max(fs/win_size, 0.025*hz)
    out = []
    for k in range(1, 7):
        target = k*hz
        if target + width >= min(4000, fs/2):
            break
        sl = np.flatnonzero((freqs >= target-width) & (freqs <= target+width))
        if sl.size:
            j = int(sl[np.argmax(mag[sl])])
            out.append((k, float(freqs[j]), float(mag[j])))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=['all', *CASES], default='all')
    ap.add_argument('--tests-dir', type=Path, default=Path('tests'))
    ap.add_argument('--out-dir', type=Path, default=Path('tests/robustness_v3/piggy_acoustics'))
    ap.add_argument('--window-ms', type=float, default=24.)
    ap.add_argument('--hop-ms', type=float, default=5.)
    args = ap.parse_args()
    if not 15 <= args.window_ms <= 60 or not 1 <= args.hop_ms <= args.window_ms:
        ap.error('Require 15 <= --window-ms <= 60 and 1 <= --hop-ms <= --window-ms')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = list(CASES) if args.case == 'all' else [args.case]
    for name in names:
        start, end, markers = CASES[name]
        matches = [args.tests_dir / (name + ext) for ext in ('.wav', '.wa') if (args.tests_dir / (name + ext)).is_file()]
        if len(matches) != 1:
            raise SystemExit(f'{name}: expected exactly one .wav or .wa in {args.tests_dir}; got {len(matches)}')
        path = matches[0]
        audio, fs = sf.read(path, dtype='float64')
        if audio.ndim != 1:
            raise SystemExit(f'{name}: mono required, found shape {audio.shape}')
        win = max(512, round(args.window_ms * fs/1000))
        hop = max(1, round(args.hop_ms * fs/1000))
        first = max(0, round(start*fs))
        last = min(len(audio)-win, round(end*fs))
        if first > last:
            raise SystemExit(f'{name}: interval outside WAV')
        rows, detail = [], []
        detail += [f'CASE {name}', f'WAV {path.resolve()}', f'SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}',
                   f'fs {fs} samples {len(audio)} window_samples {win} hop_samples {hop}',
                   f'markers_s {markers}', 'NO MIDI / NO TRACKER MODIFICATION / NO PITCH CORRECTION',
                   'WARNING: overlapping ACF maxima are not independent evidence; FFT peak spacing needs sufficient cycles.',
                   'WARNING: mono mixed WAV cannot identify direct-to-reverb ratio or establish a reverb cause.', '']
        for i in range(first, last+1, hop):
            x = audio[i:i+win]
            t0, t1 = i/fs, (i+win)/fs
            rms, candidates, spectral, spec = wave_metrics(x, fs)
            strongest = candidates[0] if candidates else None
            shortest = min(candidates, key=lambda p:p[0]) if candidates else None
            r = {'case':name,'start_s':f'{t0:.8f}','end_s':f'{t1:.8f}',
                 'center_s':f'{(t0+t1)/2:.8f}', 'marker_overlap':int(any(t0<=m<=t1 for m in markers)),
                 'rms':f'{rms:.9f}', 'acf_best_hz':fnum(strongest[1]) if strongest else '',
                 'acf_best_strength':fnum(strongest[2],5) if strongest else '',
                 'acf_shortest_peak_hz':fnum(shortest[1]) if shortest else '',
                 'acf_shortest_peak_strength':fnum(shortest[2],5) if shortest else '',
                 'fft_top_hz':fnum(spectral[0][0]) if spectral else '',
                 'fft_top_mag':fnum(spectral[0][1],6) if spectral else '',
                 'acf_measured_peaks':' | '.join(f'{hz:.2f}@lag{lag}:{val:.3f}' for lag,hz,val in candidates),
                 'fft_top_peaks':' | '.join(f'{hz:.1f}:{val:.3g}' for hz,val in spectral[:8])}
            rows.append(r)
            detail.append(f'WINDOW {t0:.6f}-{t1:.6f} center={(t0+t1)/2:.6f} marker_overlap={r["marker_overlap"]} RMS={rms:.7f}')
            detail.append('  measured ACF peaks: '+(r['acf_measured_peaks'] or 'NONE'))
            detail.append('  FFT peaks: '+(r['fft_top_peaks'] or 'NONE'))
            if spec and candidates:
                freqs, mag = spec
                for lag, hz, strength in candidates[:4]:
                    fam = spectral_family(freqs, mag, hz, fs, win)
                    detail.append('  family of MEASURED lag %d (%.2f Hz, ACF %.3f): %s' % (
                        lag, hz, strength, ' '.join(f'H{k}={freq:.1f}Hz/{amp:.3g}' for k,freq,amp in fam)))
            detail.append('')
        csv_path = args.out_dir / f'{name}_windows.csv'
        with csv_path.open('w',newline='') as fp:
            w = csv.DictWriter(fp,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
        detail_path = args.out_dir / f'{name}_details.txt'
        detail_path.write_text('\n'.join(detail)+'\n')
        print(f'{name}: {len(rows)} windows -> {csv_path}, {detail_path}')
    print('Interpretation: compare stability and harmonic coherence around markers; no direct/reverb ratio can be inferred.')

if __name__ == '__main__':
    main()
