#!/usr/bin/env python3
"""Read-only audit of every VOICED_UNRESOLVED span in an ECKF recording.

Uses WAV and pitch/status from ONE track_pitch() invocation. Never fills F0,
changes statuses, consults reference MIDI, or asserts a note boundary.
Outputs small TXT and CSV reports in tests/.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_eckf.config import ECKFConfig
from python_eckf.tracker import track_pitch
from python_eckf.pitch_status import PitchStatus, validate_pitch_status
from python_eckf.periodicity import assess_periodicity

U = int(PitchStatus.UNVOICED)
V = int(PitchStatus.VOICED_VALID)
R = int(PitchStatus.VOICED_UNRESOLVED)
NAMES = {U: 'UNVOICED', V: 'VOICED_VALID', R: 'VOICED_UNRESOLVED'}


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fmt(x, digits=3):
    return f'{x:.{digits}f}' if x is not None and math.isfinite(x) else '--'


def pitch_summary(f0):
    v = np.asarray(f0)
    v = v[np.isfinite(v) & (v > 0)]
    return (float(np.median(v)), float(np.min(v)), float(np.max(v))) if len(v) else (None, None, None)


def measured_spectrum(frame, sr, count=6):
    x = np.asarray(frame, dtype=np.float64)
    if not len(x) or not np.any(x):
        return []
    x = (x - x.mean()) * np.blackman(len(x))
    nfft = max(16384, 2 ** int(np.ceil(np.log2(max(1, len(x) * 8)))))
    mag = np.abs(np.fft.rfft(x, n=nfft))
    indices, _ = find_peaks(mag)
    indices = indices[(indices * sr / nfft >= 70) & (indices * sr / nfft <= 4000)]
    if not len(indices):
        return []
    top = indices[np.argsort(mag[indices])[-count:][::-1]]
    return [(float(i * sr / nfft), float(mag[i])) for i in top]


def inspect_frame(start, end, audio, f0, status, sr):
    segment = audio[start:end]
    statuses, counts = np.unique(status[start:end], return_counts=True)
    label = ','.join(f'{NAMES.get(int(k), str(k))}:{n}' for k, n in zip(statuses, counts))
    med, lo, hi = pitch_summary(f0[start:end])
    rms = float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.0
    try:
        p = assess_periodicity(segment, sr)
        voiced = p.voiced
        acf = p.acf_peak
        cmndf = p.cmndf_minimum
        acf_hz = p.acf_frequency_hz
        yin_hz = p.cmndf_frequency_hz
        reason = p.reason
    except (ValueError, FloatingPointError) as exc:
        voiced, acf, cmndf, acf_hz, yin_hz = None, None, None, None, None
        reason = f'ASSESSMENT_UNAVAILABLE:{type(exc).__name__}:{exc}'
    peaks = measured_spectrum(segment, sr)
    return dict(start_sample=start, end_sample=end, start_s=start/sr, end_s=end/sr,
                statuses=label, median_f0_hz=med, min_f0_hz=lo, max_f0_hz=hi,
                rms=rms, periodicity_voiced=voiced, acf_peak=acf,
                cmndf_minimum=cmndf, acf_hz=acf_hz, cmndf_hz=yin_hz,
                periodicity_reason=reason,
                strongest_peaks_hz=';'.join(f'{hz:.2f}' for hz, _ in peaks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, default=ROOT/'tests'/'PREDESTINATI.wav')
    parser.add_argument('--flank-frames', type=int, default=5,
                        help='Number of frames on each side for detailed observation (default: 5).')
    parser.add_argument('--out', type=Path, default=ROOT/'tests'/'PREDESTINATI_unresolved_islands.txt')
    args = parser.parse_args()
    if args.flank_frames < 1:
        parser.error('--flank-frames must be >= 1')
    audio, sr = sf.read(args.wav, dtype='float64')
    if audio.ndim != 1:
        raise SystemExit('Expected mono WAV; no automatic downmix')
    config = ECKFConfig(mode='offline')
    result = track_pitch(audio, sr, config)
    f0 = np.asarray(result.f0_hz)
    status = np.asarray(result.pitch_status)
    validate_pitch_status(f0, status)
    # Exclude tracker padding when classifying actual performance.
    size = len(audio)
    f0, status = f0[:size], status[:size]
    block = config.block_size
    islands = spans(status == R)
    lines = ['=== UNRESOLVED-VOICE ISLAND AUDIT: ONE TRACKER RUN ===',
             f'WAV: {args.wav.resolve()}', f'sample_rate={sr} block={block} original_samples={size}',
             f'unresolved_spans={len(islands)} unresolved_samples={int(np.sum(status == R))}',
             'No MIDI, interpolation, pitch correction, note generation, or status modification.',
             'Neighbor pitches are measured ECKF values, not proof of note identity.', '']
    csv_rows = []
    if not islands:
        lines.append('No unresolved spans. The three-state contract is not exercised by this recording.')
    for idx, (a, b) in enumerate(islands, 1):
        left = a - 1
        right = b
        left_status = NAMES.get(int(status[left]), str(status[left])) if left >= 0 else 'RECORDING_START'
        right_status = NAMES.get(int(status[right]), str(status[right])) if right < size else 'RECORDING_END'
        # Immediate contiguous VOICED_VALID runs only. Never jump across an UNVOICED wall.
        left_start = a
        while left_start > 0 and status[left_start-1] == V:
            left_start -= 1
        right_end = b
        while right_end < size and status[right_end] == V:
            right_end += 1
        left_pitch = pitch_summary(f0[max(left_start, a - args.flank_frames*block):a])[0] if left_start < a else None
        right_pitch = pitch_summary(f0[b:min(right_end, b + args.flank_frames*block)])[0] if right_end > b else None
        delta_st = 12*math.log2(right_pitch/left_pitch) if left_pitch and right_pitch else None
        lines += [f'=== ISLAND {idx:02d}: {a/sr:.6f}–{b/sr:.6f} s ({1000*(b-a)/sr:.2f} ms; {b-a} samples) ===',
                  f'immediate neighbors: LEFT={left_status} RIGHT={right_status}',
                  f'contiguous VALID context: left={1000*(a-left_start)/sr:.2f} ms right={1000*(right_end-b)/sr:.2f} ms',
                  f'nearest-side measured median F0: left={fmt(left_pitch)} Hz right={fmt(right_pitch)} Hz',
                  f'endpoint interval (observation only)={fmt(delta_st)} semitones',
                  'NOTE BOUNDARY: UNDETERMINED. Status 2 is neither a note-off nor permission to hold/split a note.',
                  'VOICE CONTINUITY: marked voiced but F0 unresolved by tracker; adjacent pitch cannot establish note identity.',
                  'Frame evidence (including flanks):',
                  'time_s    status                ECKF_med   ACF     CMNDF   ACF_Hz    YIN_Hz   RMS        spectral_peaks_Hz']
        frame_first = max(0, (a//block - args.flank_frames)*block)
        frame_stop = min(size, ((b + block-1)//block + args.flank_frames)*block)
        for s in range(frame_first, frame_stop, block):
            e = min(size, s+block)
            row = inspect_frame(s, e, audio, f0, status, sr)
            row['island_id'] = idx
            row['position'] = 'UNRESOLVED' if s < b and e > a else ('LEFT' if e <= a else 'RIGHT')
            csv_rows.append(row)
            lines.append(f"{row['start_s']:8.4f}  {row['statuses'][:20]:20s} {fmt(row['median_f0_hz']):>8s} "
                         f"{fmt(row['acf_peak']):>7s} {fmt(row['cmndf_minimum']):>7s} "
                         f"{fmt(row['acf_hz']):>9s} {fmt(row['cmndf_hz']):>9s} "
                         f"{row['rms']:10.6g}  {row['strongest_peaks_hz']}"
                         + (f"  [{row['periodicity_reason']}]" if row['position']=='UNRESOLVED' else ''))
        lines += ['INTERPRETATION: Inspect waveform periodicity, spectral peaks, and both sides together.',
                  'A matching endpoint pitch is not proof of same note; different endpoints are not proof of onset inside island.',
                  'If either side is missing/UNVOICED, do not bridge that wall.', '']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    csv_path = args.out.with_suffix('.csv')
    if csv_rows:
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print('\n'.join(lines))
    print(f'\nTXT: {args.out}\nCSV: {csv_path if csv_rows else "not written (no islands)"}')


if __name__ == '__main__':
    main()
