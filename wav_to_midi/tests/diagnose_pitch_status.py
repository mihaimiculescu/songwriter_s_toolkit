#!/usr/bin/env python3
"""Inspect actual tracker pitch-status output without altering pitches or MIDI."""
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_eckf.config import ECKFConfig
from python_eckf.tracker import track_pitch


def main():
    wav = ROOT / 'tests' / 'PREDESTINATI.wav'
    y, sr = sf.read(wav, dtype='float64')
    if y.ndim != 1:
        raise SystemExit('Expected a mono WAV')
    cfg = ECKFConfig(mode='offline')
    result = track_pitch(y, sr, cfg)
    if not hasattr(result, 'pitch_status'):
        raise SystemExit('Installed tracker result lacks pitch_status; check patch installation')
    status = np.asarray(result.pitch_status)
    f0 = np.asarray(result.f0_hz)
    if status.shape != f0.shape:
        raise SystemExit(f'Shape mismatch: status={status.shape}, f0={f0.shape}')
    lines = ['=== PITCH STATUS / F0 AUDIT (SAME RUN) ===',
             'frame_time_s  status_values  voiced_samples  median_f0_hz']
    block = cfg.block_size
    for start in range(0, len(f0), block):
        t = start / sr
        if not 58.0 <= t <= 59.1:
            continue
        s = status[start:start + block]
        f = f0[start:start + block]
        values, counts = np.unique(s, return_counts=True)
        labels = ','.join(f'{v}:{n}' for v, n in zip(values, counts))
        voiced = f[np.isfinite(f) & (f > 0)]
        med = f'{np.median(voiced):.3f}' if voiced.size else 'NO_F0'
        lines.append(f'{t:12.4f}  {labels:24s}  {voiced.size:14d}  {med}')
    values, counts = np.unique(status, return_counts=True)
    lines.extend(['', '=== ENTIRE SONG STATUS COUNTS ==='] +
                 [f'{v}: {n} samples' for v, n in zip(values, counts)])
    report = ROOT / 'tests' / 'PREDESTINATI_pitch_status_audit.txt'
    report.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved: {report}')


if __name__ == '__main__':
    main()
