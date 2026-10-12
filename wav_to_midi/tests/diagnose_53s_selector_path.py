#!/usr/bin/env python3
"""Read-only replay of actual candidate acceptance at one ECKF frame.

Uses the repository's current detector, periodicity and candidate functions.
Prints every measured local-period candidate, harmonic support, acceptance
conditions and the actual sort key. Does NOT alter tracker, WAV, MIDI or F0.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_eckf.config import ECKFConfig
from python_eckf.harmonic_change import HarmonicChangeDetector
from python_eckf.periodicity import assess_periodicity
from python_eckf.initialization_candidates import (
    _measured_period_candidates, _harmonic_support,
    choose_initialization,
)
from python_eckf.octave_reacquisition import reconcile_initialization


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wav', type=Path, default=ROOT/'tests'/'PREDESTINATI.wav')
    ap.add_argument('--target', type=float, default=53.2666)
    ap.add_argument('--previous-hz', type=float, default=None,
                    help='Only if independently measured from previous frame; never generates candidates')
    ap.add_argument('--output', type=Path, default=ROOT/'tests'/'PREDESTINATI_53s_selector_path.txt')
    args = ap.parse_args()
    config = ECKFConfig(mode='offline')
    wav, fs = sf.read(args.wav, dtype='float64')
    if wav.ndim != 1:
        ap.error('mono WAV required')
    block = config.block_size
    start = round(args.target * fs / block) * block
    frame = np.pad(wav[start:start+block], (0, max(0,start+block-len(wav))))
    detector = HarmonicChangeDetector(config)
    periodicity = assess_periodicity(frame, fs)
    proposal = detector.analyze(None, frame, fs)
    measured = _measured_period_candidates(frame, fs)
    out=[]
    out.append('=== ACTUAL CANDIDATE-SELECTION PATH: OBSERVATIONAL REPLAY ===')
    out.append(f'frame_start={start} frame_time={start/fs:.8f}s block={block} fs={fs}')
    out.append(f'periodicity voiced={periodicity.voiced} ACF={periodicity.acf_frequency_hz:.6f} Hz '
               f'YIN={periodicity.cmndf_frequency_hz:.6f} Hz ACF_peak={periodicity.acf_peak:.5f} '
               f'CMNDF={periodicity.cmndf_minimum:.5f}')
    out.append(f'raw spectral proposal={proposal.f0_hz:.6f} Hz')
    out.append('Selection implementation: initialization_candidates.choose_initialization (current installed repo)')
    out.append('Eligibility: measured-period match <=50 cents; >=2 spectral harmonics including one <=3.')
    out.append('Ordering: min(harmonic_indices), -harmonic_count, transition_penalty, measured_disagreement, source.')
    out.append('NOTE: harmonic support tests peak LOCATIONS, not their amplitude or harmonic-family uniqueness.')
    out.append(f'measured local periods={len(measured)}')
    entries=[('spectral_spacing',float(proposal.f0_hz))]+ [('waveform_periodicity',p[0]) for p in measured]
    eligible=[]
    for source,hz in entries:
        if not math.isfinite(hz) or hz<=0 or hz>=fs/2:
            out.append(f'REJECT {source} {hz:.4f}: out of physical frequency range')
            continue
        matches=[p for p in measured if abs(1200*math.log2(hz/p[0]))<=50]
        hs=_harmonic_support(detector,frame,fs,hz)
        accepted=bool(matches) and len(hs)>=2 and any(h<=3 for h in hs)
        err=min((abs(1200*math.log2(hz/p[0])) for p in matches),default=float('inf'))
        penalty=None
        if args.previous_hz and args.previous_hz>0:
            from python_eckf.trajectory_resolver import vocal_transition_penalty
            penalty=float(vocal_transition_penalty(12*math.log2(hz/args.previous_hz),1000*block/fs))
        key=(min(hs) if hs else float('inf'),-len(hs),
             float('inf') if penalty is None else penalty,err,0 if source=='spectral_spacing' else 1)
        out.append(f'{"ACCEPT" if accepted else "REJECT"} {source:21s} f0={hz:9.3f} '
                   f'matches={[(round(x[0],3),x[1]) for x in matches]} harmonics={hs} '
                   f'penalty={penalty} key={key}')
        if accepted:
            eligible.append((key,source,hz))
    eligible.sort()
    out.append(f'RECONSTRUCTED_SORT_WINNER={eligible[0] if eligible else "NONE"}')
    choice=choose_initialization(detector,frame,fs,start,proposal,periodicity,
                                 args.previous_hz,1000*block/fs if args.previous_hz else None)
    out.append(f'ACTUAL_SELECTOR_CHOICE={choice}')
    final,evidence=reconcile_initialization(detector,frame,fs,start,choice,periodicity,
                                            args.previous_hz,1000*block/fs if args.previous_hz else None)
    out.append(f'OCTAVE_AUDIT={evidence}')
    out.append(f'ACTUAL_FINAL_CHOICE={final}')
    out.append('REPLAY caveat: prior Hz is optional and may differ from actual live tracker context; compare choices to synchronized trace.')
    out.append('No frequency correction, MIDI or synthetic candidate created.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(out)+'\n')
    print('\n'.join(out))
    print(f'Wrote {args.output}')

if __name__=='__main__':
    main()
