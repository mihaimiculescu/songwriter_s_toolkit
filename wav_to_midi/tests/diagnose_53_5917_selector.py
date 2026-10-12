#!/usr/bin/env python3
"""Read-only selector replay for PREDESTINATI sample 2363392 (~53.5917 s).

Uses current installed repository modules; never changes tracker, pitch, MIDI, or WAV.
Run from repo root: python tests/diagnose_53_5917_selector.py
"""
from __future__ import annotations
import argparse
import inspect
import math
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_eckf.config import ECKFConfig
from python_eckf.harmonic_change import HarmonicChangeDetector
from python_eckf.periodicity import assess_periodicity
from python_eckf import initialization_candidates as selector
from python_eckf.octave_reacquisition import reconcile_initialization
from python_eckf.trajectory_resolver import vocal_transition_penalty


def fmt(x):
    return 'none' if x is None else f'{x:.6f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wav', type=Path, default=ROOT/'tests'/'PREDESTINATI.wav')
    ap.add_argument('--sample', type=int, default=2363392)
    ap.add_argument('--previous-hz', type=float, default=None,
                    help='Optional comparison ONLY: real live previous voiced F0; not a generated candidate')
    ap.add_argument('--elapsed-ms', type=float, default=None,
                    help='Required with --previous-hz; elapsed since actual preceding voiced sample')
    ap.add_argument('--output', type=Path, default=ROOT/'tests'/'PREDESTINATI_53_5917_selector_audit.txt')
    a = ap.parse_args()
    if (a.previous_hz is None) != (a.elapsed_ms is None):
        ap.error('--previous-hz and --elapsed-ms must be supplied together')
    config = ECKFConfig(mode='offline')
    wav, fs = sf.read(a.wav, dtype='float64')
    if wav.ndim != 1:
        ap.error('WAV must be mono')
    block = config.block_size
    if a.sample < 0 or a.sample % block or a.sample + block > len(wav):
        ap.error(f'--sample must be an in-range multiple of block_size={block}')
    frame = wav[a.sample:a.sample+block]
    detector = HarmonicChangeDetector(config)
    period = assess_periodicity(frame, fs)
    proposal = detector.analyze(None, frame, fs)
    measured = selector._measured_period_candidates(frame, fs)
    a_hz = float(period.acf_frequency_hz)
    y_hz = float(period.cmndf_frequency_hz)
    reliable = (np.isfinite(a_hz) and np.isfinite(y_hz) and a_hz > 0 and y_hz > 0
                and float(period.acf_peak) >= .90 and float(period.cmndf_minimum) <= .10
                and abs(1200*math.log2(a_hz/y_hz)) <= 50)
    lines = []
    say = lines.append
    say('=== 53.5917s SELECTOR ACCEPTANCE REPLAY: READ ONLY ===')
    say(f'installed selector: {inspect.getfile(selector.choose_initialization)}')
    say(f'frame sample={a.sample} time={a.sample/fs:.8f}s block={block} fs={fs}')
    say(f'periodicity voiced={period.voiced} ACF={a_hz:.6f} Hz peak={period.acf_peak:.6f} '
        f'YIN={y_hz:.6f} Hz CMNDF={period.cmndf_minimum:.6f} reliable_reference={reliable}')
    say(f'raw spectral proposal={proposal.f0_hz:.6f} Hz')
    say('Ground-truth note onset annotated by user at this marker; NOT used for pitch or selection.')
    say(f'previous_hz={fmt(a.previous_hz)} elapsed_ms={fmt(a.elapsed_ms)} '
        '(absent means no continuity reference across the preceding unvoiced frame)')
    source = inspect.getsource(selector.choose_initialization)
    has_v2 = ('-len(harmonics), -hz' in source and
              'reference_mismatch' in source and 'candidates.sort(key=ranking)' in source)
    say(f'v2 ranking source recognized={has_v2}; replayed key valid only when True')
    say('Eligibility: local period within 50 cents, >=2 spectral peak locations, at least one harmonic index <=3.')
    say('WARNING: harmonic locations are not proof of independent fundamental energy.')
    say('')
    say('=== MEASURED LOCAL PERIODS (ACF and CMNDF for EACH lag) ===')
    for hz, lag, acf, cmndf in measured:
        say(f'lag={lag:4d} f0={hz:10.4f} Hz ACF={acf:.6f} CMNDF={cmndf:.6f}')
    say('')
    say('=== CANDIDATES AND CURRENT V2 SORT KEYS ===')
    entries = [('spectral_spacing',float(proposal.f0_hz))] + [
        ('waveform_periodicity',p[0]) for p in measured]
    eligible=[]
    for name,hz in entries:
        if not math.isfinite(hz) or hz <= 0 or hz >= fs/2:
            say(f'REJECT {name} {hz}: invalid frequency');continue
        matches=[p for p in measured if abs(1200*math.log2(hz/p[0])) <= 50]
        hs=selector._harmonic_support(detector,frame,fs,hz)
        permitted=bool(matches) and len(hs)>=2 and any(h<=3 for h in hs)
        penalty=(float(vocal_transition_penalty(12*math.log2(hz/a.previous_hz),a.elapsed_ms))
                 if a.previous_hz is not None else None)
        err=min((abs(1200*math.log2(hz/p[0])) for p in matches),default=float('inf'))
        mismatch=abs(1200*math.log2(hz/a_hz)) if reliable else 0.
        key=(0. if penalty is None else penalty, -len(hs), -hz, mismatch, err,
             0 if name=='spectral_spacing' else 1)
        say(f'{"ACCEPT" if permitted else "REJECT"} {name:21s} f0={hz:10.4f}Hz '
            f'matches={[(round(p[0],3),p[1],round(p[2],3),round(p[3],3)) for p in matches]} '
            f'harmonics={hs} penalty={penalty} waveform_mismatch_cents={mismatch:.3f} key={key}')
        if permitted:eligible.append((key,name,hz))
    eligible.sort()
    say(f'RECONSTRUCTED_V2_WINNER={eligible[0] if eligible else None}')
    choice=selector.choose_initialization(detector,frame,fs,a.sample,proposal,period,
                                           a.previous_hz,a.elapsed_ms)
    say(f'ACTUAL_SELECTOR_CHOICE={choice}')
    final,evidence=reconcile_initialization(detector,frame,fs,a.sample,choice,period,
                                            a.previous_hz,a.elapsed_ms)
    say(f'OCTAVE_AUDIT={evidence}')
    say(f'ACTUAL_FINAL_CHOICE={final}')
    if has_v2 and eligible and choice.frequency_hz is not None:
        say(f'REPLAY_MATCH={math.isclose(eligible[0][2],choice.frequency_hz,rel_tol=1e-8)}')
    say('')
    say('=== FFT PEAK AMPLITUDES / SHARED-HARMONIC WARNING ===')
    _, mag, _ = detector._frame_spectrum(frame,fs)
    inds,_=find_peaks(mag)
    inds=sorted(inds,key=lambda idx:mag[idx],reverse=True)
    for idx in inds[:18]:
        say(f'peak={float(detector._fbins[idx]):10.3f}Hz magnitude={float(mag[idx]):.8g}')
    say('These are raw spectrum peaks, NOT evidence that a submultiple is a fundamental.')
    say('')
    say('=== INTERPRETATION SAFETY ===')
    say('This is a standalone frame replay, not a live ECKF-state replay.')
    say('If absent, previous pitch is intentionally not invented across unvoiced audio.')
    say('The user-confirmed onset is not used to dictate any pitch or create MIDI notes.')
    say('No candidate multiplication, forced bridge, tracker mutation, or MIDI reading.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(f'Wrote {a.output}')

if __name__ == '__main__':
    main()
