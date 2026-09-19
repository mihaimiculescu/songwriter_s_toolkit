#!/usr/bin/env python3
"""Read-only, WAV-only adaptive acoustic evidence experiment for FOUR recordings.

Run from repository root: python tests/diagnose_adaptive_evidence.py
Uses short probes at each hop; selectively expands only ambiguous/insufficient probes.
No production modules imported, no MIDI accessed, no pitch correction, no synthetic candidates.
Observational heuristic decisions are NOT production F0 decisions or accuracy scores.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, fftconvolve

NAMES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
HOTSPOTS = {'Ochiitai': [(42.05, 42.55)], 'PREDESTINATI': [(37.90, 38.25), (53.48, 53.80)],
            'RATATA': [(42.78, 43.20)], 'Trandafiri': [(10.60, 11.06)]}
SIZES = (24., 40., 64., 96.)
COLUMNS = ('name', 'time_s', 'probe_rms', 'window_ms', 'expanded', 'reason', 'status', 'observed_hz',
           'acf', 'cmndf', 'spectral_coverage', 'candidate_count', 'candidates', 'early_hz', 'late_hz',
           'nonstationary', 'baseline_hz', 'baseline_voiced', 'difference_semitones', 'comparison')

def analyze(x, fs, min_hz=65., max_hz=900., max_candidates=8):
    """Compute normalized overlap ACF and YIN CMNDF from actual measured signal.
    Spectral corroboration uses harmonic energy normalized by local spectral maximum;
    it does not count a peak as independent proof or infer candidate by dividing F0.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    rms = float(np.sqrt(np.mean(x*x)))
    if rms < 1e-6 or n < 64:
        return rms, []
    first = max(2, int(math.floor(fs/max_hz)))
    last = min(n-2, int(math.ceil(fs/min_hz)))
    if last <= first:
        return rms, []
    # FFT convolution O(n log n), unlike repeated direct np.correlate.
    corr = fftconvolve(x, x[::-1], mode='full')[n-1:n+last]
    cumulative = np.concatenate(([0.], np.cumsum(x*x)))
    lags = np.arange(last+1)
    left = cumulative[n-lags]
    right = cumulative[n] - cumulative[lags]
    denom = np.sqrt(np.maximum(left*right, 0.))
    acf = np.divide(corr, denom, out=np.full(last+1, -1.), where=denom > 1e-15)
    difference = np.maximum(left+right-2*corr, 0.)
    sums = np.cumsum(difference[1:])
    cmndf = np.ones(last+1)
    cmndf[1:] = np.divide(difference[1:]*lags[1:], sums,
                          out=np.ones(last), where=sums > 1e-15)
    peaks, _ = find_peaks(acf[first:last+1])
    peaks = (peaks+first).tolist()
    if acf[first] > acf[first+1]: peaks.append(first)
    if acf[last] > acf[last-1]: peaks.append(last)
    # Retain moderately strong observed peaks for competition; do not manufacture octaves.
    peaks = [p for p in peaks if acf[p] >= .35 and cmndf[p] <= .50]
    if not peaks:
        return rms, []
    spec = np.abs(np.fft.rfft(x*np.hanning(n), n=max(4096, 1 << (4*n-1).bit_length())))
    bins = np.fft.rfftfreq((len(spec)-1)*2, 1/fs)
    peak_ids, _ = find_peaks(spec)
    peak_ids = peak_ids[(bins[peak_ids] >= min_hz) & (bins[peak_ids] <= min(5000.,fs/2))]
    if len(peak_ids):
        maxmag = max(float(np.max(spec[peak_ids])), 1e-12)
    else:
        maxmag = 1e-12
    out = []
    for lag in peaks:
        hz = fs/lag
        # Distinct, actual FFT peaks near each harmonic. Harmonics alone don't certify F0.
        # Fourier resolution floor fs/n prevents artificial certainty from zero padding.
        coverage = 0
        for k in range(1, 7):
            target = k*hz
            if target > min(5000., fs/2): break
            tolerance = max(fs/n, target*.028)
            match = peak_ids[np.abs(bins[peak_ids]-target) <= tolerance]
            if len(match) and float(np.max(spec[match])) >= .06*maxmag:
                coverage += 1
        out.append(dict(hz=float(hz), lag=int(lag), acf=float(acf[lag]),
                        cmndf=float(cmndf[lag]), coverage=int(coverage)))
    out.sort(key=lambda c: (-c['acf'],c['cmndf']))
    return rms, out[:max_candidates]

def credible(c):
    return (c['acf'] >= .70 and c['cmndf'] <= .25 and c['coverage'] >= 2) or (c['acf'] >= .90 and c['cmndf'] <= .10 and c['coverage'] >= 1)

def select(cands, fs, n, cycles):
    """Explicitly conservative descriptive heuristic, not v3 selector replacement."""
    good = [c for c in cands if credible(c)]
    if not good:
        return None, 'no_credible_candidate'
    # First require sufficient directly observed cycles; never pick low-frequency result
    # just because a long-lag ACF is slightly stronger.
    enough = [c for c in good if n/c['lag'] >= cycles]
    if not enough:
        return None, 'insufficient_cycles'
    # Rank independent measured short period over its submultiples only when spectral
    # family corroborates and its ACF / YIN evidence is competitive.
    best_acf = max(c['acf'] for c in enough)
    competitive = [c for c in enough if c['acf'] >= best_acf-.045]
    competitive.sort(key=lambda c: (-c['coverage'], -c['hz'], -c['acf']))
    first = competitive[0]
    other = [c for c in competitive[1:] if abs(1200*math.log2(c['hz']/first['hz'])) > 70]
    if other and first['coverage'] <= other[0]['coverage']:
        return None, 'competing_families'
    return first, 'candidate_observed'

def read_baseline(path):
    if not path.is_file(): return None
    data = []
    with path.open(newline='') as f:
        r=csv.DictReader(f)
        needed={'sample','voiced_samples','median_f0_hz'}
        if not needed.issubset(r.fieldnames or []):
            raise ValueError(f'Unexpected baseline schema: {path}: {r.fieldnames}')
        for row in r:
            hz = row['median_f0_hz'].strip()
            data.append((int(row['sample']), float(hz) if hz else None, int(row['voiced_samples'])))
    return data

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tests-dir', type=Path, default=Path('tests'))
    ap.add_argument('--baseline-dir', type=Path, default=Path('tests/robustness_v3'))
    ap.add_argument('--out-dir', type=Path, default=Path('tests/adaptive_evidence'))
    ap.add_argument('--hop-ms', type=float, default=10.)
    ap.add_argument('--cycles', type=float, default=6.)
    ap.add_argument('--case', choices=('all',*NAMES), default='all')
    ap.add_argument('--hotspots-only', action='store_true', help='Fast diagnostic, not full four-song verification')
    args=ap.parse_args()
    if not 2 <= args.hop_ms <= 25 or not 3 <= args.cycles <= 12:
        ap.error('Require 2 <= hop-ms <= 25; 3 <= cycles <= 12')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    names=NAMES if args.case=='all' else (args.case,)
    manifest={'parameters':dict(hop_ms=args.hop_ms,cycles=args.cycles,windows_ms=SIZES,hotspots_only=args.hotspots_only),
              'midi_used':False,'production_modified':False,'files':[]}
    for name in names:
        candidates=[args.tests_dir/(name+e) for e in ('.wav','.wa') if (args.tests_dir/(name+e)).is_file()]
        if len(candidates)!=1:
            raise SystemExit(f'{name}: expected exactly one WAV .wav/.wa, got {len(candidates)}')
        path=candidates[0]
        t_start=time.perf_counter()
        audio,fs=sf.read(path,dtype='float64')
        if audio.ndim!=1: raise SystemExit(f'{name}: mono WAV required')
        baseline_file=args.baseline_dir/f'{name}_v3_frames.csv'
        baseline=read_baseline(baseline_file)
        baseline_samples=[b[0] for b in baseline] if baseline else []
        hop=max(1,round(fs*args.hop_ms/1000))
        max_win=round(fs*max(SIZES)/1000)
        centers=range(max_win//2,len(audio)-max_win//2,hop)
        rows=[]
        counts={}
        for center in centers:
            t=center/fs
            if args.hotspots_only and not any(a<=t<=b for a,b in HOTSPOTS[name]): continue
            probe=None; chosen=None; why=''; chosen_ms=SIZES[0]; current=[]; rms=0.
            for ms in SIZES:
                n=round(fs*ms/1000)
                left=center-n//2
                rms,current=analyze(audio[left:left+n],fs)
                chosen,why=select(current,fs,n,args.cycles)
                chosen_ms=ms
                if ms==SIZES[0]: probe=rms
                # Only expand on insufficient/ambiguous acoustic evidence.
                if chosen is not None: break
            # Independent short-window stationarity diagnostic, not a correction.
            n=round(fs*chosen_ms/1000)
            left=center-n//2
            piece=audio[left:left+n]
            half=n//2
            _,early=analyze(piece[:half],fs,max_candidates=5)
            _,late=analyze(piece[half:],fs,max_candidates=5)
            e,er=select(early,fs,half,max(3.,args.cycles/2))
            l,lr=select(late,fs,n-half,max(3.,args.cycles/2))
            e_hz=e['hz'] if e else None
            l_hz=l['hz'] if l else None
            nonstat=int(e_hz is not None and l_hz is not None and abs(1200*math.log2(e_hz/l_hz))>=100)
            # If divergent halves, the combined window is not treated as stationary.
            if nonstat: chosen=None; why='nonstationary_early_late'
            b_hz=None; b_voiced=None
            if baseline:
                ix=bisect.bisect_right(baseline_samples,center)-1
                if ix>=0: _,b_hz,b_voiced=baseline[ix]
            hz=chosen['hz'] if chosen else None
            delta=1200*math.log2(hz/b_hz) if hz and b_hz and b_voiced else None
            if baseline is None: comparison='baseline_missing'
            elif hz is None and b_hz is None: comparison='both_unresolved'
            elif hz is None: comparison='adaptive_unresolved'
            elif b_hz is None: comparison='baseline_unvoiced'
            elif abs(delta)>=100: comparison='pitch_disagreement_>=1st'
            else: comparison='within_1st'
            status='observed_candidate' if hz else 'unresolved'
            counts[comparison]=counts.get(comparison,0)+1
            rows.append(dict(name=name,time_s=f'{t:.8f}',probe_rms=f'{probe:.8g}',window_ms=chosen_ms,
                             expanded=int(chosen_ms>SIZES[0]),reason=why,status=status,
                             observed_hz=f'{hz:.5f}' if hz else '',acf=f'{chosen["acf"]:.5f}' if chosen else '',
                             cmndf=f'{chosen["cmndf"]:.5f}' if chosen else '',spectral_coverage=chosen['coverage'] if chosen else '',
                             candidate_count=len(current),candidates=json.dumps(current[:6],separators=(',',':')),
                             early_hz=f'{e_hz:.5f}' if e_hz else '',late_hz=f'{l_hz:.5f}' if l_hz else '',
                             nonstationary=nonstat,baseline_hz=f'{b_hz:.5f}' if b_hz else '',
                             baseline_voiced=b_voiced if b_voiced is not None else '',
                             difference_semitones=f'{delta:.4f}' if delta is not None else '',comparison=comparison))
        csv_path=args.out_dir/f'{name}_adaptive.csv'
        with csv_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
        elapsed=time.perf_counter()-t_start
        record={'name':name,'wav':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'sample_rate':fs,'samples':len(audio),'duration_s':len(audio)/fs,'baseline':str(baseline_file) if baseline else None,
                'observations':len(rows),'elapsed_s':elapsed,'observations_per_s':len(rows)/elapsed if elapsed else None,
                'expanded':sum(r['expanded'] for r in rows),'unresolved':sum(r['status']=='unresolved' for r in rows),
                'nonstationary':sum(r['nonstationary'] for r in rows),'comparison_counts':counts,'output':str(csv_path)}
        manifest['files'].append(record)
        print(f'{name}: {len(rows)} probes; expanded {record["expanded"]}; unresolved {record["unresolved"]}; '
              f'nonstationary {record["nonstationary"]}; runtime {elapsed:.1f}s; {csv_path}',flush=True)
    (args.out_dir/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('NOTE: reports are observational hypotheses, not correctness or reverb scores; no ground truth read.')

if __name__=='__main__': main()
