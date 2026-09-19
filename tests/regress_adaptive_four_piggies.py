#!/usr/bin/env python3
"""Read-only, corrected four-WAV regression comparison.

Inputs: EXISTING production v3 frames, adaptive measurements, acoustic shadow v2,
voicing/recovery review, and original mono WAV. No tracker execution, MIDI,
parameter tuning, output replacement, or continuity assumption.

All case ranges are WAV seconds; the three RATATA djuvvs are full intervals.
Reports are comparisons, NOT pitch/voicing ground-truth labels.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

CASES = {
    'RATATA': [
        ('djuvv_1', 32.307, 32.380),
        ('djuvv_2', 36.410, 36.923),
        ('djuvv_3', 44.615, 45.128),
        ('submultiple_42_965', 42.915, 43.065),
    ],
    'Ochiitai': [('transition_42_214', 42.174, 42.413)],
    'Trandafiri': [('tsis_10_774', 10.724, 10.921)],
    'PREDESTINATI': [('control_38_034', 37.984, 38.144),
                    ('control_53_592', 53.542, 53.752)],
}


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fl(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def truthy(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def lookup_at(rows, times, t, tolerance=0.0051):
    if not rows:
        return None
    i = bisect.bisect_left(times, t)
    options = [j for j in (i-1, i) if 0 <= j < len(rows)]
    if not options:
        return None
    j = min(options, key=lambda n: abs(times[n]-t))
    return rows[j] if abs(times[j]-t) <= tolerance else None


def periodic_evidence(signal, fs, candidate):
    """Measure an already reported candidate, NEVER infer a new frequency."""
    if candidate is None or candidate <= 0 or len(signal) < 8:
        return {'cycles': None, 'acf_at_candidate': None,
                'rms': None, 'projection_variance_fraction': None}
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x*x)))
    lag = int(round(fs/candidate))
    acf = None
    if 1 <= lag < len(x)-2:
        a, b = x[:-lag], x[lag:]
        denom = float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
        if denom > 0:
            acf = float(np.dot(a,b)/denom)
    fraction = None
    energy = float(np.dot(x,x))
    if energy > 0:
        t = np.arange(len(x))/fs
        basis = np.column_stack((np.cos(2*np.pi*candidate*t),
                                 np.sin(2*np.pi*candidate*t)))
        coef, *_ = np.linalg.lstsq(basis,x,rcond=None)
        modeled = basis @ coef
        fraction = float(np.dot(modeled,modeled)/energy)
    return {'cycles': round(len(x)*candidate/fs,3),
            'acf_at_candidate': acf, 'rms': rms,
            'projection_variance_fraction': fraction}


def raw_local(audio, fs, time_s, hz, duration_ms=24.):
    if hz is None:
        return {}
    n = max(8,round(duration_ms*fs/1000))
    center = round(time_s*fs)
    start = center-n//2
    if start < 0 or start+n > len(audio):
        return {}
    x = audio[start:start+n]
    thirds = np.array_split(x,3)
    return {'window_start_s':start/fs, 'window_end_s':(start+n)/fs,
            'whole':periodic_evidence(x,fs,hz),
            'thirds':[periodic_evidence(v,fs,hz) for v in thirds]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['all',*CASES], default='all')
    p.add_argument('--margin-ms', type=float,default=35,
                   help='Extra context outside each annotated interval; explicitly marked')
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--baseline-dir',type=Path,default=Path('tests/robustness_v3'))
    p.add_argument('--adaptive-dir',type=Path,default=Path('tests/adaptive_evidence'))
    p.add_argument('--shadow-dir',type=Path,default=Path('tests/shadow_acoustic_tiebreak_v2'))
    p.add_argument('--review-dir',type=Path,default=Path('tests/voicing_recovery_shadow'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/corrected_adaptive_regression'))
    a=p.parse_args()
    if a.margin_ms<0: p.error('margin-ms must be >= 0')
    names=CASES if a.case=='all' else {a.case:CASES[a.case]}
    inputs={}
    for name in names:
        wavs=[v for v in (a.wav_dir/f'{name}.wav',a.wav_dir/f'{name}.wa') if v.is_file()]
        if len(wavs)!=1: p.error(f'{name}: expected one WAV/WA in {a.wav_dir}; found {len(wavs)}')
        paths={
            'baseline':a.baseline_dir/f'{name}_v3_frames.csv',
            'adaptive':a.adaptive_dir/f'{name}_adaptive.csv',
            'shadow':a.shadow_dir/f'{name}_comparison.csv',
            'review':a.review_dir/f'{name}_review.csv',
        }
        for kind,path in paths.items():
            if not path.is_file(): p.error(f'{name}: missing {kind}: {path}')
        inputs[name]=(wavs[0],paths)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'version':1,'read_only':True,'midi_used':False,'production_modified':False,
              'pitch_or_status_changed':False,'note':'No classification is ground truth.',
              'djuvv_intervals_s':CASES['RATATA'][:3], 'cases':{}}
    for name,(wav,paths) in inputs.items():
        audio,fs=sf.read(wav,dtype='float64',always_2d=False)
        if audio.ndim!=1: p.error(f'{wav}: expected mono; no automatic downmix')
        baseline=read_csv(paths['baseline'])
        adaptive=read_csv(paths['adaptive'])
        shadow=read_csv(paths['shadow'])
        review=read_csv(paths['review'])
        for rows in (baseline,adaptive,shadow,review):
            rows.sort(key=lambda r: float(r['time_s']))
        bt=[float(r['time_s']) for r in baseline]
        st=[float(r['time_s']) for r in shadow]
        rt=[float(r['time_s']) for r in review]
        # Block interval is derived from actual adjacent baseline sample indices.
        blocks=[]
        for i,r in enumerate(baseline):
            sample=int(r['sample'])
            block=(int(baseline[i+1]['sample'])-sample if i+1<len(baseline)
                   else int(baseline[i]['sample'])-int(baseline[i-1]['sample']) if i else 2048)
            if block<=0: raise ValueError(f'{name}: nonmonotone baseline sample positions')
            blocks.append((sample/fs,(sample+block)/fs,r))
        starts=[b[0] for b in blocks]
        all_inside=[]
        summaries=[]
        for label,lo,hi in CASES[name]:
            margin=a.margin_ms/1000
            selected=[r for r in adaptive if lo-margin<=float(r['time_s'])<=hi+margin]
            rows=[]; counts=Counter()
            for r in selected:
                t=float(r['time_s'])
                idx=bisect.bisect_right(starts,t)-1
                b=blocks[idx][2] if idx>=0 and t<blocks[idx][1] else None
                sh=lookup_at(shadow,st,t)
                rev=lookup_at(review,rt,t)
                bz=bool(b and int(b['voiced_samples'])>0)
                bfreq=fl(b['median_f0_hz']) if bz else None
                ahz=fl(r['observed_hz'])
                shz=fl(sh['acoustic_only_hz']) if sh else None
                in_interval=lo<=t<=hi
                counts['inside' if in_interval else 'context']+=1
                if in_interval:
                    counts['baseline_voiced' if bz else 'baseline_unvoiced']+=1
                    counts['adaptive_candidate' if ahz is not None else 'adaptive_unresolved']+=1
                    counts['shadow_candidate' if shz is not None else 'shadow_abstained']+=1
                    if sh and truthy(sh['state_disagreement_review']): counts['state_review']+=1
                    if sh and truthy(sh['baseline_unvoiced_review']): counts['unvoiced_review']+=1
                row={
                    'case':name,'region':label,'time_s':t,
                    'region_start_s':lo,'region_end_s':hi,
                    'within_annotated_interval':int(in_interval),
                    'baseline_frame_start_s':float(b['time_s']) if b else None,
                    'baseline_frame_end_s':blocks[idx][1] if b else None,
                    'baseline_voiced_samples':int(b['voiced_samples']) if b else None,
                    'baseline_status':'voiced_samples_present' if bz else 'no_voiced_samples',
                    'baseline_median_hz':bfreq,
                    'baseline_first_hz':fl(b['first_f0_hz']) if b else None,
                    'baseline_last_hz':fl(b['last_f0_hz']) if b else None,
                    'adaptive_status':r['status'],'adaptive_reason':r['reason'],
                    'adaptive_hz':ahz,'adaptive_window_ms':fl(r['window_ms']),
                    'adaptive_probe_rms':fl(r['probe_rms']),
                    'adaptive_acf':fl(r['acf']),'adaptive_cmndf':fl(r['cmndf']),
                    'adaptive_candidates_json':r['candidates'],
                    'shadow_acoustic_hz':shz,
                    'shadow_reason':sh['acoustic_reason'] if sh else 'NO_ALIGNED_SHADOW',
                    'shadow_curve_preference_hz':fl(sh['curve_preference_hz']) if sh else None,
                    'shadow_state_review':int(truthy(sh['state_disagreement_review'])) if sh else None,
                    'shadow_unvoiced_review':int(truthy(sh['baseline_unvoiced_review'])) if sh else None,
                    'shadow_measured_candidates_json':sh['measured_candidates'] if sh else '',
                    'review_category':rev['category'] if rev else '',
                    'review_candidate_evidence_json':rev['candidate_evidence'] if rev else '',
                    'review_state_evidence_json':rev['state_evidence'] if rev else '',
                    'raw_candidate_24ms_json':json.dumps(raw_local(audio,fs,t,shz),allow_nan=False),
                }
                rows.append(row)
            if not rows: raise RuntimeError(f'{name}/{label}: no adaptive observations in interval')
            out=a.out_dir/f'{name}_{label}_timeline.csv'
            with out.open('w',newline='',encoding='utf-8') as f:
                wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
            all_inside+= [r for r in rows if r['within_annotated_interval']]
            summaries.append({'label':label,'interval_s':[lo,hi],
                              'context_margin_ms':a.margin_ms,'counts':dict(counts),
                              'output':str(out)})
            print(f'{name} {label}: {counts["inside"]} inside, {counts["context"]} context; '
                  f'{counts["baseline_voiced"]} baseline voiced, '
                  f'{counts["adaptive_candidate"]} adaptive candidates')
        manifest['cases'][name]={'wav':str(wav),'fs':fs,
             'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
             'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
             'regions':summaries}
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (a.out_dir/'README.txt').write_text(
      'CORRECTED WAV-ONLY REGRESSION. RATATA djuvvs: 32.307-32.380, 36.410-36.923, '
      '44.615-45.128 seconds. Timeline is adaptive 10ms grid; baseline is matched by '
      'actual production sample-block containment, NOT nearest frame. Other auxiliary '
      'CSV measurements are nearest-time aligned only if within 5.1ms. '
      'Window sizes and supports differ; comparisons do NOT measure accuracy. '
      'Raw candidate 24ms diagnostics use the reported shadow candidate without '
      'inventing a pitch. Tripartite windows may have insufficient cycles; '
      'neither their values nor overlapping measurements are independent votes. '
      'within_annotated_interval distinguishes actual djuvv from context. '
      'Baseline no-voiced-samples is not ground-truth silence. '
      'No MIDI, no changes to F0/status/production.\n',encoding='utf-8')
    print('Outputs:',a.out_dir)

if __name__=='__main__': main()
