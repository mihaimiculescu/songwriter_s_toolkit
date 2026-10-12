#!/usr/bin/env python3
"""Read-only temporal attribution for the THREE correctly annotated RATATA djuvvs.

Separates (A) aperture leakage, (B) locally periodic material that production
rejects, and (C) nonlocal/unstable misleading periodicity. These are evidence
patterns, NOT source-identification or ground-truth classifications.

Inputs: tests/RATATA.wav, corrected regression RATATA_djuvv_*_timeline.csv.
Production baseline is BLOCK-AGGREGATED; sample-exact voicing boundaries CANNOT
be reconstructed from it. The CSV marks that limitation explicitly.
"""
from __future__ import annotations
import argparse, csv, json, math, hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

REGIONS = {'djuvv_1':(32.307,32.380), 'djuvv_2':(36.410,36.923),
           'djuvv_3':(44.615,45.128)}

def number(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def stats(x,fs,hz):
    """Test an EXISTING measured frequency, never invent one."""
    if hz is None or not 0<hz<fs/2 or len(x)<8:return None
    z=np.asarray(x,dtype=np.float64); z=z-z.mean()
    energy=float(z@z); rms=math.sqrt(energy/len(z))
    lag=round(fs/hz); acf=None; residual=None
    cycles=len(z)*hz/fs
    if 1<=lag<len(z)-2:
        a=z[:-lag]; b=z[lag:]; d=math.sqrt(float(a@a)*float(b@b))
        if d>1e-20:
            acf=float(a@b)/d
            residual=float((a-b)@(a-b))/max(float(a@a+b@b),1e-30)
    frac=None; amplitude=None
    if energy>1e-20:
        phases=2*np.pi*hz*np.arange(len(z))/fs
        B=np.column_stack((np.cos(phases),np.sin(phases)))
        co,*_=np.linalg.lstsq(B,z,rcond=None)
        fit=B@co
        frac=float(fit@fit)/energy
        amplitude=float(np.hypot(*co))
    return {'cycles':round(cycles,3),'rms':rms,'acf':acf,
            'normalized_period_residual':residual,
            'fundamental_variance_fraction':frac,'fundamental_amplitude':amplitude,
            'period_test_feasible':acf is not None}

def segment(audio,fs,start,end,hz):
    lo=max(0,int(round(start*fs)));hi=min(len(audio),int(round(end*fs)))
    out={'start_s':lo/fs,'end_s':hi/fs,'samples':hi-lo}
    out.update(stats(audio[lo:hi],fs,hz) or {})
    return out

def strength(e):
    """Conservative DIAGNOSTIC gate; not production voiced status."""
    return (e and e.get('cycles',0)>=3 and e.get('acf') is not None
            and e['acf']>=.70 and e.get('fundamental_variance_fraction') is not None
            and e['fundamental_variance_fraction']>=.03)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav',type=Path,default=Path('tests/RATATA.wav'))
    p.add_argument('--regression-dir',type=Path,default=Path('tests/corrected_adaptive_regression'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/djuvv_temporal_support'))
    p.add_argument('--micro-ms',type=float,default=12.,help='Independent nonoverlapping local probes')
    args=p.parse_args()
    if not 8<=args.micro_ms<=24:p.error('micro-ms must be 8..24')
    if not args.wav.is_file():p.error(f'missing {args.wav}')
    audio,fs=sf.read(args.wav,dtype='float64',always_2d=False)
    if audio.ndim!=1:p.error('mono WAV required; no automatic downmix')
    inputs={k:args.regression_dir/f'RATATA_{k}_timeline.csv' for k in REGIONS}
    for k,path in inputs.items():
        if not path.is_file():p.error(f'{k}: missing {path}')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'version':1,'read_only':True,'midi_used':False,'production_modified':False,
              'baseline_resolution':'block aggregate only, NOT sample-level status',
              'wav_sha256':hashlib.sha256(args.wav.read_bytes()).hexdigest(),
              'sample_rate':fs,'micro_ms':args.micro_ms,'regions':{}}
    dt=args.micro_ms/1000
    for key,(region_lo,region_hi) in REGIONS.items():
        with inputs[key].open(newline='',encoding='utf8') as f: source=list(csv.DictReader(f))
        output=[]; counts=Counter()
        for r in source:
            t=number(r.get('time_s'))
            if t is None or not region_lo<=t<=region_hi:continue
            hz=number(r.get('shadow_acoustic_hz')) or number(r.get('adaptive_hz'))
            if hz is None:continue
            win_ms=number(r.get('adaptive_window_ms'))
            if win_ms is None:continue
            # Previous adaptive implementation used time_s as CENTER; explicitly report this assumption.
            half=win_ms/2000
            ws,we=t-half,t+half
            center=segment(audio,fs,t-dt/2,t+dt/2,hz)
            left=segment(audio,fs,t-1.5*dt,t-.5*dt,hz)
            right=segment(audio,fs,t+.5*dt,t+1.5*dt,hz)
            full=segment(audio,fs,ws,we,hz)
            # Intersections with reported aperture: these are precisely its samples outside
            # the central probe, not imaginary independent confirmation.
            outside_left=segment(audio,fs,ws,min(we,t-dt/2),hz) if ws<t-dt/2 else None
            outside_right=segment(audio,fs,max(ws,t+dt/2),we,hz) if we>t+dt/2 else None
            neighboring=[left,right]
            outer=[e for e in (outside_left,outside_right) if e]
            center_strong=bool(strength(center))
            outer_strong=any(strength(e) for e in outer)
            neighbor_strong=sum(bool(strength(e)) for e in neighboring)
            full_strong=bool(strength(full))
            # Basing status on voiced_samples>0 would conflate a partly voiced block
            # with a voiced sample; report the original aggregate transparently.
            nvoiced=number(r.get('baseline_voiced_samples'))
            block_unvoiced=nvoiced==0 if nvoiced is not None else None
            baseline_block_mixed=nvoiced is not None and nvoiced>0
            if full_strong and not center_strong and outer_strong:
                pattern='A_aperture_support_outside_center'
            elif center_strong and block_unvoiced:
                pattern='B_center_periodicity_baseline_block_unvoiced'
            elif center_strong:
                pattern='center_periodicity_block_contains_voicing'
            elif full_strong and not center_strong:
                pattern='C_full_window_only_no_local_confirmation'
            else:
                pattern='unresolved_insufficient_or_mixed_evidence'
            # These patterns don't prove vocal origin. Probe length/phase sensitivity
            # especially relevant when a centered 12ms micro window has < 3 cycles.
            counts[pattern]+=1
            result={
                'region':key,'time_s':t,'inside_region':True,
                'candidate_hz_measured':hz,'candidate_source':(
                    'shadow' if number(r.get('shadow_acoustic_hz')) is not None else 'adaptive'),
                'adaptive_window_start_s_assumed':ws,'adaptive_window_end_s_assumed':we,
                'adaptive_window_ms':win_ms,'center_probe_ms':args.micro_ms,
                'baseline_block_start_s':number(r.get('baseline_frame_start_s')),
                'baseline_block_end_s':number(r.get('baseline_frame_end_s')),
                'baseline_block_voiced_samples':nvoiced,
                'baseline_block_contains_voicing':baseline_block_mixed,
                'baseline_block_all_unvoiced':block_unvoiced,
                'exact_production_sample_status':'UNAVAILABLE_IN_AGGREGATE_CSV',
                'pattern':pattern,'whole_strong_diagnostic':full_strong,
                'center_strong_diagnostic':center_strong,
                'outer_strong_diagnostic':outer_strong,
                'adjacent_strong_probe_count':neighbor_strong,
                'whole_json':json.dumps(full,allow_nan=False),
                'center_json':json.dumps(center,allow_nan=False),
                'left_nonoverlap_json':json.dumps(left,allow_nan=False),
                'right_nonoverlap_json':json.dumps(right,allow_nan=False),
                'aperture_outside_left_json':json.dumps(outside_left,allow_nan=False),
                'aperture_outside_right_json':json.dumps(outside_right,allow_nan=False),
            }
            output.append(result)
        if not output:raise RuntimeError(f'{key}: no measured candidate rows in annotated interval')
        path=args.out_dir/f'RATATA_{key}_temporal.csv'
        with path.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
        manifest['regions'][key]={'interval_s':[region_lo,region_hi],
                                  'input_sha256':hashlib.sha256(inputs[key].read_bytes()).hexdigest(),
                                  'candidate_observations':len(output),
                                  'pattern_counts':dict(counts),'output':str(path)}
        print(f'{key}: {len(output)} candidate observations; {dict(counts)}')
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.out_dir/'README.txt').write_text(
        'PATTERNS ARE DIAGNOSTIC, NOT TRUTH LABELS.\n'
        'A: whole aperture and outer portion support a measured candidate, central probe does not.\n'
        'B: central probe supports candidate while entire production BLOCK reports zero voiced samples.\n'
        'C: whole window passes but central and outer portions do not independently pass.\n'
        'All other patterns are explicitly reported. A vs C is not exhaustive proof.\n'
        'Fundamental energy can be weak for real harmonically rich voices. Gates are exploratory.\n'
        'Production CSV contains block voiced sample counts, not sample-level status.\n'
        'Centering of adaptive aperture is an assumption from previous scripts, not verified from raw timestamps.\n'
        '12ms probe may be too short at low candidate Hz; inspect cycles and rerun --micro-ms 16/20.\n'
        'No ground-truth labels, no MIDI, no tracker modifications, no corrected F0 output.\n')
    print('Output:',args.out_dir)
if __name__=='__main__':main()
