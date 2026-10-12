#!/usr/bin/env python3
"""Read-only binocular aperture attribution audit; never estimates final F0/voicing.

Put beside tests/diagnose_adaptive_evidence.py and run from repository root.
All time windows are explicitly centered; each tested frequency MUST be measured
independently in its own aperture by the existing acoustic analyzer.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf

sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_adaptive_evidence import analyze, select, NAMES

SIZES=(24.,40.,64.,96.)
REGIONS={
 'RATATA':[('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.915,43.065)],
 'Ochiitai':[('transition',42.174,42.413)],
 'Trandafiri':[('tsis_steady_i_between_consonants',10.724,10.921)],
 'PREDESTINATI':[('control_38',37.984,38.144),('control_53',53.542,53.752)],
}

def window(x,fs,t,ms):
    n=round(fs*ms/1000); middle=round(fs*t); lo=middle-n//2; hi=lo+n
    if lo<0 or hi>len(x):return None,lo/fs,hi/fs
    return x[lo:hi],lo/fs,hi/fs

def measure(x,fs,t,ms,cycles):
    z,a,b=window(x,fs,t,ms)
    if z is None:return dict(start_s=a,end_s=b,ms=ms,hz=None,reason='edge',rms=None,candidates=[])
    rms,cands=analyze(z,fs)
    winner,reason=select(cands,fs,len(z),cycles)
    return dict(start_s=a,end_s=b,ms=ms,hz=None if winner is None else winner['hz'],
                reason=reason,rms=rms,candidates=cands)

def support(x,fs,t,ms,hz,min_cycles=3.):
    z,a,b=window(x,fs,t,ms)
    d=dict(start_s=a,end_s=b,ms=ms,hz_tested=hz,cycles=None,acf=None,
           rms=None,fundamental_fraction=None,testability='no_candidate')
    if hz is None:return d
    if z is None:d['testability']='edge';return d
    y=np.asarray(z,dtype=float)-float(np.mean(z));n=len(y);power=float(y@y)
    d['cycles']=n*hz/fs;d['rms']=math.sqrt(power/n)
    if power<1e-20:d['testability']='silent';return d
    lag=round(fs/hz)
    if lag<1 or lag>=n-2:d['testability']='lag_not_testable';return d
    u,v=y[:-lag],y[lag:];den=math.sqrt(float(u@u)*float(v@v))
    if den<=1e-20:d['testability']='zero_overlap_energy';return d
    d['acf']=float(u@v)/den
    phase=2*np.pi*hz*np.arange(n)/fs
    B=np.column_stack((np.cos(phase),np.sin(phase)))
    coef=np.linalg.lstsq(B,y,rcond=None)[0]; fit=B@coef
    d['fundamental_fraction']=float((fit@fit)/power)
    d['testability']='testable' if d['cycles']>=min_cycles else 'insufficient_cycles'
    return d

def local_class(e):
    if e['testability']!='testable':return 'inconclusive_'+e['testability']
    if e['acf']>=.70 and e['fundamental_fraction']>=.03:return 'supported_exploratory'
    return 'weak_evidence'

def at_time(audio,fs,t,look_ms,offset_ms,cycles,local_ms):
    future=measure(audio,fs,t+offset_ms/1000,look_ms,3.)
    if future['hz'] is None:
        initial=SIZES[0];initial_reason='lookahead_inconclusive_short_start'
    else:
        needed=1000*cycles/future['hz']
        initial=next((ms for ms in SIZES if ms>=needed),SIZES[-1])
        initial_reason='lookahead_measured_period_cycle_budget'
    trace=[]; accepted=None;stop='all_apertures_exhausted'
    # Inspect every level up to initial, preserving provenance. Do not treat absent
    # lookahead evidence as permission to expand blindly.
    for ms in SIZES:
        m=measure(audio,fs,t,ms,cycles)
        hz=m['hz']
        center=support(audio,fs,t,local_ms,hz)
        left=support(audio,fs,t-local_ms/1000,local_ms,hz)
        right=support(audio,fs,t+local_ms/1000,local_ms,hz)
        whole=support(audio,fs,t,ms,hz)
        # Overlapping probes are evidence LOCATION, not independent confirmations.
        cclass=local_class(center)
        leftclass=local_class(left);rightclass=local_class(right)
        measured=hz is not None
        prior=trace[-1] if trace else None
        change=None
        if prior and prior['hz'] and hz:
            change=1200*math.log2(hz/prior['hz'])
        record=dict(ms=ms,start_s=m['start_s'],end_s=m['end_s'],hz=hz,
                    measurement_reason=m['reason'],rms=m['rms'],
                    candidates=m['candidates'][:6],center=center,left=left,right=right,whole=whole,
                    center_class=cclass,left_class=leftclass,right_class=rightclass,
                    change_from_previous_cents=change,decision='not_evaluated')
        trace.append(record)
        # If locally confirmed at target, never inspect longer: no needless look-back.
        if ms<initial:
            record['decision']='below_lookahead_cycle_budget_observation_only'
            continue
        if measured and cclass=='supported_exploratory':
            accepted=record;record['decision']='stop_local_support_observed';stop=record['decision'];break
        # A longer aperture cannot repair poor support at the PRESENT by importing
        # voiced content from neighboring time. This is a measurement stop, not unvoicing.
        if measured and center['testability']=='testable' and cclass=='weak_evidence':
            neighbors=(leftclass=='supported_exploratory' or rightclass=='supported_exploratory')
            if neighbors and whole['acf'] is not None and whole['acf']>=.70:
                record['decision']='stop_temporal_attribution_conflict'
                stop=record['decision'];break
        if prior and measured and prior['hz'] is not None and change is not None and abs(change)>150:
            if cclass!='supported_exploratory':
                record['decision']='stop_aperture_candidate_instability'
                stop=record['decision'];break
        if ms==SIZES[-1]:record['decision']='stop_max_aperture';stop=record['decision'];break
        if not measured:
            record['decision']='expand_investigative_only_not_pitch'
        elif center['testability']!='testable':
            record['decision']='expand_center_insufficient_cycles'
        else:
            record['decision']='expand_no_local_confirmation_exploratory'
    chosen=accepted or trace[-1]
    status=('locally_supported_not_voicing_proof' if accepted else
            'unresolved_temporal_conflict' if 'conflict' in stop else 'unresolved_no_local_confirmation')
    return dict(time_s=t,lookahead_start_s=future['start_s'],lookahead_end_s=future['end_s'],
                lookahead_hz=future['hz'],lookahead_reason=future['reason'],
                initial_aperture_ms=initial,initial_reason=initial_reason,
                final_inspected_aperture_ms=chosen['ms'],observational_hz=chosen['hz'],
                stop_reason=stop,status=status,stages_json=json.dumps(trace,allow_nan=False),
                f0_applied=False,voicing_applied=False,tracker_reset_requested=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('all',*NAMES),default='all')
    ap.add_argument('--wav-dir',type=Path,default=Path('tests'))
    ap.add_argument('--out-dir',type=Path,default=Path('tests/binocular_attribution_v2'))
    ap.add_argument('--regions-only',action='store_true')
    ap.add_argument('--hop-ms',type=float,default=10.)
    ap.add_argument('--lookahead-ms',type=float,default=24.)
    ap.add_argument('--lookahead-offset-ms',type=float,default=12.)
    ap.add_argument('--local-ms',type=float,default=24.)
    ap.add_argument('--cycles',type=float,default=6.)
    a=ap.parse_args()
    if not (2<=a.hop_ms<=25 and 12<=a.lookahead_ms<=40 and 0<=a.lookahead_offset_ms<=48 and 12<=a.local_ms<=40 and 3<=a.cycles<=12):ap.error('Invalid parameters')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'read_only':True,'no_midi':True,'production_modified':False,
              'warning':'Exploratory acoustic categories are not vocal ground truth or final F0',
              'trandafiri_annotation':'steady i between consonants; NOT glissando', 'cases':{}}
    for name in (NAMES if a.case=='all' else (a.case,)):
        path=a.wav_dir/(name+'.wav')
        if not path.is_file():ap.error(f'Missing WAV: {path}')
        audio,fs=sf.read(path,dtype='float64',always_2d=False)
        if audio.ndim!=1:ap.error(f'Mono required: {path}')
        hop=max(1,round(fs*a.hop_ms/1000)); margin=round(fs*.15)
        if a.regions_only:
            stamps=sorted(set(round(t*fs) for _,lo,hi in REGIONS.get(name,[]) for t in np.arange(lo,hi+1e-9,a.hop_ms/1000)))
        else:stamps=range(margin,len(audio)-margin,hop)
        rows=[]; counts=Counter()
        for sample in stamps:
            if sample<margin or sample>=len(audio)-margin:continue
            t=sample/fs
            r=at_time(audio,fs,t,a.lookahead_ms,a.lookahead_offset_ms,a.cycles,a.local_ms)
            labels=[label for label,lo,hi in REGIONS.get(name,[]) if lo<=t<=hi]
            r['case']=name;r['region']=';'.join(labels);r['sample']=sample
            counts[r['stop_reason']]+=1;rows.append(r)
        dest=a.out_dir/(name+'.csv')
        if rows:
            with dest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        manifest['cases'][name]={'wav':str(path),'sample_rate':fs,'observations':len(rows),'stop_reasons':dict(counts)}
        print(name,len(rows),dict(counts),flush=True)
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: no production F0, voicing or resets applied.')
if __name__=='__main__':main()
