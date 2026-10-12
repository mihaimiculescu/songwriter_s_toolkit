#!/usr/bin/env python3
"""Read-only four-song acoustic aperture search. No MIDI, production edits or F0 application.

Place alongside tests/diagnose_adaptive_evidence.py. Run from repository root:
  python tests/binocular_evidence_search_v3.py --case all

Evidence categories and stopping rules are experimental, not voice ground truth.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_adaptive_evidence import NAMES, analyze, credible, select

SIZES = (24., 40., 64., 96.)
REGIONS = {
    'RATATA': [('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),
               ('djuvv_3',49.230,49.743),('submultiple',42.915,43.065)],
    'Ochiitai': [('transition',42.174,42.413)],
    'Trandafiri': [('tsis_steady_i_NOT_glissando',10.724,10.921)],
    'PREDESTINATI': [('control_38',37.984,38.144),('control_53',53.542,53.752)]
}

def extract(audio, fs, center, ms):
    n = max(64, round(fs * ms / 1000.))
    lo = center - n // 2
    hi = lo + n
    return (None if lo < 0 or hi > len(audio) else audio[lo:hi], lo, hi)

def period_support(audio, fs, center, ms, hz, min_cycles=3.):
    """Test a *measured* period, even if its original selector rejected it."""
    segment, lo, hi = extract(audio, fs, center, ms)
    result = {'start_sample':lo,'end_sample':hi,'cycles':None,'acf':None,
              'fundamental_fraction':None,'rms':None,'testability':'missing'}
    if segment is None or hz is None or not np.isfinite(hz) or hz <= 0:
        return result
    y = np.asarray(segment, dtype=float) - float(np.mean(segment))
    n = len(y)
    power = float(y @ y)
    result['rms'] = math.sqrt(power/n)
    result['cycles'] = n*hz/fs
    if power < 1e-18:
        result['testability'] = 'silent'
        return result
    lag = round(fs/hz)
    if lag < 2 or lag >= n-2:
        result['testability'] = 'lag_not_testable'
        return result
    u,v=y[:-lag],y[lag:]
    denominator=math.sqrt(float(u@u)*float(v@v))
    if denominator < 1e-15:
        result['testability'] = 'overlap_energy_insufficient'
        return result
    result['acf'] = float(u@v)/denominator
    angle = 2*np.pi*hz*np.arange(n)/fs
    basis=np.column_stack((np.cos(angle),np.sin(angle)))
    fit=basis@np.linalg.lstsq(basis,y,rcond=None)[0]
    result['fundamental_fraction']=float(np.clip(float(fit@fit)/power,0.,1.))
    result['testability']='testable' if result['cycles']>=min_cycles else 'insufficient_cycles'
    return result

def audit_candidate(audio, fs, center, ms, candidate, local_ms):
    hz=candidate['hz']
    target=period_support(audio,fs,center,local_ms,hz)
    # Neighbouring probes are disjoint from the central probe by construction.
    offset=round(fs*local_ms/1000.)
    earlier=period_support(audio,fs,center-offset,local_ms,hz)
    later=period_support(audio,fs,center+offset,local_ms,hz)
    whole=period_support(audio,fs,center,ms,hz)
    present_ok=(target['testability']=='testable' and target['acf'] is not None
                and target['acf']>=.70 and target['fundamental_fraction']>=.03)
    distant_ok=any(p['testability']=='testable' and p['acf'] is not None
                   and p['acf']>=.70 and p['fundamental_fraction']>=.03
                   for p in (earlier,later))
    if present_ok: attribution='local_support_exploratory'
    elif target['testability']!='testable': attribution='local_inconclusive_'+target['testability']
    elif distant_ok and whole['acf'] is not None and whole['acf']>=.70:
        attribution='possible_distant_only_support'
    else: attribution='local_not_corroborated'
    return {'candidate':candidate,'eligible':credible(candidate),'cycles_full':ms*hz/1000.,
            'present':target,'earlier':earlier,'later':later,'whole':whole,
            'attribution':attribution}

def observe(audio,fs,center,local_ms,cycle_target):
    stages=[]
    stop='max_aperture_inconclusive'
    selected=None
    previous=None
    for ms in SIZES:
        segment,lo,hi=extract(audio,fs,center,ms)
        if segment is None: break
        rms,candidates=analyze(segment,fs,max_candidates=8)
        winner,selector_reason=select(candidates,fs,len(segment),cycle_target)
        audits=[audit_candidate(audio,fs,center,ms,c,local_ms) for c in candidates]
        locally_supported=[a for a in audits if a['attribution']=='local_support_exploratory']
        eligible_local=[a for a in locally_supported if a['eligible'] and a['cycles_full']>=cycle_target]
        distant_only=[a for a in audits if a['attribution']=='possible_distant_only_support']
        insufficient=[a for a in audits if a['present']['testability']=='insufficient_cycles']
        # Only a measured candidate is tested. The selector is informational; it does
        # not authorize us to convert a rejected candidate into a final F0.
        stage={'aperture_ms':ms,'start_sample':lo,'end_sample':hi,
               'start_s':lo/fs,'end_s':hi/fs,'rms':rms,
               'raw_candidate_count':len(candidates),'raw_candidates':audits,
               'selector_hz':winner['hz'] if winner else None,
               'selector_reason':selector_reason,'local_raw_count':len(locally_supported),
               'eligible_local_count':len(eligible_local),'distant_only_count':len(distant_only),
               'insufficient_local_cycles_count':len(insufficient),'decision':None}
        stages.append(stage)
        # Report local support without silently picking one of several competing pitches.
        if len(eligible_local)==1 and winner is not None and abs(1200*math.log2(
                eligible_local[0]['candidate']['hz']/winner['hz']))<=70:
            stage['decision']='stop_one_selector_consistent_locally_supported_candidate'
            stop=stage['decision']
            selected=winner['hz']  # observational only
            break
        if distant_only and not locally_supported:
            stage['decision']='stop_distant_only_periodicity_conflict'
            stop=stage['decision']
            break
        if len(eligible_local)>1:
            stage['decision']='stop_competing_local_candidates_unresolved'
            stop=stage['decision']
            break
        if previous is not None:
            old={round(a['candidate']['hz'],1):a for a in previous['raw_candidates']}
            # Same-period comparison only; a different candidate never counts as gain.
            comparisons=[]
            for a in audits:
                p=old.get(round(a['candidate']['hz'],1))
                if p and a['present']['acf'] is not None and p['present']['acf'] is not None:
                    comparisons.append({'hz':a['candidate']['hz'],
                        'local_acf_delta':a['present']['acf']-p['present']['acf'],
                        'full_acf_delta':a['candidate']['acf']-p['candidate']['acf']})
            stage['same_period_gain']=comparisons
            # Stop if the larger window adds only remote evidence, not present support.
            if any(c['full_acf_delta']>.04 and c['local_acf_delta']<.02 for c in comparisons) and not locally_supported:
                stage['decision']='stop_full_window_gain_without_local_gain'
                stop=stage['decision']
                break
        if ms==SIZES[-1]:
            stage['decision']='stop_max_aperture'
            stop=stage['decision']
            break
        if not candidates:
            stage['decision']='expand_no_raw_period_exploratory'
        elif insufficient:
            stage['decision']='expand_raw_period_local_cycle_deficit'
        elif locally_supported:
            stage['decision']='expand_local_period_not_eligible_or_not_unique'
        else:
            stage['decision']='expand_raw_period_without_local_confirmation'
        previous=stage
    return {'time_s':center/fs,'sample':center,'stop_reason':stop,
            'observational_hz':selected,'status':'local_candidate_observed_not_final_f0' if selected is not None else 'unresolved',
            'stages_json':json.dumps(stages,allow_nan=False),'f0_applied':False,
            'voicing_applied':False,'tracker_reset_requested':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*NAMES),default='all')
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/binocular_evidence_search_v3'))
    p.add_argument('--regions-only',action='store_true',help='Fast inspection only; not golden-standard validation')
    p.add_argument('--hop-ms',type=float,default=10.)
    p.add_argument('--local-ms',type=float,default=24.)
    p.add_argument('--cycles',type=float,default=6.)
    args=p.parse_args()
    if not (2<=args.hop_ms<=25 and 12<=args.local_ms<=40 and 3<=args.cycles<=12):
        p.error('Require hop 2-25 ms, local 12-40 ms, cycles 3-12')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'binocular_evidence_search_v3','read_only':True,'midi_used':False,
              'production_modified':False,'gold_standard':not args.regions_only,
              'warning':'Heuristic local acoustic evidence is NOT final pitch or voicing ground truth.',
              'trandafiri':'steady i between consonants; NOT a glissando',
              'regions':REGIONS,'cases':{}}
    for name in (NAMES if args.case=='all' else (args.case,)):
        files=[args.wav_dir/(name+ext) for ext in ('.wav','.wa') if (args.wav_dir/(name+ext)).is_file()]
        if len(files)!=1:p.error(f'{name}: expected one WAV in {args.wav_dir}, found {files}')
        audio,fs=sf.read(files[0],dtype='float64')
        if audio.ndim!=1:p.error(f'{name}: mono required')
        hop=max(1,round(args.hop_ms*fs/1000))
        margin=round(fs*.15)
        if args.regions_only:
            centers=sorted({round(t*fs) for _,a,b in REGIONS.get(name,[])
                            for t in np.arange(a,b+1e-9,args.hop_ms/1000.)})
        else:centers=range(margin,len(audio)-margin,hop)
        rows=[];count=Counter();region_counts=Counter();start=time.perf_counter()
        for center in centers:
            if center<margin or center>=len(audio)-margin:continue
            row=observe(audio,fs,center,args.local_ms,args.cycles)
            labels=[label for label,a,b in REGIONS.get(name,[]) if a<=center/fs<=b]
            row.update(case=name,region=';'.join(labels))
            rows.append(row);count[row['stop_reason']]+=1
            for label in labels:region_counts[(label,row['stop_reason'])]+=1
        dest=args.out_dir/(name+'.csv')
        with dest.open('w',newline='') as fh:
            cols=('case','region','sample','time_s','stop_reason','status','observational_hz',
                  'stages_json','f0_applied','voicing_applied','tracker_reset_requested')
            w=csv.DictWriter(fh,fieldnames=cols);w.writeheader();w.writerows(rows)
        manifest['cases'][name]={'wav':str(files[0]),'sample_rate':fs,'observations':len(rows),
                                 'stop_reasons':dict(count),'region_reasons':
                                 {f'{k[0]}::{k[1]}':v for k,v in region_counts.items()},
                                 'elapsed_s':time.perf_counter()-start,'output':str(dest)}
        print(name,len(rows),dict(count),flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: no F0, voicing, or resets applied.')
if __name__=='__main__':main()
