#!/usr/bin/env python3
"""Read-only full-song audit of existing binocular_evidence_search_v3 output.

Run from repository root:
  python tests/audit_incremental_evidence_v4.py --case all

Requires v3 CSVs and original WAVs. Does NOT run or modify the tracker, create F0,
accept voicing, enforce continuity, or use MIDI. Proposed early stops are audited
counterfactually against later *already-measured* v3 stages, never applied.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

NAMES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
REGIONS = {
    'RATATA': [('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),
               ('djuvv_3',49.230,49.743),('submultiple',42.915,43.065)],
    'Ochiitai': [('transition',42.174,42.413)],
    'Trandafiri': [('tsis_steady_i_NOT_glissando',10.724,10.921)],
    'PREDESTINATI': [('control_38',37.984,38.144),('control_53',53.542,53.752)]
}


def spectral_probe(audio, fs, sample, duration_ms=24.):
    """Descriptive present-centered noise/energy measurements, NOT a voice label.

    Flatness is dimensionless and can be misleading in low-energy recordings.
    No F0 or candidate is inferred from spectral brightness.
    """
    n=max(64,round(fs*duration_ms/1000))
    lo=sample-n//2
    if lo<0 or lo+n>len(audio):
        return {'probe_status':'outside_audio'}
    x=np.asarray(audio[lo:lo+n],dtype=np.float64)
    x=x-x.mean()
    rms=float(np.sqrt(np.mean(x*x)))
    if rms<1e-10:
        return {'probe_status':'very_low_energy','probe_rms':rms}
    w=np.hanning(n)
    p=np.abs(np.fft.rfft(x*w))**2
    hz=np.fft.rfftfreq(n,1/fs)
    use=(hz>=80)&(hz<=min(9000.,fs/2))
    p=p[use]
    hz=hz[use]
    if not len(p) or float(p.sum())<1e-20:
        return {'probe_status':'insufficient_spectral_energy','probe_rms':rms}
    eps=max(float(p.max())*1e-12,1e-30)
    flatness=float(np.exp(np.mean(np.log(p+eps)))/np.mean(p+eps))
    brightness=float(np.sum(p[hz>=2500])/np.sum(p))
    return {'probe_status':'measured','probe_rms':rms,
            'spectral_flatness':flatness,'energy_fraction_above_2500hz':brightness}


def cents(a,b):
    return abs(1200*math.log2(a/b)) if a and b and a>0 and b>0 else math.inf


def match_periods(previous,current, tolerance_cents=65.):
    """One-to-one matching of independently measured peaks; never multiply pitches."""
    pairs=[]; used=set()
    for a in previous:
        best=None
        for j,b in enumerate(current):
            if j in used: continue
            distance=cents(a['candidate']['hz'],b['candidate']['hz'])
            if distance<=tolerance_cents and (best is None or distance<best[0]):
                best=(distance,j,b)
        if best is not None:
            used.add(best[1]);pairs.append((a,best[2],best[0]))
    return pairs


def local_evidence(a):
    p=a.get('present') or {}
    return p.get('acf'),p.get('fundamental_fraction'),p.get('cycles'),p.get('testability')


def analyze_stages(stages):
    transitions=[]
    for i in range(1,len(stages)):
        prev,now=stages[i-1],stages[i]
        old=prev.get('raw_candidates') or []
        new=now.get('raw_candidates') or []
        matches=match_periods(old,new)
        matched_before={id(a) for a,_,_ in matches}
        matched_after={id(b) for _,b,_ in matches}
        entrants=[b for b in new if id(b) not in matched_after]
        # A same candidate's present 24ms probe is unchanged across aperture sizes.
        # Its local ACF cannot constitute *incremental* information. Compare full
        # evidence AND test whether the local probe actually supports that period.
        comparisons=[]
        for a,b,dist in matches:
            old_local=local_evidence(a);new_local=local_evidence(b)
            comparisons.append({'previous_hz':a['candidate']['hz'],
                'current_hz':b['candidate']['hz'],'distance_cents':dist,
                'full_acf_delta':b['candidate']['acf']-a['candidate']['acf'],
                'full_cmndf_delta':b['candidate']['cmndf']-a['candidate']['cmndf'],
                'full_coverage_delta':b['candidate']['coverage']-a['candidate']['coverage'],
                'local_acf':new_local[0],'local_fraction':new_local[1],
                'local_cycles':new_local[2], 'local_testability':new_local[3],
                'attribution':b.get('attribution')})
        local_new=[a for a in entrants if a.get('attribution')=='local_support_exploratory']
        remote_new=[a for a in entrants if a.get('attribution')=='possible_distant_only_support']
        weak_new=[a for a in entrants if a.get('attribution','').startswith('local_inconclusive')]
        # Categories are descriptions, not claims about provenance or correctness.
        if local_new: kind='new_measured_period_with_local_support'
        elif remote_new: kind='new_period_with_distant_only_support'
        elif entrants: kind='new_period_without_local_confirmation'
        elif matches and any(c['full_acf_delta']>=.04 and c['local_acf'] is not None and
                                  c['local_acf']<.70 for c in comparisons):
            kind='full_window_gain_local_weak'
        elif matches and any(c['full_acf_delta']>=.04 or c['full_cmndf_delta']<=-.04 or
                                  c['full_coverage_delta']>0 for c in comparisons):
            kind='same_period_full_window_evidence_gain'
        elif not old and not new:kind='no_raw_period_at_either_size'
        elif old and not new:kind='raw_period_lost_on_expansion'
        else:kind='no_demonstrated_incremental_gain'
        transitions.append({'from_ms':prev['aperture_ms'],'to_ms':now['aperture_ms'],
                            'kind':kind,'old_raw_count':len(old),'new_raw_count':len(new),
                            'new_local_count':len(local_new),'new_remote_count':len(remote_new),
                            'new_inconclusive_count':len(weak_new),'matches':comparisons,
                            'new_periods':[{'hz':a['candidate']['hz'],
                                            'acf':a['candidate']['acf'],
                                            'cmndf':a['candidate']['cmndf'],
                                            'attribution':a.get('attribution')}
                                           for a in entrants]})
    return transitions


def counterfactual(stages):
    """Propose stopping after TWO empty stages, then inspect available later stages.

    No early stop is executed. Later candidate existence is an upper-bound warning,
    not evidence that candidate is correct. v3 might already have stopped early;
    such counterfactuals are explicitly unassessable.
    """
    candidate_index=None
    for i in range(1,len(stages)-1):
        p,q=stages[i-1],stages[i]
        if p.get('raw_candidate_count')==0 and q.get('raw_candidate_count')==0:
            candidate_index=i;break
    if candidate_index is None:
        return {'suggested':False,'assessment':'no_two_empty_stages'}
    later=stages[candidate_index+1:]
    if not later:return {'suggested':True,'assessment':'no_later_stages_available',
                        'at_ms':stages[candidate_index]['aperture_ms']}
    later_raw=[a for s in later for a in s.get('raw_candidates',[])]
    later_local=[a for a in later_raw if a.get('attribution')=='local_support_exploratory']
    later_eligible=[a for a in later_local if a.get('eligible')]
    return {'suggested':True,'at_ms':stages[candidate_index]['aperture_ms'],
            'assessment':('later_eligible_local_candidate' if later_eligible else
                          'later_local_raw_candidate' if later_local else
                          'later_raw_candidate_only' if later_raw else 'no_later_raw_candidate'),
            'later_raw_count':len(later_raw),'later_local_count':len(later_local),
            'later_eligible_local_count':len(later_eligible)}


def competing_periods(stages):
    """Per-aperture disagreements; no across-time smoothing or desired note prior."""
    detail=[]
    for s in stages:
        c=s.get('raw_candidates') or []
        viable=[a for a in c if a.get('attribution')=='local_support_exploratory']
        for i,a in enumerate(viable):
            for b in viable[i+1:]:
                separation=cents(a['candidate']['hz'],b['candidate']['hz'])
                if separation<70:continue
                pa,pb=a['present'],b['present']
                detail.append({'aperture_ms':s['aperture_ms'],
                    'hz_a':a['candidate']['hz'],'hz_b':b['candidate']['hz'],
                    'separation_cents':separation,
                    'acf_a':pa.get('acf'),'acf_b':pb.get('acf'),
                    'fundamental_fraction_a':pa.get('fundamental_fraction'),
                    'fundamental_fraction_b':pb.get('fundamental_fraction'),
                    'cmndf_a':a['candidate']['cmndf'],'cmndf_b':b['candidate']['cmndf']})
    return detail


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*NAMES),default='all')
    p.add_argument('--v3-dir',type=Path,default=Path('tests/binocular_evidence_search_v3'))
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/incremental_evidence_v4'))
    args=p.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    summary={'schema':'incremental_evidence_v4','full_song':args.case=='all',
             'read_only':True,'uses_midi':False,'modifies_production':False,
             'counterfactual_early_stop_applied':False,
             'note':'Early-stop hypotheses are audited against available v3 later stages, not executed. No voiced/F0 ground truth.',
             'regions':REGIONS,'cases':{}}
    for name in (NAMES if args.case=='all' else (args.case,)):
        source=args.v3_dir/(name+'.csv')
        if not source.is_file():p.error(f'Missing v3 results: {source}. Run v3 --case all first.')
        wav=args.wav_dir/(name+'.wav')
        if not wav.is_file():p.error(f'Missing original WAV: {wav}')
        audio,fs=sf.read(wav,dtype='float64')
        if audio.ndim!=1:p.error(f'Mono WAV required: {wav}')
        reasons=Counter();increments=Counter();early=Counter();regions=Counter()
        output=args.out_dir/(name+'.csv')
        columns=('case','region','time_s','sample','v3_stop_reason','v3_observational_hz',
                 'stage_count','transition_kinds','incremental_json','early_stop_audit_json',
                 'present_probe_json','local_competing_periods_json','f0_applied',
                 'voicing_applied','tracker_reset_requested')
        n=0
        with source.open(newline='') as inp,output.open('w',newline='') as out:
            reader=csv.DictReader(inp)
            needed={'sample','time_s','stages_json','stop_reason','observational_hz'}
            if not needed.issubset(reader.fieldnames or []):
                p.error(f'Unexpected v3 CSV columns: {source}')
            writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
            for row in reader:
                stages=json.loads(row['stages_json'])
                transitions=analyze_stages(stages)
                cf=counterfactual(stages)
                competition=competing_periods(stages)
                t=float(row['time_s']);sample=int(row['sample'])
                region=';'.join(label for label,a,b in REGIONS.get(name,[]) if a<=t<=b)
                probe=spectral_probe(audio,fs,sample)
                kinds=[x['kind'] for x in transitions]
                writer.writerow({'case':name,'region':region,'time_s':t,'sample':sample,
                    'v3_stop_reason':row['stop_reason'],
                    'v3_observational_hz':row['observational_hz'],
                    'stage_count':len(stages),'transition_kinds':';'.join(kinds),
                    'incremental_json':json.dumps(transitions,allow_nan=False),
                    'early_stop_audit_json':json.dumps(cf,allow_nan=False),
                    'present_probe_json':json.dumps(probe,allow_nan=False),
                    'local_competing_periods_json':json.dumps(competition,allow_nan=False),
                    'f0_applied':False,'voicing_applied':False,'tracker_reset_requested':False})
                n+=1;reasons[row['stop_reason']]+=1;early[cf['assessment']]+=1
                for k in kinds:increments[k]+=1
                if region:
                    for label in region.split(';'):regions[label]+=1
        summary['cases'][name]={'observations':n,'sample_rate':fs,
            'v3_stop_reasons':dict(reasons),'incremental_patterns':dict(increments),
            'two_empty_stage_counterfactual':dict(early),
            'region_observations':dict(regions),'output':str(output)}
        print(f'{name}: {n} observations; incremental={dict(increments)}; early-stop={dict(early)}',flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SHADOW AUDIT ONLY. No early stop, F0, voicing or reset applied.')

if __name__=='__main__':main()
