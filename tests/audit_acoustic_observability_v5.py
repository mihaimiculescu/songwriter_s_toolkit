#!/usr/bin/env python3
"""Read-only full-song audit: production comparison, counterfactual apertures,
period observability vs frequency identifiability. NO pitch/status corrections.

Run from repository root:
    python tests/audit_acoustic_observability_v5.py --case all
Uses existing v3, v4 and baseline CSVs; does not rerun ECKF, inspect MIDI or
use production as truth. Block-level baseline flags are NOT sample-level labels.
"""
from __future__ import annotations
import argparse, bisect, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

NAMES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={
 'RATATA':[('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.915,43.065)],
 'Ochiitai':[('transition',42.174,42.413)],
 'Trandafiri':[('tsis_steady_i_NOT_glissando',10.724,10.921)],
 'PREDESTINATI':[('control_38',37.984,38.144),('control_53',53.542,53.752)]}

def number(v):
 try:
  f=float(v)
  return f if math.isfinite(f) else None
 except (TypeError,ValueError): return None

def cents(a,b):
 return abs(1200*math.log2(a/b)) if a and b and a>0 and b>0 else None

def rows(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def candidates(stage):
 return stage.get('raw_candidates') or []

def local(candidate):
 return candidate.get('attribution')=='local_support_exploratory'

def quality(candidate):
 p=candidate.get('present') or {}
 return {'hz':number((candidate.get('candidate') or {}).get('hz')),
         'acf':number(p.get('acf')),'cycles':number(p.get('cycles')),
         'fundamental_fraction':number(p.get('fundamental_fraction')),
         'testability':p.get('testability'),'eligible':bool(candidate.get('eligible')),
         'attribution':candidate.get('attribution')}

def observability(stages):
 # Presence of a locally supported period is not a vocal/voiced verdict.
 allc=[c for s in stages for c in candidates(s)]
 loc=[c for c in allc if local(c)]
 eligible=[c for c in loc if c.get('eligible')]
 testable=[c for c in allc if (c.get('present') or {}).get('testability')=='testable']
 insufficient=[c for c in allc if (c.get('present') or {}).get('testability')=='insufficient_cycles']
 distant=[c for c in allc if c.get('attribution')=='possible_distant_only_support']
 if eligible:obs='local_eligible_period_evidence'
 elif loc:obs='local_raw_period_evidence_only'
 elif distant:obs='distant_only_evidence'
 elif insufficient and not testable:obs='local_cycle_insufficiency'
 elif testable:obs='testable_but_no_local_support'
 elif allc:obs='raw_period_without_local_confirmation'
 else:obs='no_raw_period_any_inspected_aperture'
 # Identifiability is assessed at a SINGLE aperture, not across time and not
 # by interpreting multiple octave/submultiple candidates as automatically tied.
 per_stage=[]
 for s in stages:
  support=[c for c in candidates(s) if local(c) and c.get('eligible')]
  vals=[quality(c) for c in support]
  vals=[v for v in vals if v['hz']]
  unique=[]
  for v in sorted(vals,key=lambda x:x['hz']):
   if not any(cents(v['hz'],u['hz']) is not None and cents(v['hz'],u['hz'])<=70 for u in unique):unique.append(v)
  per_stage.append({'aperture_ms':s.get('aperture_ms'),'raw_count':len(candidates(s)),
                    'eligible_local_distinct_count':len(unique),'eligible_local_candidates':unique,
                    'selector_hz':number(s.get('selector_hz')),
                    'stage_decision':s.get('decision')})
 if any(p['eligible_local_distinct_count']>1 for p in per_stage):ident='multiple_local_eligible_periods_unadjudicated'
 elif any(p['eligible_local_distinct_count']==1 for p in per_stage):ident='one_local_eligible_period_at_some_aperture'
 else:ident='no_locally_eligible_period_to_identify'
 return obs,ident,per_stage

def counterfactual(stages):
 """All actually observed stages. A previous v3 stop leaves later outcomes UNKNOWN."""
 out=[]
 for i,s in enumerate(stages):
  later=stages[i+1:]
  future_local=[c for z in later for c in candidates(z) if local(c) and c.get('eligible')]
  future_raw=[c for z in later for c in candidates(z)]
  here=[c for c in candidates(s) if local(c) and c.get('eligible')]
  out.append({'stop_at_ms':s.get('aperture_ms'),'local_eligible_now':len(here),
              'later_observed_stage_count':len(later),
              'later_eligible_local_observed':len(future_local),
              'later_raw_observed':len(future_raw),
              'later_evidence_status':('unobserved_after_v3_termination' if not later and i<len((24,40,64,96))-1
                else 'later_eligible_local_observed' if future_local else
                'later_raw_only_observed' if future_raw else 'no_later_raw_observed'),
              'later_eligible_hz':[quality(c)['hz'] for c in future_local]})
 return out

def production_index(base,fs):
 base=sorted(base,key=lambda r:int(r['sample']))
 starts=[int(r['sample']) for r in base]
 if len(starts)<2 or any(b<=a for a,b in zip(starts,starts[1:])):
  raise ValueError('Baseline must contain monotonically increasing sample starts')
 intervals=[]
 for i,r in enumerate(base):
  end=starts[i+1] if i+1<len(starts) else starts[i]+(starts[i]-starts[i-1])
  intervals.append((starts[i],end,r))
 return starts,intervals

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--case',choices=('all',*NAMES),default='all')
 p.add_argument('--v3-dir',type=Path,default=Path('tests/binocular_evidence_search_v3'))
 p.add_argument('--v4-dir',type=Path,default=Path('tests/incremental_evidence_v4'))
 p.add_argument('--baseline-dir',type=Path,default=Path('tests/robustness_v3'))
 p.add_argument('--out-dir',type=Path,default=Path('tests/acoustic_observability_v5'))
 a=p.parse_args(); names=NAMES if a.case=='all' else (a.case,)
 inputs={}
 for name in names:
  paths={'v3':a.v3_dir/f'{name}.csv','v4':a.v4_dir/f'{name}.csv',
         'baseline':a.baseline_dir/f'{name}_v3_frames.csv'}
  for kind,path in paths.items():
   if not path.is_file():p.error(f'{name}: missing {kind} at {path}')
  inputs[name]=paths
 a.out_dir.mkdir(parents=True,exist_ok=True)
 report={'version':'v5','read_only':True,'case_all':a.case=='all','production_is_ground_truth':False,
         'changes_pitch_or_status':False,'uses_midi':False,
         'warning':'Production status is BLOCK-level; overlap is not sample-level voicing. Local period evidence is not verified F0 or voice.',
         'regions':REGIONS,'cases':{}}
 for name,paths in inputs.items():
  v3=rows(paths['v3']);v4=rows(paths['v4']);base=rows(paths['baseline'])
  # V3/V4 must correspond exactly: fail rather than silently nearest-neighbor join.
  v4_by_sample={int(r['sample']):r for r in v4}
  if len(v4_by_sample)!=len(v4) or len(v3)!=len(v4):raise ValueError(f'{name}: v3/v4 length or duplicate-sample mismatch')
  # Native rate inferred from baseline time/sample pair (no forced 44.1k assumption).
  fs_values=[int(r['sample'])/float(r['time_s']) for r in base if number(r['time_s']) and int(r['sample'])>0]
  fs=round(fs_values[0]) if fs_values else None
  if fs is None or any(abs(f-fs)>2 for f in fs_values[:100]):raise ValueError(f'{name}: cannot establish baseline native sample rate')
  starts,blocks=production_index(base,fs)
  counts=Counter();region_counts=defaultdict(Counter)
  cols=['case','region','time_s','sample','production_block_start_s','production_block_end_s','production_voiced_samples',
        'production_status','production_median_hz','adaptive_status','adaptive_observational_hz',
        'comparison','hz_distance_cents','observability','identifiability','v3_stop_reason',
        'v4_transition_kinds','v4_early_stop_audit_json','v4_competition_json',
        'stage_evidence_json','counterfactual_apertures_json','f0_applied','status_applied','tracker_reset_requested']
  dest=a.out_dir/f'{name}.csv'
  with dest.open('w',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
   for r in v3:
    sample=int(r['sample']);audit=v4_by_sample.get(sample)
    if audit is None or abs(float(audit['time_s'])-float(r['time_s']))>1e-6:
     raise ValueError(f'{name}: v3/v4 mismatch at sample {sample}')
    j=bisect.bisect_right(starts,sample)-1
    block=blocks[j] if j>=0 and sample<blocks[j][1] else None
    stages=json.loads(r['stages_json']);obs,ident,stage_detail=observability(stages)
    hz=number(r['observational_hz']);base_hz=number(block[2].get('median_f0_hz')) if block else None
    voiced=int(block[2]['voiced_samples']) if block else None
    prod=('block_has_voiced_samples' if voiced and voiced>0 else 'block_has_no_voiced_samples' if voiced==0 else 'no_matching_block')
    if prod=='no_matching_block':comparison='unaligned_production'
    elif voiced>0:
     if hz is None:comparison='production_voiced_adaptive_unresolved'
     elif base_hz is None:comparison='production_voiced_no_baseline_pitch_summary'
     elif cents(hz,base_hz)<=70:comparison='production_voiced_adaptive_agree_70c'
     else:comparison='production_voiced_adaptive_different_70c'
    else:comparison='production_unvoiced_adaptive_candidate' if hz is not None else 'both_no_candidate_or_voicing'
    t=float(r['time_s']); region=';'.join(l for l,lo,hi in REGIONS.get(name,[]) if lo<=t<=hi)
    record={'case':name,'region':region,'time_s':t,'sample':sample,
            'production_block_start_s':block[0]/fs if block else '',
            'production_block_end_s':block[1]/fs if block else '',
            'production_voiced_samples':voiced,'production_status':prod,'production_median_hz':base_hz,
            'adaptive_status':r['status'],'adaptive_observational_hz':hz,'comparison':comparison,
            'hz_distance_cents':cents(hz,base_hz),'observability':obs,'identifiability':ident,
            'v3_stop_reason':r['stop_reason'],'v4_transition_kinds':audit['transition_kinds'],
            'v4_early_stop_audit_json':audit['early_stop_audit_json'],
            'v4_competition_json':audit['local_competing_periods_json'],
            'stage_evidence_json':json.dumps(stage_detail,allow_nan=False),
            'counterfactual_apertures_json':json.dumps(counterfactual(stages),allow_nan=False),
            'f0_applied':False,'status_applied':False,'tracker_reset_requested':False}
    writer.writerow(record)
    counts['total']+=1;counts['comparison:'+comparison]+=1;counts['observability:'+obs]+=1;counts['identifiability:'+ident]+=1
    if region:
     for label in region.split(';'):region_counts[label]['total']+=1;region_counts[label]['comparison:'+comparison]+=1
  report['cases'][name]={'observations':counts['total'],'sample_rate':fs,'comparison':{k[11:]:v for k,v in counts.items() if k.startswith('comparison:')},
   'observability':{k[14:]:v for k,v in counts.items() if k.startswith('observability:')},
   'identifiability':{k[16:]:v for k,v in counts.items() if k.startswith('identifiability:')},
   'regions':{key:dict(value) for key,value in region_counts.items()},
   'unaligned_production_observations':counts['comparison:unaligned_production'],
   'warning_if_unaligned':'Production reference may cover only part of the song; do not interpret unmatched observations as disagreement.' if counts['comparison:unaligned_production'] else None,
   'output':str(dest)}
  print(f'{name}: {counts["total"]} observations; '+str(report['cases'][name]['comparison']),flush=True)
 (a.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print('READ-ONLY audit: no pitch, voicing, early stop or reset applied.')
if __name__=='__main__':main()
