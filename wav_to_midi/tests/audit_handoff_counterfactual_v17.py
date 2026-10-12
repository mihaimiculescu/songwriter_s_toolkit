#!/usr/bin/env python3
"""V17: read-only counterfactual of V16 fallback and return hypotheses.

Requires V16 decision CSVs, V13, V14, V15, original WAVs, the unchanged
repository penalty, and V15's independent waveform-screen functions.
No production edits; a held frequency is only a TEST HYPOTHESIS, never an
emitted production F0 or an accepted pitch. No vocal source certification.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
ROOT=Path(__file__).resolve().parent.parent

def num(v):
 try:
  x=float(v);return x if math.isfinite(x) and x>0 else None
 except (ValueError,TypeError):return None

def yes(v):return str(v).lower() in ('true','1','yes')
def read_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def indexed(rows):
 out={int(x['sample']):x for x in rows}
 if len(out)!=len(rows):raise ValueError('Duplicate samples in input')
 return out

def replay(r, dark, with_curve):
 """EXACT V16 fallback branch; change only whether the joint criterion sees penalty.
 Do not retune the curve or treat positive penalty as an unconditional veto.
 """
 if r['v13_status']=='acoustic_period_supported_source_unverified' and num(r['adaptive_hz']):
  return 'adaptive_primary_supported_provisional'
 if dark is None:return 'no_pitch_reference_non_dark'
 if num(r['production_block_hz']) is None:return 'no_production_proposal'
 if not yes(r['production_provisionally_supported']):
  return 'production_acoustically_ineligible_or_ambiguous'
 from audit_cross_layer_handoffs_v16 import candidate_competition,j
 eligible=j(dark['eligible_candidates_json'],[])
 competition=candidate_competition(eligible,num(r['production_block_hz']))
 p=num(r['production_penalty'])
 if with_curve and p is not None and p>0 and yes(r['energy_low_flag']) and yes(r['energy_decay_flag']):
  return 'jointly_challenged_switch'
 if competition=='eligible_competition_not_resolved':return 'insufficient_competing_eligible_periods'
 if r['v15_screen']=='production_frequency_already_measured_but_adaptive_rejected_or_remote':
  return 'insufficient_adaptive_eligibility_or_temporal_disagreement'
 return 'conditionally_credible_production_switch_source_unverified'

def current_screen(audio,sr,t,hz):
 from audit_cross_layer_handoffs_v16 import views_for_pitch
 return views_for_pitch(audio,sr,t,hz) if hz else None

def support(screen):
 return bool(screen and screen['screen']=='independent_present_period_support_not_voice_proof')

def curve_value(curve,prior,hz,elapsed):
 from audit_cross_layer_handoffs_v16 import evaluate
 return evaluate(curve,prior,hz,elapsed) if prior and hz and elapsed else None

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--case',choices=('all',*CASES),default='all')
 p.add_argument('--repo-root',type=Path,default=ROOT)
 p.add_argument('--v16',type=Path,default=Path('tests/cross_layer_handoffs_v16'))
 p.add_argument('--v14',type=Path,default=Path('tests/failure_mechanisms_v14'))
 p.add_argument('--wav-dir',type=Path,default=Path('tests'))
 p.add_argument('--out-dir',type=Path,default=Path('tests/handoff_counterfactual_v17'))
 args=p.parse_args();repo=args.repo_root.resolve()
 for directory in (repo,repo/'tests'):
  if str(directory) not in sys.path:sys.path.insert(0,str(directory))
 from audit_cross_layer_handoffs_v16 import load_curve
 curve=load_curve(repo)
 args.out_dir.mkdir(parents=True,exist_ok=True)
 report={'schema':'v17_counterfactual_v16_exact_fallback_replay_and_return_hypothesis',
         'penalty':'existing_function_imported_unchanged',
         'fallback_counterfactual':'V16 decisions replayed with only joint penalty disabled',
         'return_hypothesis':'last accepted production pitch independently remeasured at return; NOT a current production output',
         'not_ground_truth':True,'active_vocal_source_validated':False,
         'production_modified':False,'cases':{}}
 for case in (CASES if args.case=='all' else (args.case,)):
  decisions=read_csv(args.v16/f'{case}_decisions.csv')
  original_returns=read_csv(args.v16/f'{case}_returns.csv')
  dark=indexed(read_csv(args.v14/f'{case}_dark.csv'))
  aud,sr=sf.read(args.wav_dir/f'{case}.wav',dtype='float64',always_2d=False)
  if aud.ndim==2:aud=aud.mean(axis=1)
  fall=[]; returns=[];cf=Counter();rc=Counter()
  for r in decisions:
   s=int(r['sample']); d=dark.get(s)
   on=replay(r,d,True);off=replay(r,d,False)
   if on!=r['diagnostic']:
    raise AssertionError(f'{case} sample {s}: replay={on}, V16={r["diagnostic"]}; abort instead of reporting an invalid counterfactual')
   if d and num(r['production_block_hz']):
    changed=(on!=off);cf['proposals']+=1;cf['changed' if changed else 'unchanged']+=1
    cf['on_'+on]+=1;cf['off_'+off]+=1
    if r['production_penalty'] not in ('',None):
     cf['penalty_evaluated']+=1
     if float(r['production_penalty'])>0:cf['penalty_positive']+=1
    fall.append({'case':case,'sample':s,'time_s':r['time_s'],
      'adaptive_hz':r['adaptive_hz'],'production_block_hz':r['production_block_hz'],
      'prior_hz':r['prior_hz'],'prior_layer':r['prior_layer'],
      'acoustic_screen':r['v15_screen'],'production_penalty':r['production_penalty'],
      'low_energy':r['energy_low_flag'],'decay':r['energy_decay_flag'],
      'decision_penalty_on':on,'decision_penalty_off':off,
      'penalty_changes_diagnostic':changed,'vocal_f0_verified':False})
  for e in original_returns:
   s=int(e['sample']);t=float(e['time_s']);left=num(e['previous_hz'])
   adaptive=num(e['adaptive_hz']); actual=num(e['production_block_hz'])
   if not left or not adaptive:raise AssertionError(f'{case} bad return row {s}')
   # V16 documented lack of current production candidate. Test historical pitch
   # against CURRENT waveform, not as fictitious production output.
   hypothesis=actual if actual else left
   from audit_cross_layer_handoffs_v16 import cents
   av=current_screen(aud,sr,t,adaptive)
   hv=current_screen(aud,sr,t,hypothesis)
   ag=support(av);hg=support(hv)
   if actual:origin='current_production_block_median'
   else:origin='previous_accepted_production_frequency_hypothesis_only'
   # Use exact previous/current sample times from complete V16 decision timeline.
   pos=next((i for i,r in enumerate(decisions) if int(r['sample'])==s),None)
   if pos is None or pos==0:raise AssertionError('Return absent from decision timeline')
   prev=decisions[pos-1]
   dt=1000*(s-int(prev['sample']))/sr
   adjacent=abs(dt-10)<=1.5
   ap=curve_value(curve,left,adaptive,dt) if adjacent and ag else None
   hp=curve_value(curve,left,hypothesis,dt) if adjacent and hg else None
   if ag and hg:
    if cents(adaptive,hypothesis)<=70:classify='both_supported_same_frequency_no_switch_conflict'
    elif ap is None or hp is None:classify='both_supported_penalty_unavailable'
    elif ap<hp:classify='both_supported_penalty_prefers_adaptive'
    elif hp<ap:classify='both_supported_penalty_prefers_historical_frequency'
    else:classify='both_supported_equal_penalty'
   elif ag:classify='only_adaptive_present_support'
   elif hg:classify='only_historical_frequency_present_support'
   else:classify='neither_present_support'
   rc[classify]+=1;rc['total_return_events']+=1
   if actual:rc['actual_current_production_pitch_available']+=1
   else:rc['historical_hypothesis_not_current_production_output']+=1
   returns.append({'case':case,'sample':s,'time_s':t,'previous_accepted_production_hz':left,
    'adaptive_current_hz':adaptive,'current_production_block_hz':actual or '',
    'tested_other_hz':hypothesis,'tested_other_origin':origin,
    'adaptive_present_support':ag,'other_present_support':hg,
    'adaptive_penalty':ap,'other_penalty':hp,
    'frequency_difference_cents':cents(adaptive,hypothesis),
    'diagnostic':classify,
    'adaptive_evidence_json':json.dumps(av,separators=(',',':')),
    'other_evidence_json':json.dumps(hv,separators=(',',':')),
    'historical_pitch_not_current_tracker_output':not bool(actual),
    'source_identity_verified':False})
  if cf['proposals']!=sum(bool(num(r['production_block_hz'])) and int(r['sample']) in dark for r in decisions):
   raise AssertionError('Proposal count mismatch')
  write_csv(args.out_dir/f'{case}_fallback_counterfactual.csv',fall,
   ['case','sample','time_s','adaptive_hz','production_block_hz','prior_hz','prior_layer','acoustic_screen','production_penalty','low_energy','decay','decision_penalty_on','decision_penalty_off','penalty_changes_diagnostic','vocal_f0_verified'])
  write_csv(args.out_dir/f'{case}_return_hypotheses.csv',returns,
   ['case','sample','time_s','previous_accepted_production_hz','adaptive_current_hz','current_production_block_hz','tested_other_hz','tested_other_origin','adaptive_present_support','other_present_support','adaptive_penalty','other_penalty','frequency_difference_cents','diagnostic','adaptive_evidence_json','other_evidence_json','historical_pitch_not_current_tracker_output','source_identity_verified'])
  report['cases'][case]={'fallback':dict(cf),'returns':dict(rc)}
  print(case,json.dumps(report['cases'][case],sort_keys=True),flush=True)
 (args.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Diagnostic counterfactual complete; no validated vocal F0 rescue claimed.')
if __name__=='__main__':main()
