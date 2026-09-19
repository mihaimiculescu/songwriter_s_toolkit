#!/usr/bin/env python3
"""V16 read-only cross-layer handoff audit (NOT a final vocal F0 selector).

Inputs: V13 full timeline, V14 dark-candidate inventories, V15 independent
waveform screening, and original WAVs. All four recordings by default.

Key distinctions:
 * Acoustic support for a period != active-vocal-source certification.
 * V15 production is a BLOCK MEDIAN, not an instantaneous tracker output.
 * Curve evaluates only adjacent observations with a provisionally admitted left
   pitch and an eligible right proposal. No bridging gaps or layer-based veto.
 * Every observation rechecks adaptive FIRST; no permanent production authority.
 * Counterfactual production handoffs are CONDITIONAL, not validated rescues.
 * Both return candidates get current-time independent acoustic tests.
No edits to production, pitch exports, harmonics, curve, or source WAVs.
"""
from __future__ import annotations
import argparse,csv,io,json,math,sys,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from importlib import import_module
from importlib.util import spec_from_file_location,module_from_spec

# Allow script in tests/ (normal) or anywhere with --repo-root.
def load_curve(repo_root):
 if str(repo_root) not in sys.path:sys.path.insert(0,str(repo_root))
 try:return import_module('python_eckf.trajectory_resolver').vocal_transition_penalty
 except Exception as exc:raise RuntimeError('Cannot import unchanged vocal_transition_penalty from repository; aborting: '+repr(exc)) from exc

def n(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None

def truth(x):return str(x).strip().lower() in ('true','1','yes')
def cents(a,b):return abs(1200*math.log2(a/b)) if a and b and a>0 and b>0 else None
def j(x,default):
 try:return json.loads(x) if x else default
 except (ValueError,TypeError):return default

def load(src,case,suffix):
 name=f'{case}_{suffix}.csv'
 if src.is_file() and src.suffix.lower()=='.zip':
  with zipfile.ZipFile(src) as z:
   matches=[v for v in z.namelist() if v==name or v.endswith('/'+name)]
   if len(matches)!=1:raise ValueError(f'{src}: need one {name}; got {matches}')
   with z.open(matches[0]) as f:return list(csv.DictReader(io.TextIOWrapper(f)))
 with (src/name).open(newline='') as f:return list(csv.DictReader(f))

def views_for_pitch(audio,sr,t,hz):
 # Explicit reuse of V15's *identical* independent screen; load its source from
 # the neighboring tests directory, never use its output as pitch ground truth.
 from audit_two_pass_contribution_v15 import view_evidence,summarize_views
 v=view_evidence(audio,sr,t,hz)
 s,nt,ng,nf=summarize_views(v)
 return dict(screen=s,testable=nt,supported=ng,flank=nf,
             current_views={f'{ms}_present':v[f'{ms}_present'] for ms in (24,40,64)})

def evaluate(curve,left,right,dt):
 if left is None or right is None:return None
 try:
  value=float(curve(12*math.log2(right/left),dt))
  if not math.isfinite(value):raise ValueError('nonfinite curve output')
  return value
 except Exception as exc:raise RuntimeError(f'Existing curve failed for {left}->{right} at {dt}ms: {exc}') from exc

def candidate_competition(candidates,hz):
 """Keep unresolved if alternate acoustically supported candidate is comparable.
 Mirrors V13 *diagnostic* competition conditions, not a new pitch ranking.
 """
 if not candidates:return 'no_adaptive_eligible_candidates'
 matching=[c for c in candidates if n(c.get('hz')) and cents(n(c['hz']),hz)<=70]
 if not matching:return 'production_not_in_adaptive_eligible_candidates'
 match=max(matching,key=lambda c:n(c.get('present_acf')) or -1.)
 for other in candidates:
  oh=n(other.get('hz'))
  if not oh or cents(oh,hz)<=70:continue
  ac=n(other.get('present_acf'));mac=n(match.get('present_acf'))
  frac=n(other.get('fundamental_fraction'));mfrac=n(match.get('fundamental_fraction'))
  if ac is not None and mac is not None and ac>=mac-.07 and (frac or 0)>=(mfrac or 0)*.5:
   return 'eligible_competition_not_resolved'
 return 'no_comparable_eligible_rival_in_v13_geometry'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--case',choices=('all',*CASES),default='all')
 p.add_argument('--repo-root',type=Path,default=ROOT)
 p.add_argument('--v13',type=Path,default=Path('tests/full_timeline_v13'))
 p.add_argument('--v14',type=Path,default=Path('tests/failure_mechanisms_v14'))
 p.add_argument('--v15',type=Path,default=Path('tests/two_pass_contribution_v15'))
 p.add_argument('--wav-dir',type=Path,default=Path('tests'))
 p.add_argument('--out-dir',type=Path,default=Path('tests/cross_layer_handoffs_v16'))
 args=p.parse_args(); curve=load_curve(args.repo_root.resolve());args.out_dir.mkdir(parents=True,exist_ok=True)
 # V15 functions live in tests/ when this script is installed alongside them.
 tests=args.repo_root.resolve()/'tests'
 if str(tests) not in sys.path:sys.path.insert(0,str(tests))
 report={'schema':'v16_cross_layer_conditional_handoff','curve':'imported_unchanged',
 'production_is_block_median':True,'adaptive_primary_each_observation':True,
 'penalty_is_not_a_hard_veto':True,'conditional_not_validated_f0':True,
 'source_identity_unverified':True,'production_modified':False,'cases':{}}
 for case in (CASES if args.case=='all' else (args.case,)):
  timeline=load(args.v13,case,'full_timeline');dark=load(args.v14,case,'dark');v15=load(args.v15,case,'two_pass')
  bydark={int(r['sample']):r for r in dark};byv15={int(r['sample']):r for r in v15}
  if len(bydark)!=len(dark) or len(byv15)!=len(v15) or set(bydark)!=set(byv15):raise ValueError(f'{case}: V14/V15 misalignment')
  samples=[int(r['sample']) for r in timeline]
  if samples!=sorted(set(samples)) or not set(bydark).issubset(samples):raise ValueError(f'{case}: full timeline misaligned')
  audio,sr=sf.read(args.wav_dir/f'{case}.wav',dtype='float64',always_2d=False)
  if audio.ndim==2:audio=audio.mean(axis=1)
  counts=Counter();outputs=[];returns=[];left=None; prev_sample=None
  for r in timeline:
   sample=int(r['sample']);t=float(r['time_s']);d=bydark.get(sample);v=byv15.get(sample)
   if abs(sample/sr-t)>.003:raise ValueError(f'{case}: time mismatch {sample}')
   # Explicit adjacency: observation step can vary slightly with sample rounding.
   adjacent=prev_sample is not None and abs((sample-prev_sample)/sr-.010)<=.0015
   prior=left if adjacent else None;elapsed=1000*(sample-prev_sample)/sr if prior else None
   adaptive_hz=n(r['selected_measured_hz'])
   adaptive_supported=(r['evidence_status']=='acoustic_period_supported_source_unverified'
                       and adaptive_hz is not None)
   prod_hz=n(v['production_median_hz']) if v else None
   available=bool(v and truth(v['production_pitch_available']) and prod_hz and prod_hz>0)
   current_screen=v['screen'] if v else ''
   acoustically_supported_production=available and current_screen in (
    'present_support_already_in_adaptive_eligible_candidates',
    'production_frequency_already_measured_but_adaptive_rejected_or_remote',
    'production_period_adds_testable_present_support')
   p_pen=evaluate(curve,prior['hz'],prod_hz,elapsed) if prior and acoustically_supported_production else None
   a_pen=evaluate(curve,prior['hz'],adaptive_hz,elapsed) if prior and adaptive_supported else None
   # V13 flags evidence, not a final voice certification. Recheck both sides
   # at returning boundary independently; do not inherit production's quality.
   event=bool(adaptive_supported and prior and prior['layer']=='production')
   ac_now=None;pr_now=None;return_class=''
   if event:
    ac_now=views_for_pitch(audio,sr,t,adaptive_hz)
    if available:pr_now=views_for_pitch(audio,sr,t,prod_hz)
    ac_good=ac_now['screen']=='independent_present_period_support_not_voice_proof'
    pr_good=bool(pr_now and pr_now['screen']=='independent_present_period_support_not_voice_proof')
    if ac_good and not pr_good:return_class='adaptive_current_support_production_not_supported'
    elif ac_good and pr_good:
     if cents(adaptive_hz,prod_hz)<=70:return_class='both_supported_same_frequency'
     elif a_pen is None or p_pen is None:return_class='both_supported_no_comparable_penalty'
     elif a_pen<p_pen:return_class='both_supported_penalty_favors_adaptive'
     elif p_pen<a_pen:return_class='both_supported_penalty_favors_production'
     else:return_class='both_supported_equal_penalty'
    elif pr_good:return_class='adaptive_return_not_independently_supported_production_supported'
    else:return_class='neither_current_candidate_independently_supported'
    returns.append(dict(case=case,sample=sample,time_s=t,previous_layer='production',previous_hz=prior['hz'],
                        adaptive_hz=adaptive_hz,production_block_hz=prod_hz,adaptive_penalty=a_pen,
                        production_penalty=p_pen,adaptive_waveform_json=json.dumps(ac_now,separators=(',',':')),
                        production_waveform_json=json.dumps(pr_now,separators=(',',':')),
                        return_diagnostic=return_class,final_source_identity_verified=False))
    counts['return_'+return_class]+=1
   category='';new_left=None;layer=''
   if adaptive_supported:
    # Primary layer recovers at EACH observation. Do not allow a curve-based veto.
    # V13's locally supported state is provisional, not verified singing.
    category='adaptive_primary_supported_provisional';new_left=adaptive_hz;layer='adaptive'
   elif not d:
    category='no_pitch_reference_non_dark';counts['not_dark_no_f0']+=1
   elif not available:
    category='no_production_proposal';counts['no_production_proposal']+=1
   elif not acoustically_supported_production:
    category='production_acoustically_ineligible_or_ambiguous';counts['production_acoustically_ineligible_or_ambiguous']+=1
   else:
    eligible=j(d['eligible_candidates_json'],[])
    competition=candidate_competition(eligible,prod_hz)
    low=truth(r['low_energy_flag']);decay=truth(r['decaying_flag'])
    # Don't invent a cutoff or retune curve: use V13's existing joint criterion.
    joint=p_pen is not None and p_pen>0 and low and decay
    if joint:
     category='jointly_challenged_switch';counts[category]+=1
    elif competition=='eligible_competition_not_resolved':
     category='insufficient_competing_eligible_periods';counts[category]+=1
    elif current_screen=='production_frequency_already_measured_but_adaptive_rejected_or_remote':
     category='insufficient_adaptive_eligibility_or_temporal_disagreement';counts[category]+=1
    else:
     category='conditionally_credible_production_switch_source_unverified';counts[category]+=1
     new_left=prod_hz;layer='production'
   if d and available:counts['all_production_opportunities']+=1
   if event:counts['return_events']+=1
   rec=dict(case=case,sample=sample,time_s=t,v13_status=r['evidence_status'],
            v14_mechanism=d['mechanism'] if d else '',v15_screen=current_screen,
            prior_layer=prior['layer'] if prior else '',prior_hz=prior['hz'] if prior else '',
            adjacent_reference=bool(prior),adaptive_hz=adaptive_hz,production_block_hz=prod_hz,
            production_provisionally_supported=bool(acoustically_supported_production),
            adaptive_penalty=a_pen,production_penalty=p_pen,
            energy_low_flag=r['low_energy_flag'],energy_decay_flag=r['decaying_flag'],
            diagnostic=category,return_event=event,return_diagnostic=return_class,
            provisional_next_layer=layer,provisional_next_hz=new_left,
            confirmed_active_vocal_f0=False,production_modified=False)
   outputs.append(rec)
   left={'hz':new_left,'layer':layer} if new_left else None
   prev_sample=sample
  assert len(outputs)==len(timeline)
  if counts['all_production_opportunities']!=sum(truth(r['production_pitch_available']) for r in v15):raise AssertionError('opportunity mismatch')
  for suffix,rows,fields in [('decisions',outputs,list(outputs[0])),('returns',returns,list(returns[0]) if returns else ['case','sample','time_s','return_diagnostic'])]:
   with (args.out_dir/f'{case}_{suffix}.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
  report['cases'][case]={'timeline_observations':len(timeline),'adaptive_unresolved':len(dark),
                         'production_proposals':counts['all_production_opportunities'],
                         'categories':dict(counts),'return_events':len(returns),
                         'validated_vocal_f0_rescues':0}
  print(case,json.dumps(report['cases'][case],separators=(',',':')),flush=True)
 (args.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Conditional shadow analysis only: neither source identity nor definitive F0 rescues certified.')
if __name__=='__main__':main()
