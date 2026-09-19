#!/usr/bin/env python3
"""V12: read-only source-observability and strict adjacent-interval audit.

Input: V11 energy_observability CSV (which embeds V9/V10 candidates) and
unchanged python_eckf.trajectory_resolver.vocal_transition_penalty.
No MIDI, no tracker edits, no synthetic pitches, no interpolation.

IMPORTANT: V11 covers only review observations. Consequently an adjacent
reference is allowed ONLY if the preceding review observation is exactly one
nominal 10-ms hop away. If full-song adjacency cannot be established, the
penalty is unavailable. No skipped or rejected observation can be bridged.

All classifications are shadow hypotheses; even 'candidate_rejected' does
NOT assign UNVOICED or claim a source is noise/reverb.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={'RATATA':[('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.90,43.10)],
'Ochiitai':[('transition',42.19,42.40)],'Trandafiri':[('steady_i',10.70,10.95)],
'PREDESTINATI':[('control_38',37.97,38.13),('control_53',53.52,53.73)]}

def number(v):
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError): return None

def decode(v,default):
    try:return json.loads(v)
    except (ValueError,TypeError):return default

def cent(a,b):return abs(1200*math.log2(a/b))

def candidates(row):
    stages=decode(row.get('aperture_evidence_json'),[])
    stage=next((s for s in stages if number(s.get('aperture_ms'))==40),None)
    if not stage:return [],'40ms_not_available'
    return [c for c in stage.get('candidates',[]) if number(c.get('hz')) and number(c['hz'])>0],stage.get('temporal_attribution','unspecified')

def solid(c):
    return bool(c.get('eligible')) and c.get('shift_category') in ('shift_stable_acoustic_candidate','mostly_shift_stable_candidate')

def acoustic_assessment(row,items,attribution):
    """Compare evidence; thresholds are EXPLORATORY flags, never voicing rules."""
    ratio=number(row.get('rms_present_vs_context_p80'))
    slope=number(row.get('energy_slope_past_to_future_db'))
    eligible=[c for c in items if c.get('eligible')]
    stable=[c for c in eligible if solid(c)]
    flags=[]
    if ratio is not None and ratio<.25:flags.append('relative_energy_low')
    if slope is not None and slope < -4.:flags.append('energy_falling')
    if attribution in ('past_only','future_only','distant_only_periodicity'):
        flags.append('temporal_support_asymmetric')
    if not eligible:flags.append('no_eligible_period')
    if eligible and not stable:flags.append('shift_robustness_unestablished')
    if len(eligible)>1:flags.append('eligible_period_competition')
    # A provisional reference is deliberately very conservative: ONE eligible,
    # shift-robust, non-asymmetric candidate and no low/decay warning.
    credible=(len(eligible)==1 and len(stable)==1 and
              not any(f in flags for f in ('relative_energy_low','energy_falling',
                     'temporal_support_asymmetric')))
    return eligible,stable,flags,credible

def evaluate_curve(func,prior,current,dt_ms):
    if func is None:return None,'curve_unavailable'
    if prior is None:return None,'not_two_adjacent_provisional_f0_candidates'
    try:
        result=float(func(12*math.log2(current/prior),dt_ms))
        if not math.isfinite(result):return None,'nonfinite_curve_result'
        return result,'evaluated_unchanged'
    except Exception as exc:return None,'curve_error:'+str(exc)

def analyze_case(rows,curve,fs,hop_ms):
    out=[];previous=None;counts=Counter()
    for r in rows:
        t=number(r.get('time_s')); sample=int(r['sample'])
        items,attrib=candidates(r)
        eligible,stable,flags,credible=acoustic_assessment(r,items,attrib)
        # No pitch from ECKF is ever eligible as an acoustic reference.
        # A predecessor rejected/unresolved at ANY step breaks the chain.
        adjacent=(previous is not None and sample>previous['sample'] and
                  abs((sample-previous['sample'])/fs*1000-hop_ms)<=1.5)
        prior=(previous['hz'] if adjacent and previous['provisional'] else None)
        dt=(sample-previous['sample'])/fs*1000 if adjacent else None
        penalty_results=[]
        for c in eligible:
            p,status=evaluate_curve(curve,prior,c['hz'],dt)
            penalty_results.append(dict(measured_hz=c['hz'],penalty=p,status=status,
                                        shift_robust=solid(c)))
        # The curve PARTICIPATES in the assessment when it has a legitimate
        # adjacent pair; it never decides by itself, nor promotes another pitch.
        # Numerical value is an observation, not a threshold for rejection.
        tested=[x for x in penalty_results if x['penalty'] is not None]
        curve_disagreement=(len(tested)>1 and max(x['penalty'] for x in tested)-
                            min(x['penalty'] for x in tested)>0.0)
        if not eligible:
            diagnostic='no_eligible_local_candidate'
        elif len(stable)==0:
            diagnostic='local_period_not_shift_confirmed'
        elif 'temporal_support_asymmetric' in flags:
            diagnostic='temporal_attribution_unresolved'
        elif 'relative_energy_low' in flags and 'energy_falling' in flags:
            diagnostic='low_decaying_periodic_material_source_unresolved'
        elif len(eligible)>1:
            diagnostic='fundamental_competition_with_curve_comparison' if curve_disagreement else 'fundamental_competition'
        elif credible:
            diagnostic='single_provisional_acoustic_candidate'
        else:
            diagnostic='source_observability_unresolved'
        # Penalizer CAN change diagnostic priority: if a legitimate preceding
        # acoustic pitch exists, a strong penalty is added as a challenge,
        # without fixed threshold or altering the acoustic classification.
        # Use relative candidate penalty ordering only when >=2 measured rivals.
        favored=[]
        if curve_disagreement:
            lo=min(x['penalty'] for x in tested)
            favored=[x['measured_hz'] for x in tested if abs(x['penalty']-lo)<1e-10]
        if curve_disagreement:flags.append('curve_distinguishes_measured_competitors')
        if prior is None:flags.append('curve_disabled_no_adjacent_valid_pair')
        # Rejection history only keeps descriptive evidence and is NEVER used
        # to auto-reject the next observation.
        r_out=dict(case=r.get('case'),sample=sample,time_s=t,
            production_median_hz=r.get('production_median_hz'),
            production_block_voiced_samples=r.get('production_block_voiced_samples'),
            review_comparison=r.get('review_comparison'),
            diagnostic=diagnostic,flags_json=json.dumps(flags),
            acoustic_candidates_json=json.dumps(items),
            curve_pair_eligible=prior is not None,curve_prior_hz=prior,
            curve_elapsed_ms=dt if prior is not None else None,
            curve_outputs_json=json.dumps(penalty_results),
            curve_relative_preference_json=json.dumps(favored),
            credible_provisional_anchor=credible,
            provisional_reference_hz=eligible[0]['hz'] if credible else None,
            predecessor_adjacency_verified=adjacent,
            rejection_evidence_json=json.dumps({'low_relative_energy':'relative_energy_low' in flags,
               'decay':'energy_falling' in flags,'temporal_conflict':'temporal_support_asymmetric' in flags,
               'failed_period':'no_eligible_period' in flags,
               'prior_diagnostic':previous['diagnostic'] if adjacent else None}),
            region_tags=';'.join(tag for tag,a,b in REGIONS.get(r.get('case'),[]) if t is not None and a<=t<=b),
            output_f0_modified=False,voicing_modified=False)
        out.append(r_out);counts[diagnostic]+=1
        previous={'sample':sample,'hz':eligible[0]['hz'] if credible else None,
                  'provisional':credible,'diagnostic':diagnostic}
    return out,counts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('all',*CASES),default='all')
    ap.add_argument('--v11-dir',type=Path,default=Path('tests/energy_observability_v11'))
    ap.add_argument('--out-dir',type=Path,default=Path('tests/source_observability_v12'))
    ap.add_argument('--hop-ms',type=float,default=10.,help='Expected observation spacing, NOT an aperture or BPM setting')
    a=ap.parse_args()
    names=CASES if a.case=='all' else (a.case,)
    try:
        from python_eckf.trajectory_resolver import vocal_transition_penalty
        curve=vocal_transition_penalty;curve_status='existing_function_imported_unchanged'
    except (ImportError,AttributeError) as exc:
        curve=None;curve_status='unavailable:'+str(exc)
        print('WARNING: existing curve unavailable; no replacement function supplied',file=sys.stderr)
    inputs={n:a.v11_dir/f'{n}_energy_observability.csv' for n in names}
    for file in inputs.values():
        if not file.is_file():ap.error(f'missing V11 file: {file}')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=dict(version='v12_shadow',curve_status=curve_status,cases={},
                  no_midi=True,production_modified=False,penalty_coefficients_unchanged=True,
                  no_noise_propagation=True,only_adjacent_candidate_pairs=True,
                  v11_is_review_only=True)
    for n,file in inputs.items():
        with file.open(newline='') as f:rows=sorted(csv.DictReader(f),key=lambda r:int(r['sample']))
        # Exact sample rate available in V11 manifest, but default WAV native fs
        # is looked up from explicit V11 manifest; never infer fs from BPM.
        mpath=a.v11_dir/'manifest.json'
        if not mpath.exists():ap.error(f'missing V11 manifest: {mpath}')
        metadata=json.loads(mpath.read_text())
        fs=number(metadata.get('cases',{}).get(n,{}).get('sample_rate'))
        if not fs or fs<=0:ap.error(f'V11 manifest missing native sample rate for {n}')
        out,counts=analyze_case(rows,curve,fs,a.hop_ms)
        if len(out)!=len(rows):raise AssertionError('review coverage lost')
        path=a.out_dir/f'{n}_source_observability.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
        manifest['cases'][n]=dict(review_observations=len(out),diagnostics=dict(counts),
                   curve_eligible_pairs=sum(x['curve_pair_eligible'] for x in out),
                   curve_evaluations=sum(len([p for p in decode(x['curve_outputs_json'],[]) if p['penalty'] is not None]) for x in out),
                   file=str(path))
        print(n,manifest['cases'][n],flush=True)
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: no F0/voicing decisions applied; no rejected-state propagation.')
if __name__=='__main__':main()
