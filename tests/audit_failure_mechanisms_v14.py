#!/usr/bin/env python3
"""V14: full-song, READ-ONLY audit of V13 dark cases and supported periods.

Inputs: V13 full timeline, V3 acoustic stages, V5 production comparison.
V5 production block status is NOT sample-level truth. No WAV or MIDI reference
pitch is read, no F0/state is changed, and no previous pitch is propagated.
This is an evidence inventory, NOT a voice-vs-reverb classifier.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
REGIONS = {
 'RATATA': [('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.915,43.065)],
 'Ochiitai': [('transition',42.19,42.40)],
 'PREDESTINATI': [('control_38',37.97,38.13),('control_53',53.52,53.73)],
 'Trandafiri': [('steady_i_not_glissando',10.70,10.95)],
}

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required input: {path}')
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def truth(value):
    return str(value).strip().lower() == 'true'

def json_list(value):
    try:
        v=json.loads(value or '[]')
        return v if isinstance(v,list) else []
    except (ValueError,TypeError):
        return []

def cents(a,b):
    return abs(1200*math.log2(a/b)) if a and b and a>0 and b>0 else None

def index_rows(rows, label):
    out={}
    for r in rows:
        k=int(r['sample'])
        if k in out:
            raise ValueError(f'Duplicate sample {k} in {label}')
        out[k]=r
    return out

def evidence_from_stages(stages):
    last=stages[-1] if stages else {}
    raw=last.get('raw_candidates') or []
    local=[]; testable=[]; insuff=[]; remote=[]; ineligible=[]
    for item in raw:
        candidate=item.get('candidate') or {}
        hz=number(candidate.get('hz'))
        if hz is None or hz<=0: continue
        p=item.get('present') or {}
        entry={'hz':hz,'acf':number(candidate.get('acf')),
               'cmndf':number(candidate.get('cmndf')),
               'fundamental_fraction':number(p.get('fundamental_fraction')),
               'present_acf':number(p.get('acf')),
               'present_cycles':number(p.get('cycles')),
               'attribution':item.get('attribution'),
               'eligible':bool(item.get('eligible')),
               'testability':p.get('testability')}
        if entry['testability']=='testable': testable.append(entry)
        else: insuff.append(entry)
        if entry['eligible'] and entry['attribution']=='local_support_exploratory' and entry['testability']=='testable':
            local.append(entry)
        if 'distant' in str(entry['attribution']): remote.append(entry)
        if not entry['eligible']: ineligible.append(entry)
    all_raw=sum(len(s.get('raw_candidates') or []) for s in stages)
    later_new=any((s.get('raw_candidate_count') or 0)>0 for s in stages[1:])
    first_raw=(stages[0].get('raw_candidate_count') or 0)>0 if stages else False
    # Apertures never inspected remain UNKNOWN, not failures or opportunities.
    return dict(last=last,raw=raw,local=local,testable=testable,insuff=insuff,
                remote=remote,ineligible=ineligible,stage_count=len(stages),
                total_raw_across_stages=all_raw,first_raw=first_raw,
                later_raw_observed=later_new,uninspected_count=max(0,4-len(stages)))

def temporal_energy(e):
    p=e.get('present') or {}; left=e.get('earlier') or {}; right=e.get('later') or {}
    pr=number(p.get('rms')); lr=number(left.get('rms')); rr=number(right.get('rms'))
    if pr is None or lr is None or rr is None: return 'not_measured',pr,lr,rr
    if lr>1e-12 and rr>1e-12:
        if pr<0.5*lr and rr<0.6*lr: return 'earlier_energy_dominant',pr,lr,rr
        if pr<0.5*rr and lr<0.6*rr: return 'later_energy_dominant',pr,lr,rr
    if lr>1e-12 and rr<0.6*lr: return 'energy_decreasing',pr,lr,rr
    if rr>1e-12 and lr<0.6*rr: return 'energy_increasing',pr,lr,rr
    return 'no_clear_energy_direction',pr,lr,rr

def classify_dark(v13,e):
    status=v13['evidence_status']
    if truth(v13.get('interval_energy_joint_challenge')):
        return 'joint_interval_energy_challenge','Inspect measured candidates and unchanged penalty; do not carry rejected pitch as next reference.'
    if not e['raw']:
        if e['total_raw_across_stages']:
            return 'candidate_lost_by_final_aperture','Compare earlier measured apertures and local support; later apertures not necessarily better.'
        return 'no_period_any_inspected_aperture','Examine local cycle sufficiency and acoustic energy; uninspected apertures remain unknown.'
    if e['remote'] and not e['local']:
        return 'periodicity_not_at_present','Compare actual earlier/present/later sample intervals; avoid importing distant pitch.'
    if e['local'] and len(e['local'])>1:
        return 'multiple_locally_supported_periods','Compare measured ACF/CMNDF, fundamental/harmonic support and shift robustness; no automatic winner.'
    if e['insuff'] and not e['local'] and not e['testable']:
        return 'insufficient_local_cycles','Check whether earlier inspected aperture gains local cycles; no claim of unvoicing.'
    if e['ineligible'] and not e['local']:
        return 'measured_but_not_eligible','Inspect independent rejection criteria, energy and present-time periodicity.'
    if e['testable'] and not e['local']:
        return 'testable_period_without_local_support','Compare present and flank support at the measured period; inspect candidate instability.'
    if e['local']:
        return 'supported_period_not_selected_or_unresolved','Inspect V13 choice/competition criteria and candidate membership; retain all measured alternatives.'
    return 'other_unresolved_acoustic_evidence','Inspect all inspected apertures and explicit candidate rejection reasons.'

def classify_supported(v13,e):
    selected=number(v13.get('selected_measured_hz'))
    match=min((c for c in e['local'] if cents(c['hz'],selected) is not None and cents(c['hz'],selected)<=35),
              key=lambda c:cents(c['hz'],selected),default=None)
    if match is None:
        return 'missing_selected_candidate_in_v3','Verify V13/V3 files belong to same run and same samples.',None
    direction,pr,lr,rr=temporal_energy(next((r for r in e['raw'] if number((r.get('candidate') or {}).get('hz')) is not None and cents(number((r.get('candidate') or {}).get('hz')),selected) is not None and cents(number((r.get('candidate') or {}).get('hz')),selected)<=35),{}))
    ratio=number(v13.get('relative_energy')); slope=number(v13.get('energy_slope_db'))
    # These labels are review flags ONLY. Low energy is never an unvoiced verdict.
    if ratio is not None and ratio<.25 and slope is not None and slope<-4:
        return 'low_relative_energy_and_decay','Compare an independently measured quiet sung control; test onset and local support. Tail identity unproven.',direction
    if direction=='earlier_energy_dominant':
        return 'earlier_energy_dominant','Check whether preceding periodic sound is being borrowed; do not infer reverb from mono WAV.',direction
    if direction=='later_energy_dominant':
        return 'later_energy_dominant','Check onset timing against future-only and center views; no onset declared.',direction
    if len(e['local'])>1:
        return 'supported_with_additional_measured_periods','Inspect relative fundamental and harmonic evidence for the other eligible members.',direction
    if match['fundamental_fraction'] is not None and match['fundamental_fraction']<.015:
        return 'weak_fundamental_component','Inspect higher harmonics and independent period evidence; weak fundamental is not automatic rejection.',direction
    return 'locally_supported_source_identity_open','Period measurable; active voice versus quiet note versus residual sound is not identifiable from this evidence alone.',direction

def run(name,v13rows,v3rows,v5rows):
    v3=index_rows(v3rows,'V3 '+name);v5=index_rows(v5rows,'V5 '+name)
    v13=index_rows(v13rows,'V13 '+name)
    if set(v13)!=set(v3) or set(v13)!=set(v5):
        raise ValueError(f'{name}: sample-key sets differ: V13={len(v13)} V3={len(v3)} V5={len(v5)}; regenerate matched runs')
    full=[]; dark=[]; supported=[]; counts=Counter(); matrix=Counter(); regions=defaultdict(Counter)
    for sample in sorted(v13):
        a=v13[sample]; b=v3[sample]; p=v5[sample]
        if abs(float(a['time_s'])-float(b['time_s']))>0.0015 or abs(float(a['time_s'])-float(p['time_s']))>0.0015:
            raise ValueError(f'{name}: inconsistent time at sample {sample}')
        stages=json_list(b.get('stages_json'));e=evidence_from_stages(stages)
        label=a['evidence_status']
        if label=='acoustic_period_supported_source_unverified':
            group='supported_period';mechanism,route,direction=classify_supported(a,e)
        elif label=='no_measured_period_source_unverified':
            group='no_raw_period';mechanism='no_measured_period_not_proof_of_silence';route='Check energy and cycles if required; do not emit pitch or assert silence.';direction='not_applicable'
        else:
            group='dark';mechanism,route=classify_dark(a,e);direction='not_applicable'
        prod_status=p.get('production_status','')
        prod_hz=number(p.get('production_median_hz'))
        voiced=number(p.get('production_voiced_samples'))
        if voiced is None: overlap='unavailable'
        elif voiced==0: overlap='production_block_no_voiced_samples'
        else: overlap='production_block_some_voiced_samples'
        # Production is contextual evidence, NEVER a decision source.
        if group=='dark':
            contribution='production_candidate_for_independent_acoustic_test' if prod_hz and voiced and voiced>0 else 'no_production_pitch_available'
        else: contribution='not_dark'
        t=float(a['time_s']);tags=[tag for tag,start,end in REGIONS.get(name,[]) if start<=t<=end]
        record=dict(case=name,sample=sample,time_s=t,population=group,v13_evidence_status=label,
            mechanism=mechanism,next_investigation=route,source_identity_verified=False,final_f0_decided=False,
            production_block_status=prod_status,production_voiced_samples=voiced,production_median_hz=prod_hz,
            production_candidate_contribution=contribution,production_is_ground_truth=False,
            v13_measured_hz=number(a.get('selected_measured_hz')),
            production_vs_measured_cents=cents(prod_hz,number(a.get('selected_measured_hz'))),
            v13_relative_energy=number(a.get('relative_energy')),
            v13_slope_db=number(a.get('energy_slope_db')),
            v13_interval_penalty=number(a.get('penalty')),
            v13_joint_challenge=truth(a.get('interval_energy_joint_challenge')),
            final_aperture_ms=number((e['last'] or {}).get('aperture_ms')),
            inspected_apertures=e['stage_count'],uninspected_apertures_unknown=e['uninspected_count'],
            raw_at_final=len(e['raw']),raw_total_inspected=e['total_raw_across_stages'],
            locally_supported_measured=len(e['local']),
            insufficient_cycle_candidates=len(e['insuff']),
            remote_period_candidates=len(e['remote']),
            eligible_candidates_json=json.dumps(e['local'],separators=(',',':')),
            all_final_raw_candidates_json=json.dumps(e['raw'],separators=(',',':')),
            energy_temporal_pattern=direction,annotation_tags=';'.join(tags),
            decision_applied=False,production_modified=False)
        full.append(record);counts[(group,mechanism)]+=1;matrix[(group,overlap)]+=1
        if group=='dark':dark.append(record)
        if group=='supported_period':supported.append(record)
        for tag in tags:regions[tag][group]+=1
    assert len(full)==len(v13rows)
    assert len(dark)+len(supported)+sum(r['population']=='no_raw_period' for r in full)==len(full)
    return full,dark,supported,counts,matrix,regions

def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('all',*CASES),default='all')
    ap.add_argument('--v13-dir',type=Path,default=Path('tests/full_timeline_v13'))
    ap.add_argument('--v3-dir',type=Path,default=Path('tests/binocular_evidence_search_v3'))
    ap.add_argument('--v5-dir',type=Path,default=Path('tests/acoustic_observability_v5'))
    ap.add_argument('--out-dir',type=Path,default=Path('tests/failure_mechanisms_v14'))
    args=ap.parse_args()
    names=CASES if args.case=='all' else (args.case,)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'v14_read_only_failure_mechanisms','production_is_ground_truth':False,
              'active_source_classification':'NOT ESTABLISHED: review flags only',
              'final_pitch_decisions_applied':False,'inputs':{'v13':str(args.v13_dir),'v3':str(args.v3_dir),'v5':str(args.v5_dir)},
              'cases':{}}
    totals=Counter()
    for name in names:
        v13=read_csv(args.v13_dir/f'{name}_full_timeline.csv')
        v3=read_csv(args.v3_dir/f'{name}.csv')
        v5=read_csv(args.v5_dir/f'{name}.csv')
        full,dark,supported,counts,matrix,regions=run(name,v13,v3,v5)
        fields=list(full[0])
        write_csv(args.out_dir/f'{name}_full_audit.csv',full,fields)
        write_csv(args.out_dir/f'{name}_dark.csv',dark,fields)
        write_csv(args.out_dir/f'{name}_supported_source_audit.csv',supported,fields)
        no_raw=len(full)-len(dark)-len(supported)
        manifest['cases'][name]={'total':len(full),'dark':len(dark),'supported_period_source_unverified':len(supported),
            'no_raw_period_not_silence':no_raw,
            'dark_with_production_voiced_block':sum(r['population']=='dark' and r['production_voiced_samples'] is not None and r['production_voiced_samples']>0 for r in full),
            'dark_with_production_pitch_available_for_testing':sum(r['population']=='dark' and r['production_candidate_contribution']=='production_candidate_for_independent_acoustic_test' for r in full),
            'mechanisms':{f'{g}/{m}':n for (g,m),n in sorted(counts.items())},
            'production_overlap':{f'{g}/{p}':n for (g,p),n in sorted(matrix.items())},
            'regions':{k:dict(v) for k,v in regions.items()}}
        totals.update(total=len(full),dark=len(dark),supported=len(supported),no_raw=no_raw)
        print(f'{name}: total={len(full)} dark={len(dark)} supported_period_source_unverified={len(supported)} no_raw_period={no_raw} production_pitch_in_dark={manifest["cases"][name]["dark_with_production_pitch_available_for_testing"]}',flush=True)
    manifest['totals']=dict(totals)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY. Production pitch is an independent TEST CANDIDATE, never automatic rescue or ground truth.')
if __name__=='__main__':main()
