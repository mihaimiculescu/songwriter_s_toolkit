#!/usr/bin/env python3
"""V18 adaptive-only cross-aperture evidence audit; READ ONLY.

Inputs: complete V3 stages and V14 full audit. No production fields are read.
This is a hypothesis inventory, NOT a new F0 detector or a proof of source identity.
Every reported candidate was measured by V3; no smoothing, pitch propagation,
new frequencies, MIDI reference or interval prior. Uninspected views stay unknown.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import Counter, defaultdict

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={'RATATA':[('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.90,43.10)],'Ochiitai':[('transition',42.19,42.40)],'PREDESTINATI':[('control38',37.97,38.13),('control53',53.52,53.73)],'Trandafiri':[('steady_i_not_glissando',10.70,10.95)]}

def read(p):
    if not p.is_file(): raise FileNotFoundError(f'Required input missing: {p}')
    with p.open(newline='') as f:return list(csv.DictReader(f))
def number(v):
    try:
        v=float(v);return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None
def parse(v):
    try:return json.loads(v)
    except (ValueError,TypeError):return []
def distance(a,b):return abs(1200*math.log2(a/b))
def peaks(stages):
    out=[]
    for si,s in enumerate(stages):
        for entry in s.get('raw_candidates') or []:
            c=entry.get('candidate') or {};p=entry.get('present') or {};hz=number(c.get('hz'))
            if hz is None or hz<=0:continue
            out.append(dict(stage_index=si,aperture_ms=number(s.get('aperture_ms')),
                hz=hz,eligible=bool(entry.get('eligible')),
                local=entry.get('attribution')=='local_support_exploratory',
                attribution=entry.get('attribution'),testable=p.get('testability')=='testable',
                cycles=number(p.get('cycles')),acf=number(p.get('acf')),
                cmndf=number(c.get('cmndf')),
                fundamental_fraction=number(p.get('fundamental_fraction'))))
    return out

def families(items):
    # Frequency clustering at 35 cents, measured periods only; integer-related
    # periods stay separate hypotheses (never silently merged as same F0).
    result=[]
    for p in sorted(items,key=lambda x:x['hz']):
        matches=[f for f in result if distance(p['hz'],f['representative_hz'])<=35]
        if matches:matches[0]['members'].append(p)
        else:result.append({'representative_hz':p['hz'],'members':[p]})
    for f in result:
        m=f['members'];f['apertures_ms']=sorted({x['aperture_ms'] for x in m if x['aperture_ms'] is not None})
        f['locally_eligible_apertures_ms']=sorted({x['aperture_ms'] for x in m if x['eligible'] and x['local'] and x['testable'] and x['aperture_ms'] is not None})
        f['present_testable_apertures_ms']=sorted({x['aperture_ms'] for x in m if x['testable'] and x['aperture_ms'] is not None})
        f['best_present_acf']=max((x['acf'] for x in m if x['acf'] is not None),default=None)
        f['best_fundamental_fraction']=max((x['fundamental_fraction'] for x in m if x['fundamental_fraction'] is not None),default=None)
        f['temporal_attributions']=sorted({str(x['attribution']) for x in m})
        f['all_observed_eligible']=any(x['eligible'] for x in m)
        del f['members']
    return result

def audit(case,v3,v14):
    d3={int(r['sample']):r for r in v3};d14={int(r['sample']):r for r in v14}
    if len(d3)!=len(v3) or len(d14)!=len(v14) or set(d3)!=set(d14):
        raise ValueError(f'{case}: V3/V14 samples mismatch or duplicates')
    rows=[]; counts=Counter();mechanisms=defaultdict(Counter);regions=defaultdict(Counter)
    for sample in sorted(d3):
        a=d3[sample];b=d14[sample]
        if abs(float(a['time_s'])-float(b['time_s']))>.0015:raise ValueError(f'{case} time mismatch {sample}')
        population=b['population'];stages=parse(a['stages_json']);observed=peaks(stages);fs=families(observed)
        final=stages[-1] if stages else {};final_index=len(stages)-1
        final_local=[x for x in observed if x['stage_index']==final_index and x['eligible'] and x['local'] and x['testable']]
        previous_local=[x for x in observed if x['stage_index']<final_index and x['eligible'] and x['local'] and x['testable']]
        multistage=[f for f in fs if len(f['locally_eligible_apertures_ms'])>=2]
        final_family=[f for f in fs if (number(final.get('aperture_ms')) in f['locally_eligible_apertures_ms'])]
        measured=number(b.get('v13_measured_hz'))
        selected_family=min((f for f in fs if measured and distance(f['representative_hz'],measured)<=35),key=lambda f:distance(f['representative_hz'],measured),default=None)
        near_integer=[]
        for i,f in enumerate(fs):
            for g in fs[i+1:]:
                ratio=max(f['representative_hz'],g['representative_hz'])/min(f['representative_hz'],g['representative_hz'])
                for n in (2,3,4):
                    if abs(1200*math.log2(ratio/n))<=45:
                        near_integer.append({'lower_hz':min(f['representative_hz'],g['representative_hz']), 'higher_hz':max(f['representative_hz'],g['representative_hz']),'ratio':ratio,'near_integer':n});break
        # The following labels are investigative priorities, never acceptance/rejection.
        if population=='dark':
            if previous_local and not final_local:priority='earlier_aperture_local_evidence_lost_in_final'
            elif multistage and len(final_local)>1:priority='persistent_competing_measured_families'
            elif near_integer and len(final_local)>1:priority='integer_related_competition'
            elif final_local:priority='final_local_evidence_selection_unresolved'
            elif observed and not any(x['testable'] for x in observed):priority='cycle_testability_gap'
            elif observed and any(x['testable'] and not x['local'] for x in observed):priority='temporal_attribution_gap'
            elif observed:priority='eligibility_gap'
            else:priority='no_period_measured_in_inspected_apertures'
        elif population=='supported_period':
            if selected_family is None:priority='selected_frequency_missing_from_stage_families'
            elif near_integer:priority='supported_integer_family_source_check'
            elif (number(b.get('v13_relative_energy')) is not None and number(b.get('v13_relative_energy'))<.25 and number(b.get('v13_slope_db')) is not None and number(b.get('v13_slope_db'))< -4):priority='supported_low_energy_decay_source_check'
            elif not selected_family['locally_eligible_apertures_ms']:priority='supported_not_reproduced_by_stage_locality'
            else:priority='supported_period_source_identity_unverified'
        else:priority='no_period_measured_source_unknown'
        tag=[x for x,start,end in REGIONS.get(case,[]) if start<=float(a['time_s'])<=end]
        result=dict(case=case,sample=sample,time_s=a['time_s'],population=population,
            v14_mechanism=b['mechanism'],investigation_priority=priority,
            inspected_aperture_count=len(stages),inspected_apertures_ms=json.dumps([s.get('aperture_ms') for s in stages]),
            final_aperture_ms=final.get('aperture_ms'),all_measured_periods=len(observed),
            unique_measured_families=len(fs),final_local_eligible=len(final_local),earlier_local_eligible=len(previous_local),
            cross_aperture_locally_eligible_families=len(multistage),integer_related_pairs=len(near_integer),
            v13_selected_measured_hz=measured,selected_family_aperture_support=json.dumps(selected_family['locally_eligible_apertures_ms'] if selected_family else []),
            families_json=json.dumps(fs,separators=(',',':')),integer_pairs_json=json.dumps(near_integer,separators=(',',':')),
            region_tags=';'.join(tag),new_pitch_selected=False,active_vocal_source_verified=False,
            adjudication_changed=False,production_data_used=False)
        rows.append(result);counts[population]+=1;mechanisms[population][priority]+=1
        for t in tag:regions[t][priority]+=1
    return rows,counts,mechanisms,regions

def write(path,records):
    if not records:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('all',*CASES),default='all')
    ap.add_argument('--v3-dir',type=Path,default=Path('tests/binocular_evidence_search_v3'))
    ap.add_argument('--v14-dir',type=Path,default=Path('tests/failure_mechanisms_v14'))
    ap.add_argument('--out-dir',type=Path,default=Path('tests/adaptive_criteria_v18'))
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'v18_adaptive_only_cross_aperture_hypothesis_audit',
        'inputs':{'v3':str(args.v3_dir),'v14':str(args.v14_dir)},
        'production_data_used':False,'wav_remeasured':False,'new_pitch_selected':False,
        'source_identity_verified':False,'final_f0_rescues':0,'production_modified':False,'cases':{}}
    totals=Counter()
    for name in CASES if args.case=='all' else (args.case,):
        rows,groups,mechs,regions=audit(name,read(args.v3_dir/f'{name}.csv'),read(args.v14_dir/f'{name}_full_audit.csv'))
        write(args.out_dir/f'{name}_all.csv',rows)
        write(args.out_dir/f'{name}_dark.csv',[r for r in rows if r['population']=='dark'])
        write(args.out_dir/f'{name}_supported_source.csv',[r for r in rows if r['population']=='supported_period'])
        manifest['cases'][name]={'total':len(rows),'populations':dict(groups),
            'priorities':{g:dict(c) for g,c in mechs.items()},'regions':{g:dict(c) for g,c in regions.items()}}
        totals.update(groups)
        print(f'{name}: total={len(rows)} dark={groups["dark"]} supported_source_unverified={groups["supported_period"]} no_raw={groups["no_raw_period"]}',flush=True)
        for label,n in mechs['dark'].most_common():print(f'  dark {label}: {n}',flush=True)
    manifest['totals']=dict(totals)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('V18: measured-candidate evidence inventory only. No new F0, no certified noise, no production input.',flush=True)
if __name__=='__main__':main()
