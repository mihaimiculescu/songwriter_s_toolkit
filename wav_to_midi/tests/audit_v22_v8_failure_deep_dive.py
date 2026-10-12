#!/usr/bin/env python3
"""Read-only V8 deep dive: tournament gaps, margin shortfalls, and acoustic gate failures.

Inputs: V8 pair/observation CSVs and the ORIGINAL V21 temporal-pairs CSVs.
No WAV analysis, parameter changes, winner selection, or ground-truth claims.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
WINDOWS = ('24ms_-8ms','24ms_+0ms','24ms_+8ms','40ms_-8ms','40ms_+0ms','40ms_+8ms','64ms_-8ms','64ms_+0ms','64ms_+8ms')

def read(path):
    if not path.is_file(): raise FileNotFoundError(f'Required input missing: {path}')
    with path.open(newline='',encoding='utf-8') as fh: return list(csv.DictReader(fh))

def write(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def number(value):
    try:
        x=float(value);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None

def yes(value):return str(value).lower().strip() in ('true','1','yes')
def ts(row):return round(float(row['time_s']),6)
def pairkey(row):return (ts(row),round(float(row['low_hz']),5),round(float(row['high_hz']),5))

def measurement_analysis(row,side,threshold,minimum):
    blob=json.loads(row.get('evidence_json') or '{}')
    windows=blob.get('shift_windows',{})
    if not isinstance(windows,dict):windows={}
    total=untestable=missing=malformed=passed=failed=0; values=[]; details=[]
    for window,payload in windows.items():
        total+=1
        entry=(payload or {}).get(side) if isinstance(payload,dict) else None
        if not isinstance(entry,dict):
            missing+=1;details.append(f'{window}:missing_candidate');continue
        if not yes(entry.get('testable')):
            untestable+=1;details.append(f'{window}:untestable');continue
        v=number(entry.get('acf'))
        if v is None:
            malformed+=1;details.append(f'{window}:missing_acf');continue
        values.append(v)
        if v>=threshold:passed+=1;details.append(f'{window}:pass:{v:.4f}')
        else:failed+=1;details.append(f'{window}:below:{v:.4f}')
    if not windows: status='no_shift_windows'
    elif not values:status='no_testable_acf' if not malformed else 'no_numeric_acf'
    elif passed>=minimum:status='passes_gate'
    elif max(values)<threshold:status='all_measured_acf_below_threshold'
    else:status='some_acf_pass_but_too_few'
    return {'gate_cause':status,'windows_present':total,'untestable_windows':untestable,
            'missing_candidate_windows':missing,'missing_numeric_acf':malformed,'testable_numeric_acf':len(values),
            'below_threshold_acf':failed,'above_threshold_acf':passed,
            'acf_min':min(values) if values else '', 'acf_max':max(values) if values else '',
            'acf_median':sorted(values)[len(values)//2] if values else '',
            'acf_values_json':json.dumps(values),'window_audit_json':json.dumps(details)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v8',type=Path,default=Path('tests/shadow_adjudicator_v22_fixed_v8'))
    p.add_argument('--v21',type=Path,default=Path('tests/temporal_integer_families_v21'))
    p.add_argument('--out',type=Path,default=Path('tests/v8_failure_deep_dive'))
    p.add_argument('--case',choices=('all',)+CASES,default='all')
    p.add_argument('--min-acf',type=float,default=.72)
    p.add_argument('--min-supported-measurements',type=int,default=2)
    p.add_argument('--margins',type=float,nargs='+',default=[.175,.15,.10])
    a=p.parse_args()
    if a.out.resolve() in (a.v8.resolve(),a.v21.resolve()):p.error('Output must differ from input directories')
    if not 0<=a.min_acf<=1 or a.min_supported_measurements<1 or any(x<0 for x in a.margins):p.error('Invalid thresholds')
    selected=CASES if a.case=='all' else (a.case,)
    acoustic=[]; observations=[]; comparisons=[]; per_song=[]; cause_count=Counter()
    for case in selected:
        v8p=read(a.v8/f'{case}_pair_decisions.csv');v8o=read(a.v8/f'{case}_observation_summary.csv')
        v21p=read(a.v21/f'{case}_temporal_pairs.csv')
        lookup=defaultdict(list)
        for r in v21p: lookup[pairkey(r)].append(r)
        obs_pairs=defaultdict(list); song=Counter(); matched=0
        for row in v8p:
            key=pairkey(row); prior=lookup.get(key)
            if not prior:raise ValueError(f'{case}: V8/V21 pair cannot be matched: {key}')
            source=prior.pop(0);matched+=1
            for side in ('low','high'):
                d=measurement_analysis(source,side,a.min_acf,a.min_supported_measurements)
                admitted=yes(row.get(side+'_range_admitted'))
                declared=yes(row.get(side+'_admissible'))
                if admitted and (d['gate_cause']=='passes_gate')!=declared:
                    raise ValueError(f'{case} {key} {side}: reconstructed gate disagrees with V8')
                acoustic.append({'case':case,'time_s':row['time_s'],'low_hz':row['low_hz'],
                    'high_hz':row['high_hz'],'candidate':side,'candidate_hz':row[side+'_hz'],
                    'range_admitted':admitted,'v8_admissible':declared,'v8_reason':row.get(side+'_admissibility_reason'),
                    **d})
                song[f'candidate_{d["gate_cause"]}']+=1
                if admitted:song[f'admitted_{d["gate_cause"]}']+=1
            obs_pairs[ts(row)].append(row)
        if matched!=len(v21p) - sum(len(v) for v in lookup.values()):raise ValueError('Pair join accounting mismatch')
        if len(obs_pairs)!=len(v8o):raise ValueError(f'{case}: observation accounting mismatch')
        for obs in v8o:
            t=ts(obs);rows=obs_pairs[t]
            if len(rows)!=int(obs['pair_records']):raise ValueError(f'{case} {t}: pair count mismatch')
            outcome=obs['observation_outcome'];reason=obs.get('abstention_cause') or obs.get('abstention_reason') or ''
            admitted=[(r,s) for r in rows for s in ('low','high') if yes(r[s+'_range_admitted'])]
            valid=[(r,s) for r,s in admitted if yes(r[s+'_admissible'])]
            decisive=[r for r in rows if str(r['outcome']).startswith('provisional_')]
            direct=[r for r in rows if r['outcome'] in ('provisional_high','provisional_low')]
            margins=[]
            for r in rows:
                lo=number(r.get('low_score'));hi=number(r.get('high_score'))
                if lo is not None and hi is not None:margins.append(abs(hi-lo))
            # Classifications are descriptive; NEVER create counterfactual winners from scores.
            if outcome=='provisional_tournament_champion':cause='champion_reference'
            elif outcome=='abstain_all_candidates_out_of_range':cause='all_range_rejected'
            elif outcome=='abstain_single_candidate_no_acoustic_support':cause='unsupported_singleton'
            elif outcome=='abstain_tournament_no_unique_champion':cause='tournament_no_unique_champion'
            elif outcome=='abstain_no_decisive_pair':
                if not valid:cause='no_acoustically_admissible_candidate'
                elif any(r['outcome']=='abstain_insufficient_separation' for r in rows):cause='insufficient_score_margin'
                elif any('weak' in r['outcome'] for r in rows):cause='absolute_score_failure'
                elif decisive and not direct:cause='no_direct_head_to_head_victory'
                else:cause='other_no_decisive_pair'
            else:cause='other_'+outcome
            song[cause]+=1;cause_count[cause]+=1
            shortfall=min([max(0,.2-m) for m in margins],default=None)
            audit={'case':case,'time_s':t,'v8_outcome':outcome,'cause':cause,'v8_abstention_detail':reason,
                'pair_count':len(rows),'admitted_candidate_records':len(admitted),
                'acoustically_admissible_candidate_records':len(valid),
                'provisional_pair_wins':len(decisive),'direct_head_to_head_wins':len(direct),
                'comparable_scored_pairs':len(margins),'best_margin':max(margins) if margins else '',
                'closest_margin_to_0_20':min(margins,key=lambda v:abs(v-.2)) if margins else '',
                'minimal_margin_shortfall':shortfall if shortfall is not None else '',
                'candidate_gate_causes_json':json.dumps(dict(Counter(r['gate_cause'] for r in acoustic if r['case']==case and round(float(r['time_s']),6)==t and r['range_admitted'])),sort_keys=True),
                'pair_outcomes_json':json.dumps(dict(Counter(r['outcome'] for r in rows)),sort_keys=True)}
            observations.append(audit)
            if cause in ('insufficient_score_margin','no_direct_head_to_head_victory','tournament_no_unique_champion'):
                for r in rows:
                    lo=number(r.get('low_score'));hi=number(r.get('high_score'))
                    margin=abs(hi-lo) if lo is not None and hi is not None else None
                    comparisons.append({'case':case,'time_s':t,'observation_cause':cause,
                        'low_hz':r['low_hz'],'high_hz':r['high_hz'],'low_admissible':r['low_admissible'],
                        'high_admissible':r['high_admissible'],'low_range_admitted':r['low_range_admitted'],
                        'high_range_admitted':r['high_range_admitted'],'pair_outcome':r['outcome'],
                        'low_score':r.get('low_score'),'high_score':r.get('high_score'),
                        'absolute_margin':margin if margin is not None else '',
                        'shortfall_0_20':max(0,.2-margin) if margin is not None else '',
                        **{f'clears_margin_{m:g}':margin is not None and margin>=m for m in a.margins},
                        'note':'Margin diagnostic only: cannot infer tournament winner from pair margin alone.'})
        per_song.append({'case':case,'observations':len(v8o),'pair_records':len(v8p),**dict(song)})
    write(a.out/'acoustic_candidate_audit.csv',acoustic)
    write(a.out/'observation_decomposition.csv',observations)
    write(a.out/'head_to_head_and_margin_audit.csv',comparisons)
    write(a.out/'per_song.csv',per_song)
    write(a.out/'cause_summary.csv',[{'cause':k,'observations':v} for k,v in sorted(cause_count.items())])
    manifest={'schema':'v8_deep_dive_read_only','cases':list(selected),'observations':len(observations),
        'candidate_records':len(acoustic),'comparison_records':len(comparisons),
        'cause_counts':dict(cause_count),'acf_threshold':a.min_acf,
        'min_supported_measurements':a.min_supported_measurements,'alternative_margins_observational_only':a.margins,
        'decision_changes':False,'production_modified':False,'wav_remeasured':False,
        'warning':'V13 references are source-unverified. Scores and alternate margins are NOT proof of true F0.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=='__main__':main()
