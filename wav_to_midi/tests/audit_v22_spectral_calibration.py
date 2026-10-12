#!/usr/bin/env python3
"""Read-only V4/V5 spectral-scale and abstention audit. No F0 decisions made.

Run from repository root. Historical V13 references are SOURCE-UNVERIFIED controls,
not ground truth. This tool does not optimize coefficients or thresholds.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
REGRESSIONS = {'Ochiitai': 3.55, 'PREDESTINATI': 4.78}

def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def yes(x):
    return str(x).lower() in ('true', '1', 'yes')

def read(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required input missing: {path}')
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def quantiles(values):
    a = sorted(x for x in values if x is not None and math.isfinite(x))
    if not a:
        return {'n': 0, 'p10': None, 'p50': None, 'p90': None}
    def q(f):
        p = (len(a)-1)*f
        i = int(p)
        return a[i] + (a[min(i+1,len(a)-1)]-a[i])*(p-i)
    return {'n': len(a), 'p10': q(.1), 'p50': q(.5), 'p90': q(.9)}

def direction(x, epsilon=1e-10):
    if x is None: return 'unavailable'
    return 'high' if x > epsilon else 'low' if x < -epsilon else 'tie'

def exclusive_bin(x):
    if x is None: return 'missing'
    if x < .002: return 'below_0.2pct_floor'
    if x < .01: return '0.2_to_1pct'
    if x < .04: return '1_to_4pct'
    return 'at_least_4pct'

def acf_bin(x):
    if x is None: return 'missing'
    if x < -.12: return 'low_strong'
    if x < -.02: return 'low_modest'
    if x <= .02: return 'near_equal'
    if x <= .12: return 'high_modest'
    return 'high_strong'

def get_details(r, candidate):
    try:
        return json.loads(r.get('details_json') or '{}').get(candidate, {})
    except (ValueError, AttributeError):
        return {}

def margin(r):
    a,b = number(r.get('low_score')), number(r.get('high_score'))
    return b-a if a is not None and b is not None else None

def load_controls(path):
    controls = defaultdict(list)
    for r in read(path):
        if r.get('evidence_status') != 'acoustic_period_supported_source_unverified': continue
        if not yes(r.get('usable_as_next_reference')) or not yes(r.get('diagnostic_settled')): continue
        t,hz = number(r.get('time_s')),number(r.get('selected_measured_hz'))
        if t is not None and hz and hz > 0: controls[round(t,6)].append(hz)
    return controls

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tests-root',default='tests')
    p.add_argument('--v4',default='tests/shadow_adjudicator_v22_fixed_v4')
    p.add_argument('--v5',default='tests/shadow_adjudicator_v22_fixed_v5')
    p.add_argument('--out',default='tests/shadow_spectral_calibration_audit')
    p.add_argument('--case',choices=('all',)+CASES,default='all')
    args=p.parse_args()
    root,out=Path(args.tests_root),Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary={'schema':'v22_spectral_observational_audit_v1','read_only':True,'weights_modified':False,'production_modified':False,'midi_used':False,'controls_are_ground_truth':False,'cases':{}}
    for case in CASES if args.case=='all' else (args.case,):
        old=read(Path(args.v4)/f'{case}_pair_decisions.csv')
        new=read(Path(args.v5)/f'{case}_pair_decisions.csv')
        obs4=read(Path(args.v4)/f'{case}_observation_summary.csv')
        obs5=read(Path(args.v5)/f'{case}_observation_summary.csv')
        controls=load_controls(root/'full_timeline_v13'/f'{case}_full_timeline.csv')
        if len(old)!=len(new): raise ValueError(f'{case}: V4/V5 row counts differ; cannot align')
        results=[]; bins=defaultdict(list); counts=Counter(); regression=[]
        for index,(a,b) in enumerate(zip(old,new)):
            # Enforce identical pair identity, rather than silently joining on nonunique time.
            ident=('time_s','low_hz','high_hz')
            if any(number(a.get(k))!=number(b.get(k)) for k in ident):
                raise ValueError(f'{case}: mismatched V4/V5 pair at row {index+1}')
            t=number(b['time_s']); low=number(b['low_hz']); high=number(b['high_hz'])
            m4,m5=margin(a),margin(b)
            ex=number(b.get('low_exclusive_energy_median'))
            acf=number(b.get('acf_advantage_high_minus_low'))
            admissible=yes(b.get('low_admissible')) and yes(b.get('high_admissible'))
            vd=get_details(b,'high'); ld=get_details(b,'low')
            spec_h=number(vd.get('spectral_component')); spec_l=number(ld.get('spectral_component'))
            temp_h=number(vd.get('temporal_component')); temp_l=number(ld.get('temporal_component'))
            inter_h=number(vd.get('interval_component')); inter_l=number(ld.get('interval_component'))
            range_h=number(vd.get('range_component')); range_l=number(ld.get('range_component'))
            def delta(h,l): return h-l if h is not None and l is not None else None
            control_values=controls.get(round(t,6),[])
            control_hz=control_values[0] if len(control_values)==1 else None
            c_low=abs(1200*math.log2(low/control_hz)) if control_hz and low else None
            c_high=abs(1200*math.log2(high/control_hz)) if control_hz and high else None
            # Describe which hypothesis is closer to source-unverified V13; no label of correctness.
            closer='high' if c_low is not None and c_high is not None and c_high<c_low else 'low' if c_low is not None and c_high is not None and c_low<c_high else 'unavailable_or_tie'
            sdiff=delta(spec_h,spec_l)
            row={'case':case,'pair_index':index,'time_s':t,'low_hz':low,'high_hz':high,
                 'both_admissible':admissible,'v4_outcome':a.get('outcome'),'v5_outcome':b.get('outcome'),
                 'v4_margin_high_minus_low':m4,'v5_margin_high_minus_low':m5,
                 'margin_change_v5_minus_v4':delta(m5,m4),'v4_direction':direction(m4),'v5_direction':direction(m5),
                 'spectral_difference_v5':sdiff,'spectral_direction':direction(sdiff),
                 'acf_advantage_high_minus_low':acf,'low_exclusive_fraction':ex,
                 'exclusive_bin':exclusive_bin(ex),'acf_bin':acf_bin(acf),
                 'temporal_difference':delta(temp_h,temp_l),'interval_difference':delta(inter_h,inter_l),
                 'range_difference':delta(range_h,range_l),'v13_same_time_reference_hz':control_hz,
                 'v13_reference_count_at_time':len(control_values),
                 'low_distance_to_v13_cents':c_low,'high_distance_to_v13_cents':c_high,
                 'v13_closer_candidate_source_unverified':closer}
            results.append(row)
            counts[f'v4_{a.get("outcome")}']+=1
            counts[f'v5_{b.get("outcome")}']+=1
            if m4 is not None and m5 is not None:
                counts['both_scored']+=1
                if direction(m4)!=direction(m5):counts['score_direction_changed']+=1
                if a.get('outcome','').startswith('provisional') and not b.get('outcome','').startswith('provisional'):
                    counts['v4_decisive_v5_not']+=1
            if control_hz is not None: counts['same_time_v13_reference_pairs']+=1
            bins[(row['exclusive_bin'],row['acf_bin'],closer)].append(row)
            if case in REGRESSIONS and abs(t-REGRESSIONS[case])<.000001:regression.append(row)
        write(out/f'{case}_pair_comparison.csv',results)
        write(out/f'{case}_regression_pairs.csv',regression)
        bucket=[]
        for (eb,ab,cl), rr in sorted(bins.items()):
            bucket.append({'case':case,'exclusive_bin':eb,'acf_bin':ab,'v13_closer_candidate_source_unverified':cl,
                           'n_pairs':len(rr),'n_both_admissible':sum(r['both_admissible'] for r in rr),
                           'v4_margin':json.dumps(quantiles([r['v4_margin_high_minus_low'] for r in rr])),
                           'v5_margin':json.dumps(quantiles([r['v5_margin_high_minus_low'] for r in rr])),
                           'spectral_difference':json.dumps(quantiles([r['spectral_difference_v5'] for r in rr]))})
        write(out/f'{case}_evidence_bins.csv',bucket)
        # Capture V5's own abstention taxonomy intact; do not invent causal labels.
        abst=Counter(r.get('abstention_category') or r.get('observation_outcome') or 'unspecified' for r in obs5)
        changes=Counter()
        if len(obs4)==len(obs5):
            for a,b in zip(obs4,obs5):
                if a.get('time_s')!=b.get('time_s'):raise ValueError(f'{case}: observation alignment differs')
                changes[(a.get('observation_outcome'),b.get('observation_outcome'))]+=1
        else: raise ValueError(f'{case}: observation counts differ')
        write(out/f'{case}_observation_transitions.csv',[
            {'case':case,'v4_outcome':k[0],'v5_outcome':k[1],'count':n} for k,n in sorted(changes.items())])
        summary['cases'][case]={'pair_count':len(results),'observations':len(obs5),'pair_counts':dict(counts),
                                'v5_abstention_categories':dict(abst),'v4_to_v5_observation_outcomes':[
                                    {'from':k[0],'to':k[1],'count':n} for k,n in sorted(changes.items())],
                                'regression_pair_count':len(regression),
                                'v13_same_time_control_rows':len(controls),
                                'warning':'V13 controls are acoustically supported but source-unverified; counts are descriptive, not accuracy estimates.'}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
