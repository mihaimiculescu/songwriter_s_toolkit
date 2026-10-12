#!/usr/bin/env python3
"""Read-only exclusive-energy calibration study for V22/V5.

Default: descriptive V13 proxy analysis ONLY; no calibrated model is claimed.
Optional --labels: independently reviewed pair labels low/high/uncertain.
One observation per time-block and case is sampled deterministically to reduce
10-ms-frame pseudoreplication; holdout Ochiitai 3.550 / PREDESTINATI 4.780.
No production imports, MIDI, F0 writes, or changes to V5 scoring.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
REGRESSION = {'Ochiitai':3.550, 'PREDESTINATI':4.780}
EDGES = (0.0, .0005, .001, .002, .004, .01, .02, .04, .08, 1.01)

def num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def rows(path):
    if not path.is_file(): raise FileNotFoundError(f'Missing required input: {path}')
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def output(path, data, names=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    names=names or list(dict.fromkeys(k for r in data for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=names); w.writeheader(); w.writerows(data)

def cents(a,b): return abs(1200*math.log2(a/b))

def bin_index(v):
    for i in range(len(EDGES)-1):
        if EDGES[i] <= v < EDGES[i+1]: return i
    return len(EDGES)-2

def quant(v,p):
    v=sorted(v)
    if not v:return None
    k=(len(v)-1)*p; lo=int(k);hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(k-lo)

def wilson(k,n):
    if n==0:return (None,None)
    z=1.96; p=k/n; d=1+z*z/n; center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return center-half,center+half

def pava(values,weights):
    blocks=[]
    for i,(value,weight) in enumerate(zip(values,weights)):
        if not weight:continue
        blocks.append([i,i,value*weight,weight])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks[-1];a[1]=b[1];a[2]+=b[2];a[3]+=b[3]
    out=[None]*len(values)
    for a,b,s,n in blocks:
        for i in range(a,b+1):out[i]=s/n
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v5',default='tests/shadow_adjudicator_v22_fixed_v5')
    ap.add_argument('--tests-root',default='tests')
    ap.add_argument('--out',default='tests/exclusive_energy_calibration')
    ap.add_argument('--labels',help='Optional independent-review CSV: case,time_s,low_hz,high_hz,label (low/high/uncertain).')
    ap.add_argument('--block-s',type=float,default=.50)
    ap.add_argument('--case',choices=('all',)+CASES,default='all')
    args=ap.parse_args()
    if args.block_s <=0:ap.error('--block-s must be positive')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    requested=CASES if args.case=='all' else (args.case,)
    labeled={}
    if args.labels:
        for r in rows(Path(args.labels)):
            if r.get('label') not in ('low','high','uncertain'):raise ValueError('Label must be low/high/uncertain')
            key=(r['case'],num(r['time_s']),num(r['low_hz']),num(r['high_hz']))
            if None in key:raise ValueError(f'Invalid label identity: {r}')
            if key in labeled:raise ValueError(f'Duplicate label: {key}')
            labeled[key]=r['label']
    records=[]; summary={'schema':'exclusive_energy_calibration_diagnostic_v1',
      'production_modified':False,'v5_modified':False,'midi_used':False,
      'v13_is_ground_truth':False,'independent_labels_supplied':bool(args.labels),
      'block_seconds':args.block_s,'regression_cases_excluded_from_fitting':REGRESSION,
      'warning':'Proxy agreement is not accuracy. No calibration is valid without independently reviewed low AND high controls.',
      'cases':{}}
    for case in requested:
        v5=rows(Path(args.v5)/f'{case}_pair_decisions.csv')
        v13=rows(Path(args.tests_root)/'full_timeline_v13'/f'{case}_full_timeline.csv')
        refs=defaultdict(list)
        for r in v13:
            if r.get('evidence_status')!='acoustic_period_supported_source_unverified':continue
            if str(r.get('usable_as_next_reference')).lower() not in ('true','1','yes'):continue
            t=num(r.get('time_s'));hz=num(r.get('selected_measured_hz'))
            if t is not None and hz is not None and hz>0:refs[round(t,6)].append(hz)
        n_valid=0
        for r in v5:
            t=num(r.get('time_s'));low=num(r.get('low_hz'));high=num(r.get('high_hz'))
            ex=num(r.get('low_exclusive_energy_median'))
            if t is None or low is None or high is None:continue
            ref=refs.get(round(t,6),[])
            closer='unavailable'
            if len(ref)==1:
                dl,dh=cents(low,ref[0]),cents(high,ref[0])
                if abs(dl-dh)>35:closer='low' if dl<dh else 'high'
            key=(case,t,low,high)
            label=labeled.get(key,'unreviewed')
            holdout=case in REGRESSION and abs(t-REGRESSION[case])<=.025
            valid=ex is not None and 0<=ex<=1
            n_valid+=valid
            records.append({'case':case,'time_s':t,'low_hz':low,'high_hz':high,
                'low_exclusive_fraction':ex,'low_exclusive_percent':ex*100 if ex is not None else None,
                'energy_bin':bin_index(ex) if valid else '',
                'acf_advantage_high_minus_low':num(r.get('acf_advantage_high_minus_low')),
                'low_admissible':r.get('low_admissible'),'high_admissible':r.get('high_admissible'),
                'v5_outcome':r.get('outcome'),'v13_proxy_closer':closer,
                'independent_label':label,'regression_holdout':holdout,
                'block':math.floor(t/args.block_s),'valid_energy':valid})
        summary['cases'][case]={'v5_pair_rows':len(v5),'valid_energy_rows':n_valid,
                                'v13_supported_times':len(refs)}
    output(out/'pair_evidence.csv',records)
    # Deterministic block-level sampling: at most one pair per case+time block,
    # separately for each proxy class. Does not treat adjacent frames as independent.
    representatives={}
    for r in records:
        if not r['valid_energy'] or r['regression_holdout']:continue
        key=(r['case'],r['block'],r['v13_proxy_closer'])
        representatives.setdefault(key,[]).append(r)
    sample=[]
    for rr in representatives.values():
        rr=sorted(rr,key=lambda r:(r['time_s'],r['low_hz'],r['high_hz']))
        sample.append(rr[len(rr)//2])
    output(out/'block_proxy_sample.csv',sample)
    profiles=[]
    for name,dataset in [('all_pairs',records),('block_proxy_sample',sample)]:
        for klass in ('low','high','unavailable'):
            subset=[r['low_exclusive_fraction'] for r in dataset if r['valid_energy'] and r['v13_proxy_closer']==klass]
            profiles.append({'set':name,'v13_proxy_closer':klass,'n':len(subset),
               'p10':quant(subset,.1),'p25':quant(subset,.25),'p50':quant(subset,.5),
               'p75':quant(subset,.75),'p90':quant(subset,.9)})
    output(out/'proxy_distributions.csv',profiles)
    counts=defaultdict(Counter)
    for r in sample:
        if r['v13_proxy_closer'] in ('low','high'):
            counts[r['energy_bin']][r['v13_proxy_closer']]+=1
    proxy=[]
    for i in range(len(EDGES)-1):
        c=counts[i];n=c['low']+c['high'];lo,hi=wilson(c['low'],n)
        proxy.append({'bin':i,'min_fraction':EDGES[i],'max_fraction':EDGES[i+1],
           'blocks_proxy_low':c['low'],'blocks_proxy_high':c['high'],
           'proxy_low_fraction':c['low']/n if n else None,'wilson_low':lo,'wilson_high':hi,
           'interpretation':'SOURCE-UNVERIFIED PROXY; not calibration'})
    output(out/'proxy_energy_bins.csv',proxy)
    # Independently reviewed labels are optional and must not be guessed from V13.
    reviewed=[r for r in records if r['independent_label']!='unreviewed' and not r['regression_holdout'] and r['valid_energy']]
    # One representative per case+block+label to avoid counting a dense voiced region repeatedly.
    grouped=defaultdict(list)
    for r in reviewed:grouped[(r['case'],r['block'],r['independent_label'])].append(r)
    independent=[]
    for rr in grouped.values():
        rr=sorted(rr,key=lambda r:(r['time_s'],r['low_hz'],r['high_hz']))
        independent.append(rr[len(rr)//2])
    output(out/'independent_label_sample.csv',independent, list(records[0]) if records else ['case'])
    bcounts=defaultdict(Counter)
    for r in independent:
        if r['independent_label'] in ('low','high'):bcounts[r['energy_bin']][r['independent_label']]+=1
    nlow=sum(c['low'] for c in bcounts.values());nhigh=sum(c['high'] for c in bcounts.values())
    sufficient=nlow>=10 and nhigh>=10 and sum(bool(c['low'] and c['high']) for c in bcounts.values())>=2
    raw=[bcounts[i]['low']/(bcounts[i]['low']+bcounts[i]['high']) if bcounts[i]['low']+bcounts[i]['high'] else 0 for i in range(len(EDGES)-1)]
    weights=[sum(bcounts[i].values()) for i in range(len(EDGES)-1)]
    iso=pava(raw,weights) if sufficient else [None]*len(raw)
    fit=[]
    for i in range(len(raw)):
        c=bcounts[i];n=weights[i];lo,hi=wilson(c['low'],n)
        fit.append({'bin':i,'min_fraction':EDGES[i],'max_fraction':EDGES[i+1],
           'independent_low_blocks':c['low'],'independent_high_blocks':c['high'],
           'raw_fraction_low':raw[i] if n else None,'wilson_low':lo,'wilson_high':hi,
           'monotonic_fraction_low_exploratory':iso[i] if sufficient else None})
    output(out/'independent_calibration_bins.csv',fit)
    template=[{'case':r['case'],'time_s':r['time_s'],'low_hz':r['low_hz'],
               'high_hz':r['high_hz'],'label':'uncertain'} for r in records
              if r['valid_energy'] and (r['regression_holdout'] or (r['case'],r['block'],r['v13_proxy_closer']) in representatives)]
    # Template is a review queue, not a source of automatically assigned labels.
    output(out/'independent_review_template.csv',template,['case','time_s','low_hz','high_hz','label'])
    summary['independent_review']={'low_block_controls':nlow,'high_block_controls':nhigh,
       'fit_exploratory_only':sufficient,'deployment_approved':False,
       'note':'Even with enough labels, evaluate held-out SONGS and genuine low-note controls before deploying any formula.'}
    summary['outputs']=['pair_evidence.csv','block_proxy_sample.csv','proxy_distributions.csv',
        'proxy_energy_bins.csv','independent_label_sample.csv','independent_calibration_bins.csv',
        'independent_review_template.csv','manifest.json']
    (out/'manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
