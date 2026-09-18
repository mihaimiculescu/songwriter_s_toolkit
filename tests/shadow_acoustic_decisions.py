#!/usr/bin/env python3
"""Four-piggy read-only shadow evaluation using existing WAV-only evidence.

Run from songwriter_s_toolkit repository root:
  python tests/shadow_acoustic_decisions.py --case all

Inputs: tests/adaptive_evidence/*_adaptive.csv (all-song WAV analysis),
        tests/robustness_v3/*_v3_frames.csv (optional, observational baseline),
        tests/production_decision_trace/*_decision_path.jsonl (optional, hotspot-only).
No WAV modifications, no MIDI, no production import, no ECKF execution, no resets.
Shadow outputs are HYPOTHESES, never applied to production F0 or pitch status.
The adaptive evidence rows overlap; adjacent rows are NOT independent votes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

CASES = {
    'Ochiitai': (42.214,42.260,42.307,42.353),
    'PREDESTINATI': (38.034,53.592),
    'RATATA': (36.923,41.026,42.965,43.008,49.231),
    'Trandafiri': (10.774,10.821),
}
FIELDS = ('case','time_s','probe_rms','window_ms','adaptive_status','adaptive_hz',
          'adaptive_reason','measured_candidate_count','shadow_initialization',
          'shadow_hz','shadow_reason','competing_hz','shadow_active_state',
          'baseline_hz','baseline_voiced','baseline_comparison','acf','cmndf',
          'spectral_coverage','early_hz','late_hz','nonstationary')

def num(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):
        return None

def close_ratio(a,b,cents=55):
    if not (a and b and a>0 and b>0): return None
    r=max(a,b)/min(a,b)
    multiple=round(r)
    if 2<=multiple<=5 and abs(1200*math.log2(r/multiple))<=cents:
        return multiple
    return None

def assess(row):
    """Advisory only: abstain on insufficient/ambiguous measured evidence.

    Never synthesize a candidate from multiples; only reports provided measured Hz.
    'coverage' is the prototype's spectral-harmonic count, NOT cycle count.
    """
    candidates=json.loads(row.get('candidates') or '[]')
    measured=[c for c in candidates if num(c.get('hz')) and num(c.get('acf')) is not None
              and num(c.get('cmndf')) is not None]
    observed=num(row.get('observed_hz'))
    if row.get('nonstationary')=='1':
        return 'unresolved',None,'nonstationary_window',measured
    if row.get('status')!='observed_candidate' or observed is None:
        return 'unresolved',None,'adaptive_evidence_unresolved',measured
    chosen=min(measured,key=lambda c:abs(1200*math.log2(float(c['hz'])/observed))) if measured else None
    if chosen is None or abs(1200*math.log2(float(chosen['hz'])/observed))>55:
        return 'unresolved',None,'reported_hz_not_measured_in_candidate_list',measured
    if float(chosen['acf'])<.70 or float(chosen['cmndf'])>.25:
        return 'unresolved',None,'periodicity_insufficient',measured
    if int(chosen.get('coverage') or 0)<3:
        return 'unresolved',None,'weak_spectral_corrobation',measured
    # Competing measured octave/multiple: abstain unless acoustic period measurements
    # themselves distinguish the alternatives. No prior, MIDI, or pitch multiplication.
    for rival in measured:
        if rival is chosen or close_ratio(float(chosen['hz']),float(rival['hz'])) is None: continue
        if float(rival['acf'])<.70 or float(rival['cmndf'])>.25: continue
        if int(rival.get('coverage') or 0)<3: continue
        if abs(float(chosen['acf'])-float(rival['acf']))<.06 and abs(float(chosen['cmndf'])-float(rival['cmndf']))<.045:
            return 'unresolved',None,'competing_measured_integer_related_periods',measured
    return 'measured_hypothesis',float(chosen['hz']),'observed_local_period_with_support',measured

def read_baseline(path):
    if not path.is_file():return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['all',*CASES],default='all')
    p.add_argument('--adaptive-dir',type=Path,default=Path('tests/adaptive_evidence'))
    p.add_argument('--baseline-dir',type=Path,default=Path('tests/robustness_v3'))
    p.add_argument('--trace-dir',type=Path,default=Path('tests/production_decision_trace'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/shadow_acoustic_decisions'))
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    names=CASES if a.case=='all' else {a.case:CASES[a.case]}
    manifest={'midi_used':False,'production_modified':False,'pitch_modified':False,
              'decisions_applied':False,'input':'previously measured WAV-only adaptive observations',
              'no_independent_vote_from_overlapping_windows':True,'cases':{}}
    for name,sites in names.items():
        source=a.adaptive_dir/f'{name}_adaptive.csv'
        if not source.is_file():p.error(f'missing {source}; run adaptive evidence experiment first')
        with source.open(newline='') as f: data=list(csv.DictReader(f))
        base=read_baseline(a.baseline_dir/f'{name}_v3_frames.csv')
        base_idx=0; out=[]; count=Counter(); hotspots={str(s):[] for s in sites}
        for row in data:
            t=num(row.get('time_s'))
            if t is None:continue
            while base_idx+1<len(base) and num(base[base_idx+1]['time_s']) is not None and float(base[base_idx+1]['time_s'])<=t:
                base_idx+=1
            b=base[base_idx] if base else {}
            baseline_hz=num(b.get('median_f0_hz'))
            voiced=int(b.get('voiced_samples') or 0)>0 if b else None
            status,hz,reason,candidates=assess(row)
            # The active-state question is recorded without carrying or correcting pitch.
            # A mismatch is a review flag, NOT a reset recommendation.
            active='not_available'
            if voiced is True and status=='measured_hypothesis' and baseline_hz and hz:
                delta=abs(1200*math.log2(hz/baseline_hz))
                active=('acoustic_state_disagreement_review' if delta>=100 else 'measured_state_agreement')
            elif voiced is True and status=='unresolved':active='acoustic_evidence_unresolved_no_reset'
            elif voiced is False and status=='measured_hypothesis':active='baseline_unvoiced_candidate_review'
            elif voiced is False:active='both_unresolved_or_unvoiced'
            result=dict(case=name,time_s=t,probe_rms=row.get('probe_rms'),window_ms=row.get('window_ms'),
              adaptive_status=row.get('status'),adaptive_hz=row.get('observed_hz'),
              adaptive_reason=row.get('reason'),measured_candidate_count=len(candidates),
              shadow_initialization=status,shadow_hz=hz,shadow_reason=reason,
              competing_hz=json.dumps([{'hz':round(float(c['hz']),4),'acf':round(float(c['acf']),4),
                'cmndf':round(float(c['cmndf']),4),'spectral_coverage':c.get('coverage')} for c in candidates],separators=(',',':')),
              shadow_active_state=active,baseline_hz=baseline_hz,baseline_voiced=voiced,
              baseline_comparison=row.get('comparison'),acf=row.get('acf'),cmndf=row.get('cmndf'),
              spectral_coverage=row.get('spectral_coverage'),early_hz=row.get('early_hz'),
              late_hz=row.get('late_hz'),nonstationary=row.get('nonstationary'))
            out.append(result);count[reason]+=1;count[active]+=1
            for site in sites:
                if abs(t-site)<=.15:hotspots[str(site)].append(result)
        dest=a.out_dir/f'{name}_shadow.csv'
        with dest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(out)
        (a.out_dir/f'{name}_hotspots.json').write_text(json.dumps(hotspots,indent=2,allow_nan=False)+'\n')
        trace=a.trace_dir/f'{name}_decision_path.jsonl'
        manifest['cases'][name]={'rows':len(out),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'baseline_present':bool(base),'trace_present':trace.is_file(),
            'counts':dict(count),'csv':str(dest),'hotspots':str(a.out_dir/f'{name}_hotspots.json')}
        print(f'{name}: {len(out)} shadow observations; baseline={bool(base)}; {dict(count)}',flush=True)
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Shadow outputs only. No production, WAV, MIDI or pitch modifications.')
if __name__=='__main__':main()
