#!/usr/bin/env python3
"""Read-only V8 abstention forensics. Does not adjudicate or alter upstream evidence.

Reads tests/shadow_adjudicator_v22_fixed_v8/*_pair_decisions.csv and
*_observation_summary.csv. Produces mutually exclusive observation-level causes,
pair-level mechanisms, review queues and per-song accounting. No rescue decisions.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')

def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        path.write_text('', encoding='utf-8'); return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def yes(v):
    return str(v).strip().lower() in ('true','1','yes')

def classify(r, pairs, margin):
    """Exclusive immediate failure cause, not a claim of counterfactual recoverability."""
    status = r['observation_outcome']
    if status == 'provisional_tournament_champion': return 'champion_reference'
    if status == 'abstain_all_candidates_out_of_range': return 'all_range_rejected'
    if status == 'abstain_single_candidate_no_acoustic_support':
        admitted = [(p,side) for p in pairs for side in ('low','high') if yes(p.get(side+'_range_admitted'))]
        reasons = Counter(p.get(side+'_admissibility_reason','') for p,side in admitted)
        if not admitted: return 'singleton_no_admitted_candidate_in_pair_records'
        if reasons and all(reason == 'no_shifted_acf' for reason in reasons): return 'singleton_no_acf_measurements'
        if reasons and all(reason.startswith('acf_support_') for reason in reasons): return 'singleton_insufficient_acf_support'
        if any('weak' in p['outcome'] for p in pairs): return 'singleton_weak_absolute_score'
        return 'singleton_mixed_or_other_acoustic_failure'
    if status == 'abstain_no_decisive_pair':
        if all(p['outcome'] == 'abstain_all_candidates_out_of_range' for p in pairs): return 'no_decisive_all_pairs_range_rejected'
        eligible = [p for p in pairs if yes(p.get('low_range_admitted')) or yes(p.get('high_range_admitted'))]
        if not eligible: return 'no_decisive_no_admitted_pairs'
        if all(not yes(p.get('low_admissible')) and not yes(p.get('high_admissible')) for p in eligible):
            return 'no_decisive_no_acoustically_admissible_candidate'
        if any(p['outcome'] == 'abstain_insufficient_separation' for p in eligible):
            return 'no_decisive_score_margin'
        if any('weak' in p['outcome'] for p in eligible): return 'no_decisive_absolute_score'
        if all(p['outcome'] in ('provisional_low_only','provisional_high_only','abstain_no_admissible_candidate','abstain_all_candidates_out_of_range') for p in eligible):
            return 'no_decisive_only_singleton_pair_wins_no_head_to_head'
        return 'no_decisive_other_or_uncompared'
    if status == 'abstain_tournament_no_unique_champion':
        return 'tournament_multiple_undefeated'
    if status == 'abstain_tournament_cycle': return 'tournament_cycle'
    if status == 'abstain_tournament_incomplete_comparisons': return 'tournament_missing_decisive_path'
    if status == 'abstain_tournament_champion_ambiguous': return 'tournament_champion_direct_ambiguity'
    if status == 'abstain_tournament_unmapped': return 'tournament_unmapped_frequency'
    return 'other_'+status

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('tests/shadow_adjudicator_v22_fixed_v8'))
    ap.add_argument('--out',type=Path,default=Path('tests/v8_abstention_forensics'))
    ap.add_argument('--case',choices=('all',)+CASES,default='all')
    ap.add_argument('--min-score-margin',type=float,default=0.20,help='Diagnostic reference only; does not change decisions.')
    args=ap.parse_args()
    if args.out.resolve()==args.input.resolve(): ap.error('Output directory must differ from V8 input directory')
    selected = CASES if args.case == 'all' else (args.case,)
    pair_rows=[]; obs_rows=[]; breakdown=[]; review=[]; by_song=[]; causes=Counter()
    for case in selected:
        op=args.input/f'{case}_observation_summary.csv'; pp=args.input/f'{case}_pair_decisions.csv'
        if not op.is_file() or not pp.is_file(): ap.error(f'Missing V8 input: {op if not op.is_file() else pp}')
        observations=read(op); pairs=read(pp)
        grouped=defaultdict(list)
        for p in pairs:
            t=num(p.get('time_s'))
            if t is None: raise ValueError(f'{pp}: invalid time_s')
            grouped[round(t,6)].append(p)
        if len(grouped)!=len(observations):
            raise ValueError(f'{case}: observation count mismatch: {len(grouped)} pair timestamps vs {len(observations)} summaries')
        song_causes=Counter()
        for o in observations:
            t=num(o.get('time_s'))
            if t is None: raise ValueError(f'{op}: invalid time_s')
            ps=grouped.get(round(t,6),[])
            if len(ps)!=int(o['pair_records']): raise ValueError(f'{case} {t}: pair count mismatch')
            cause=classify(o,ps,args.min_score_margin)
            margins=[]; min_margin_gap=None
            for p in ps:
                lo=num(p.get('low_score')); hi=num(p.get('high_score'))
                if lo is not None and hi is not None:
                    difference=abs(hi-lo); margins.append(difference)
                details=json.loads(p.get('details_json') or '{}')
                row={**p,'observation_outcome':o['observation_outcome'], 'primary_observation_cause':cause,
                     'score_margin':difference if lo is not None and hi is not None else '',
                     'margin_shortfall':max(0,args.min_score_margin-difference) if lo is not None and hi is not None else '',
                     'low_interval_component':details.get('low',{}).get('interval_component'),
                     'high_interval_component':details.get('high',{}).get('interval_component'),
                     'low_range_component':details.get('low',{}).get('range_component'),
                     'high_range_component':details.get('high',{}).get('range_component')}
                pair_rows.append(row)
            margin_shortfalls=[args.min_score_margin-m for m in margins if m<args.min_score_margin]
            out={**o, 'primary_cause':cause,
                 'pair_outcomes_json':json.dumps(dict(Counter(p['outcome'] for p in ps)),sort_keys=True),
                 'minimum_scored_pair_margin':min(margins) if margins else '',
                 'maximum_scored_pair_margin':max(margins) if margins else '',
                 'smallest_margin_shortfall':min(margin_shortfalls) if margin_shortfalls else '',
                 'admitted_candidate_occurrences':sum(yes(p.get(s+'_range_admitted')) for p in ps for s in ('low','high')),
                 'range_rejected_candidate_occurrences':sum(not yes(p.get(s+'_range_admitted')) for p in ps for s in ('low','high')),
                 'acoustically_admissible_candidate_occurrences':sum(yes(p.get(s+'_admissible')) for p in ps for s in ('low','high'))}
            obs_rows.append(out); song_causes[cause]+=1; causes[cause]+=1
            if cause!='champion_reference':
                review.append({k:out.get(k,'') for k in ('case','time_s','observation_outcome','primary_cause','pair_records','pairs_both_inadmissible','pairs_one_admissible','pairs_both_admissible','pairs_insufficient_separation','pairs_weak_winner','pairs_without_any_interval_component','minimum_scored_pair_margin','smallest_margin_shortfall','tournament_candidate_frequencies_hz')})
        for cause,count in sorted(song_causes.items()):
            breakdown.append({'case':case,'primary_cause':cause,'observations':count})
        by_song.append({'case':case,'observations':len(observations),'pair_records':len(pairs),
                        'champions':song_causes['champion_reference'],
                        'abstentions':len(observations)-song_causes['champion_reference'],
                        'primary_causes_accounted_for':sum(song_causes.values())})
    args.out.mkdir(parents=True,exist_ok=True)
    write(args.out/'observation_causes.csv',obs_rows)
    write(args.out/'pair_mechanisms.csv',pair_rows)
    write(args.out/'cause_summary.csv',breakdown)
    write(args.out/'review_queue.csv',review)
    write(args.out/'per_song.csv',by_song)
    manifest={'schema':'v8_read_only_abstention_forensics_v1','input_dir':str(args.input),
              'decisions_modified':False,'production_modified':False,'midi_used':False,
              'score_margin_reference_only':args.min_score_margin,'cases':by_song,
              'primary_cause_totals':dict(sorted(causes.items())),
              'observations':len(obs_rows),'pairs':len(pair_rows),
              'qualification':'Categories describe immediate failure mechanisms, not proven recoverable F0s. V8 inputs only; no WAV remeasurement.'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=='__main__': main()
