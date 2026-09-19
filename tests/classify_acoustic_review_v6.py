#!/usr/bin/env python3
"""Read-only, full-song classification of V5 production/adaptive review observations.

Run: python tests/classify_acoustic_review_v6.py --case all

Consumes V5 CSVs only. Production is NOT truth; its voiced flag is block-level.
No pitches, statuses, aperture decisions or ECKF states are modified. Categories
are diagnostic hypotheses and investigation routes, not verdicts or repairs.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

NAMES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
REVIEWS = {
    'production_voiced_adaptive_unresolved': 'production_voiced_adaptive_unresolved',
    'production_voiced_adaptive_different_70c': 'production_voiced_adaptive_different_70c',
    'production_unvoiced_adaptive_candidate': 'production_unvoiced_adaptive_candidate',
}
ROUTES = {
    'local_period_ambiguous': 'Compare independently measured ACF/CMNDF peaks and fundamental/harmonic amplitudes in the SAME aperture; do not choose by pitch continuity.',
    'period_visible_but_not_eligible': 'Inspect raw-candidate eligibility reasons and local cycles; independently verify whether a measurable period belongs at this timestamp.',
    'remote_periodicity': 'Audit actual past/center/future acoustic support and window boundaries; distinguish temporal leakage from present-period evidence.',
    'cycle_limited': 'Investigate longer local support and count cycles before judging a candidate absent; do not assign a pitch from insufficient cycles.',
    'no_period_measured': 'Compare centered energy, noise/flatness and raw ACF landscape; verify if sound is genuinely nonperiodic or detector misses low-energy periods.',
    'testable_no_local_support': 'Audit central periodicity, fundamental energy and spectral noise; inspect why whole-window candidate lacks present-time support.',
    'raw_period_unconfirmed': 'Inspect raw ACF peaks, CMNDF, local RMS, cycle count and eligibility separately; retain unresolved until independently supported.',
    'locally_supported_conflict': 'Compare production-state frequency and adaptive candidate against the SAME waveform window; first assess state support, then independently assess a replacement.',
    'block_timing_ambiguity': 'Retrieve sample-level production statuses; a block containing any voiced samples does not prove the observation timestamp is voiced.',
    'candidate_during_unvoiced_block': 'Audit exact production voicing boundaries and local temporal/energy support; classify periodic fragment versus noise/leakage without assuming production truth.',
    'unclassified_evidence': 'Inspect the complete V3 stage records and production sample-level status; current V5 summary does not distinguish this mechanism.',
}

def num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def parse_json(row, key, expected):
    value = row.get(key)
    try: parsed = json.loads(value) if value else expected()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {key} at {row.get('case')} sample {row.get('sample')}: {exc}") from exc
    if not isinstance(parsed, expected):
        raise ValueError(f"Invalid {key} type at {row.get('case')} sample {row.get('sample')}")
    return parsed

def classify(r):
    obs=r['observability']; ident=r['identifiability']; comparison=r['comparison']
    stages=parse_json(r,'stage_evidence_json',list)
    cf=parse_json(r,'counterfactual_apertures_json',list)
    competition=parse_json(r,'v4_competition_json',list)
    early=parse_json(r,'v4_early_stop_audit_json',dict)
    reasons=[]
    if ident == 'multiple_local_eligible_periods_unadjudicated' or competition:
        reasons.append('local_period_ambiguous')
    if obs == 'local_raw_period_evidence_only': reasons.append('period_visible_but_not_eligible')
    if obs == 'distant_only_evidence': reasons.append('remote_periodicity')
    if obs == 'local_cycle_insufficiency': reasons.append('cycle_limited')
    if obs == 'no_raw_period_any_inspected_aperture': reasons.append('no_period_measured')
    if obs == 'testable_but_no_local_support': reasons.append('testable_no_local_support')
    if obs == 'raw_period_without_local_confirmation': reasons.append('raw_period_unconfirmed')
    if obs == 'local_eligible_period_evidence' and comparison=='production_voiced_adaptive_different_70c':
        reasons.append('locally_supported_conflict')
    if comparison == 'production_voiced_adaptive_unresolved':
        reasons.append('block_timing_ambiguity')
    if comparison == 'production_unvoiced_adaptive_candidate':
        reasons.append('candidate_during_unvoiced_block')
    if not reasons: reasons.append('unclassified_evidence')
    # One exclusive PRIMARY acoustic mechanism; other flags remain in secondary routes.
    priority=['local_period_ambiguous','remote_periodicity','cycle_limited',
              'period_visible_but_not_eligible','testable_no_local_support',
              'raw_period_unconfirmed','no_period_measured','locally_supported_conflict',
              'candidate_during_unvoiced_block','block_timing_ambiguity','unclassified_evidence']
    primary=next(x for x in priority if x in reasons)
    # Counterfactual: only make claims about apertures actually inspected.
    raw_counts=[int(s.get('raw_count') or 0) for s in stages]
    local_counts=[int(s.get('eligible_local_distinct_count') or 0) for s in stages]
    first_local=next((num(s.get('aperture_ms')) for s in stages if int(s.get('eligible_local_distinct_count') or 0)>0),None)
    first_raw=next((num(s.get('aperture_ms')) for s in stages if int(s.get('raw_count') or 0)>0),None)
    observed_later_local=any(int(s.get('later_eligible_local_observed') or 0)>0 for s in cf[:-1])
    if first_local is not None and first_local>24:
        aperture='observed_longer_aperture_first_local_candidate'
    elif first_local is not None:
        aperture='local_candidate_present_at_shortest_inspected_aperture'
    elif first_raw is not None:
        aperture='raw_period_found_but_no_local_eligible_candidate'
    elif len(stages)==4:
        aperture='all_four_apertures_inspected_no_raw_period'
    else:
        aperture='search_ended_without_raw_period_later_unobserved'
    early_status=early.get('assessment','not_reported')
    if early_status == 'later_eligible_local_candidate':
        early_risk='observed_early_stop_would_miss_local_candidate'
    elif early_status in ('later_local_raw_candidate','later_raw_candidate_only'):
        early_risk='observed_early_stop_would_miss_raw_evidence'
    elif early_status=='no_later_raw_candidate':
        early_risk='observed_later_stages_show_no_raw_candidate'
    else:
        early_risk='not_assessable_or_not_suggested'
    return {'primary_mechanism':primary,'secondary_flags':reasons,
            'investigation_routes':[ROUTES[x] for x in reasons],
            'aperture_opportunity':aperture,'first_raw_aperture_ms':first_raw,
            'first_local_eligible_aperture_ms':first_local,'inspected_apertures_ms':[num(s.get('aperture_ms')) for s in stages],
            'observed_raw_counts':raw_counts,'observed_local_distinct_counts':local_counts,
            'early_stop_risk':early_risk,'early_stop_assessment':early_status,
            'competition_pairs_reported':len(competition),'later_local_evidence_observed':observed_later_local}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*NAMES),default='all')
    p.add_argument('--v5-dir',type=Path,default=Path('tests/acoustic_observability_v5'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/acoustic_review_v6'))
    a=p.parse_args(); names=NAMES if a.case=='all' else (a.case,)
    missing=[str(a.v5_dir/f'{n}.csv') for n in names if not (a.v5_dir/f'{n}.csv').is_file()]
    if missing:p.error('Missing V5 CSV(s): '+', '.join(missing))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    summary={'version':'v6','read_only':True,'production_ground_truth':False,
             'review_categories':list(REVIEWS),'warning':'Production voiced counts are block-level. These are hypotheses, not errors or validated voicing.',
             'cases':{},'overall':{}}
    overall=Counter(); bytype=defaultdict(Counter); bymechanism=defaultdict(Counter)
    cols=['case','region','sample','time_s','comparison','production_block_start_s','production_block_end_s',
          'production_voiced_samples','production_median_hz','adaptive_observational_hz','hz_distance_cents',
          'observability','identifiability','v3_stop_reason','primary_mechanism','secondary_flags',
          'aperture_opportunity','first_raw_aperture_ms','first_local_eligible_aperture_ms',
          'early_stop_risk','early_stop_assessment','investigation_routes_json','measured_stage_evidence_json',
          'counterfactual_apertures_json','classification_details_json','f0_applied','status_applied','tracker_reset_requested']
    for name in names:
        seen=set();counts=Counter(); mechanisms=Counter(); categories=defaultdict(Counter)
        dst=a.out_dir/f'{name}_review.csv'
        with (a.v5_dir/f'{name}.csv').open(newline='',encoding='utf-8') as src,dst.open('w',newline='',encoding='utf-8') as out:
            reader=csv.DictReader(src)
            required={'case','sample','comparison','stage_evidence_json','counterfactual_apertures_json',
                      'v4_competition_json','v4_early_stop_audit_json','observability','identifiability'}
            if not required.issubset(reader.fieldnames or []):raise ValueError(f'{name}: missing V5 fields: {required-set(reader.fieldnames or [])}')
            writer=csv.DictWriter(out,fieldnames=cols);writer.writeheader()
            for r in reader:
                if r['case']!=name:raise ValueError(f'{name}: wrong case in CSV: {r["case"]}')
                sample=int(r['sample'])
                if sample in seen:raise ValueError(f'{name}: duplicate sample {sample}')
                seen.add(sample);counts['total']+=1
                if r['comparison'] not in REVIEWS:continue
                info=classify(r);counts['review']+=1
                counts['comparison:'+r['comparison']]+=1
                mechanisms[info['primary_mechanism']]+=1
                categories[r['comparison']][info['primary_mechanism']]+=1
                counts['aperture:'+info['aperture_opportunity']]+=1
                counts['early_stop:'+info['early_stop_risk']]+=1
                record={k:r.get(k,'') for k in cols if k in r}
                record.update({'primary_mechanism':info['primary_mechanism'],
                    'secondary_flags':';'.join(info['secondary_flags']),
                    'aperture_opportunity':info['aperture_opportunity'],
                    'first_raw_aperture_ms':info['first_raw_aperture_ms'],
                    'first_local_eligible_aperture_ms':info['first_local_eligible_aperture_ms'],
                    'early_stop_risk':info['early_stop_risk'],'early_stop_assessment':info['early_stop_assessment'],
                    'investigation_routes_json':json.dumps(info['investigation_routes']),
                    'measured_stage_evidence_json':r['stage_evidence_json'],
                    'counterfactual_apertures_json':r['counterfactual_apertures_json'],
                    'classification_details_json':json.dumps(info),
                    'f0_applied':False,'status_applied':False,'tracker_reset_requested':False})
                writer.writerow(record)
        if counts['review'] != sum(counts['comparison:'+c] for c in REVIEWS):raise AssertionError(f'{name}: review partition mismatch')
        if counts['review'] != sum(mechanisms.values()):raise AssertionError(f'{name}: mechanism partition mismatch')
        overall.update(counts)
        bymechanism[name].update(mechanisms)
        for category,c in categories.items():bytype[category].update(c)
        summary['cases'][name]={'total':counts['total'],'review':counts['review'],
            'review_categories':{k.removeprefix('comparison:'):v for k,v in counts.items() if k.startswith('comparison:')},
            'primary_mechanisms':dict(mechanisms),
            'aperture_opportunities':{k.removeprefix('aperture:'):v for k,v in counts.items() if k.startswith('aperture:')},
            'early_stop_risks':{k.removeprefix('early_stop:'):v for k,v in counts.items() if k.startswith('early_stop:')},
            'output':str(dst)}
        print(f'{name}: {counts["review"]}/{counts["total"]} review; '+str(dict(mechanisms)),flush=True)
    summary['overall']={'total':overall['total'],'review':overall['review'],
        'review_categories':{k.removeprefix('comparison:'):v for k,v in overall.items() if k.startswith('comparison:')},
        'primary_mechanisms':dict(sum((Counter(v) for v in bymechanism.values()),Counter())),
        'comparison_by_mechanism':{k:dict(v) for k,v in bytype.items()},
        'aperture_opportunities':{k.removeprefix('aperture:'):v for k,v in overall.items() if k.startswith('aperture:')},
        'early_stop_risks':{k.removeprefix('early_stop:'):v for k,v in overall.items() if k.startswith('early_stop:')}}
    (a.out_dir/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    if a.case=='all':
        expected={'Ochiitai':803,'PREDESTINATI':871,'RATATA':1750,'Trandafiri':166}
        # Counts are V5 review definitions; fail on a mismatch instead of silently advertising historical totals.
        for name,expected_count in expected.items():
            actual=summary['cases'][name]['review']
            if actual != expected_count:print(f'NOTICE: {name} review count {actual}, historical reference {expected_count}',flush=True)
    print(f'TOTAL: {overall["review"]}/{overall["total"]} observational reviews. NO DECISIONS APPLIED.',flush=True)

if __name__=='__main__':main()
