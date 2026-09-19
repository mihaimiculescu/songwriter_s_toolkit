#!/usr/bin/env python3
"""V8 read-only candidate-set investigation over ALL V7 temporal-view rows.

No BPM, MIDI, production changes, pitch selection, voiced verdicts or continuity.
Run: python tests/audit_candidate_geometry_v8.py --case all
Requires tests/temporal_views_v7/*_temporal_views.csv (the --all-four-apertures run).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

NAMES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
VIEWS = ('past', 'center', 'future')
DURATIONS = (24., 40., 64., 96.)
FIELDS = ('case','sample','time_s','region','comparison','primary_mechanism',
          'view_ms','v7_geometry','v7_frequency_relation','reference_hz',
          'raw_candidate_count','eligible_candidate_count','cluster_count',
          'center_raw_count','center_eligible_count','center_supported_clusters',
          'flank_only_clusters','shared_three_view_clusters','competing_center_clusters',
          'octave_related_pairs','candidate_geometry','investigation_route',
          'clusters_json','view_candidates_json','no_pitch_applied','no_voicing_applied')

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def cents(a,b):
    return abs(1200*math.log2(a/b))

def candidates(row):
    """Every candidate comes from original V7 independently measured waveform peaks."""
    records=[]
    views={}
    for view in VIEWS:
        data=json.loads(row[view+'_json'])
        views[view]=data
        raw=data.get('raw_candidates') or []
        eligible=data.get('eligible_candidates') or []
        # Eligibility is associated by measured lag (not recomputed or loosened).
        eligible_lags={finite(x.get('lag')) for x in eligible}
        for c in raw:
            hz=finite(c.get('hz'))
            if hz is None or hz<=0:continue
            records.append({'view':view,'hz':hz,'lag':finite(c.get('lag')),
                            'acf':finite(c.get('acf')),'cmndf':finite(c.get('cmndf')),
                            'coverage':c.get('coverage'),
                            'eligible':finite(c.get('lag')) in eligible_lags})
    return views,records

def cluster(records,tolerance=70.):
    """Connected measured-frequency groups; this does not propose a new frequency."""
    ordered=sorted(records,key=lambda r:r['hz'])
    groups=[]
    for r in ordered:
        if groups and cents(r['hz'],groups[-1][-1]['hz'])<=tolerance:
            groups[-1].append(r)
        else:groups.append([r])
    result=[]
    for group in groups:
        by={v:[r for r in group if r['view']==v] for v in VIEWS}
        elig={v:[r for r in by[v] if r['eligible']] for v in VIEWS}
        # Representative is an existing measured peak; never synthesize averages/multiples.
        representative=min(group,key=lambda r:(r['cmndf'] if r['cmndf'] is not None else 10,
                                               -(r['acf'] if r['acf'] is not None else -1)))
        result.append({'representative_measured_hz':representative['hz'],
                       'min_measured_hz':min(r['hz'] for r in group),
                       'max_measured_hz':max(r['hz'] for r in group),
                       'view_raw_counts':{v:len(by[v]) for v in VIEWS},
                       'view_eligible_counts':{v:len(elig[v]) for v in VIEWS},
                       'center_eligible':bool(elig['center']),
                       'all_three_eligible':all(elig.values()),
                       'flank_eligible_only':not elig['center'] and bool(elig['past'] or elig['future']),
                       'best_acf_by_view':{v:max((r['acf'] for r in by[v] if r['acf'] is not None),default=None) for v in VIEWS},
                       'best_cmndf_by_view':{v:min((r['cmndf'] for r in by[v] if r['cmndf'] is not None),default=None) for v in VIEWS}})
    return result

def geometry(groups,views):
    supported=[g for g in groups if g['center_eligible']]
    all_three=[g for g in groups if g['all_three_eligible']]
    remote=[g for g in groups if g['flank_eligible_only']]
    if not groups:return 'no_raw_period_in_any_view','inspect_observability_and_energy; preserve_unresolved'
    if not supported:
        if remote:return 'flank_eligible_without_center','inspect_temporal_attribution_and_cycles; no_instantaneous_pitch'
        return 'raw_periods_only_no_center_eligible','inspect_rejection_reasons_and_cycle_sufficiency'
    if len(supported)>1:
        return 'multiple_independently_measured_center_periods','inspect_acf_cmndf_and_fundamental_harmonic_support; do_not_select_by_continuity'
    if all_three:return 'one_center_period_supported_across_views','audit_voicing_and_existing_eckf_state_separately'
    return 'one_center_period_with_view_asymmetry','inspect_onsets_offsets_and_temporal_support'

def inspect(row):
    views, records=candidates(row)
    groups=cluster(records)
    kind,route=geometry(groups,views)
    centers=[g for g in groups if g['center_eligible']]
    octave_pairs=[]
    for i,g in enumerate(groups):
        for h in groups[i+1:]:
            ratio=h['representative_measured_hz']/g['representative_measured_hz']
            if any(abs(1200*math.log2(ratio)-1200*math.log2(n))<=70 for n in (2,3,4)):
                octave_pairs.append([g['representative_measured_hz'],h['representative_measured_hz'],round(ratio,4)])
    result={'case':row['case'],'sample':row['sample'],'time_s':row['time_s'],
            'region':row['region'],'comparison':row['comparison'],
            'primary_mechanism':row['primary_mechanism'],'view_ms':row['view_ms'],
            'v7_geometry':row['geometry'],'v7_frequency_relation':row['frequency_relation'],
            'reference_hz':row['period_hz'],
            'raw_candidate_count':len(records),'eligible_candidate_count':sum(r['eligible'] for r in records),
            'cluster_count':len(groups),
            'center_raw_count':sum(r['view']=='center' for r in records),
            'center_eligible_count':sum(r['view']=='center' and r['eligible'] for r in records),
            'center_supported_clusters':len(centers),
            'flank_only_clusters':sum(g['flank_eligible_only'] for g in groups),
            'shared_three_view_clusters':sum(g['all_three_eligible'] for g in groups),
            'competing_center_clusters':max(0,len(centers)-1),
            'octave_related_pairs':len(octave_pairs),
            'candidate_geometry':kind,'investigation_route':route,
            'clusters_json':json.dumps({'clusters':groups,'integer_related_measured_pairs':octave_pairs},allow_nan=False),
            'view_candidates_json':json.dumps(records,allow_nan=False),
            'no_pitch_applied':True,'no_voicing_applied':True}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*NAMES),default='all')
    p.add_argument('--v7-dir',type=Path,default=Path('tests/temporal_views_v7'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/candidate_geometry_v8'))
    args=p.parse_args()
    names=NAMES if args.case=='all' else (args.case,)
    for name in names:
        if not (args.v7_dir/f'{name}_temporal_views.csv').is_file():
            p.error(f'missing {args.v7_dir / (name+"_temporal_views.csv")}')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'candidate_geometry_v8','read_only':True,'not_ground_truth':True,
              'no_bpm_or_grid_changes':True,'no_pitch_or_voicing_applied':True,
              'correct_ratata_djuvvs':[[36.923,37.435],[41.025,41.538],[49.230,49.743]],
              'trandafiri':'steady i between consonants, not a glissando','cases':{}}
    for name in names:
        infile=args.v7_dir/f'{name}_temporal_views.csv'
        outfile=args.out_dir/f'{name}_candidate_geometry.csv'
        counts=Counter();by_duration=defaultdict(Counter);keyset=set();duration_per_key=defaultdict(set)
        with infile.open(newline='') as fi, outfile.open('w',newline='') as fo:
            reader=csv.DictReader(fi)
            missing=set(('past_json','center_json','future_json','sample','view_ms','geometry'))-set(reader.fieldnames or [])
            if missing:p.error(f'{infile}: missing columns {sorted(missing)}')
            writer=csv.DictWriter(fo,fieldnames=FIELDS);writer.writeheader()
            for row in reader:
                row['case']=name
                report=inspect(row)
                writer.writerow(report)
                ms=finite(row['view_ms']); key=row['sample']
                counts[report['candidate_geometry']]+=1
                by_duration[str(ms)][report['candidate_geometry']]+=1
                keyset.add(key);duration_per_key[key].add(ms)
        missing_durations={key:sorted(set(DURATIONS)-dur) for key,dur in duration_per_key.items() if set(DURATIONS)-dur}
        manifest['cases'][name]={'unique_review_observations':len(keyset),'rows':sum(counts.values()),
             'geometry_counts_all_durations':dict(counts),
             'geometry_counts_by_duration':{ms:dict(c) for ms,c in by_duration.items()},
             'observations_missing_one_or_more_durations':len(missing_durations),
             'output':str(outfile)}
        print(name, 'review observations',len(keyset),'rows',sum(counts.values()),
              '40ms classifications',dict(by_duration['40.0']),flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: measured candidate clusters are diagnostic; no pitch, voicing, or state changes.')
if __name__=='__main__':main()
