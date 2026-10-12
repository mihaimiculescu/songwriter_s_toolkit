#!/usr/bin/env python3
"""Read-only HARMONIC DETECTIVE V2: consult only when V12 is hung.

Consumes the independently measured V1 harmonic CSVs and the *matching* V12
observation summaries. Does not edit V12 or manufacture an F0 for any candidate
failing V12 acoustic/range admission. Only V12 note-group representatives are
eligible for a cross-note tiebreak. Existing V12 champions remain untouched.

From repository root:
 python tests/audit_harmonic_detective_v2.py \
   --v12-zip tests/shadow_adjudicator_v22_fixed_v12.zip \
   --detective-v1 tests/harmonic_detective_v1 \
   --out tests/harmonic_detective_v2

The V1 input may be a directory or ZIP. No WAV processing is repeated.
"""
from __future__ import annotations
import argparse,csv,io,json,math,zipfile
from pathlib import Path
from collections import Counter

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')

def read_csv(source, filename):
    if source.is_dir():
        path=source/filename
        if not path.is_file(): raise FileNotFoundError(path)
        return list(csv.DictReader(path.open(newline='',encoding='utf-8-sig')))
    with zipfile.ZipFile(source) as z:
        matches=[n for n in z.namelist() if n==filename or n.endswith('/'+filename)]
        if len(matches)!=1: raise ValueError(f'{filename}: found {len(matches)} entries; expected one')
        return list(csv.DictReader(io.StringIO(z.read(matches[0]).decode('utf-8-sig'))))

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def fnum(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError): return None

def hzmatch(a,b):return a is not None and b is not None and abs(a-b)<.02

def evaluate_group_reps(vrow,candidates,args):
    """Strictly use V12's representatives, never promote a defeated same-note candidate."""
    groups=json.loads(vrow.get('note_groups_json') or '[]')
    qualified=[];issues=[]
    for g in groups:
        hz=fnum(g.get('representative_hz'))
        if hz is None:
            issues.append('group_has_no_representative:'+str(g.get('reason','unknown')))
            continue
        matched=[c for c in candidates if hzmatch(hz,fnum(c.get('candidate_hz')))]
        if len(matched)!=1:
            issues.append('missing_or_duplicate_representative_measurement')
            continue
        c=matched[0]
        if c.get('range_and_acoustic_admitted')!='1' or c.get('within_49c')!='1':
            issues.append('representative_fails_existing_admission')
            continue
        score=fnum(c.get('harmonic_score'));support=fnum(c.get('harmonic_support'))
        try: n=int(c.get('valid_windows','0'))
        except ValueError:n=0
        if score is None or support is None or n<args.min_valid_windows:
            issues.append('insufficient_independent_fft_windows')
            continue
        qualified.append({'hz':hz,'score':score,'support':support,'windows':n,
                          'midi':next((m.get('nearest_midi') for m in g['members'] if hzmatch(hz,fnum(m.get('hz')))),None),
                          'v12_acf_median':fnum(c.get('v12_acf_median'))})
    return groups,qualified,issues

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v12-zip',type=Path,default=Path('tests/shadow_adjudicator_v22_fixed_v12.zip'))
    p.add_argument('--detective-v1',type=Path,default=Path('tests/harmonic_detective_v1'))
    p.add_argument('--out',type=Path,default=Path('tests/harmonic_detective_v2'))
    p.add_argument('--min-valid-windows',type=int,default=2)
    p.add_argument('--min-support',type=float,default=.5,
                   help='Minimum fraction of weighted harmonic bands supported by best candidate')
    p.add_argument('--min-score',type=float,default=.45)
    p.add_argument('--min-margin',type=float,default=.08,
                   help='Difference between highest and second highest independent harmonic template scores')
    p.add_argument('--require-complete-field',action='store_true',default=True,
                   help='Abstain unless every V12 note group has a measured, admissible representative (default on)')
    p.add_argument('--allow-incomplete-field',action='store_false',dest='require_complete_field',
                   help='Diagnostic sensitivity only: permit absent representatives (not recommended)')
    a=p.parse_args()
    if a.min_valid_windows<1 or not 0<=a.min_support<=1 or not 0<=a.min_score<=1 or a.min_margin<0:p.error('Invalid thresholds')
    for source in (a.v12_zip,a.detective_v1):
        if not source.exists():p.error(f'Missing input: {source}')
    a.out.mkdir(parents=True,exist_ok=True)
    allrows=[]; song_summary=[]; consistency=[]
    for case in CASES:
        v12=read_csv(a.v12_zip,f'{case}_observation_summary.csv')
        dc=read_csv(a.detective_v1,f'{case}_harmonic_candidates.csv')
        do=read_csv(a.detective_v1,f'{case}_harmonic_observations.csv')
        bytime={};observed={}
        for c in dc:bytime.setdefault(round(float(c['time_s']),6),[]).append(c)
        for o in do:
            if round(float(o['time_s']),6) in observed:raise ValueError('Duplicate detective observation: '+case+' '+o['time_s'])
            observed[round(float(o['time_s']),6)]=o
        if len(v12)!=len(observed):raise ValueError(f'{case}: mismatched V12/V1 observations {len(v12)}/{len(observed)}')
        records=[]
        for v in v12:
            t=round(float(v['time_s']),6);champ=fnum(v.get('selected_provisional_hz'))
            if t not in observed:raise ValueError(f'{case} {t}: unmatched source input')
            if bool(champ is not None)!=bool(fnum(observed[t].get('v12_champion_hz')) is not None):
                raise ValueError(f'{case} {t}: mismatched V12/V1 champion status')
            if champ is not None and not hzmatch(champ,fnum(observed[t]['v12_champion_hz'])):
                raise ValueError(f'{case} {t}: mismatched V12/V1 champion frequency')
            groups,qualified,issues=evaluate_group_reps(v,bytime.get(t,[]),a)
            qualified.sort(key=lambda c:(-c['score'],c['hz']))
            new=None;reason='';best=qualified[0] if qualified else None
            runner=qualified[1] if len(qualified)>1 else None
            margin=(best['score']-runner['score']) if runner else None
            if champ is not None:reason='not_called_v12_already_has_champion'
            elif not groups:reason='no_v12_candidates'
            elif not qualified:reason='no_supported_group_representative'
            elif a.require_complete_field and len(qualified)!=len(groups):reason='incomplete_competitor_field'
            elif len(qualified)<2:reason='no_cross_note_contest_to_break'
            elif best['support']<a.min_support or best['score']<a.min_score:reason='harmonic_evidence_too_weak'
            elif margin is None or margin<a.min_margin:reason='harmonic_margin_insufficient'
            elif best['midi'] == runner['midi']:reason='same_note_representatives_unexpected'
            else:
                reason='diagnostic_tiebreak_proposal_NOT_VALIDATED';new=best['hz']
            rec={'case':case,'time_s':t,'v12_outcome':v.get('observation_outcome',''),
                'v12_abstention_category':v.get('abstention_category',''),
                'v12_champion_hz':champ if champ is not None else '',
                'detective_called':int(champ is None),
                'group_count':len(groups),'eligible_representatives':len(qualified),
                'best_representative_hz':best['hz'] if best else '',
                'best_midi':best['midi'] if best else '',
                'best_score':best['score'] if best else '',
                'best_support':best['support'] if best else '',
                'runnerup_representative_hz':runner['hz'] if runner else '',
                'runnerup_score':runner['score'] if runner else '',
                'harmonic_margin':round(margin,6) if margin is not None else '',
                'diagnostic_proposed_hz':new if new is not None else '',
                'reason':reason,'group_issues':'|'.join(issues),
                'NOTE':'SHADOW_ONLY_V12_UNCHANGED_NO_ACCURACY_CLAIM'}
            records.append(rec)
            # On resolved controls, report concordance without permitting an override.
            if champ is not None and best and len(qualified)>=2 and margin is not None and margin>=a.min_margin:
                consistency.append({'case':case,'time_s':t,'v12_champion_hz':champ,
                    'detective_best_hz':best['hz'],'agrees':int(hzmatch(champ,best['hz'])),
                    'margin':round(margin,6)})
        if len(records)!=len(v12):raise AssertionError('Observation lost')
        fields=list(records[0].keys())
        write_csv(a.out/f'{case}_hung_jury_audit.csv',records,fields)
        counted=Counter(r['reason'] for r in records if r['detective_called'])
        summary={'case':case,'observations':len(records),'original_champions':sum(not r['detective_called'] for r in records),
                 'original_unresolved':sum(r['detective_called'] for r in records),
                 'diagnostic_proposals':sum(bool(r['diagnostic_proposed_hz']) for r in records),
                 'reason_counts':dict(counted)}
        song_summary.append(summary);allrows.extend(records)
        print(case,summary)
    write_csv(a.out/'resolved_control_agreement.csv',consistency,
              ['case','time_s','v12_champion_hz','detective_best_hz','agrees','margin'])
    write_csv(a.out/'summary_by_song.csv',song_summary,
              ['case','observations','original_champions','original_unresolved','diagnostic_proposals','reason_counts'])
    manifest={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'total_observations':len(allrows),
              'v12_champions_unchanged':sum(not r['detective_called'] for r in allrows),
              'v12_unresolved_unchanged':sum(r['detective_called'] for r in allrows),
              'diagnostic_tiebreak_proposals':sum(bool(r['diagnostic_proposed_hz']) for r in allrows),
              'resolved_control_agreement':{'evaluated':len(consistency),'agree':sum(c['agrees'] for c in consistency)},
              'limitations':['Proposals are not verified correctness or adopted V12 champions.',
                'FFT reuses spectral information; agreement/disagreement controls cannot prove independence.',
                'Does not rescue failed acoustic admission or unrepresented note groups.',
                'No new WAV measurements; reuse matching V1 FFT data and V12 grouping.']}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(allrows)==5068 and manifest['v12_champions_unchanged']==3003 and manifest['v12_unresolved_unchanged']==2065, 'Unexpected source corpus; inspect manifest / remove fixed-coverage assertion for another dataset'
    print('TOTAL',manifest)
if __name__=='__main__':main()
