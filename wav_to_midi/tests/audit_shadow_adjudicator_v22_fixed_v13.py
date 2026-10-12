#!/usr/bin/env python3
"""V13 shadow replay: V12 verdicts + conditional Harmonic Detective V2 tie-break.

Read-only replay over all four songs. Does not infer silence from abstention and does
not introduce a relative-energy or noise threshold. Requires the conservative V2
qualification, including acoustic/range admission and complete note-group field.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
FIELDS = ('case','time_s','v12_outcome','v12_abstention_category','v12_champion_hz',
          'detective_called','group_count','eligible_representatives','best_representative_hz',
          'best_midi','best_score','best_support','runnerup_representative_hz','runnerup_score',
          'harmonic_margin','diagnostic_proposed_hz','reason','group_issues')

def load(source: Path, name: str):
    if source.is_dir():
        with (source/name).open(newline='',encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    with zipfile.ZipFile(source) as z:
        matches=[n for n in z.namelist() if n==name or n.endswith('/'+name)]
        if len(matches)!=1: raise ValueError(f'{source}: {name}: expected one, got {len(matches)}')
        return list(csv.DictReader(io.StringIO(z.read(matches[0]).decode('utf-8-sig'))))

def number(x):
    try:
        n=float(x)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError): return None

def key(row): return round(float(row['time_s']),6)

def write(path, rows):
    if not rows: raise ValueError(f'No records: {path}')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def valid_representative(v12, proposed):
    groups=json.loads(v12.get('note_groups_json') or '[]')
    if len(groups)<2: return False, 'not_a_cross_note_contest'
    representatives=[]
    for group in groups:
        hz=number(group.get('representative_hz'))
        if hz is None: return False, 'incomplete_note_group_field'
        representatives.append(hz)
    if len({round(x,4) for x in representatives})!=len(representatives):
        return False,'duplicate_representatives'
    if not any(abs(x-proposed)<.02 for x in representatives):
        return False,'proposal_not_a_v12_representative'
    return True,'passed'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v12-zip',type=Path,default=Path('tests/shadow_adjudicator_v22_fixed_v12.zip'))
    p.add_argument('--detective-v2',type=Path,default=Path('tests/harmonic_detective_v2.zip'))
    p.add_argument('--out',type=Path,default=Path('tests/shadow_adjudicator_v22_fixed_v13'))
    args=p.parse_args()
    for src in (args.v12_zip,args.detective_v2):
        if not src.exists():p.error(f'Input missing: {src}')
    args.out.mkdir(parents=True,exist_ok=True)
    summaries=[];combined=[];watch=None
    for case in CASES:
        base=load(args.v12_zip,f'{case}_observation_summary.csv')
        proposed=load(args.detective_v2,f'{case}_hung_jury_audit.csv')
        base_by_time={key(r):r for r in base}
        prop_by_time={key(r):r for r in proposed}
        if len(base_by_time)!=len(base) or len(prop_by_time)!=len(proposed) or set(base_by_time)!=set(prop_by_time):
            raise ValueError(f'{case}: missing / duplicate timestamps across V12 and V2')
        out=[]
        for t,v in sorted(base_by_time.items()):
            d=prop_by_time[t]
            for field in ('v12_champion_hz','v12_outcome'):
                actual=v.get('selected_provisional_hz') if field=='v12_champion_hz' else v.get('observation_outcome')
                observed=d.get(field)
                if field=='v12_champion_hz':
                    if (number(actual) is None)!=(number(observed) is None) or (number(actual) is not None and abs(number(actual)-number(observed))>=.02):
                        raise ValueError(f'{case} {t}: V12 champion differs in V2')
                elif actual!=observed: raise ValueError(f'{case} {t}: V12 outcome differs in V2')
            original=number(v.get('selected_provisional_hz'))
            candidate=number(d.get('diagnostic_proposed_hz'))
            decision=original;provenance='v12_jury' if original is not None else 'unresolved'
            reason='v12_champion_preserved' if original is not None else d.get('reason','')
            if original is not None and candidate is not None:
                raise ValueError(f'{case} {t}: detective attempted to override existing champion')
            if candidate is not None:
                if d.get('reason')!='diagnostic_tiebreak_proposal_NOT_VALIDATED' or d.get('detective_called')!='1':
                    raise ValueError(f'{case} {t}: unexpected proposal provenance')
                permitted,why=valid_representative(v,candidate)
                if not permitted: raise ValueError(f'{case} {t}: invalid proposal: {why}')
                if d.get('group_issues'):
                    raise ValueError(f'{case} {t}: unresolved note-group qualification: {d["group_issues"]}')
                if int(d['eligible_representatives']) != int(d['group_count']):
                    raise ValueError(f'{case} {t}: missing admissible opponent')
                if number(d.get('best_score')) is None or number(d.get('best_support')) is None or number(d.get('harmonic_margin')) is None:
                    raise ValueError(f'{case} {t}: missing harmonic evidence')
                if abs(candidate-number(d.get('best_representative_hz')))>=.02:
                    raise ValueError(f'{case} {t}: candidate not top harmonic representative')
                decision=candidate;provenance='harmonic_detective_tiebreak';reason='v12_hung_detective_v2_qualified'
            record={**v,'v13_selected_provisional_hz':decision if decision is not None else '',
                    'v13_outcome':'champion' if decision is not None else 'abstain',
                    'v13_provenance':provenance,'v13_reason':reason,
                    'detective_score':d.get('best_score','') if candidate is not None else '',
                    'detective_support':d.get('best_support','') if candidate is not None else '',
                    'detective_margin':d.get('harmonic_margin','') if candidate is not None else ''}
            out.append(record)
            if case=='RATATA' and abs(t-24.34)<.000001:watch={'case':case,'time_s':t,'v12_outcome':v.get('observation_outcome'),
                'v12_champion_hz':original,'detective_proposed_hz':candidate,
                'v13_selected_hz':decision,'provenance':provenance,'reason':reason,
                'note':'No silence veto exists in the V12/V2 data; V12 abstention is not a positive silence decision.'}
        write(args.out/f'{case}_v13_observation_summary.csv',out)
        stats=Counter(r['v13_provenance'] for r in out)
        result={'case':case,'observations':len(out),'v12_champions':stats['v12_jury'],
                'detective_assisted_champions':stats['harmonic_detective_tiebreak'],
                'v13_champions':stats['v12_jury']+stats['harmonic_detective_tiebreak'],
                'v13_unresolved':stats['unresolved']}
        summaries.append(result);combined.extend(out)
        print(case,result)
    assert len(combined)==5068, f'Unexpected observation count {len(combined)}'
    assert sum(r['v12_champions'] for r in summaries)==3003, 'Unexpected V12 baseline'
    assert sum(r['detective_assisted_champions'] for r in summaries)==467, 'Detective proposal count changed: check V2 inputs'
    write(args.out/'summary_by_song.csv',summaries)
    if watch is None:raise ValueError('RATATA 24.340 watch timestamp not found')
    (args.out/'RATATA_HR0141_watch.json').write_text(json.dumps(watch,indent=2)+'\n')
    manifest={'mode':'read_only_v12_plus_conditional_detective_v2',
              'source_v12':str(args.v12_zip),'source_detective_v2':str(args.detective_v2),
              'weights_and_thresholds':'V12 unchanged; V2 existing conservative FFT qualification unchanged',
              'relative_energy_gate':'not_added_as_requested',
              'silence_policy':'No explicit V12 silence verdict exists in these summaries. Abstention alone cannot veto the detective.',
              'total_observations':len(combined),'summary_by_song':summaries,'watch_HR0141':watch,
              'limitations':['Shadow replay only, not production MIDI output.',
                  'A measurable F0 is not automatically an audible note or MIDI note.',
                  'HR0141 is watched, not manually excluded or specially penalized.',
                  'Diagnostic V2 proposals were validated on a limited RATATA sample, not fully independently across all four recordings.']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('WATCH_HR0141',json.dumps(watch))
    print('TOTAL',sum(s['v13_champions'] for s in summaries),'champions;',sum(s['v13_unresolved'] for s in summaries),'unresolved')
if __name__=='__main__':main()
