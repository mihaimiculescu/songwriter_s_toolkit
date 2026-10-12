#!/usr/bin/env python3
"""Read-only V7 acoustic-admissibility threshold sweep; run from repository root.
Runs the existing V7 adjudicator for each ACF cutoff/support-count combination.
Preserves all trial CSVs, compares observation champions with baseline, and writes
summary.csv, per_song.csv, manifest.json. Never edits source or production outputs.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
ACFS = (0.60, 0.66, 0.72, 0.78, 0.84, 0.90)
COUNTS = (1, 2, 3)

def csvrows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def decisions(folder, song):
    rows = csvrows(folder / (song + '_observation_summary.csv'))
    result = {}
    for r in rows:
        if r['observation_outcome'] == 'provisional_tournament_champion' and r['selected_provisional_hz']:
            result[round(float(r['time_s']), 6)] = float(r['selected_provisional_hz'])
    return result

def same_pitch(a, b, cents=35.0):
    return a > 0 and b > 0 and abs(1200.0 * math.log2(a / b)) <= cents

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--script', default='tests/audit_shadow_adjudicator_v22_fixed_v7.py')
    ap.add_argument('--tests-root', default='tests')
    ap.add_argument('--out', default='tests/acf_gate_sensitivity')
    args = ap.parse_args()
    source = Path(args.script).resolve()
    root = Path(args.tests_root).resolve()
    out = Path(args.out).resolve()
    if not source.is_file():
        ap.error(f'V7 script missing: {source}')
    for folder in ('full_timeline_v13', 'temporal_integer_families_v21'):
        if not (root / folder).is_dir():
            ap.error(f'Missing input directory: {root / folder}')
    out.mkdir(parents=True, exist_ok=True)
    settings = [(0.72, 2)] + [(a, n) for a in ACFS for n in COUNTS if (a, n) != (0.72, 2)]
    baseline = None
    summary = []
    per_song = []
    for index, (acf, count) in enumerate(settings, 1):
        trial = f'acf_{acf:.2f}_n_{count}'
        folder = out / trial
        cmd = [sys.executable, str(source), '--case', 'all', '--tests-root', str(root),
               '--out', str(folder), '--min-acf', str(acf), '--min-supported-measurements', str(count)]
        print(f'[{index}/{len(settings)}] {trial}', flush=True)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            (out / 'failure.log').write_text(f'Command: {cmd!r}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n')
            raise SystemExit(f'{trial} failed. See {out / "failure.log"}')
        manifest = json.loads((folder / 'manifest.json').read_text())
        if not manifest.get('curve_imported_unchanged'):
            raise SystemExit(f'{trial}: transition curve unavailable; refusing partial results')
        decisions_by_song = {song: decisions(folder, song) for song in CASES}
        if baseline is None:
            baseline = decisions_by_song
        total = dict(champions=0, unresolved=0, no_admissible_pairs=0, gained=0, lost=0, changed=0, retained=0)
        for song in CASES:
            item = manifest['cases'][song]
            obs = item['observation_outcomes']
            now, old = decisions_by_song[song], baseline[song]
            both = set(now) & set(old)
            row = dict(trial=trial, min_acf=acf, min_supported_measurements=count,
                       song=song, observations=item['observations_with_pairs'],
                       champions=len(now), unresolved=item['observations_with_pairs'] - len(now),
                       no_admissible_pairs=item['pair_outcomes'].get('abstain_no_admissible_candidate', 0),
                       gained=len(set(now) - set(old)), lost=len(set(old) - set(now)),
                       changed=sum(not same_pitch(now[t], old[t]) for t in both),
                       retained=sum(same_pitch(now[t], old[t]) for t in both),
                       no_decisive_pair=obs.get('abstain_no_decisive_pair', 0),
                       tournament_abstentions=sum(v for k,v in obs.items() if k.startswith('abstain_tournament')))
            per_song.append(row)
            for k in total:
                total[k] += row[k]
        summary.append(dict(trial=trial, min_acf=acf, min_supported_measurements=count,
                            observations=sum(manifest['cases'][s]['observations_with_pairs'] for s in CASES), **total))
        print('  champions:',total['champions'],'unresolved:',total['unresolved'],
              'lost:',total['lost'],'changed:',total['changed'],flush=True)
        write_csv(out / 'summary.csv', summary)
        write_csv(out / 'per_song.csv', per_song)
    metadata = dict(schema='v22_v7_acf_gate_sensitivity', trials=len(settings),
                    baseline={'min_acf':0.72,'min_supported_measurements':2},
                    fixed='All V7 scoring weights, other thresholds, curves, and tournament logic',
                    outputs='trial subdirectories contain original V7 CSVs and manifests',
                    note='Champion count is decisiveness, not independently verified accuracy.')
    (out / 'sweep_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Complete:',out,flush=True)

if __name__ == '__main__':
    main()
