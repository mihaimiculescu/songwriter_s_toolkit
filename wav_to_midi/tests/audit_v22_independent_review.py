#!/usr/bin/env python3
"""Blinded, human-led independent F0 review; no pitch inference or model fitting.

Workflow:
  prepare: build blinded review queue and per-reviewer label sheets from targeted queue.
  finalize: validate human labels, combine independent reviewers conservatively,
            and export clean calibration controls separately from holdouts and tails.

Requires Python standard library only. Does not modify the source queue or WAV files.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {'low', 'high', 'neither', 'uncertain'}
CONFIDENCES = {'high', 'medium', 'low'}
TAIL_CUTOFF = {'RATATA': 87.081}  # performance-end annotation; NOT an automatic energy detector
PUBLIC = ['review_id', 'case', 'time_s', 'low_hz', 'high_hz', 'clip_path', 'phrase_group', 'regression_holdout', 'review_partition']
SHEET = ['review_id', 'reviewer', 'review_label', 'review_confidence', 'independent_evidence', 'notes']


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def is_true(x):
    return str(x).strip().lower() in ('true', '1', 'yes')


def finite_pos(x):
    try:
        z = float(x)
        return math.isfinite(z) and z > 0
    except (TypeError, ValueError):
        return False


def partition(row):
    # A clip overlapping the annotated performance-end is also contaminated.
    # We conservatively exclude it from active-vocal calibration and review it separately.
    if is_true(row.get('regression_holdout')):
        return 'regression_holdout'
    if row['case'] == 'RATATA':
        cutoff = TAIL_CUTOFF['RATATA']
        end = row.get('clip_end_s')
        if float(row['time_s']) >= cutoff or (end not in (None, '') and float(end) > cutoff):
            return 'tail_or_boundary'
    return 'active_candidate'


def prepare(args):
    source = Path(args.input)
    output = Path(args.out)
    if not source.is_file():
        raise SystemExit(f'Missing queue: {source}')
    rows = read_csv(source)
    if not rows:
        raise SystemExit('Empty review queue')
    if len({r['review_id'] for r in rows}) != len(rows):
        raise SystemExit('Duplicate review IDs in source')
    index = {}
    clip_index = source.parent / 'clip_index.csv'
    if clip_index.exists():
        index = {r['review_id']: r for r in read_csv(clip_index)}
    blinded = []
    enriched = []
    for row in rows:
        if not finite_pos(row.get('low_hz')) or not finite_pos(row.get('high_hz')):
            raise SystemExit(f'Invalid candidate frequencies at {row["review_id"]}')
        if float(row['low_hz']) >= float(row['high_hz']):
            raise SystemExit(f'Candidate order invalid at {row["review_id"]}')
        merged = dict(row)
        merged.update({k: v for k, v in index.get(row['review_id'], {}).items() if k not in merged})
        merged['review_partition'] = partition(merged)
        # Do NOT expose energy, ACF, selection stratum, V13, or V5 to the reviewer.
        blinded.append({k: merged.get(k, '') for k in PUBLIC})
        enriched.append(merged)
    write_csv(output / 'blind_review_queue.csv', blinded, PUBLIC)
    write_csv(output / 'reviewer_1.csv', [{'review_id': r['review_id']} for r in blinded], SHEET)
    write_csv(output / 'reviewer_2.csv', [{'review_id': r['review_id']} for r in blinded], SHEET)
    instructions = '''INDEPENDENT F0 REVIEW — BLINDED PROCEDURE\n\nOpen blind_review_queue.csv, and listen to each referenced clip under the original\ntargeted-controls directory. For difficult clips listen to the entire ORIGINAL\nWAV/phrase. Do not use the original targeted_review_queue.csv, V13, V5,\nexclusive-energy measurements, or the selection reasons to decide the label.\n\nAssign low / high / neither / uncertain for the ACTUAL actively sung fundamental.\nIf no active sung note exists (silence, reverb, instrumental residue), mark\nneither and note 'no_active_vocal'. If you cannot determine the F0, uncertain.\nHigh/medium/low confidence, independent_evidence, and reviewer identity are\nmandatory for decisive labels. Distinguish an octave harmonic from an active\nfundamental via harmonic spacing, vowel continuity, listening, and neighboring\naudio. Do not infer F0 from the written candidate frequencies alone.\n\nUse reviewer_1.csv and reviewer_2.csv as separate forms; reviewers should not\nsee one another's labels until both forms are complete. One reviewer is allowed\nfor preliminary inspection but cannot establish independently corroborated\ncalibration controls. Holdouts cannot enter calibration. RATATA audio overlapping\n87.081 seconds or later belongs to tail/boundary review, never active controls.\nThe 0.5-s block/phrase grouping is a dependence safeguard, not a true musical\nphrase segmentation. Finalize exports a group identifier; downstream training\nmust group splits by phrase_group and recording. No score curve is fitted.\n'''
    (output / 'REVIEW_INSTRUCTIONS.txt').write_text(instructions, encoding='utf-8')
    manifest = {'schema': 'v22_blind_independent_review_v1', 'source': str(source),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'rows': len(rows), 'partitions': dict(Counter(r['review_partition'] for r in blinded)),
                'human_labels_created': 0, 'automatic_labels': False,
                'f0_inferred': False, 'calibration_fitted': False, 'production_modified': False,
                'tail_boundary_s': TAIL_CUTOFF}
    (output / 'prepare_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def load_sheet(path, ids, expected_reviewer):
    records = {}
    for row in read_csv(path):
        rid = row.get('review_id', '').strip()
        if rid not in ids or rid in records:
            raise SystemExit(f'Unknown or duplicate review ID {rid!r} in {path}')
        label = row.get('review_label', '').strip().lower()
        if label and label not in LABELS:
            raise SystemExit(f'Invalid label {label!r} for {rid}')
        if label:
            reviewer = row.get('reviewer', '').strip()
            conf = row.get('review_confidence', '').strip().lower()
            evidence = row.get('independent_evidence', '').strip()
            if not reviewer or reviewer != expected_reviewer:
                raise SystemExit(f'Reviewer identity mismatch/missing for {rid} in {path}')
            if conf not in CONFIDENCES:
                raise SystemExit(f'Invalid/missing confidence for {rid}')
            if label in {'low', 'high'} and not evidence:
                raise SystemExit(f'Independent evidence required for decisive label at {rid}')
            row['review_label'] = label
            row['review_confidence'] = conf
            records[rid] = row
    return records


def finalize(args):
    blind = read_csv(Path(args.blind))
    ids = {r['review_id']: r for r in blind}
    if len(ids) != len(blind):
        raise SystemExit('Duplicate IDs in blinded queue')
    if args.reviewer1 == args.reviewer2:
        raise SystemExit('Reviewer identifiers must differ')
    one = load_sheet(args.sheet1, ids, args.reviewer1)
    two = load_sheet(args.sheet2, ids, args.reviewer2)
    out = Path(args.out)
    combined = []
    for r in blind:
        rid = r['review_id']
        a, b = one.get(rid, {}), two.get(rid, {})
        la, lb = a.get('review_label', ''), b.get('review_label', '')
        if not la or not lb:
            status, label = 'pending_independent_review', ''
        elif la == lb and la in ('low', 'high', 'neither') and all(
            x.get('review_confidence') in ('high', 'medium') for x in (a, b)
        ):
            status, label = 'agreed_independent', la
        elif la == lb == 'uncertain':
            status, label = 'agreed_uncertain', ''
        elif la == lb:
            status, label = 'low_confidence', ''
        else:
            status, label = 'reviewer_disagreement', ''
        merged = dict(r)
        merged.update({'reviewer1_label': la, 'reviewer2_label': lb,
                       'reviewer1_confidence': a.get('review_confidence', ''),
                       'reviewer2_confidence': b.get('review_confidence', ''),
                       'reviewer1_evidence': a.get('independent_evidence', ''),
                       'reviewer2_evidence': b.get('independent_evidence', ''),
                       'consensus_status': status, 'consensus_label': label})
        combined.append(merged)
    fields = PUBLIC + ['reviewer1_label', 'reviewer2_label', 'reviewer1_confidence',
                       'reviewer2_confidence', 'reviewer1_evidence', 'reviewer2_evidence',
                       'consensus_status', 'consensus_label']
    write_csv(out / 'review_consensus.csv', combined, fields)
    controls = [r for r in combined if r['review_partition'] == 'active_candidate'
                and r['consensus_status'] == 'agreed_independent'
                and r['consensus_label'] in ('low', 'high')]
    write_csv(out / 'independent_active_controls.csv', controls, fields)
    write_csv(out / 'tail_and_boundary_review.csv',
              [r for r in combined if r['review_partition'] == 'tail_or_boundary'], fields)
    write_csv(out / 'regression_holdout_review.csv',
              [r for r in combined if r['review_partition'] == 'regression_holdout'], fields)
    write_csv(out / 'unresolved_review.csv',
              [r for r in combined if r['consensus_status'] != 'agreed_independent'], fields)
    groups = defaultdict(list)
    for r in controls:
        groups[(r['case'], r['phrase_group'])].append(r['consensus_label'])
    group_rows = [{'case': k[0], 'phrase_group': k[1], 'reviewed_pairs': len(v),
                   'low_count': v.count('low'), 'high_count': v.count('high'),
                   'contradictory_labels': len(set(v)) > 1} for k, v in sorted(groups.items())]
    write_csv(out / 'control_group_accounting.csv', group_rows,
              ['case', 'phrase_group', 'reviewed_pairs', 'low_count', 'high_count', 'contradictory_labels'])
    manifest = {'schema': 'v22_independent_review_consensus_v1', 'rows': len(combined),
                'status_counts': dict(Counter(r['consensus_status'] for r in combined)),
                'active_controls': len(controls), 'active_low_controls': sum(r['consensus_label']=='low' for r in controls),
                'active_high_controls': sum(r['consensus_label']=='high' for r in controls),
                'independent_active_groups': len(groups), 'contradictory_groups': sum(x['contradictory_labels'] for x in group_rows),
                'tail_rows_excluded': sum(r['review_partition']=='tail_or_boundary' for r in combined),
                'holdout_rows_excluded': sum(r['review_partition']=='regression_holdout' for r in combined),
                'calibration_fitted': False, 'production_modified': False,
                'warning': 'These are independent human review labels, not externally instrumented ground truth.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--input', default='tests/exclusive_energy_targeted_controls/targeted_review_queue.csv')
    a.add_argument('--out', default='tests/independent_f0_review')
    a.set_defaults(func=prepare)
    b = sub.add_parser('finalize')
    b.add_argument('--blind', default='tests/independent_f0_review/blind_review_queue.csv')
    b.add_argument('--sheet1', default='tests/independent_f0_review/reviewer_1.csv')
    b.add_argument('--sheet2', default='tests/independent_f0_review/reviewer_2.csv')
    b.add_argument('--reviewer1', required=True)
    b.add_argument('--reviewer2', required=True)
    b.add_argument('--out', default='tests/independent_f0_review/results')
    b.set_defaults(func=finalize)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
