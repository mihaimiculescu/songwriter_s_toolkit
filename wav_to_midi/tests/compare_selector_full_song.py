#!/usr/bin/env python3
"""Read-only, aligned comparison of synchronized full-song v2/v3 ECKF outputs.

No MIDI, frequency corrections, or musical note inference. Both runs must use
identical WAV, block size, sample rate, window and config; validate separately.
"""
import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

FIELDS = ('first_f0_hz', 'median_f0_hz', 'last_f0_hz')
EVENT = re.compile(r'^\s*([0-9]+\.[0-9]+)s\s+(INITIALIZATION_(?:CANDIDATES|REJECTED|PROPOSED)|RESET_COMPLETED)\s+(.*)$')
KV = re.compile(r'([A-Za-z_]+)=([^\s]+)')


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows or not {'sample', 'time_s', 'voiced_samples', *FIELDS} <= rows[0].keys():
        raise ValueError(f'Empty or incompatible synchronized CSV: {path}')
    out = {}
    for row in rows:
        sample = int(row['sample'])
        if sample in out:
            raise ValueError(f'Duplicate sample {sample} in {path}')
        out[sample] = row
    return out


def read_events(path):
    result = {}
    header = path.read_text(encoding='utf-8')
    for line in header.splitlines():
        match = EVENT.match(line)
        if not match:
            continue
        time, name, details = match.groups()
        data = dict(KV.findall(details))
        if 'frame_start' not in data:
            continue
        sample = int(data['frame_start'])
        result.setdefault(sample, []).append((name, data))
    return result, header


def event_summary(events, sample):
    chunks = []
    for name, data in events.get(sample, []):
        if name == 'RESET_COMPLETED':
            chunks.append('RESET:' + data.get('init_f0_hz', '?'))
        elif name == 'INITIALIZATION_REJECTED':
            chunks.append('REJECT:' + data.get('reason', '?'))
        elif name == 'INITIALIZATION_CANDIDATES':
            chunks.append('CANDIDATES:' + data.get('reason', '?'))
    return ','.join(chunks) or '-'


def fmt(x):
    return f'{x:.3f}' if x is not None else '--'


def analyze(v2_csv, v3_csv, v2_txt, v3_txt, threshold, output):
    old, new = read_csv(v2_csv), read_csv(v3_csv)
    old_events, old_head = read_events(v2_txt)
    new_events, new_head = read_events(v3_txt)
    if set(old) != set(new):
        raise ValueError(f'Frame sample grids differ: v2-only={len(set(old)-set(new))}, v3-only={len(set(new)-set(old))}. Regenerate both with same window.')
    for token in ('WAV:', 'Sample rate:', 'Window:'):
        old_line = next((l for l in old_head.splitlines() if l.startswith(token)), None)
        new_line = next((l for l in new_head.splitlines() if l.startswith(token)), None)
        if not old_line or old_line != new_line:
            raise ValueError(f'Run metadata mismatch/missing for {token}: {old_line!r} != {new_line!r}')
    samples = sorted(old)
    rows = []
    classifications = Counter()
    status_changes = []
    pitch_changes = []
    reset_changes = []
    for sample in samples:
        a, b = old[sample], new[sample]
        if abs(float(a['time_s']) - float(b['time_s'])) > 1e-6:
            raise ValueError(f'Time-grid mismatch at sample {sample}')
        ap, bp = number(a['median_f0_hz']), number(b['median_f0_hz'])
        av, bv = int(a['voiced_samples']) > 0, int(b['voiced_samples']) > 0
        if av != bv:
            classification = 'VOICING_CHANGED'
        elif ap is not None and bp is not None:
            delta = 12 * math.log2(bp / ap)
            classification = 'LARGE_PITCH_CHANGE' if abs(delta) >= threshold else 'similar_voiced'
        else:
            delta = None
            classification = 'both_unvoiced'
        classifications[classification] += 1
        entry = {'sample': sample, 'time_s': float(a['time_s']), 'v2_voiced': int(a['voiced_samples']),
                 'v3_voiced': int(b['voiced_samples']), 'v2_median_hz': ap, 'v3_median_hz': bp,
                 'delta_st': delta, 'v2_events': event_summary(old_events, sample),
                 'v3_events': event_summary(new_events, sample), 'classification': classification}
        rows.append(entry)
        if av != bv:
            status_changes.append(entry)
        if classification == 'LARGE_PITCH_CHANGE':
            pitch_changes.append(entry)
        if entry['v2_events'] != entry['v3_events'] and ('RESET' in entry['v2_events'] or 'RESET' in entry['v3_events'] or 'REJECT' in entry['v2_events'] or 'REJECT' in entry['v3_events']):
            reset_changes.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_csv = output.with_suffix('.csv')
    with out_csv.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['=== FULL-SONG SELECTOR v2 → v3 COMPARISON: READ ONLY ===',
             f'v2: {v2_csv}', f'v3: {v3_csv}', f'frames: {len(rows)}',
             'Same WAV, sample rate, window and exact sample grid: VERIFIED',
             'Voiced means nonempty positive F0 in the synchronized frame, not a note boundary.',
             f'Large pitch difference threshold: {threshold:.2f} semitones',
             'No MIDI or reference melody was consulted. Difference != automatically improvement.',
             '', '=== COUNTS ===']
    lines += [f'{key}: {value}' for key, value in sorted(classifications.items())]
    lines += [f'frames with changed reset/rejection events: {len(reset_changes)}', '',
              '=== EVERY VOICING STATUS CHANGE ===']
    def show(entry):
        return (f"{entry['time_s']:.4f}s sample={entry['sample']} "
                f"v2={fmt(entry['v2_median_hz'])}Hz ({entry['v2_voiced']} voiced samples) "
                f"v3={fmt(entry['v3_median_hz'])}Hz ({entry['v3_voiced']} voiced samples) "
                f"delta={fmt(entry['delta_st'])}st "
                f"v2=[{entry['v2_events']}] v3=[{entry['v3_events']}]")
    lines += [show(r) for r in status_changes] or ['none']
    lines += ['', '=== EVERY LARGE F0 DIFFERENCE ===']
    lines += [show(r) for r in pitch_changes] or ['none']
    lines += ['', '=== EVERY RESET/REJECTION EVENT CHANGE ===']
    lines += [show(r) for r in reset_changes] or ['none']
    lines += ['', '=== INTERPRETATION ===',
              'Investigate every changed frame using its same-run trace and waveform evidence.',
              'This comparison identifies regressions/candidates; it cannot score pitch accuracy without an independent acoustic check.',
              'Do not interpret status changes as MIDI note boundaries.']
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return classifications, output, out_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tests = Path(__file__).resolve().parent
    parser.add_argument('--v2', type=Path, default=tests / 'PREDESTINATI_full_v2_frames.csv')
    parser.add_argument('--v3', type=Path, default=tests / 'PREDESTINATI_full_v3_frames.csv')
    parser.add_argument('--v2-report', type=Path, default=tests / 'PREDESTINATI_full_v2.txt')
    parser.add_argument('--v3-report', type=Path, default=tests / 'PREDESTINATI_full_v3.txt')
    parser.add_argument('--threshold-st', type=float, default=1.0)
    parser.add_argument('--output', type=Path, default=tests / 'PREDESTINATI_full_selector_comparison.txt')
    args = parser.parse_args()
    if not 0 < args.threshold_st < 24:
        parser.error('threshold must be between 0 and 24 semitones')
    counts, report, csv_out = analyze(args.v2, args.v3, args.v2_report, args.v3_report, args.threshold_st, args.output)
    print(report.read_text(encoding='utf-8'))
    print(f'Saved: {report}\nSaved: {csv_out}')


if __name__ == '__main__':
    main()
