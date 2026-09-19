#!/usr/bin/env python3
"""Read-only PREDESTINATI octave/transition-penalty audit.

Run from repository root: python tests/diagnose_predestinati_transition_penalty.py
Uses the EXISTING rapid_interval_penalty function from
  tests/diagnose_ratata_harmonic_locks.py
and the synchronized diagnostic files. Does not modify production code,
create pitch estimates, repair discontinuities, or output MIDI.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'


def number(s):
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def st(a, b):
    return 12.0 * math.log2(b / a)


def load_existing_penalty(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'Existing penalty source not found: {path}\n'
            'Supply --penalty-source PATH to the existing script. '
            'This diagnostic will NOT substitute a fabricated curve.'
        )
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TESTS))
    spec = importlib.util.spec_from_file_location('_existing_ratata_penalty', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import existing source: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    func = getattr(module, 'rapid_interval_penalty', None)
    if not callable(func):
        raise RuntimeError(f'{path} does not define callable rapid_interval_penalty')
    signature = inspect.signature(func)
    try:
        signature.bind(12.0, 46.0)
    except TypeError as e:
        raise RuntimeError(
            f'Existing function signature is {signature}, not compatible '
            f'with (interval_semitones, duration_ms): {e}'
        ) from e
    return func, signature


def load_frames(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing synchronized frames CSV: {path}')
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    required = {'time_s', 'voiced_samples', 'first_f0_hz', 'median_f0_hz', 'last_f0_hz'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Wrong or empty synchronized frames CSV; regenerate with diagnose_synchronized_eckf.py')
    return rows


def parse_events(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing synchronized report: {path}')
    events = []
    pattern = re.compile(r'^\s*(\d+\.\d+)s\s+INITIALIZATION_PROPOSED\s+.*?init_f0_hz=([0-9.eE+-]+)')
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            events.append((float(match.group(1)), float(match.group(2))))
    if not events:
        raise ValueError('No INITIALIZATION_PROPOSED entries in synchronized report')
    return events


def near(rows, time):
    return min(rows, key=lambda row: abs(float(row['time_s']) - time))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--penalty-source', type=Path, default=TESTS / 'diagnose_ratata_harmonic_locks.py')
    p.add_argument('--frames', type=Path, default=TESTS / 'PREDESTINATI_synchronized_diagnostic_frames.csv')
    p.add_argument('--report', type=Path, default=TESTS / 'PREDESTINATI_synchronized_diagnostic.txt')
    p.add_argument('--output', type=Path, default=TESTS / 'PREDESTINATI_transition_penalty_audit.txt')
    p.add_argument('--start', type=float, default=58.0)
    p.add_argument('--end', type=float, default=59.1)
    a = p.parse_args()
    func, signature = load_existing_penalty(a.penalty_source)
    rows = load_frames(a.frames)
    events = parse_events(a.report)
    out = []
    def say(text=''):
        out.append(str(text))

    say('=== EXISTING INTERVAL/TIME PENALTY AUDIT (READ ONLY) ===')
    say(f'Existing implementation: {a.penalty_source.resolve()}')
    say(f'Function: rapid_interval_penalty{signature}')
    say('Sources: synchronized trace and frame CSV from the same ECKF run.')
    say('Penalty applied to raw proposals for OBSERVATION ONLY; no frequency is altered.')
    say()
    say('=== INITIALIZER PROPOSALS ===')
    for event_time, proposed in events:
        if not a.start <= event_time <= a.end:
            continue
        current_index = min(range(len(rows)), key=lambda i: abs(float(rows[i]['time_s']) - event_time))
        if abs(float(rows[current_index]['time_s']) - event_time) > 0.03:
            say(f'{event_time:.4f}s proposal={proposed:.3f} Hz; missing matching frame')
            continue
        prior = next((rows[i] for i in range(current_index - 1, -1, -1)
                      if number(rows[i]['last_f0_hz']) is not None
                      and number(rows[i]['last_f0_hz']) > 0), None)
        if prior is None:
            say(f'{event_time:.4f}s proposal={proposed:.3f} Hz; no prior voiced F0')
            continue
        previous = number(prior['last_f0_hz'])
        prev_t = float(prior['time_s'])
        curr_t = float(rows[current_index]['time_s'])
        elapsed_ms = (curr_t - prev_t) * 1000.0
        delta = st(previous, proposed)
        penalty = func(abs(delta), elapsed_ms)
        say(f'{event_time:.4f}s previous_last={previous:.3f} Hz '
            f'proposal={proposed:.3f} Hz interval={delta:+.3f} st '
            f'time={elapsed_ms:.3f} ms EXISTING_penalty={penalty!r}')
        say(f'  observed_new_frame_first={rows[current_index]["first_f0_hz"]} Hz '
            f'median={rows[current_index]["median_f0_hz"]} Hz')
    say()
    say('=== FRAME-TO-FRAME OUTPUT TRANSITIONS (NO EXTRA PENALTY FORMULA) ===')
    previous = None
    for row in rows:
        time = float(row['time_s'])
        if not a.start <= time <= a.end:
            continue
        current = number(row['median_f0_hz'])
        if current is not None and current > 0 and previous is not None:
            old_t, old_f = previous
            dt = 1000.0 * (time - old_t)
            interval = st(old_f, current)
            penalty = func(abs(interval), dt)
            say(f'{old_t:.4f}->{time:.4f}s {old_f:.2f}->{current:.2f} Hz '
                f'interval={interval:+.3f} st dt={dt:.1f} ms penalty={penalty!r}')
        previous = (time, current) if current is not None and current > 0 else None
    say()
    say('NOTE: Penalty is diagnostic evidence, NOT permission to reject, bridge,')
    say('halve, interpolate, or reinterpret a sung pitch or MIDI note boundary.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(out) + '\n'
    a.output.write_text(text)
    print(text, end='')
    print(f'Report saved: {a.output}')


if __name__ == '__main__':
    main()
