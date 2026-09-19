#!/usr/bin/env python3
"""Read-only source audit: locate the ACTUAL pitch candidate selection and penalty wiring.

Run: python tests/inspect_existing_candidate_selection.py
Output: tests/PREDESTINATI_candidate_selection_source_audit.txt
No tracker changes, pitch estimates, guessed ranking rules, or MIDI output.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'tests' / 'PREDESTINATI_candidate_selection_source_audit.txt'
KEYWORDS = ('candidate', 'select', 'score', 'rank', 'penalty', 'transition', 'harmonic', 'octave', 'pitch', 'resolve', 'choose')
IMPORTANT = ('rapid_interval_penalty', 'candidate', 'penalty', 'argmin', 'argmax', 'sorted', 'min', 'max')


def qualified_functions(tree):
    result = []
    def walk(node, parents):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.append((child, '.'.join((*parents, child.name))))
                walk(child, (*parents, child.name))
            elif isinstance(child, ast.ClassDef):
                walk(child, (*parents, child.name))
            else:
                walk(child, parents)
    walk(tree, ())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--max-source-lines', type=int, default=180,
                   help='Maximum lines printed per relevant function (default: 180)')
    args = p.parse_args()
    tests = ROOT / 'tests'
    preferred = tests / 'diagnose_ratata_harmonic_locks.py'
    if not preferred.is_file():
        raise SystemExit(f'Expected implementation missing: {preferred}; cannot substitute a new curve.')
    paths = sorted(set(ROOT.glob('*.py')) | set(tests.glob('*.py')) | set((ROOT / 'python_eckf').glob('*.py')))
    own_path = Path(__file__).resolve()
    paths = [path for path in paths if path.resolve() != own_path]
    lines = ['=== EXISTING CANDIDATE SELECTION / PENALTY SOURCE AUDIT ===',
             f'Repository: {ROOT}', f'Primary source: {preferred}',
             'Read-only AST inspection; reported source remains authoritative.', '']
    found_penalty = False
    for path in paths:
        try:
            content = path.read_text(encoding='utf-8-sig')
            tree = ast.parse(content, filename=str(path))
        except (UnicodeError, OSError, SyntaxError) as exc:
            lines.append(f'UNREADABLE {path.relative_to(ROOT)}: {exc}')
            continue
        source_lines = content.splitlines()
        funcs = qualified_functions(tree)
        interesting = []
        for node, name in funcs:
            section = '\n'.join(source_lines[node.lineno-1:node.end_lineno])
            score = sum(term in name.lower() for term in KEYWORDS)
            if 'rapid_interval_penalty' in section:
                score += 6
            if any(term in section.lower() for term in ('candidate', 'penalty', 'argmin', 'argmax')):
                score += 1
            if score >= 2 or name.endswith('rapid_interval_penalty'):
                interesting.append((node, name, score, section))
            if name.endswith('rapid_interval_penalty'):
                found_penalty = True
        call_sites = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                try:
                    call = ast.unparse(node.func)
                except Exception:
                    continue
                if any(term in call.lower() for term in IMPORTANT):
                    if 'rapid_interval_penalty' in call or any(term in call.lower() for term in ('candidate', 'penalty', 'argmin', 'argmax')):
                        call_sites.append((node.lineno, call))
        if not interesting and not call_sites:
            continue
        lines.extend((f'\n--- FILE: {path.relative_to(ROOT)} ---',
                      f'Candidate-related functions: {len(interesting)}; relevant call sites: {len(call_sites)}'))
        for line_no, call in sorted(set(call_sites)):
            lines.append(f'CALL {line_no}: {call}')
        for node, name, score, section in sorted(interesting, key=lambda t: (-t[2], t[0].lineno)):
            lines.append(f'\nFUNCTION {name} lines {node.lineno}-{node.end_lineno} (relevance {score})')
            src = section.splitlines()
            for i, text in enumerate(src[:args.max_source_lines], start=node.lineno):
                lines.append(f'{i:5d} | {text}')
            if len(src) > args.max_source_lines:
                lines.append(f'... OMITTED {len(src)-args.max_source_lines} lines. Read original source before editing.')
    if not found_penalty:
        lines.append('\nWARNING: rapid_interval_penalty definition not located by AST. Verify source manually.')
    lines.extend(('', '=== INTEGRATION QUESTIONS (NOT ANSWERED BY PENALTY ALONE) ===',
                  '1. Which function constructs alternatives from actual WAV evidence?',
                  '2. Where is rapid_interval_penalty added to candidate costs, and in what units?',
                  '3. What comparison determines the selected candidate?',
                  '4. Does an unresolved voiced frame preserve a separate status for MIDI?',
                  '5. Does any code silently halve, bridge, or interpolate a frequency?',
                  'No production code was changed.'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Source audit written: {args.output}')
    print(f'Lines: {len(lines)}')
    print('Send the report for a source-verified integration patch.')


if __name__ == '__main__':
    main()
