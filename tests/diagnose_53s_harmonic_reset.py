#!/usr/bin/env python3
"""Focused, read-only audit of the suspicious reset around 53.2666 s.

Purpose
-------
Observe whether the EXISTING interval/time penalty would oppose the reset from the
preceding ECKF trajectory to a low harmonic/submultiple candidate, while keeping
waveform periodicity, initializer proposal, selected reset, and actual output
separate.

NO MASSAGING:
- does not alter tracker output
- does not multiply/divide candidate frequencies
- does not consult MIDI
- does not create note boundaries
- does not interpolate or smooth F0

Run from repository root after generating the 52.5..53.6 synchronized diagnostic:

    python tests/diagnose_synchronized_eckf.py \
        --start 52.9 --end 53.6 \
        --output tests/PREDESTINATI_reacquisition_53s.txt

    python tests/diagnose_53s_harmonic_reset.py

Optional explicit paths are available below via CLI arguments.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def finite_float(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def semitones(a: float, b: float) -> float:
    return 12.0 * math.log2(b / a)


def load_existing_penalty(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Existing penalty source not found: {path}\n"
            "This diagnostic will not invent a replacement penalty curve."
        )
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TESTS))
    spec = importlib.util.spec_from_file_location("_existing_interval_penalty", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    func = getattr(module, "rapid_interval_penalty", None)
    if not callable(func):
        raise RuntimeError(f"{path} does not define rapid_interval_penalty")
    signature = inspect.signature(func)
    try:
        signature.bind(12.0, 46.0)
    except TypeError as exc:
        raise RuntimeError(
            f"rapid_interval_penalty has incompatible signature {signature}: {exc}"
        ) from exc
    return func, signature


def load_frames(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    needed = {
        "sample", "time_s", "voiced_samples", "first_f0_hz", "median_f0_hz",
        "last_f0_hz", "min_f0_hz", "max_f0_hz",
    }
    if not rows or not needed.issubset(rows[0]):
        raise ValueError("Unexpected synchronized frame CSV schema")
    if len({int(r["sample"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate frame sample indices")
    return rows


@dataclass
class Event:
    time: float
    name: str
    fields: dict[str, str]


EVENT_RE = re.compile(r"^\s*(\d+\.\d+)s\s+([A-Z0-9_]+)\s*(.*)$")
FIELD_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def load_events(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing synchronized report: {path}")
    out: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = EVENT_RE.match(line)
        if not m:
            continue
        fields = {k: v for k, v in FIELD_RE.findall(m.group(3))}
        out.append(Event(float(m.group(1)), m.group(2), fields))
    if not out:
        raise ValueError("No trace events parsed from synchronized report")
    return out


def nearest_frame(rows, t: float):
    return min(rows, key=lambda r: abs(float(r["time_s"]) - t))


def previous_voiced_frame(rows, target_sample: int):
    candidates = []
    for row in rows:
        rt = float(row["time_s"])
        f = finite_float(row["last_f0_hz"])
        if int(row["sample"]) < target_sample and f is not None and f > 0:
            candidates.append(row)
    return max(candidates, key=lambda r: float(r["time_s"])) if candidates else None


def next_voiced_frame(rows, t: float):
    candidates = []
    for row in rows:
        rt = float(row["time_s"])
        f = finite_float(row["first_f0_hz"])
        if rt > t and f is not None and f > 0:
            candidates.append(row)
    return min(candidates, key=lambda r: float(r["time_s"])) if candidates else None


def event_number(event: Event, key: str) -> Optional[float]:
    return finite_float(event.fields.get(key))


def fmt(x: Optional[float], digits=3):
    return "--" if x is None else f"{x:.{digits}f}"


def penalty_line(func, reference_hz: float, candidate_hz: float, dt_ms: float):
    interval = semitones(reference_hz, candidate_hz)
    penalty = func(abs(interval), dt_ms)
    return interval, penalty


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", type=Path,
        default=TESTS / "PREDESTINATI_reacquisition_53s.txt",
    )
    p.add_argument(
        "--frames", type=Path,
        default=TESTS / "PREDESTINATI_reacquisition_53s_frames.csv",
    )
    p.add_argument(
        "--penalty-source", type=Path,
        default=TESTS / "diagnose_ratata_harmonic_locks.py",
    )
    p.add_argument("--target", type=float, default=53.2666)
    p.add_argument("--sample-rate", type=float, default=44100.0)
    p.add_argument("--radius", type=float, default=0.18)
    p.add_argument(
        "--output", type=Path,
        default=TESTS / "PREDESTINATI_53s_harmonic_reset_audit.txt",
    )
    a = p.parse_args()

    penalty_func, penalty_sig = load_existing_penalty(a.penalty_source)
    rows = load_frames(a.frames)
    events = load_events(a.report)

    lo = a.target - a.radius
    hi = a.target + a.radius
    relevant_events = [e for e in events if lo <= e.time <= hi]
    if not relevant_events:
        raise ValueError(f"No events found in {lo:.4f}..{hi:.4f}s")

    # Locate the closest reset event to target. We deliberately inspect what
    # production actually selected, rather than constructing a substitute.
    reset_events = [e for e in relevant_events if e.name == "RESET_COMPLETED"]
    if not reset_events:
        raise ValueError("No RESET_COMPLETED event near target")
    reset = min(reset_events, key=lambda e: abs(e.time - a.target))

    proposed_candidates = [
        e for e in relevant_events
        if e.name == "INITIALIZATION_PROPOSED" and abs(e.time - reset.time) < 0.002
    ]
    periodicity_events = [
        e for e in relevant_events
        if e.name == "PERIODICITY_DECISION" and abs(e.time - reset.time) < 0.002
    ]
    audit_events = [
        e for e in relevant_events
        if e.name in {"OCTAVE_INITIALIZATION_AUDIT", "INITIALIZATION_CANDIDATES"}
        and abs(e.time - reset.time) < 0.002
    ]

    reset_hz = event_number(reset, "init_f0_hz")
    proposal_hz = (
        event_number(proposed_candidates[0], "init_f0_hz")
        if proposed_candidates else None
    )
    periodicity = periodicity_events[0] if periodicity_events else None
    acf_hz = event_number(periodicity, "acf_frequency_hz") if periodicity else None
    yin_hz = event_number(periodicity, "cmndf_frequency_hz") if periodicity else None
    acf_peak = event_number(periodicity, "acf_peak") if periodicity else None
    cmndf = event_number(periodicity, "cmndf_minimum") if periodicity else None

    reset_sample = int(reset.fields["frame_start"])
    sample_rate = float(a.sample_rate)
    prev = previous_voiced_frame(rows, reset_sample)
    curr = next((r for r in rows if int(r["sample"]) == reset_sample), None)
    if curr is None:
        raise ValueError(f"No exact reset frame sample={reset_sample} in CSV")
    assert prev is None or int(prev["sample"]) < reset_sample, "Previous frame overlaps reset"
    nxt = next_voiced_frame(rows, reset.time)

    lines: list[str] = []
    say = lines.append

    say("=== 53.2666s HARMONIC / INTERVAL-PENALTY AUDIT ===")
    say("READ ONLY. No MIDI. No frequency replacement. No interpolation.")
    say(f"Report:  {a.report.resolve()}")
    say(f"Frames:  {a.frames.resolve()}")
    say(f"Penalty: {a.penalty_source.resolve()}")
    say(f"Function: rapid_interval_penalty{penalty_sig}")
    say(f"Target reset inspected: {reset.time:.4f}s")
    say("")

    say("=== SAME-FRAME RAW EVIDENCE ===")
    if periodicity:
        say(
            f"periodicity: ACF={fmt(acf_hz)} Hz peak={fmt(acf_peak)} | "
            f"YIN={fmt(yin_hz)} Hz CMNDF={fmt(cmndf)} | "
            f"reason={periodicity.fields.get('reason', '--')}"
        )
    else:
        say("periodicity: --")
    say(f"initializer raw proposal: {fmt(proposal_hz)} Hz")
    say(f"selected reset frequency: {fmt(reset_hz)} Hz")
    for e in audit_events:
        say(f"{e.name}: reason={e.fields.get('reason', '--')}")
    say("")

    say("=== ACTUAL ECKF OUTPUT CONTEXT ===")
    if prev:
        say(
            f"previous voiced frame: t={float(prev['time_s']):.4f}s "
            f"first={prev['first_f0_hz']} median={prev['median_f0_hz']} "
            f"last={prev['last_f0_hz']} Hz"
        )
    else:
        say("previous voiced frame: --")
    say(
        f"reset frame: t={float(curr['time_s']):.4f}s "
        f"first={curr['first_f0_hz']} median={curr['median_f0_hz']} "
        f"last={curr['last_f0_hz']} min={curr['min_f0_hz']} "
        f"max={curr['max_f0_hz']} Hz"
    )
    if nxt:
        say(
            f"next voiced frame: t={float(nxt['time_s']):.4f}s "
            f"first={nxt['first_f0_hz']} median={nxt['median_f0_hz']} "
            f"last={nxt['last_f0_hz']} Hz"
        )
    else:
        say("next voiced frame: --")
    say("")

    say("=== EXISTING INTERVAL/TIME PENALTY — OBSERVATION ONLY ===")
    if prev:
        prev_last = finite_float(prev["last_f0_hz"])
        # Frame-start separation, not the rounded trace timestamp and not a
        # same-frame last sample. This is a frame-resolution transition prior.
        dt_ms = (reset_sample - int(prev["sample"])) * 1000.0 / sample_rate
        if prev_last is not None and prev_last > 0:
            say(f"reference = actual previous ECKF last sample: {prev_last:.3f} Hz")
            say(f"elapsed to reset event: {dt_ms:.3f} ms")
            for label, hz in (
                ("raw initializer proposal", proposal_hz),
                ("selected reset", reset_hz),
                ("ACF estimate", acf_hz),
                ("YIN estimate", yin_hz),
            ):
                if hz is None or hz <= 0:
                    say(f"{label:24s}: --")
                    continue
                interval, penalty = penalty_line(penalty_func, prev_last, hz, dt_ms)
                say(
                    f"{label:24s}: {hz:9.3f} Hz  interval={interval:+8.3f} st  "
                    f"penalty={penalty!r}"
                )
        else:
            say("No usable previous ECKF F0.")
    else:
        say("No previous voiced frame available.")
    say("")

    say("=== FRAME-TO-FRAME OUTPUT TRANSITIONS IN LOCAL WINDOW ===")
    local_rows = [r for r in rows if lo <= float(r["time_s"]) <= hi]
    last_voiced = None
    for row in local_rows:
        t = float(row["time_s"])
        med = finite_float(row["median_f0_hz"])
        if med is None or med <= 0:
            say(f"{t:.4f}s UNVOICED/NO_F0")
            last_voiced = None
            continue
        if last_voiced is None:
            say(f"{t:.4f}s median={med:.3f} Hz (start/reacquisition in this printed run)")
        else:
            pt, pf = last_voiced
            dt = (t - pt) * 1000.0
            interval = semitones(pf, med)
            penalty = penalty_func(abs(interval), dt)
            say(
                f"{pt:.4f}->{t:.4f}s {pf:.3f}->{med:.3f} Hz "
                f"interval={interval:+.3f} st dt={dt:.2f} ms penalty={penalty!r}"
            )
        last_voiced = (t, med)
    say("")

    say("=== RAW TRACE EVENTS IN LOCAL WINDOW ===")
    for e in relevant_events:
        if e.name in {
            "PERIODICITY_DECISION", "OCTAVE_INITIALIZATION_AUDIT",
            "INITIALIZATION_CANDIDATES", "INITIALIZATION_PROPOSED",
            "INITIALIZATION_REJECTED", "RESET_COMPLETED",
        }:
            details = " ".join(f"{k}={v}" for k, v in e.fields.items())
            say(f"{e.time:9.4f}s {e.name:28s} {details}")
    say("")

    say("INTERPRETATION RULES")
    say("- A large penalty is evidence against a rapid jump, not permission to replace pitch.")
    say("- ACF/YIN agreement is independent waveform evidence, not ground truth by itself.")
    say("- A consonant/short-vowel disturbance may make the frame genuinely ambiguous.")
    say("- This report intentionally does NOT triple/halve any candidate or create MIDI boundaries.")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    a.output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
