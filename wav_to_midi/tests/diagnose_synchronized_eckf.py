#!/usr/bin/env python3
"""One-run ECKF diagnostic: trace events and F0 from the SAME execution.

Read-only: does not change tracker.py, MIDI, WAV, or the pitch estimates.
Produces a small human-readable report and a frame CSV; no giant NPZ.
Save in the repository tests/ directory. Run from any working directory:
    python tests/diagnose_synchronized_eckf.py
Optional:
    python tests/diagnose_synchronized_eckf.py --start 57.9 --end 59.2
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
TESTS_DIR = REPO_ROOT / "tests"

from python_eckf.config import ECKFConfig
import python_eckf.tracker as tracker_module


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def fmt(value, digits=3):
    value = number(value)
    return f"{value:.{digits}f}" if value is not None else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, default=TESTS_DIR / "PREDESTINATI.wav")
    parser.add_argument("--start", type=float, default=58.0)
    parser.add_argument("--end", type=float, default=59.1)
    parser.add_argument("--output", type=Path, default=TESTS_DIR / "PREDESTINATI_synchronized_diagnostic.txt")
    args = parser.parse_args()
    if not args.start < args.end:
        parser.error("--start must be earlier than --end")

    audio, sample_rate = sf.read(args.wav, dtype="float64")
    if audio.ndim != 1:
        parser.error("Input WAV must be mono")

    config = ECKFConfig(mode="offline")
    block = config.block_size
    events = []
    real_trace_class = tracker_module.ECKFTrace

    class RecordingTrace:
        """Capture emitted events without relying on the trace CSV's schema."""

        def __init__(self, sr):
            self.sr = sr

        def emit(self, event_name, sample, **fields):
            timestamp = int(sample) / self.sr
            if args.start - 2 * block / self.sr <= timestamp <= args.end + 2 * block / self.sr:
                events.append((timestamp, str(event_name), dict(fields)))

        def close(self):
            pass

    # The tracker resolves ECKFTrace when track_pitch() runs. Restore it even
    # if the tracker raises an exception. Only the trace sink is substituted;
    # the algorithm and numerical computations are untouched.
    tracker_module.ECKFTrace = RecordingTrace
    try:
        result = tracker_module.track_pitch(audio, sample_rate, config)
    finally:
        tracker_module.ECKFTrace = real_trace_class

    frame_rows = []
    for start in range(0, len(result.f0_hz), block):
        timestamp = start / sample_rate
        if timestamp < args.start or timestamp > args.end:
            continue
        values = result.f0_hz[start:start + block]
        voiced = values[np.isfinite(values) & (values > 0)]
        row = {
            "time_s": round(timestamp, 8),
            "sample": start,
            "voiced_samples": len(voiced),
            "first_f0_hz": float(voiced[0]) if len(voiced) else "",
            "median_f0_hz": float(np.median(voiced)) if len(voiced) else "",
            "last_f0_hz": float(voiced[-1]) if len(voiced) else "",
            "min_f0_hz": float(voiced.min()) if len(voiced) else "",
            "max_f0_hz": float(voiced.max()) if len(voiced) else "",
        }
        frame_rows.append(row)

    csv_path = args.output.with_name(args.output.stem + "_frames.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]) if frame_rows else [
            "time_s", "sample", "voiced_samples", "first_f0_hz",
            "median_f0_hz", "last_f0_hz", "min_f0_hz", "max_f0_hz"
        ])
        writer.writeheader()
        writer.writerows(frame_rows)

    lines = [
        "=== SYNCHRONIZED ECKF DIAGNOSTIC ===",
        f"WAV: {args.wav}",
        f"Sample rate: {sample_rate} Hz | block: {block} samples | mode: offline",
        f"Window: {args.start:.4f}–{args.end:.4f} s",
        "Trace events and F0 belong to the SAME track_pitch() call.",
        "No pitch was changed; no MIDI was used; no previous trace was read.",
        "",
        "=== OUTPUT FRAMES ===",
        "time_s     voiced   first_Hz    median_Hz   last_Hz     min_Hz      max_Hz",
    ]
    for row in frame_rows:
        lines.append(
            f"{row['time_s']:9.4f} {row['voiced_samples']:7d} "
            f"{fmt(row['first_f0_hz']):>11} {fmt(row['median_f0_hz']):>11} "
            f"{fmt(row['last_f0_hz']):>11} {fmt(row['min_f0_hz']):>11} "
            f"{fmt(row['max_f0_hz']):>11}"
        )

    lines.extend(["", "=== ACTUAL EVENTS IN EMISSION ORDER ==="])
    interesting = ("PERIODICITY", "INITIALIZATION", "RESET", "LOOKAHEAD", "SILENCE", "FRAME_ANALYSIS")
    relevant = [(t, name, fields) for t, name, fields in events if any(word in name.upper() for word in interesting)]
    if not relevant:
        lines.append("No matching diagnostic events emitted in the requested interval.")
    keys = (
        "frame_start", "init_f0_hz", "init_amplitude", "peak_1_hz", "peak_2_hz",
        "peak_3_hz", "acf_peak", "cmndf_minimum", "acf_frequency_hz",
        "cmndf_frequency_hz", "voiced", "reason", "reset_accepted", "count",
        "flag", "harm_prev", "harm_cur",
    )
    for t, name, fields in relevant:
        details = " ".join(f"{key}={fields[key]}" for key in keys if fields.get(key) is not None)
        lines.append(f"{t:9.4f}s {name:26s} {details}")

    lines.extend([
        "", "=== READING THE RESULT ===",
        "If INITIALIZATION_PROPOSED is already ~2x observed F0, the proposed octave is wrong.",
        "If proposed F0 is ~correct but first_f0_hz is ~2x, investigate state initialization/filtering.",
        "Events are listed in emission order, not sorted time: a reset may rewind the cursor.",
        "This report alone does not establish performed F0; compare with WAV periodicity evidence.",
    ])
    text = "\n".join(lines) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved report: {args.output}")
    print(f"Saved frame CSV: {csv_path}")


if __name__ == "__main__":
    main()
