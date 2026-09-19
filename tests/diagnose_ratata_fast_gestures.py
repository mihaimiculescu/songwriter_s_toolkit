#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_eckf.validity import (
    PitchValidityConfig,
    analyse_pitch_validity,
)


NOTE_NAMES = [
    "C", "C#", "D", "D#",
    "E", "F", "F#", "G",
    "G#", "A", "A#", "B",
]


def hz_to_midi(f0):
    f0 = np.asarray(f0, dtype=np.float64)

    out = np.full_like(f0, np.nan)

    good = np.isfinite(f0) & (f0 > 0.0)

    out[good] = (
        69.0
        + 12.0 * np.log2(f0[good] / 440.0)
    )

    return out


def midi_name(value):
    if not np.isfinite(value):
        return "---"

    n = int(round(value))
    octave = n // 12 - 1

    return f"{NOTE_NAMES[n % 12]}{octave}"


def true_runs(mask):
    """
    Return contiguous True runs as [start, end),
    end exclusive.
    """

    mask = np.asarray(mask, dtype=bool)

    runs = []
    i = 0

    while i < len(mask):

        if not mask[i]:
            i += 1
            continue

        start = i

        while i < len(mask) and mask[i]:
            i += 1

        runs.append((start, i))

    return runs


@dataclass
class Gesture:
    kind: str
    run_start: int
    run_end: int

    center: int
    window_start: int
    window_end: int

    span: float
    endpoint_change: float
    max_step: float


def best_window_in_run(
    midi,
    run_start,
    run_end,
    radius_steps,
):
    """
    Find the strongest local pitch movement wholly contained
    inside one contiguous run.

    No samples outside the run participate.
    """

    best = None

    for center in range(run_start, run_end):

        start = max(
            run_start,
            center - radius_steps,
        )

        end = min(
            run_end,
            center + radius_steps + 1,
        )

        values = midi[start:end]

        values = values[
            np.isfinite(values)
        ]

        if len(values) < 2:
            continue

        span = float(
            np.max(values) - np.min(values)
        )

        endpoint = float(
            values[-1] - values[0]
        )

        diffs = np.abs(
            np.diff(values)
        )

        max_step = (
            float(np.max(diffs))
            if len(diffs)
            else 0.0
        )

        candidate = Gesture(
            kind="",
            run_start=run_start,
            run_end=run_end,
            center=center,
            window_start=start,
            window_end=end,
            span=span,
            endpoint_change=endpoint,
            max_step=max_step,
        )

        if (
            best is None
            or candidate.span > best.span
        ):
            best = candidate

    return best


def classify_shape(delta):
    if delta > 0.25:
        return "UP"

    if delta < -0.25:
        return "DOWN"

    return "RETURN/OSCILLATORY"


def print_context(
    t,
    f0,
    midi,
    valid,
    reason,
    center,
    radius,
):
    start = max(
        0,
        center - radius,
    )

    end = min(
        len(t),
        center + radius + 1,
    )

    print(
        "    time_s      f0_hz    midi     note   "
        "valid   reason"
    )

    print(
        "    --------   --------   -------   ----   "
        "-----   ----------------------------"
    )

    for i in range(start, end):

        marker = ">>" if i == center else "  "

        print(
            f"{marker}  "
            f"{t[i]:8.3f}   "
            f"{f0[i]:8.2f}   "
            f"{midi[i]:7.2f}   "
            f"{midi_name(midi[i]):4s}   "
            f"{str(bool(valid[i])):5s}   "
            f"{reason[i]}"
        )


def rank_gestures(
    gestures,
    top_n,
    minimum_span,
):
    gestures = [
        g
        for g in gestures
        if g is not None
        and g.span >= minimum_span
    ]

    gestures.sort(
        key=lambda g: g.span,
        reverse=True,
    )

    return gestures[:top_n]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--window-ms",
        type=float,
        default=80.0,
        help="half-window for gesture measurement",
    )

    parser.add_argument(
        "--context-ms",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--min-span",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    data = np.genfromtxt(
        args.csv,
        delimiter=",",
        names=True,
    )

    t = np.asarray(
        data["time_s"],
        dtype=np.float64,
    )

    f0 = np.asarray(
        data["f0_hz"],
        dtype=np.float64,
    )

    dt = float(
        np.median(
            np.diff(t)
        )
    )

    analysis_hz = 1.0 / dt

    midi = hz_to_midi(f0)

    config = PitchValidityConfig(
        analysis_hz=analysis_hz,
        min_vocal_hz=120.0,
    )

    result = analyse_pitch_validity(
        f0_hz=f0,
        sample_rate=analysis_hz,
        config=config,
    )

    valid = result.valid
    reason = result.reason

    radius_steps = max(
        1,
        int(
            round(
                args.window_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    context_steps = max(
        1,
        int(
            round(
                args.context_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    # ============================================================
    # ACCEPTED RUNS
    #
    # Only contiguous valid rows participate.
    # ============================================================

    accepted_runs = true_runs(
        valid
        & np.isfinite(midi)
    )

    accepted_gestures = []

    for start, end in accepted_runs:

        gesture = best_window_in_run(
            midi=midi,
            run_start=start,
            run_end=end,
            radius_steps=radius_steps,
        )

        if gesture is not None:
            gesture.kind = "ACCEPTED"
            accepted_gestures.append(
                gesture
            )

    accepted_gestures = rank_gestures(
        accepted_gestures,
        args.top,
        args.min_span,
    )

    # ============================================================
    # REJECTED ABOVE-FLOOR RUNS
    #
    # Only contiguous rejected-but-numerically-plausible rows.
    #
    # Below-floor garbage and above-ceiling garbage are excluded.
    # ============================================================

    rejected_candidate = (
        (~valid)
        & np.isfinite(f0)
        & (f0 >= config.min_vocal_hz)
        & (f0 <= config.max_vocal_hz)
    )

    rejected_runs = true_runs(
        rejected_candidate
    )

    rejected_gestures = []

    for start, end in rejected_runs:

        gesture = best_window_in_run(
            midi=midi,
            run_start=start,
            run_end=end,
            radius_steps=radius_steps,
        )

        if gesture is not None:
            gesture.kind = "REJECTED"
            rejected_gestures.append(
                gesture
            )

    rejected_gestures = rank_gestures(
        rejected_gestures,
        args.top,
        args.min_span,
    )

    print()
    print("=" * 108)
    print("RATATA FAST-GESTURE / VALIDITY STRESS TEST — RUN-BASED")
    print("=" * 108)

    print()
    print(f"Rows:                {len(t)}")
    print(f"Analysis rate:       {analysis_hz:.3f} Hz")
    print(f"Gesture half-window: {args.window_ms:.1f} ms")
    print(f"Accepted runs:       {len(accepted_runs)}")
    print(f"Rejected runs:       {len(rejected_runs)}")
    print(
        f"Valid rows:          "
        f"{np.count_nonzero(valid)} / {len(valid)} "
        f"({100.0 * np.mean(valid):.1f}%)"
    )

    def report(title, gestures):

        print()
        print("=" * 108)
        print(title)
        print("=" * 108)

        if not gestures:
            print("\nNo qualifying gestures.")
            return

        for rank, g in enumerate(
            gestures,
            start=1,
        ):

            duration_ms = (
                (g.run_end - g.run_start)
                / analysis_hz
                * 1000.0
            )

            center = g.center

            print()
            print("-" * 108)

            print(
                f"{g.kind} #{rank:02d}"
            )

            print(
                f"Center time:       {t[center]:.3f} s"
            )

            print(
                f"Center pitch:      "
                f"{midi[center]:.2f} "
                f"({midi_name(midi[center])})"
            )

            print(
                f"Run:               "
                f"{t[g.run_start]:.3f} .. "
                f"{t[g.run_end - 1]:.3f} s"
            )

            print(
                f"Run duration:      "
                f"{duration_ms:.1f} ms"
            )

            print(
                f"Local valid span:  "
                f"{g.span:.2f} semitones"
            )

            print(
                f"Endpoint change:   "
                f"{g.endpoint_change:+.2f} semitones"
            )

            print(
                f"Max adjacent step: "
                f"{g.max_step:.2f} semitones / "
                f"{1000.0 / analysis_hz:.1f} ms"
            )

            print(
                f"Shape hint:        "
                f"{classify_shape(g.endpoint_change)}"
            )

            print(
                f"Center reason:     "
                f"{reason[center]}"
            )

            print()

            print_context(
                t=t,
                f0=f0,
                midi=midi,
                valid=valid,
                reason=reason,
                center=center,
                radius=context_steps,
            )

    report(
        "A. LARGEST MOVEMENTS WHOLLY INSIDE ACCEPTED RUNS",
        accepted_gestures,
    )

    report(
        "B. LARGEST MOVEMENTS WHOLLY INSIDE REJECTED ABOVE-FLOOR RUNS",
        rejected_gestures,
    )

    print()
    print("=" * 108)


if __name__ == "__main__":
    main()