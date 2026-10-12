#!/usr/bin/env python3
"""
Read-only ECKF harmonic-consistency diagnostic.

Uses the production FFT and spectral peak-selection implementation.
Replays INITIALIZATION_PROPOSED frames from an existing ECKF trace.

For each frame:
  - ranks spectral peaks by magnitude;
  - generates F0 hypotheses from peaks / harmonic numbers;
  - evaluates integer-harmonic alignment;
  - reports candidate support and matched peaks;
  - includes the production F0 as an explicit reference.

Does NOT:
  - change the tracker or initializer;
  - select a replacement F0;
  - apply temporal smoothing;
  - use ground-truth MIDI;
  - assume the strongest candidate is the correct vocal pitch.

The candidate score is diagnostic, not a calibrated probability.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

from python_eckf.config import ECKFConfig
from python_eckf.harmonic_change import HarmonicChangeDetector


def read_wav(path):
    fs, audio = wavfile.read(path)

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono WAV; received shape {audio.shape}"
        )

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)

        if info.min == 0:
            midpoint = (info.max + 1) / 2.0
            audio = (
                audio.astype(np.float64) - midpoint
            ) / midpoint
        else:
            audio = (
                audio.astype(np.float64)
                / max(abs(info.min), info.max)
            )
    else:
        audio = audio.astype(np.float64)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or Inf")

    return float(fs), audio


def load_events(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "event",
            "sample",
            "time_s",
            "init_f0_hz",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing trace columns: {sorted(missing)}"
            )

        return [
            row
            for row in reader
            if row["event"] == "INITIALIZATION_PROPOSED"
        ]


def spectral_peaks(detector, frame, fs, top_n):
    _, magnitude, _ = detector._frame_spectrum(
        frame,
        fs,
    )

    positions, _ = find_peaks(magnitude)

    if len(positions) < detector.config.npeaks:
        raise RuntimeError(
            "Insufficient spectral peaks"
        )

    values = magnitude[positions]

    ordering = np.argsort(
        -values,
        kind="stable",
    )

    selected = ordering[:top_n]

    ranked_positions = positions[selected]

    frequencies = detector._fbins[ranked_positions]
    magnitudes = magnitude[ranked_positions]

    production_positions, _, _ = (
        detector._select_peaks(magnitude)
    )

    production_set = set(
        int(value)
        for value in production_positions
    )

    peaks = []

    for rank, (position, frequency, mag) in enumerate(
        zip(
            ranked_positions,
            frequencies,
            magnitudes,
        ),
        start=1,
    ):
        peaks.append({
            "rank": rank,
            "position": int(position),
            "frequency": float(frequency),
            "magnitude": float(mag),
            "production_selected": (
                int(position) in production_set
            ),
        })

    return peaks


def generate_candidates(
    peaks,
    minimum_f0,
    maximum_f0,
    maximum_harmonic,
    production_f0,
):
    """
    Generate hypotheses f_peak / harmonic_number.

    Quantization to 0.5 Hz prevents a huge number of nearly
    identical hypotheses. This is diagnostic discretization,
    not production pitch quantization.
    """

    candidates = set()

    for peak in peaks:
        frequency = peak["frequency"]

        for harmonic in range(
            1,
            maximum_harmonic + 1,
        ):
            candidate = frequency / harmonic

            if minimum_f0 <= candidate <= maximum_f0:
                candidates.add(
                    round(candidate * 2.0) / 2.0
                )

    if (
        math.isfinite(production_f0)
        and production_f0 > 0
    ):
        candidates.add(float(production_f0))

    return sorted(candidates)


def evaluate_candidate(
    f0,
    peaks,
    tolerance_cents,
    maximum_harmonic,
):
    """
    Match each peak to its nearest integer harmonic.

    A peak is accepted if its deviation from that harmonic
    is within tolerance_cents.

    Metrics:
      matched_count:
          Number of matched spectral peaks.

      distinct_harmonics:
          Number of distinct harmonic indices represented.

      explained_magnitude_fraction:
          Fraction of magnitude among the examined peaks
          that matches the candidate harmonic series.

      weighted_alignment:
          Magnitude-weighted alignment, penalizing
          frequency deviation within the tolerance.

      low_harmonic_support:
          Number of distinct matched harmonics among 1..5.

    These are independent diagnostics, not a winner score.
    """

    total_magnitude = sum(
        peak["magnitude"]
        for peak in peaks
    )

    matches = []

    matched_magnitude = 0.0
    weighted_alignment = 0.0

    for peak in peaks:
        frequency = peak["frequency"]

        harmonic = int(
            math.floor(
                frequency / f0 + 0.5
            )
        )

        if (
            harmonic < 1
            or harmonic > maximum_harmonic
        ):
            continue

        expected_frequency = harmonic * f0

        cents = (
            1200.0
            * math.log2(
                frequency / expected_frequency
            )
        )

        if abs(cents) > tolerance_cents:
            continue

        magnitude = peak["magnitude"]

        matched_magnitude += magnitude

        alignment = max(
            0.0,
            1.0 - abs(cents) / tolerance_cents,
        )

        weighted_alignment += (
            magnitude * alignment
        )

        matches.append({
            **peak,
            "harmonic": harmonic,
            "expected_frequency": expected_frequency,
            "error_hz": (
                frequency - expected_frequency
            ),
            "error_cents": cents,
        })

    harmonic_indices = {
        match["harmonic"]
        for match in matches
    }

    if total_magnitude > 0:
        explained_fraction = (
            matched_magnitude / total_magnitude
        )

        weighted_fraction = (
            weighted_alignment / total_magnitude
        )
    else:
        explained_fraction = 0.0
        weighted_fraction = 0.0

    return {
        "f0": f0,
        "matched_count": len(matches),
        "distinct_harmonics": len(harmonic_indices),
        "low_harmonic_support": len(
            harmonic_indices.intersection(
                {1, 2, 3, 4, 5}
            )
        ),
        "explained_fraction": explained_fraction,
        "weighted_alignment": weighted_fraction,
        "matches": matches,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav",
        type=Path,
        default=Path("tests/PREDESTINATI.wav"),
    )

    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "tests/PREDESTINATI_spectral_trace.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tests/PREDESTINATI_harmonic_consistency.csv"
        ),
    )

    parser.add_argument(
        "--matches-output",
        type=Path,
        default=Path(
            "tests/PREDESTINATI_harmonic_matches.csv"
        ),
    )

    parser.add_argument(
        "--top-peaks",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--min-f0",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--max-f0",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--max-harmonic",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--tolerance-cents",
        type=float,
        default=35.0,
    )

    parser.add_argument(
        "--report-candidates",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    if args.top_peaks < 3:
        parser.error("--top-peaks must be >= 3")

    if args.min_f0 <= 0:
        parser.error("--min-f0 must be positive")

    if args.max_f0 <= args.min_f0:
        parser.error("--max-f0 must exceed --min-f0")

    if args.max_harmonic < 1:
        parser.error("--max-harmonic must be >= 1")

    if args.tolerance_cents <= 0:
        parser.error(
            "--tolerance-cents must be positive"
        )

    if args.report_candidates < 1:
        parser.error(
            "--report-candidates must be >= 1"
        )

    fs, audio = read_wav(args.wav)

    config = ECKFConfig(mode="offline")
    config.validate()

    detector = HarmonicChangeDetector(config)

    block = config.block_size

    # Mirror the previous peak-ranking diagnostic.
    padded_length = (
        int(np.ceil(len(audio) / block)) * block
    )

    audio = np.pad(
        audio,
        (0, padded_length - len(audio)),
    )

    events = load_events(args.trace)

    if not events:
        raise RuntimeError(
            "No INITIALIZATION_PROPOSED events found"
        )

    summary_rows = []
    match_rows = []

    print()
    print("=" * 84)
    print("ECKF HARMONIC-CONSISTENCY DIAGNOSTIC")
    print("=" * 84)

    print(f"WAV:                 {args.wav}")
    print(f"Sample rate:         {fs:g} Hz")
    print(f"Block size:          {block}")
    print(f"Initialization rows: {len(events)}")
    print(f"Top spectral peaks:  {args.top_peaks}")
    print(
        f"Candidate F0 range:  "
        f"{args.min_f0:g}–{args.max_f0:g} Hz"
    )
    print(
        f"Harmonic tolerance:  "
        f"±{args.tolerance_cents:g} cents"
    )
    print(
        f"Maximum harmonic:    "
        f"{args.max_harmonic}"
    )

    for event_index, event in enumerate(
        events,
        start=1,
    ):
        start = int(event["sample"])

        frame = audio[start:start + block]

        if len(frame) != block:
            raise RuntimeError(
                f"Incomplete frame at sample {start}"
            )

        production_f0 = float(
            event["init_f0_hz"]
        )

        peaks = spectral_peaks(
            detector,
            frame,
            fs,
            args.top_peaks,
        )

        candidates = generate_candidates(
            peaks,
            args.min_f0,
            args.max_f0,
            args.max_harmonic,
            production_f0,
        )

        evaluations = [
            evaluate_candidate(
                candidate,
                peaks,
                args.tolerance_cents,
                args.max_harmonic,
            )
            for candidate in candidates
        ]

        # Presentation order only. No candidate is fed
        # into the tracker or designated as correct.
        evaluations.sort(
            key=lambda item: (
                -item["weighted_alignment"],
                -item["distinct_harmonics"],
                -item["explained_fraction"],
                item["f0"],
            )
        )

        print()
        print("-" * 84)
        print(
            f"INITIALIZATION {event_index} "
            f"| time={start / fs:.6f}s "
            f"| production F0={production_f0:.3f} Hz"
        )
        print("-" * 84)

        print(
            f"Evaluated {len(evaluations)} "
            "candidate hypotheses."
        )

        print()
        print(
            f"{'F0 Hz':>9} "
            f"{'Peaks':>6} "
            f"{'Harmonics':>9} "
            f"{'H1-H5':>6} "
            f"{'Mag %':>8} "
            f"{'Align %':>9} "
            f"{'Original':>9}"
        )

        print("-" * 75)

        production_result = None

        for result in evaluations:
            if result["f0"] == production_f0:
                production_result = result

        display = evaluations[
            :args.report_candidates
        ]

        if (
            production_result is not None
            and production_result not in display
        ):
            display = [
                *display,
                production_result,
            ]

        for result in display:
            original = (
                "YES"
                if result["f0"] == production_f0
                else ""
            )

            print(
                f"{result['f0']:>9.2f} "
                f"{result['matched_count']:>6} "
                f"{result['distinct_harmonics']:>9} "
                f"{result['low_harmonic_support']:>6} "
                f"{100 * result['explained_fraction']:>7.2f}% "
                f"{100 * result['weighted_alignment']:>8.2f}% "
                f"{original:>9}"
            )

        print()
        print("Production F0 is marked YES.")
        print(
            "Candidate ordering is diagnostic only; "
            "it is not a pitch decision."
        )

        for result in evaluations:
            summary_rows.append({
                "initialization_index": event_index,
                "time_s": f"{start / fs:.9f}",
                "sample": start,
                "production_f0_hz": production_f0,
                "candidate_f0_hz": result["f0"],
                "is_production_f0": int(
                    result["f0"] == production_f0
                ),
                "matched_peak_count": (
                    result["matched_count"]
                ),
                "distinct_harmonics": (
                    result["distinct_harmonics"]
                ),
                "low_harmonic_support": (
                    result["low_harmonic_support"]
                ),
                "explained_magnitude_fraction": (
                    result["explained_fraction"]
                ),
                "weighted_alignment": (
                    result["weighted_alignment"]
                ),
            })

            for match in result["matches"]:
                match_rows.append({
                    "initialization_index": event_index,
                    "time_s": f"{start / fs:.9f}",
                    "candidate_f0_hz": result["f0"],
                    "peak_rank": match["rank"],
                    "peak_frequency_hz": (
                        match["frequency"]
                    ),
                    "peak_magnitude": (
                        match["magnitude"]
                    ),
                    "harmonic_number": (
                        match["harmonic"]
                    ),
                    "expected_frequency_hz": (
                        match["expected_frequency"]
                    ),
                    "error_hz": match["error_hz"],
                    "error_cents": (
                        match["error_cents"]
                    ),
                    "production_selected_peak": int(
                        match["production_selected"]
                    ),
                })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.matches_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_fields = [
        "initialization_index",
        "time_s",
        "sample",
        "production_f0_hz",
        "candidate_f0_hz",
        "is_production_f0",
        "matched_peak_count",
        "distinct_harmonics",
        "low_harmonic_support",
        "explained_magnitude_fraction",
        "weighted_alignment",
    ]

    match_fields = [
        "initialization_index",
        "time_s",
        "candidate_f0_hz",
        "peak_rank",
        "peak_frequency_hz",
        "peak_magnitude",
        "harmonic_number",
        "expected_frequency_hz",
        "error_hz",
        "error_cents",
        "production_selected_peak",
    ]

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_fields,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with args.matches_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=match_fields,
        )
        writer.writeheader()
        writer.writerows(match_rows)

    print()
    print("=" * 84)
    print(f"Candidate report: {args.output}")
    print(f"Individual matches: {args.matches_output}")
    print("=" * 84)


if __name__ == "__main__":
    main()
