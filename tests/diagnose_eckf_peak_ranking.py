#!/usr/bin/env python3
"""
Observational ECKF spectral peak-ranking diagnostic.

Replays initialization frames recorded in an ECKF spectral trace.
Uses the production detector's spectrum and peak-selection methods.

Does not modify production code or change tracking behavior.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

from python_eckf.config import ECKFConfig
from python_eckf.harmonic_change import HarmonicChangeDetector


def read_mono_wav(path: Path):
    fs, audio = wavfile.read(path)

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono WAV, got shape {audio.shape}"
        )

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)

        if info.min == 0:
            # Unsigned PCM, such as uint8.
            midpoint = (info.max + 1) / 2
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
        raise ValueError("WAV contains NaN or Inf")

    return float(fs), audio


def load_initializations(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "event",
        "sample",
        "time_s",
        "init_f0_hz",
    }

    if not rows:
        raise ValueError("Trace contains no rows")

    missing = required - set(rows[0])

    if missing:
        raise ValueError(
            f"Missing trace columns: {sorted(missing)}"
        )

    return [
        row
        for row in rows
        if row["event"] == "INITIALIZATION_PROPOSED"
    ]


def analyze_frame(
    detector,
    audio,
    fs,
    start,
    block,
    top_n,
):
    frame = audio[start:start + block]

    if len(frame) != block:
        raise ValueError(
            f"Incomplete frame at sample {start}"
        )

    # The production spectrum implementation.
    _, mag, _ = detector._frame_spectrum(frame, fs)

    # All local maxima, using the same SciPy operation
    # as the production _select_peaks method.
    peak_positions, _ = find_peaks(mag)

    if len(peak_positions) < detector.config.npeaks:
        raise RuntimeError(
            f"Only {len(peak_positions)} peaks detected"
        )

    peak_magnitudes = mag[peak_positions]

    # Same stable descending-magnitude ranking used
    # by the production peak selector.
    ranked_indices = np.argsort(
        -peak_magnitudes,
        kind="stable",
    )

    # Obtain the actual production selection independently.
    selected_positions, selected_values, selected_freqs = (
        detector._select_peaks(mag)
    )

    selected_set = set(
        int(position)
        for position in selected_positions
    )

    results = []

    for rank, index in enumerate(
        ranked_indices[:top_n],
        start=1,
    ):
        position = int(peak_positions[index])

        results.append({
            "rank": rank,
            "frequency_hz": float(
                detector._fbins[position]
            ),
            "magnitude": float(mag[position]),
            "fft_bin": position,
            "selected": position in selected_set,
        })

    return {
        "ranked": results,
        "selected_frequencies": selected_freqs,
        "selected_magnitudes": selected_values,
        "total_peaks": len(peak_positions),
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
            "tests/PREDESTINATI_peak_ranking.csv"
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    if args.top < 3:
        parser.error("--top must be at least 3")

    fs, audio = read_mono_wav(args.wav)
    initializations = load_initializations(args.trace)

    config = ECKFConfig(mode="offline")
    config.validate()

    detector = HarmonicChangeDetector(config)
    block = config.block_size

    # Match the tracker's zero-padded audio representation.
    padded_length = (
        int(np.ceil(len(audio) / block)) * block
    )

    audio = np.pad(
        audio,
        (0, padded_length - len(audio)),
        mode="constant",
    )

    output_rows = []

    print()
    print("=" * 76)
    print("ECKF SPECTRAL PEAK-RANKING DIAGNOSTIC")
    print("=" * 76)

    print(f"WAV:         {args.wav}")
    print(f"Sample rate: {fs:g} Hz")
    print(f"Block size:  {block}")
    print(f"Trace:       {args.trace}")
    print(f"Frames:      {len(initializations)}")

    for event_index, event in enumerate(
        initializations,
        start=1,
    ):
        start = int(event["sample"])
        time_s = start / fs

        result = analyze_frame(
            detector,
            audio,
            fs,
            start,
            block,
            args.top,
        )

        frequencies = result["selected_frequencies"]
        spacings = np.diff(frequencies)

        print()
        print("-" * 76)
        print(
            f"INITIALIZATION {event_index} "
            f"| sample={start} "
            f"| time={time_s:.6f}s"
        )
        print("-" * 76)

        print(
            f"Recorded F0: {event['init_f0_hz']} Hz"
        )

        print(
            "Selected peaks: "
            + ", ".join(
                f"{value:.3f}"
                for value in frequencies
            )
            + " Hz"
        )

        print(
            "Spacings:       "
            + ", ".join(
                f"{value:.3f}"
                for value in spacings
            )
            + " Hz"
        )

        print(
            f"Total local spectral peaks: "
            f"{result['total_peaks']}"
        )

        print()
        print(
            f"{'Rank':>4} "
            f"{'Frequency Hz':>14} "
            f"{'Magnitude':>15} "
            f"{'Selected':>10}"
        )

        print("-" * 49)

        for peak in result["ranked"]:
            selected = (
                "*** YES ***"
                if peak["selected"]
                else ""
            )

            print(
                f"{peak['rank']:>4} "
                f"{peak['frequency_hz']:>14.3f} "
                f"{peak['magnitude']:>15.8g} "
                f"{selected:>10}"
            )

            output_rows.append({
                "initialization_index": event_index,
                "time_s": f"{time_s:.9f}",
                "sample": start,
                "recorded_f0_hz": event["init_f0_hz"],
                "rank": peak["rank"],
                "frequency_hz": (
                    f"{peak['frequency_hz']:.9f}"
                ),
                "magnitude": (
                    f"{peak['magnitude']:.12g}"
                ),
                "fft_bin": peak["fft_bin"],
                "selected": int(peak["selected"]),
            })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "initialization_index",
        "time_s",
        "sample",
        "recorded_f0_hz",
        "rank",
        "frequency_hz",
        "magnitude",
        "fft_bin",
        "selected",
    ]

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("=" * 76)
    print(f"Saved: {args.output}")
    print("=" * 76)


if __name__ == "__main__":
    main()
