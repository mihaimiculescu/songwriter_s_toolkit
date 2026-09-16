#!/usr/bin/env python3
"""
Observational waveform-periodicity diagnostic.

Independent of ECKF pitch estimates and spectral initialization.

For each analysis window:
    - calculate normalized autocorrelation;
    - report local autocorrelation maxima;
    - calculate normalized difference-function minima (YIN-style);
    - inspect periodicity in consecutive subframes;
    - export full correlation curves.

No voiced/unvoiced decisions.
No pitch replacement.
No production-code modifications.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks


def read_mono_wav(path):
    fs, audio = wavfile.read(path)

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono WAV, got {audio.shape}"
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


def normalized_autocorrelation(x, max_lag):
    """
    Lag-dependent energy normalization:

        r(tau) =
          sum x[n] x[n+tau]
          -----------------------
          sqrt(sum x[n]^2 sum x[n+tau]^2)

    The overlap changes with lag, but both energies
    are calculated over exactly that overlap.
    """

    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    result = np.full(
        max_lag + 1,
        np.nan,
        dtype=np.float64,
    )

    for lag in range(max_lag + 1):
        if lag == 0:
            a = x
            b = x
        else:
            a = x[:-lag]
            b = x[lag:]

        denominator = np.sqrt(
            np.dot(a, a) * np.dot(b, b)
        )

        if denominator > 0:
            result[lag] = (
                np.dot(a, b) / denominator
            )

    return result


def normalized_difference(x, max_lag):
    """
    YIN-style cumulative mean normalized
    difference function (CMNDF).

    This is calculated independently from
    normalized autocorrelation.

    No threshold is applied.
    """

    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    difference = np.zeros(
        max_lag + 1,
        dtype=np.float64,
    )

    for lag in range(1, max_lag + 1):
        delta = x[:-lag] - x[lag:]

        difference[lag] = np.dot(
            delta,
            delta,
        )

    cmndf = np.ones_like(difference)

    cumulative = np.cumsum(
        difference[1:]
    )

    lags = np.arange(
        1,
        max_lag + 1,
    )

    valid = cumulative > 0

    cmndf[1:][valid] = (
        difference[1:][valid]
        * lags[valid]
        / cumulative[valid]
    )

    return cmndf


def analyze_window(
    x,
    fs,
    minimum_f0,
    maximum_f0,
    top_n,
):
    min_lag = max(
        1,
        int(np.floor(fs / maximum_f0)),
    )

    max_lag = int(
        np.ceil(fs / minimum_f0)
    )

    if len(x) <= max_lag:
        raise ValueError(
            "Window too short for requested F0 range"
        )

    acf = normalized_autocorrelation(
        x,
        max_lag,
    )

    cmndf = normalized_difference(
        x,
        max_lag,
    )

    search_acf = acf[
        min_lag:max_lag + 1
    ]

    search_cmndf = cmndf[
        min_lag:max_lag + 1
    ]

    # Local maxima of normalized autocorrelation.
    acf_positions, _ = find_peaks(
        search_acf
    )

    acf_positions = (
        acf_positions + min_lag
    )

    # Include endpoints when they are valid maxima.
    for endpoint in (min_lag, max_lag):
        if np.isfinite(acf[endpoint]):
            acf_positions = np.append(
                acf_positions,
                endpoint,
            )

    acf_positions = np.unique(
        acf_positions
    )

    acf_positions = sorted(
        acf_positions,
        key=lambda lag: (
            -acf[lag],
            lag,
        ),
    )

    # Local minima of CMNDF.
    yin_positions, _ = find_peaks(
        -search_cmndf
    )

    yin_positions = (
        yin_positions + min_lag
    )

    for endpoint in (min_lag, max_lag):
        yin_positions = np.append(
            yin_positions,
            endpoint,
        )

    yin_positions = np.unique(
        yin_positions
    )

    yin_positions = sorted(
        yin_positions,
        key=lambda lag: (
            cmndf[lag],
            lag,
        ),
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.asarray(x) ** 2
            )
        )
    )

    return {
        "rms": rms,
        "acf": acf,
        "cmndf": cmndf,
        "min_lag": min_lag,
        "max_lag": max_lag,
        "acf_candidates": [
            {
                "lag": int(lag),
                "frequency": fs / lag,
                "value": float(acf[lag]),
            }
            for lag in acf_positions[:top_n]
        ],
        "yin_candidates": [
            {
                "lag": int(lag),
                "frequency": fs / lag,
                "value": float(cmndf[lag]),
            }
            for lag in yin_positions[:top_n]
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav",
        type=Path,
        default=Path("tests/PREDESTINATI.wav"),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=58.15,
    )

    parser.add_argument(
        "--end",
        type=float,
        default=59.05,
    )

    parser.add_argument(
        "--window-ms",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--hop-ms",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--min-f0",
        type=float,
        default=70.0,
    )

    parser.add_argument(
        "--max-f0",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "tests/PREDESTINATI_periodicity_summary.csv"
        ),
    )

    parser.add_argument(
        "--curves",
        type=Path,
        default=Path(
            "tests/PREDESTINATI_periodicity_curves.csv"
        ),
    )

    args = parser.parse_args()

    if args.start < 0 or args.end <= args.start:
        parser.error("Invalid time interval")

    if args.window_ms <= 0 or args.hop_ms <= 0:
        parser.error(
            "Window and hop must be positive"
        )

    if args.min_f0 <= 0:
        parser.error("--min-f0 must be positive")

    if args.max_f0 <= args.min_f0:
        parser.error(
            "--max-f0 must exceed --min-f0"
        )

    if args.top < 1:
        parser.error("--top must be >= 1")

    fs, audio = read_mono_wav(args.wav)

    window_samples = int(
        round(
            fs * args.window_ms / 1000.0
        )
    )

    hop_samples = int(
        round(
            fs * args.hop_ms / 1000.0
        )
    )

    first_sample = int(
        round(args.start * fs)
    )

    last_sample = min(
        len(audio),
        int(round(args.end * fs)),
    )

    summary_rows = []
    curve_rows = []

    print()
    print("=" * 92)
    print("ECKF RAW-WAVEFORM PERIODICITY DIAGNOSTIC")
    print("=" * 92)

    print(f"WAV:          {args.wav}")
    print(f"Sample rate:  {fs:g} Hz")
    print(
        f"Interval:     "
        f"{args.start:.3f}–{args.end:.3f} s"
    )
    print(
        f"Window / hop: "
        f"{args.window_ms:g} / "
        f"{args.hop_ms:g} ms"
    )
    print(
        f"Lag F0 range: "
        f"{args.min_f0:g}–"
        f"{args.max_f0:g} Hz"
    )

    print()
    print(
        f"{'Center s':>10} "
        f"{'RMS':>12} "
        f"{'ACF Hz':>11} "
        f"{'ACF':>9} "
        f"{'CMNDF Hz':>11} "
        f"{'CMNDF':>9}"
    )

    print("-" * 92)

    for start in range(
        first_sample,
        last_sample - window_samples + 1,
        hop_samples,
    ):
        frame = audio[
            start:start + window_samples
        ]

        result = analyze_window(
            frame,
            fs,
            args.min_f0,
            args.max_f0,
            args.top,
        )

        center = (
            start + window_samples / 2
        ) / fs

        acf_candidates = result[
            "acf_candidates"
        ]

        yin_candidates = result[
            "yin_candidates"
        ]

        acf_best = (
            acf_candidates[0]
            if acf_candidates
            else None
        )

        yin_best = (
            yin_candidates[0]
            if yin_candidates
            else None
        )

        print(
            f"{center:>10.4f} "
            f"{result['rms']:>12.6g} "
            f"{acf_best['frequency'] if acf_best else float('nan'):>11.2f} "
            f"{acf_best['value'] if acf_best else float('nan'):>9.4f} "
            f"{yin_best['frequency'] if yin_best else float('nan'):>11.2f} "
            f"{yin_best['value'] if yin_best else float('nan'):>9.4f}"
        )

        summary_rows.append({
            "center_s": center,
            "start_s": start / fs,
            "end_s": (
                start + window_samples
            ) / fs,
            "rms": result["rms"],
            "acf_frequency_hz": (
                acf_best["frequency"]
                if acf_best
                else ""
            ),
            "acf_peak": (
                acf_best["value"]
                if acf_best
                else ""
            ),
            "cmndf_frequency_hz": (
                yin_best["frequency"]
                if yin_best
                else ""
            ),
            "cmndf_minimum": (
                yin_best["value"]
                if yin_best
                else ""
            ),
            "acf_top_candidates": "; ".join(
                f"{item['frequency']:.2f}:"
                f"{item['value']:.4f}"
                for item in acf_candidates
            ),
            "cmndf_top_candidates": "; ".join(
                f"{item['frequency']:.2f}:"
                f"{item['value']:.4f}"
                for item in yin_candidates
            ),
        })

        acf = result["acf"]
        cmndf = result["cmndf"]

        for lag in range(
            result["min_lag"],
            result["max_lag"] + 1,
        ):
            curve_rows.append({
                "center_s": center,
                "lag_samples": lag,
                "lag_ms": (
                    1000.0 * lag / fs
                ),
                "frequency_hz": fs / lag,
                "acf": acf[lag],
                "cmndf": cmndf[lag],
            })

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.curves.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.summary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "center_s",
                "start_s",
                "end_s",
                "rms",
                "acf_frequency_hz",
                "acf_peak",
                "cmndf_frequency_hz",
                "cmndf_minimum",
                "acf_top_candidates",
                "cmndf_top_candidates",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    with args.curves.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "center_s",
                "lag_samples",
                "lag_ms",
                "frequency_hz",
                "acf",
                "cmndf",
            ],
        )

        writer.writeheader()
        writer.writerows(curve_rows)

    print()
    print("=" * 92)
    print(f"Summary: {args.summary}")
    print(f"Curves:  {args.curves}")
    print("=" * 92)


if __name__ == "__main__":
    main()
