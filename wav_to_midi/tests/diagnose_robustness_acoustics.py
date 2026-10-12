#!/usr/bin/env python3
"""Read-only acoustic audit of the three v3 robustness conflicts.

Run from the songwriter_s_toolkit repository root.

This is a standalone selector replay, NOT a live ECKF-state replay.
Ground-truth MIDI is deliberately not loaded.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_eckf.config import ECKFConfig
from python_eckf.harmonic_change import HarmonicChangeDetector
from python_eckf.periodicity import assess_periodicity
from python_eckf import initialization_candidates as selector


CASES = {
    "RATATA": {
        "wav": "RATATA.wav",
        "start": 42.78,
        "end": 43.20,
        "markers": (42.965, 43.008),
    },
    "Ochiitai": {
        "wav": "Ochiitai.wav",
        "start": 42.05,
        "end": 42.55,
        "markers": (42.214, 42.353),
    },
    "Trandafiri": {
        "wav": "Trandafiri.wav",
        "start": 10.60,
        "end": 11.05,
        "markers": (10.774, 10.821),
    },
}


def fmt(x, digits=4):
    if x is None:
        return "--"
    try:
        x = float(x)
        return f"{x:.{digits}f}" if math.isfinite(x) else "--"
    except (ValueError, TypeError):
        return "--"


def spectral_peaks(frame, fs, count=12):
    """Raw FFT peaks; NOT fundamental-frequency decisions."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)

    window = np.hanning(len(x))
    nfft = max(16384, 2 ** math.ceil(math.log2(len(x) * 4)))

    magnitude = np.abs(np.fft.rfft(x * window, n=nfft))
    frequencies = np.fft.rfftfreq(nfft, 1 / fs)

    peaks, _ = find_peaks(magnitude)
    peaks = [
        int(p) for p in peaks
        if 65.0 <= frequencies[p] <= min(6000.0, fs / 2)
    ]

    peaks.sort(key=lambda p: magnitude[p], reverse=True)

    return [
        (float(frequencies[p]), float(magnitude[p]))
        for p in peaks[:count]
    ]


def read_real_trace(path, start, end):
    """Extract existing live-run events; never reconstruct live state."""
    if not path.is_file():
        return ["Existing synchronized trace not found."]

    lines = path.read_text(errors="replace").splitlines()
    output = []

    for line in lines:
        if "frame_start=" not in line:
            continue

        try:
            timestamp = float(line.strip().split("s ", 1)[0])
        except (ValueError, IndexError):
            continue

        if start <= timestamp <= end:
            output.append(line)

    return output or ["No matching trace events found."]


def read_real_frames(path, start, end):
    """Read the CSV produced by the actual synchronized ECKF run."""
    if not path.is_file():
        return ["Existing frame CSV not found."]

    output = []

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = float(row["time_s"])
            except (KeyError, ValueError):
                continue

            if start <= timestamp <= end:
                output.append(
                    f"time={timestamp:.6f} "
                    f"voiced_samples={row.get('voiced_samples', '?')} "
                    f"first={row.get('first_hz', '?')} "
                    f"median={row.get('median_hz', '?')} "
                    f"last={row.get('last_hz', '?')}"
                )

    return output or ["No matching frames found."]


def audit_window(frame, fs, start_sample, detector, label):
    if len(frame) < 512:
        return

    frame = np.asarray(frame, dtype=np.float64)

    print()
    print(f"--- {label} ---")
    print(
        f"sample={start_sample} "
        f"time={start_sample / fs:.8f}s "
        f"length={len(frame)} "
        f"duration_ms={1000 * len(frame) / fs:.3f}"
    )

    rms = float(np.sqrt(np.mean(frame ** 2)))
    peak = float(np.max(np.abs(frame)))

    print(f"RMS={rms:.8f} peak_amplitude={peak:.8f}")

    # Same periodicity measurement used by the production system.
    period = assess_periodicity(frame, fs)

    print(
        f"PERIODICITY voiced={period.voiced} "
        f"ACF_hz={fmt(period.acf_frequency_hz)} "
        f"ACF_peak={fmt(period.acf_peak, 6)} "
        f"YIN_hz={fmt(period.cmndf_frequency_hz)} "
        f"CMNDF={fmt(period.cmndf_minimum, 6)}"
    )

    # Independent measured local periods, rather than frequency divisions.
    measured = selector._measured_period_candidates(frame, fs)

    print("MEASURED LOCAL PERIODS:")

    if not measured:
        print("  NONE")

    for hz, lag, acf, cmndf in sorted(
        measured, key=lambda p: p[1]
    ):
        print(
            f"  lag={lag:4d} "
            f"frequency={hz:10.4f}Hz "
            f"ACF={acf:.6f} "
            f"CMNDF={cmndf:.6f}"
        )

    print("SPECTRAL PEAKS (raw, not F0 verdicts):")

    for hz, amplitude in spectral_peaks(frame, fs):
        print(
            f"  frequency={hz:10.3f}Hz "
            f"magnitude={amplitude:.7g}"
        )

    # The production selector is designed for full-sized tracker frames.
    # Never misrepresent a short-window result as an actual live reset.
    if len(frame) != detector.config.block_size:
        print(
            "SHORT WINDOW: observational periodicity/spectrum only; "
            "no selector replay."
        )
        return

    proposal = detector.analyze(None, frame, fs)

    print(
        f"RAW SPECTRAL PROPOSAL: "
        f"frequency={fmt(proposal.f0_hz)}Hz"
    )

    print("SPECTRAL HARMONIC SUPPORT:")

    for hz, lag, acf, cmndf in sorted(
        measured, key=lambda p: p[1]
    ):
        harmonics = selector._harmonic_support(
            detector, frame, fs, hz
        )

        print(
            f"  candidate={hz:.4f}Hz "
            f"harmonic_indices={harmonics}"
        )

    choice = selector.choose_initialization(
        detector,
        frame,
        fs,
        start_sample,
        proposal,
        period,
        previous_hz=None,
        elapsed_ms=None,
    )

    print(f"STANDALONE SELECTOR CHOICE: {choice}")
    print(
        "IMPORTANT: previous_hz=None. This is not the actual "
        "live tracker reset; see the synchronized trace below."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--case",
        choices=(*CASES, "all"),
        default="all",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "robustness_v3"
        / "acoustic_microscope.txt",
    )

    args = parser.parse_args()

    config = ECKFConfig(mode="offline")
    block = config.block_size

    names = CASES if args.case == "all" else {
        args.case: CASES[args.case]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Capture stdout in the report without altering any production module.
    original_stdout = sys.stdout

    try:
        with args.output.open("w") as report:

            class Tee:
                def write(self, text):
                    original_stdout.write(text)
                    report.write(text)

                def flush(self):
                    original_stdout.flush()
                    report.flush()

            sys.stdout = Tee()

            print("=== ROBUSTNESS ACOUSTIC MICROSCOPE ===")
            print("WAV ONLY. GROUND-TRUTH MIDI NOT LOADED.")
            print("NO TRACKER OR SELECTOR MODIFICATIONS.")
            print(f"block_size={block}")

            for name, case in names.items():
                print()
                print("=" * 90)
                print(f"CASE: {name}")
                print("=" * 90)

                wav_path = ROOT / "tests" / case["wav"]

                if not wav_path.is_file():
                    alternative = wav_path.with_suffix(".wa")
                    if alternative.is_file():
                        wav_path = alternative
                    else:
                        raise FileNotFoundError(wav_path)

                wav, fs = sf.read(wav_path, dtype="float64")

                if wav.ndim != 1:
                    raise RuntimeError(
                        f"Expected mono WAV: {wav_path}"
                    )

                detector = HarmonicChangeDetector(config)

                print(f"WAV: {wav_path}")
                print(f"Sample rate: {fs}")
                print(f"Duration: {len(wav) / fs:.6f}s")
                print(f"Markers: {case['markers']}")

                start = case["start"]
                end = case["end"]

                first = max(
                    0,
                    int(math.floor(start * fs / block)) * block
                )

                last = min(
                    len(wav) - block,
                    int(math.floor(end * fs / block)) * block
                )

                print()
                print("=== COMPLETE PRODUCTION-SIZED FRAMES ===")

                for sample in range(first, last + 1, block):
                    audit_window(
                        wav[sample:sample + block],
                        fs,
                        sample,
                        detector,
                        "FULL FRAME",
                    )

                print()
                print("=== MARKER-CENTERED 1024-SAMPLE SUBWINDOWS ===")
                print(
                    "These expose time-local acoustic changes. "
                    "They do not prescribe new note boundaries."
                )

                for marker in case["markers"]:
                    center = int(round(marker * fs))
                    sample = max(0, center - 512)
                    sample = min(
                        sample,
                        len(wav) - 1024
                    )

                    audit_window(
                        wav[sample:sample + 1024],
                        fs,
                        sample,
                        detector,
                        f"MARKER {marker:.6f}s",
                    )

                base = ROOT / "tests" / "robustness_v3"

                trace = base / f"{name}_v3.txt"
                frames = base / f"{name}_v3_frames.csv"

                print()
                print("=== ACTUAL LIVE ECKF OUTPUT FRAMES ===")

                for line in read_real_frames(frames, start, end):
                    print(line)

                print()
                print("=== ACTUAL LIVE INITIALIZATION TRACE ===")

                for line in read_real_trace(trace, start, end):
                    print(line)

                print()
                print("=== INTERPRETATION SAFETY ===")
                print(
                    "Standalone selector choices are NOT live "
                    "tracker-state reconstructions."
                )
                print(
                    "MIDI, intended notes, and user annotations "
                    "were never inputs to acoustic calculations."
                )
                print(
                    "No pitch multiplication, smoothing, "
                    "interpolation, or fabricated note boundaries."
                )

    finally:
        sys.stdout = original_stdout

    print(f"\nReport written: {args.output}")


if __name__ == "__main__":
    main()
    