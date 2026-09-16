#!/usr/bin/env python3
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GLISS_MIDI_TIMES = [
    32.307,
    36.410,
    44.615,
]


def midi_name(midi_float):

    if not np.isfinite(
        midi_float
    ):
        return "---"

    n = int(
        round(midi_float)
    )

    names = [
        "C", "C#", "D", "D#",
        "E", "F", "F#", "G",
        "G#", "A", "A#", "B",
    ]

    octave = n // 12 - 1

    return (
        f"{names[n % 12]}{octave}"
    )


def hz_to_midi(f):

    if (
        not np.isfinite(f)
        or f <= 0
    ):
        return np.nan

    return (
        69.0
        + 12.0
        * np.log2(
            f / 440.0
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=4.616,
    )

    args = parser.parse_args()

    data = np.genfromtxt(
        args.csv,
        delimiter=",",
        names=True,
    )

    # Handle the existing diagnostic CSV column names.
    names = data.dtype.names

    if "time_s" not in names:
        raise ValueError(
            f"time_s not found; columns are {names}"
        )

    if "f0_hz" not in names:
        raise ValueError(
            f"f0_hz not found; columns are {names}"
        )

    t = np.asarray(
        data["time_s"],
        dtype=np.float64,
    )

    f0 = np.asarray(
        data["f0_hz"],
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # For this test we run the validity logic on the already exported
    # 100-Hz trajectory.
    #
    # Therefore pretend sample_rate=100 Hz.
    # -------------------------------------------------------------

    from python_eckf.validity import (
        PitchValidityConfig,
        analyse_pitch_validity,
    )

    config = PitchValidityConfig(
        analysis_hz=100.0,
        min_vocal_hz=120.0,
    )

    result = analyse_pitch_validity(
        f0_hz=f0,
        sample_rate=100.0,
        config=config,
    )

    print()
    print("=" * 92)
    print(
        "RATATA OFFLINE PITCH-VALIDITY DIAGNOSTIC"
    )
    print("=" * 92)

    print()
    print(
        f"Rows: {len(result.f0_hz)}"
    )

    print(
        f"Valid: {np.count_nonzero(result.valid)} "
        f"({100.0 * np.mean(result.valid):.1f}%)"
    )

    # Reason counts.
    print()
    print("Reason counts:")

    unique, counts = np.unique(
        result.reason,
        return_counts=True,
    )

    order = np.argsort(
        counts
    )[::-1]

    for j in order:
        print(
            f"  {unique[j]:24s} "
            f"{counts[j]:6d}"
        )

    for number, midi_t in enumerate(
        GLISS_MIDI_TIMES,
        start=1,
    ):

        center = (
            midi_t
            + args.offset
        )

        start = center - 0.300
        end = center + 1.200

        mask = (
            (t >= start)
            & (t <= end)
        )

        idx = np.flatnonzero(
            mask
        )

        print()
        print("-" * 92)

        print(
            f"DJWWW {number}"
        )

        print(
            f"Reference MIDI time: {midi_t:.3f} s"
        )

        print(
            f"WAV/ECKF time:       {center:.3f} s"
        )

        print()
        print(
            "time_s      f0_hz    midi     note   "
            "valid   reason"
        )

        print(
            "--------   --------   -------   ----   "
            "-----   ------------------------"
        )

        for i in idx:

            # print every 20 ms
            if (
                i % 2
                != 0
            ):
                continue

            f = f0[i]

            m = hz_to_midi(
                f
            )

            v = result.valid[i]

            print(
                f"{t[i]:8.3f}   "
                f"{f:8.2f}   "
                f"{m:7.2f}   "
                f"{midi_name(m):4s}   "
                f"{str(bool(v)):5s}   "
                f"{result.reason[i]}"
            )

    print()
    print("=" * 92)


if __name__ == "__main__":
    main()