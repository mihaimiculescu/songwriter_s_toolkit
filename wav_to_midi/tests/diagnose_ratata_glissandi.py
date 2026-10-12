#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MIDI_GLISS_TIMES = [
    32.307,
    36.410,
    44.615,
]

DEFAULT_OFFSET_S = 4.616


def hz_to_midi(f):
    f = np.asarray(f, dtype=np.float64)

    out = np.full_like(
        f,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(f)
        & (f > 0)
    )

    out[valid] = (
        69.0
        + 12.0
        * np.log2(
            f[valid] / 440.0
        )
    )

    return out


def midi_to_name(value):
    if not np.isfinite(value):
        return "----"

    note = int(round(value))

    names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B",
    ]

    return (
        names[note % 12]
        + str(note // 12 - 1)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "eckf_csv",
        type=Path,
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_OFFSET_S,
    )

    parser.add_argument(
        "--before",
        type=float,
        default=0.50,
        help="Seconds before each confirmed glissando time.",
    )

    parser.add_argument(
        "--after",
        type=float,
        default=1.00,
        help="Seconds after each confirmed glissando time.",
    )

    args = parser.parse_args()

    data = np.genfromtxt(
        args.eckf_csv,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    t = np.asarray(
        data["time_s"],
        dtype=np.float64,
    )

    f0 = np.asarray(
        data["f0_hz"],
        dtype=np.float64,
    )

    amp = np.asarray(
        data["amplitude"],
        dtype=np.float64,
    )

    midi = hz_to_midi(
        f0
    )

    print()
    print("=" * 88)
    print("RATATA CONFIRMED DOWNWARD-GLISSANDO DIAGNOSTIC")
    print("=" * 88)
    print()
    print(
        f"Alignment: WAV_time = MIDI_time + {args.offset:.6f} s"
    )
    print()

    for idx, midi_time in enumerate(
        MIDI_GLISS_TIMES,
        start=1,
    ):
        wav_time = (
            midi_time
            + args.offset
        )

        start = (
            wav_time
            - args.before
        )

        end = (
            wav_time
            + args.after
        )

        mask = (
            (t >= start)
            & (t <= end)
        )

        tt = t[mask]
        ff = f0[mask]
        mm = midi[mask]
        aa = amp[mask]

        valid = (
            np.isfinite(ff)
            & (ff > 0)
        )

        print("-" * 88)
        print(
            f"GLISS {idx}"
        )
        print(
            f"MIDI/reference time: {midi_time:.3f} s"
        )
        print(
            f"WAV/ECKF time:       {wav_time:.3f} s"
        )
        print(
            f"Window:              {start:.3f} .. {end:.3f} s"
        )

        if not np.any(valid):
            print("No voiced ECKF samples in window.")
            print()
            continue

        vf = ff[valid]
        vm = mm[valid]

        print()
        print(
            f"RAW F0 range:        "
            f"{np.min(vf):.2f} .. {np.max(vf):.2f} Hz"
        )

        print(
            f"RAW MIDI-f range:    "
            f"{np.min(vm):.2f} .. {np.max(vm):.2f}"
        )

        print(
            f"RAW semitone span:   "
            f"{np.max(vm) - np.min(vm):.2f}"
        )

        plausible = (
            valid
            & (ff >= 60.0)
        )

        if np.any(plausible):
            pf = ff[plausible]
            pm = mm[plausible]

            print(
                f">=60 Hz F0 range:    "
                f"{np.min(pf):.2f} .. {np.max(pf):.2f} Hz"
            )

            print(
                f">=60 Hz MIDI range:  "
                f"{np.min(pm):.2f} .. {np.max(pm):.2f}"
            )

            print(
                f">=60 Hz span:        "
                f"{np.max(pm) - np.min(pm):.2f} semitones"
            )

        below_floor_count = np.count_nonzero(
            valid & (ff < 60.0)
        )

        print(
            f"Samples below 60 Hz: "
            f"{below_floor_count} / {np.count_nonzero(valid)} "
            f"({100.0 * below_floor_count / np.count_nonzero(valid):.1f}%)"
        )
        # Compare beginning and ending pitch over short robust windows.
        left_mask = (
            valid
            & (tt >= start)
            & (tt < start + 0.20)
        )

        right_mask = (
            valid
            & (tt > end - 0.20)
            & (tt <= end)
        )

        if np.any(left_mask):
            left_pitch = float(
                np.median(
                    mm[left_mask]
                )
            )
        else:
            left_pitch = np.nan

        if np.any(right_mask):
            right_pitch = float(
                np.median(
                    mm[right_mask]
                )
            )
        else:
            right_pitch = np.nan

        if (
            np.isfinite(left_pitch)
            and np.isfinite(right_pitch)
        ):
            delta = (
                right_pitch
                - left_pitch
            )

            print(
                f"Median start pitch:  "
                f"{left_pitch:.2f} "
                f"({midi_to_name(left_pitch)})"
            )

            print(
                f"Median end pitch:    "
                f"{right_pitch:.2f} "
                f"({midi_to_name(right_pitch)})"
            )

            print(
                f"Net movement:        "
                f"{delta:+.2f} semitones"
            )

        # Crude monotonicity measure:
        #
        # percentage of successive valid pitch steps that move downward.
        valid_idx = np.flatnonzero(
            valid
        )

        if len(valid_idx) >= 3:
            diffs = np.diff(
                midi[valid_idx]
            )

            nonzero = (
                np.abs(diffs) > 0.02
            )

            if np.any(nonzero):
                downward_fraction = (
                    np.count_nonzero(
                        diffs[nonzero] < 0
                    )
                    / np.count_nonzero(nonzero)
                )

                print(
                    f"Downward-step ratio: "
                    f"{100.0 * downward_fraction:.1f}%"
                )

        print()
        print(
            "time_s      f0_hz    midi_f   note   amplitude    status"
        )
        print(
            "--------   --------   ------   ----   ---------    -----------"
        )

        # Print every 50 ms from our 100-Hz CSV.
        if len(tt):
            stride = max(
                1,
                int(round(0.050 / 0.010)),
            )

            for j in range(
                0,
                len(tt),
                stride,
            ):

                if (
                    np.isfinite(ff[j])
                    and ff[j] > 0
                    and ff[j] < 60.0
                ):
                    status = "BELOW_FLOOR"
                elif (
                    np.isfinite(ff[j])
                    and ff[j] >= 60.0
                ):
                    status = "VOCAL_RANGE"
                else:
                    status = "UNVOICED"

                print(
                    f"{tt[j]:8.3f}   "
                    f"{ff[j]:8.2f}   "
                    f"{mm[j]:7.2f}   "
                    f"{midi_to_name(mm[j]):4s}   "
                    f"{aa[j]:.6f}   "
                    f"{status}"
                )
        print()

    print("=" * 88)
    print()


if __name__ == "__main__":
    main()