#!/usr/bin/env python3
"""Diagnose ECKF octave doubling without correcting or modifying anything."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def midi_notes(path):
    try:
        import mido
    except ImportError as exc:
        raise SystemExit("Install MIDI reader: pip install mido") from exc

    midi = mido.MidiFile(path)
    events = []

    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            events.append((tick, track_index, msg))

    events.sort(key=lambda item: (item[0], item[1]))

    tempo = 500000
    previous_tick = 0
    seconds = 0.0
    active = {}
    notes = []

    for tick, track_index, msg in events:
        seconds += (
            (tick - previous_tick)
            * tempo
            / (midi.ticks_per_beat * 1_000_000)
        )
        previous_tick = tick

        if msg.type == "set_tempo":
            tempo = msg.tempo

        elif msg.type == "note_on" and msg.velocity > 0:
            key = (track_index, msg.channel, msg.note)
            active.setdefault(key, []).append(seconds)

        elif msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        ):
            key = (track_index, msg.channel, msg.note)
            starts = active.get(key, [])
            if starts:
                start = starts.pop(0)
                notes.append((start, seconds, msg.note, track_index))

    return sorted(notes)


def note_name(number):
    names = (
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B",
    )
    return f"{names[number % 12]}{number // 12 - 1}"


def spectral_evidence(frame, sr, candidate_hz, harmonics=12):
    """Measure observed spectral energy near integer harmonics.

    Does not infer, replace, or correct F0.
    """
    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)

    if len(x) < 8:
        return float("nan"), float("nan"), float("nan")

    fft_size = max(65536, 2 ** int(np.ceil(np.log2(len(x) * 8))))
    spectrum = np.abs(np.fft.rfft(x * np.blackman(len(x)), n=fft_size)) ** 2
    frequencies = np.fft.rfftfreq(fft_size, 1 / sr)

    # A fixed measurement bandwidth, not a frequency correction.
    bandwidth_hz = max(8.0, sr / len(x))

    energies = []
    for harmonic in range(1, harmonics + 1):
        target = harmonic * candidate_hz
        if target >= sr / 2:
            break

        mask = np.abs(frequencies - target) <= bandwidth_hz / 2
        energies.append(float(np.max(spectrum[mask])) if np.any(mask) else 0.0)

    if not energies:
        return float("nan"), float("nan"), float("nan")

    fundamental = energies[0]
    odd = sum(energies[::2])
    even = sum(energies[1::2])
    total = sum(energies)

    return (
        fundamental,
        odd / total if total else 0.0,
        even / total if total else 0.0,
    )


def read_trace(path, start_s, end_s, sr):
    if path is None:
        return []

    if not path.exists():
        print(f"Trace not found: {path}; continuing without it.")
        return []

    events = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event = (
                row.get("event")
                or row.get("event_type")
                or row.get("type")
                or ""
            )

            if not any(
                term in event.upper()
                for term in (
                    "INITIALIZATION",
                    "RESET",
                    "PERIODICITY",
                    "KALMAN",
                )
            ):
                continue

            try:
                if row.get("time_s"):
                    t = float(row["time_s"])
                elif row.get("sample"):
                    t = float(row["sample"]) / sr
                else:
                    continue
            except (ValueError, TypeError):
                continue

            if start_s <= t <= end_s:
                events.append((t, event, row))

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="tests/PREDESTINATI.wav")
    parser.add_argument(
        "--midi",
        default="tests/GroundTruthPREDESTINATItype1.mid",
    )
    parser.add_argument(
        "--npz",
        default="tests/PREDESTINATI_periodicity_regression.npz",
    )
    parser.add_argument("--trace", default="tests/PREDESTINATI_spectral_trace.csv")
    parser.add_argument("--start", type=float, default=58.0)
    parser.add_argument("--end", type=float, default=59.1)
    parser.add_argument(
        "--midi-offset",
        type=float,
        default=0.0,
        help="Seconds added to MIDI times; default assumes shared timeline.",
    )
    parser.add_argument("--block", type=int, default=2048)
    args = parser.parse_args()

    audio, sr = sf.read(args.wav, dtype="float64")
    if audio.ndim != 1:
        raise SystemExit("WAV must be mono")

    data = np.load(args.npz)
    f0 = data["f0_hz"]

    if len(f0) < int(args.end * sr):
        raise SystemExit("NPZ does not cover the requested interval")

    notes = midi_notes(args.midi)

    print("\n=== INTENDED MIDI NOTES ===")
    for onset, offset, pitch, track in notes:
        onset += args.midi_offset
        offset += args.midi_offset
        if offset >= args.start and onset <= args.end:
            print(
                f"{note_name(pitch):4s} MIDI={pitch:3d} "
                f"on={onset:9.4f}s off={offset:9.4f}s "
                f"track={track}"
            )

    print("\n=== WAV / ECKF OCTAVE EVIDENCE ===")
    print(
        "frame_s    ECKF_med   WAV_F4_fund   WAV_F4_odd% "
        "WAV_G4_fund   WAV_G4_odd%   F4_vs_F5   G4_vs_G5"
    )

    for frame_start in range(0, len(f0), args.block):
        t = frame_start / sr
        if not args.start <= t <= args.end:
            continue

        frame = audio[frame_start:frame_start + args.block]
        output = f0[frame_start:frame_start + args.block]
        valid = output[np.isfinite(output) & (output > 0)]
        median = float(np.median(valid)) if len(valid) else float("nan")

        f4 = spectral_evidence(frame, sr, 349.228231)
        f5 = spectral_evidence(frame, sr, 698.456463)
        g4 = spectral_evidence(frame, sr, 391.995436)
        g5 = spectral_evidence(frame, sr, 783.990872)

        def ratio(a, b):
            return a / b if np.isfinite(a) and b > 0 else float("nan")

        print(
            f"{t:8.4f} "
            f"{median:11.2f} "
            f"{f4[0]:13.4g} {100*f4[1]:12.1f} "
            f"{g4[0]:13.4g} {100*g4[1]:12.1f} "
            f"{ratio(f4[0], f5[0]):10.3f} "
            f"{ratio(g4[0], g5[0]):10.3f}"
        )

    print("\n=== TRACKER TRACE EVENTS ===")
    events = read_trace(
        Path(args.trace) if args.trace else None,
        args.start,
        args.end,
        sr,
    )

    if not events:
        print("No matching trace events found.")
    else:
        for t, event, row in events:
            if "KALMAN" in event.upper():
                continue  # Frame summary above already covers its output.

            fields = (
                "init_f0_hz",
                "acf_peak",
                "cmndf_minimum",
                "voiced",
                "reason",
                "reset_accepted",
                "flag",
            )
            details = " ".join(
                f"{key}={row[key]}"
                for key in fields
                if row.get(key) not in (None, "")
            )
            print(f"{t:9.4f}s {event:28s} {details}")

    print("\nINTERPRETATION:")
    print("- MIDI describes intended notes only; no MIDI pitch is used to alter audio.")
    print("- ECKF_med is the actual tracker output.")
    print("- F4_vs_F5 / G4_vs_G5 compare observed energy near the")
    print("  candidate fundamental with energy near its second harmonic.")
    print("- Ratios alone cannot prove F0: a real fundamental may have")
    print("  a weak first harmonic. Inspect the full harmonic evidence.")
    print("- INITIALIZATION_PROPOSED shows whether doubling began at reset.")
    print("- If initialization is correct but ECKF_med doubles later,")
    print("  investigate the Kalman state evolution instead.")


if __name__ == "__main__":
    main()
