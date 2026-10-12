#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MidiNote:
    note: int
    start_s: float
    end_s: float
    velocity: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# MIDI parsing
# ---------------------------------------------------------------------------

def read_midi_notes(path: Path):
    """
    Read MIDI note timing in real seconds, respecting the MIDI tempo map.

    This deliberately does NOT assume 117 BPM.  If the MIDI contains tempo
    information, that is the authoritative timing source.
    """

    mid = mido.MidiFile(path)

    ticks_per_beat = mid.ticks_per_beat

    # Standard MIDI default if no tempo event has yet appeared.
    tempo = 500000

    merged = mido.merge_tracks(mid.tracks)

    abs_time_s = 0.0

    # key -> queue of active note-ons
    active = defaultdict(deque)

    notes = []
    tempo_events = []

    for msg in merged:

        # msg.time is delta ticks in a merged MIDI track.
        abs_time_s += mido.tick2second(
            msg.time,
            ticks_per_beat,
            tempo,
        )

        if msg.type == "set_tempo":
            tempo = msg.tempo

            bpm = mido.tempo2bpm(tempo)

            tempo_events.append(
                (
                    abs_time_s,
                    tempo,
                    bpm,
                )
            )

            continue

        if msg.type == "note_on" and msg.velocity > 0:

            key = (
                getattr(msg, "channel", 0),
                msg.note,
            )

            active[key].append(
                (
                    abs_time_s,
                    msg.velocity,
                )
            )

            continue

        is_note_off = (
            msg.type == "note_off"
            or (
                msg.type == "note_on"
                and msg.velocity == 0
            )
        )

        if is_note_off:

            key = (
                getattr(msg, "channel", 0),
                msg.note,
            )

            if not active[key]:
                continue

            start_s, velocity = active[key].popleft()

            if abs_time_s > start_s:
                notes.append(
                    MidiNote(
                        note=msg.note,
                        start_s=start_s,
                        end_s=abs_time_s,
                        velocity=velocity,
                    )
                )

    notes.sort(
        key=lambda n: (
            n.start_s,
            n.note,
        )
    )

    return mid, notes, tempo_events


# ---------------------------------------------------------------------------
# ECKF CSV
# ---------------------------------------------------------------------------

def read_eckf_csv(path: Path):
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )

    time_s = np.asarray(
        data["time_s"],
        dtype=np.float64,
    )

    f0_hz = np.asarray(
        data["f0_hz"],
        dtype=np.float64,
    )

    amp = np.asarray(
        data["amplitude"],
        dtype=np.float64,
    )

    midi_float = np.full_like(
        f0_hz,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(f0_hz)
        & (f0_hz > 0.0)
    )

    midi_float[valid] = (
        69.0
        + 12.0
        * np.log2(
            f0_hz[valid] / 440.0
        )
    )

    return (
        time_s,
        f0_hz,
        amp,
        midi_float,
    )


# ---------------------------------------------------------------------------
# Alignment sampling
# ---------------------------------------------------------------------------

def note_probe_times(note: MidiNote):
    """
    Probe the central part of a handwritten MIDI note.

    We intentionally stay away from the boundaries because:
        * sung attacks can anticipate/posticipate;
        * portamento can happen around transitions;
        * the handwritten ground truth omits some ornaments;
        * ECKF itself has a small causal lag.

    Five interior samples make the alignment depend mostly on the stable
    pitch target rather than exact note-on placement.
    """

    fractions = np.array(
        [
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
        ],
        dtype=np.float64,
    )

    return (
        note.start_s
        + fractions * note.duration_s
    )


def sample_pitch(
    eckf_time,
    eckf_midi,
    query_time,
):
    """
    Linear interpolation of the 100-Hz diagnostic trajectory.

    NaN / unvoiced samples remain invalid rather than being replaced by
    arbitrary pitches.
    """

    # np.interp cannot correctly propagate internal NaNs, so interpolate
    # only from actually voiced trajectory samples.

    valid = (
        np.isfinite(eckf_time)
        & np.isfinite(eckf_midi)
    )

    if np.count_nonzero(valid) < 2:
        return np.full(
            np.shape(query_time),
            np.nan,
            dtype=np.float64,
        )

    t = eckf_time[valid]
    p = eckf_midi[valid]

    result = np.interp(
        query_time,
        t,
        p,
        left=np.nan,
        right=np.nan,
    )

    # Do NOT bridge large silent holes merely because np.interp can.
    #
    # For each query, find distance to nearest actual voiced ECKF point.
    indices = np.searchsorted(
        t,
        query_time,
    )

    nearest_dist = np.full(
        np.shape(query_time),
        np.inf,
        dtype=np.float64,
    )

    right_ok = indices < len(t)

    nearest_dist[right_ok] = np.minimum(
        nearest_dist[right_ok],
        np.abs(
            t[indices[right_ok]]
            - query_time[right_ok]
        ),
    )

    left_indices = indices - 1
    left_ok = left_indices >= 0

    nearest_dist[left_ok] = np.minimum(
        nearest_dist[left_ok],
        np.abs(
            t[left_indices[left_ok]]
            - query_time[left_ok]
        ),
    )

    # CSV is 100 Hz.  Anything more than 30 ms away from real voiced
    # evidence is treated as unvoiced rather than interpolated through.
    result[nearest_dist > 0.030] = np.nan

    return result


# ---------------------------------------------------------------------------
# Offset scoring
# ---------------------------------------------------------------------------

def score_offset(
    notes,
    eckf_time,
    eckf_midi,
    offset_s,
):
    """
    Score one global mapping:

        WAV time = MIDI time + offset_s

    No stretch.  No affine warp.

    Each MIDI note contributes one robust pitch error derived from five
    central probes.

    Because the handmade MIDI intentionally omits ornamentation, errors are
    capped at 300 cents so one ornament does not dominate the alignment.
    """

    note_errors = []
    note_coverage = []

    for note in notes:

        midi_times = note_probe_times(note)

        wav_times = (
            midi_times
            + offset_s
        )

        observed = sample_pitch(
            eckf_time,
            eckf_midi,
            wav_times,
        )

        valid = np.isfinite(observed)

        coverage = (
            np.count_nonzero(valid)
            / len(observed)
        )

        note_coverage.append(
            coverage
        )

        if not np.any(valid):
            # Missing voiced evidence.
            note_errors.append(
                300.0
            )
            continue

        cents = (
            100.0
            * (
                observed[valid]
                - float(note.note)
            )
        )

        # Median protects against an ornament on one probe point.
        err = float(
            np.median(
                np.abs(cents)
            )
        )

        # Ground truth is deliberately simplified.
        # Do not allow one omitted ornament to dominate global alignment.
        err = min(
            err,
            300.0,
        )

        # Partial coverage should cost something.
        missing_fraction = (
            1.0 - coverage
        )

        err += (
            missing_fraction
            * 150.0
        )

        note_errors.append(
            err
        )

    note_errors = np.asarray(
        note_errors,
        dtype=np.float64,
    )

    note_coverage = np.asarray(
        note_coverage,
        dtype=np.float64,
    )

    return {
        "score": float(
            np.mean(note_errors)
        ),
        "median_error": float(
            np.median(note_errors)
        ),
        "coverage": float(
            np.mean(note_coverage)
        ),
    }


def search_offset(
    notes,
    eckf_time,
    eckf_midi,
    min_offset,
    max_offset,
    step,
):
    offsets = np.arange(
        min_offset,
        max_offset + step * 0.5,
        step,
        dtype=np.float64,
    )

    results = []

    for offset in offsets:

        stats = score_offset(
            notes,
            eckf_time,
            eckf_midi,
            offset,
        )

        results.append(
            (
                offset,
                stats["score"],
                stats["median_error"],
                stats["coverage"],
            )
        )

    results.sort(
        key=lambda x: x[1]
    )

    return results


def refine_offset(
    notes,
    eckf_time,
    eckf_midi,
    coarse_offset,
):
    """
    Refine around the coarse 10-ms solution.

    The CSV itself is 10 ms, so this is not claiming sub-millisecond
    measurement precision.  Interpolation merely avoids a staircase score.
    """

    return search_offset(
        notes,
        eckf_time,
        eckf_midi,
        coarse_offset - 0.050,
        coarse_offset + 0.050,
        0.001,
    )


# ---------------------------------------------------------------------------
# Section diagnostics
# ---------------------------------------------------------------------------

def split_notes_into_thirds(notes):
    """
    Split by MIDI musical time, not by number of notes.
    """

    start = min(
        n.start_s
        for n in notes
    )

    end = max(
        n.end_s
        for n in notes
    )

    span = end - start

    edge1 = start + span / 3.0
    edge2 = start + 2.0 * span / 3.0

    early = []
    middle = []
    late = []

    for note in notes:

        center = (
            note.start_s
            + note.end_s
        ) * 0.5

        if center < edge1:
            early.append(note)
        elif center < edge2:
            middle.append(note)
        else:
            late.append(note)

    return (
        early,
        middle,
        late,
    )


# ---------------------------------------------------------------------------
# Detailed per-note report
# ---------------------------------------------------------------------------

def note_diagnostics(
    notes,
    eckf_time,
    eckf_midi,
    offset_s,
):
    rows = []

    for index, note in enumerate(
        notes,
        start=1,
    ):

        probe_midi_times = (
            note_probe_times(note)
        )

        probe_wav_times = (
            probe_midi_times
            + offset_s
        )

        observed = sample_pitch(
            eckf_time,
            eckf_midi,
            probe_wav_times,
        )

        valid = np.isfinite(
            observed
        )

        if np.any(valid):

            median_pitch = float(
                np.median(
                    observed[valid]
                )
            )

            median_f0 = (
                440.0
                * 2.0
                ** (
                    (
                        median_pitch
                        - 69.0
                    )
                    / 12.0
                )
            )

            cents_error = (
                100.0
                * (
                    median_pitch
                    - note.note
                )
            )

        else:

            median_pitch = np.nan
            median_f0 = np.nan
            cents_error = np.nan

        rows.append(
            (
                index,
                note.note,
                note.start_s,
                note.end_s,
                note.duration_s,
                note.start_s + offset_s,
                note.end_s + offset_s,
                median_f0,
                median_pitch,
                cents_error,
                np.count_nonzero(valid)
                / len(valid),
            )
        )

    return rows


def write_note_report(
    path,
    rows,
):
    header = (
        "index,"
        "midi_note,"
        "midi_start_s,"
        "midi_end_s,"
        "duration_s,"
        "wav_start_s,"
        "wav_end_s,"
        "eckf_median_f0_hz,"
        "eckf_median_midi,"
        "cents_error,"
        "probe_coverage"
    )

    arr = np.asarray(
        rows,
        dtype=np.float64,
    )

    np.savetxt(
        path,
        arr,
        delimiter=",",
        header=header,
        comments="",
        fmt=[
            "%d",
            "%d",
            "%.6f",
            "%.6f",
            "%.6f",
            "%.6f",
            "%.6f",
            "%.6f",
            "%.6f",
            "%.3f",
            "%.3f",
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare RATATA ECKF trajectory "
            "against handmade MIDI ground truth."
        )
    )

    parser.add_argument(
        "midi",
        type=Path,
    )

    parser.add_argument(
        "eckf_csv",
        type=Path,
    )

    parser.add_argument(
        "--min-offset",
        type=float,
        default=0.0,
        help="Minimum WAV-minus-MIDI offset in seconds.",
    )

    parser.add_argument(
        "--max-offset",
        type=float,
        default=8.0,
        help="Maximum WAV-minus-MIDI offset in seconds.",
    )

    parser.add_argument(
        "--coarse-step",
        type=float,
        default=0.010,
        help="Coarse offset-search step in seconds.",
    )

    args = parser.parse_args()

    (
        mid,
        notes,
        tempo_events,
    ) = read_midi_notes(
        args.midi
    )

    (
        eckf_time,
        eckf_f0,
        eckf_amp,
        eckf_midi,
    ) = read_eckf_csv(
        args.eckf_csv
    )

    if not notes:
        raise RuntimeError(
            "No MIDI notes found."
        )

    print()
    print(
        "=" * 80
    )
    print(
        "RATATA MIDI ↔ ECKF ALIGNMENT DIAGNOSTIC"
    )
    print(
        "=" * 80
    )

    print()
    print("MIDI")
    print("----")
    print(
        f"file:             {args.midi}"
    )
    print(
        f"type:             {mid.type}"
    )
    print(
        f"ticks per beat:   {mid.ticks_per_beat}"
    )
    print(
        f"tracks:           {len(mid.tracks)}"
    )
    print(
        f"notes:            {len(notes)}"
    )

    midi_start = min(
        n.start_s
        for n in notes
    )

    midi_end = max(
        n.end_s
        for n in notes
    )

    print(
        f"first note:       {midi_start:.6f} s"
    )
    print(
        f"last note end:    {midi_end:.6f} s"
    )

    print()
    print("Tempo map:")

    if tempo_events:

        for (
            time_s,
            tempo,
            bpm,
        ) in tempo_events:

            print(
                f"  t={time_s:9.6f} s  "
                f"tempo={tempo} us/qn  "
                f"BPM={bpm:.6f}"
            )

    else:

        print(
            "  no explicit tempo event "
            "(MIDI default 120 BPM)"
        )

    print()
    print("ECKF")
    print("----")
    print(
        f"file:             {args.eckf_csv}"
    )
    print(
        f"samples:          {len(eckf_time)}"
    )
    print(
        f"time span:        "
        f"{eckf_time[0]:.3f} .. "
        f"{eckf_time[-1]:.3f} s"
    )

    voiced = np.isfinite(
        eckf_midi
    )

    print(
        f"voiced rows:      "
        f"{np.count_nonzero(voiced)}"
    )

    # ---------------------------------------------------------------
    # Global coarse search
    # ---------------------------------------------------------------

    print()
    print(
        "GLOBAL OFFSET SEARCH"
    )
    print(
        "--------------------"
    )
    print(
        "Model: WAV_time = MIDI_time + offset"
    )
    print(
        "Time scale is FIXED at 1.0."
    )

    coarse = search_offset(
        notes,
        eckf_time,
        eckf_midi,
        args.min_offset,
        args.max_offset,
        args.coarse_step,
    )

    best_coarse = coarse[0]

    print()
    print(
        f"Best coarse offset: "
        f"{best_coarse[0]:.6f} s"
    )

    # ---------------------------------------------------------------
    # Refine
    # ---------------------------------------------------------------

    refined = refine_offset(
        notes,
        eckf_time,
        eckf_midi,
        best_coarse[0],
    )

    best = refined[0]

    best_offset = best[0]

    print(
        f"Best refined offset: "
        f"{best_offset:.6f} s"
    )
    print(
        f"Mean robust score:   "
        f"{best[1]:.3f} cents"
    )
    print(
        f"Median error score:  "
        f"{best[2]:.3f} cents"
    )
    print(
        f"Probe coverage:      "
        f"{100.0 * best[3]:.2f}%"
    )

    print()
    print("Top 10 refined candidates:")
    print()
    print(
        " offset_s    score_cents   "
        "median_cents   coverage"
    )
    print(
        " --------    -----------   "
        "------------   --------"
    )

    for (
        offset,
        score,
        median,
        coverage,
    ) in refined[:10]:

        print(
            f" {offset:8.4f}    "
            f"{score:11.3f}   "
            f"{median:12.3f}   "
            f"{coverage * 100:7.2f}%"
        )

    # ---------------------------------------------------------------
    # Independent early / middle / late searches
    # ---------------------------------------------------------------

    print()
    print(
        "TIME-DRIFT TEST"
    )
    print(
        "---------------"
    )
    print(
        "Each third is aligned independently."
    )
    print(
        "If these offsets remain similar, "
        "there is no evidence for time stretching."
    )
    print()

    (
        early,
        middle,
        late,
    ) = split_notes_into_thirds(
        notes
    )

    section_results = []

    for name, subset in (
        ("EARLY ", early),
        ("MIDDLE", middle),
        ("LATE  ", late),
    ):

        coarse_section = search_offset(
            subset,
            eckf_time,
            eckf_midi,
            args.min_offset,
            args.max_offset,
            args.coarse_step,
        )

        refined_section = (
            refine_offset(
                subset,
                eckf_time,
                eckf_midi,
                coarse_section[0][0],
            )
        )

        section_best = (
            refined_section[0]
        )

        section_results.append(
            (
                name,
                len(subset),
                section_best,
            )
        )

        print(
            f"{name}: "
            f"notes={len(subset):3d}  "
            f"offset={section_best[0]:8.4f} s  "
            f"score={section_best[1]:8.2f} c  "
            f"coverage="
            f"{section_best[3] * 100:6.2f}%"
        )

    early_offset = (
        section_results[0][2][0]
    )

    late_offset = (
        section_results[2][2][0]
    )

    drift = (
        late_offset
        - early_offset
    )

    print()
    print(
        f"Late-minus-early offset drift: "
        f"{drift:+.6f} s"
    )

    midi_span = (
        midi_end - midi_start
    )

    if midi_span > 0:
        implied_scale = (
            1.0
            + drift / midi_span
        )

        implied_percent = (
            (implied_scale - 1.0)
            * 100.0
        )

        print(
            f"Equivalent rough scale difference: "
            f"{implied_percent:+.4f}%"
        )

    # ---------------------------------------------------------------
    # Detailed note report
    # ---------------------------------------------------------------

    rows = note_diagnostics(
        notes,
        eckf_time,
        eckf_midi,
        best_offset,
    )

    output_path = (
        args.eckf_csv.parent
        / (
            args.eckf_csv.stem
            + ".alignment.csv"
        )
    )

    write_note_report(
        output_path,
        rows,
    )

    cents_errors = np.asarray(
        [
            row[9]
            for row in rows
        ],
        dtype=np.float64,
    )

    valid_errors = cents_errors[
        np.isfinite(
            cents_errors
        )
    ]

    print()
    print(
        "PITCH AGREEMENT AT BEST GLOBAL OFFSET"
    )
    print(
        "-------------------------------------"
    )

    if len(valid_errors):

        absolute = np.abs(
            valid_errors
        )

        print(
            f"Notes with voiced evidence: "
            f"{len(valid_errors)} / {len(notes)}"
        )

        print(
            f"Median |pitch error|:      "
            f"{np.median(absolute):.2f} cents"
        )

        for threshold in (
            25,
            50,
            100,
            200,
        ):
            count = np.count_nonzero(
                absolute <= threshold
            )

            print(
                f"Within {threshold:3d} cents:          "
                f"{count:3d} / "
                f"{len(valid_errors):3d} "
                f"({100.0 * count / len(valid_errors):5.1f}%)"
            )

    print()
    print(
        f"Detailed note report: {output_path}"
    )

    print()
    print(
        "IMPORTANT"
    )
    print(
        "---------"
    )
    print(
        "This diagnostic treats the handmade MIDI "
        "as structural pitch-target ground truth."
    )
    print(
        "It does NOT assume that every ECKF deviation "
        "is an error, because ornamentation was "
        "intentionally omitted from the MIDI."
    )
    print()


if __name__ == "__main__":
    main()