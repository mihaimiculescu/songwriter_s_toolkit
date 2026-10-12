#!/usr/bin/env python3
"""
diagnose_midi_wav_event_correspondence.py

Observational diagnostic comparing:

    ground-truth MIDI intended note events

against

    independently discovered WAV/ECKF pitch events.

IMPORTANT
---------
This diagnostic MUST NOT alter the frozen WAV-analysis pipeline.

The MIDI is used only after WAV analysis has already happened.

This is NOT:
    - a MIDI quantizer
    - a note classifier
    - a gesture classifier
    - a scoring system
    - a training target generator

Its purpose is to distinguish evidence for:

    1. global MIDI <-> WAV synchronization
    2. local human timing variation
    3. expressive pitch realization
    4. possible tracking failure

No fixed "correct timing" window is imposed in V1.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mido
import numpy as np


# ---------------------------------------------------------------------------
# Make repo root importable when script is run directly from tests/
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass(frozen=True)
class Config:
    analysis_hz: float = 100.0

    # Search range used ONLY to estimate global MIDI -> WAV offset.
    global_offset_min_s: float = -15.0
    global_offset_max_s: float = 15.0
    global_offset_step_s: float = 0.010

    # Local observation aperture around each aligned MIDI onset.
    local_before_ms: float = 500.0
    local_after_ms: float = 500.0

    # Wider aperture used when searching for nearest WAV pitch event.
    event_search_ms: float = 750.0

    # Pitch relationships are OBSERVATIONAL only.
    target_near_st: float = 0.50
    harmonic_near_st: float = 0.75

    # Detect when a local trajectory begins moving.
    movement_step_st: float = 0.12

    # Stable arrival evidence around intended pitch.
    target_arrival_st: float = 0.50
    target_settle_ms: float = 60.0

    # Minimum valid coverage used when evaluating a local window.
    min_local_valid_fraction: float = 0.25


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass(frozen=True)
class MidiNote:
    index: int
    pitch: int
    onset_s: float
    offset_s: float
    duration_s: float
    channel: int
    velocity: int
    track: int


@dataclass(frozen=True)
class MidiData:
    notes: tuple[MidiNote, ...]
    ticks_per_beat: int
    midi_type: int


@dataclass(frozen=True)
class ECKFData:
    time_s: np.ndarray
    raw_f0_hz: np.ndarray
    clean_f0_hz: np.ndarray
    corrected_valid: np.ndarray
    pitch_st: np.ndarray


@dataclass(frozen=True)
class WavEvent:
    index: int
    time_s: float
    left_pitch_st: float
    right_pitch_st: float
    interval_st: float
    kind: str


@dataclass(frozen=True)
class LocalObservation:
    midi_note: MidiNote
    aligned_onset_s: float

    nearest_event: Optional[WavEvent]
    nearest_event_delta_ms: float

    valid_fraction: float

    pitch_at_onset_st: float
    pitch_error_at_onset_st: float

    nearest_target_time_s: float
    nearest_target_delta_ms: float
    nearest_target_pitch_st: float
    nearest_target_pitch_error_st: float

    movement_start_s: float
    movement_start_delta_ms: float

    target_arrival_s: float
    target_arrival_delta_ms: float

    target_settle_s: float
    target_settle_delta_ms: float

    local_min_st: float
    local_max_st: float
    local_span_st: float

    harmonic_relation: str


# ===========================================================================
# Basic pitch helpers
# ===========================================================================

def hz_to_midi_float(hz: float) -> float:
    if not np.isfinite(hz) or hz <= 0.0:
        return np.nan
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def signed_ms(value_s: float) -> float:
    if not np.isfinite(value_s):
        return np.nan
    return 1000.0 * value_s


def fmt_float(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_signed(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:+.{digits}f}"


def midi_name(note: int) -> str:
    names = (
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B",
    )
    octave = note // 12 - 1
    return f"{names[note % 12]}{octave}"


# ===========================================================================
# MIDI parser
# ===========================================================================

def load_midi_notes(path: Path) -> MidiData:
    """
    Parse all note-on/note-off pairs using the complete tempo map.

    Times are absolute seconds in the MIDI's own timeline.

    Track number is retained only diagnostically.
    """

    mid = mido.MidiFile(path)

    # ------------------------------------------------------------------
    # First collect all messages in absolute tick coordinates.
    # ------------------------------------------------------------------

    absolute_messages = []

    for track_index, track in enumerate(mid.tracks):
        tick = 0

        for msg in track:
            tick += msg.time
            absolute_messages.append(
                (
                    int(tick),
                    int(track_index),
                    msg,
                )
            )

    # ------------------------------------------------------------------
    # Global tempo map.
    # ------------------------------------------------------------------

    tempo_events = [(0, 500000)]

    for tick, _, msg in absolute_messages:
        if msg.type == "set_tempo":
            tempo_events.append((tick, int(msg.tempo)))

    tempo_events.sort(key=lambda x: x[0])

    # If multiple tempo messages occur at the same tick, later one wins.
    collapsed = []

    for tick, tempo in tempo_events:
        if collapsed and collapsed[-1][0] == tick:
            collapsed[-1] = (tick, tempo)
        else:
            collapsed.append((tick, tempo))

    tempo_events = collapsed

    def tick_to_seconds(target_tick: int) -> float:
        seconds = 0.0
        prev_tick = 0
        tempo = 500000

        for event_tick, event_tempo in tempo_events:
            if event_tick > target_tick:
                break

            if event_tick > prev_tick:
                seconds += mido.tick2second(
                    event_tick - prev_tick,
                    mid.ticks_per_beat,
                    tempo,
                )

            prev_tick = event_tick
            tempo = event_tempo

        if target_tick > prev_tick:
            seconds += mido.tick2second(
                target_tick - prev_tick,
                mid.ticks_per_beat,
                tempo,
            )

        return float(seconds)

    # ------------------------------------------------------------------
    # Pair note-on/off.
    #
    # Key includes track and channel because Type-1 files can contain
    # overlapping same-pitch material on different tracks.
    # ------------------------------------------------------------------

    active: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    raw_notes = []

    absolute_messages.sort(key=lambda x: (x[0], x[1]))

    for tick, track_index, msg in absolute_messages:
        if msg.type not in ("note_on", "note_off"):
            continue

        channel = int(getattr(msg, "channel", 0))
        pitch = int(msg.note)

        key = (track_index, channel, pitch)

        is_on = msg.type == "note_on" and msg.velocity > 0
        is_off = msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        )

        if is_on:
            active.setdefault(key, []).append(
                (tick, int(msg.velocity))
            )

        elif is_off:
            queue = active.get(key)

            if not queue:
                continue

            onset_tick, velocity = queue.pop(0)

            if not queue:
                active.pop(key, None)

            if tick <= onset_tick:
                continue

            raw_notes.append(
                (
                    onset_tick,
                    tick,
                    pitch,
                    channel,
                    velocity,
                    track_index,
                )
            )

    raw_notes.sort(
        key=lambda x: (
            x[0],
            x[2],
            x[5],
        )
    )

    notes = []

    for index, (
        onset_tick,
        offset_tick,
        pitch,
        channel,
        velocity,
        track_index,
    ) in enumerate(raw_notes, start=1):

        onset_s = tick_to_seconds(onset_tick)
        offset_s = tick_to_seconds(offset_tick)

        notes.append(
            MidiNote(
                index=index,
                pitch=pitch,
                onset_s=onset_s,
                offset_s=offset_s,
                duration_s=offset_s - onset_s,
                channel=channel,
                velocity=velocity,
                track=track_index,
            )
        )

    if not notes:
        raise RuntimeError(f"No MIDI notes found in {path}")

    return MidiData(
        notes=tuple(notes),
        ticks_per_beat=int(mid.ticks_per_beat),
        midi_type=int(mid.type),
    )


# ===========================================================================
# ECKF CSV
# ===========================================================================

def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in (
        "1", "true", "yes", "y", "t",
    )


def load_eckf_csv(path: Path) -> ECKFData:
    """
    Load the production offline CSV.

    Required:
        time_s
        f0_hz
        corrected_valid
        clean_f0_hz
    """

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header.")

        required = {
            "time_s",
            "f0_hz",
            "corrected_valid",
            "clean_f0_hz",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(
                "CSV missing required columns: "
                + ", ".join(sorted(missing))
            )

        times = []
        raw_f0 = []
        clean_f0 = []
        valid = []

        for row in reader:
            times.append(float(row["time_s"]))
            raw_f0.append(float(row["f0_hz"]))
            clean_text = row["clean_f0_hz"].strip()
            clean_f0.append(
                float(clean_text) if clean_text else np.nan
            )
            valid.append(_parse_bool(row["corrected_valid"]))

    time_s = np.asarray(times, dtype=float)
    raw_f0_hz = np.asarray(raw_f0, dtype=float)
    clean_f0_hz = np.asarray(clean_f0, dtype=float)
    corrected_valid = np.asarray(valid, dtype=bool)

    if len(time_s) < 2:
        raise RuntimeError("ECKF CSV is too short.")

    pitch_st = np.full(len(time_s), np.nan, dtype=float)

    usable = (
        corrected_valid
        & np.isfinite(clean_f0_hz)
        & (clean_f0_hz > 0.0)
    )

    pitch_st[usable] = (
        69.0
        + 12.0 * np.log2(clean_f0_hz[usable] / 440.0)
    )

    return ECKFData(
        time_s=time_s,
        raw_f0_hz=raw_f0_hz,
        clean_f0_hz=clean_f0_hz,
        corrected_valid=corrected_valid,
        pitch_st=pitch_st,
    )


# ===========================================================================
# Independent WAV event extraction
# ===========================================================================

def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Inclusive-exclusive True runs.
    """

    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0:
        return []

    padded = np.concatenate(
        ([False], mask, [False])
    ).astype(np.int8)

    diff = np.diff(padded)

    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    return list(zip(starts.tolist(), ends.tolist()))


def extract_wav_events(
    data: ECKFData,
    config: Config,
) -> list[WavEvent]:
    """
    Build a deliberately conservative event set directly from the
    corrected pitch trajectory.

    V1 event sources:

        VALIDITY_START
        VALIDITY_END
        PITCH_STEP

    PITCH_STEP is NOT declared to be a MIDI-note boundary. It is simply
    a local structural pitch event suitable for alignment.

    We deliberately avoid importing the frozen structural splitter here:
    the first alignment estimate should be based on the most primitive
    independent pitch evidence possible.
    """

    events: list[WavEvent] = []
    next_index = 1

    runs = true_runs(data.corrected_valid)

    for start, end in runs:
        if end <= start:
            continue

        start_pitch = data.pitch_st[start]

        events.append(
            WavEvent(
                index=next_index,
                time_s=float(data.time_s[start]),
                left_pitch_st=np.nan,
                right_pitch_st=float(start_pitch),
                interval_st=np.nan,
                kind="VALIDITY_START",
            )
        )
        next_index += 1

        if end < len(data.time_s):
            left_pitch = data.pitch_st[end - 1]

            events.append(
                WavEvent(
                    index=next_index,
                    time_s=float(data.time_s[end]),
                    left_pitch_st=float(left_pitch),
                    right_pitch_st=np.nan,
                    interval_st=np.nan,
                    kind="VALIDITY_END",
                )
            )
            next_index += 1

        # --------------------------------------------------------------
        # Candidate local pitch changes.
        #
        # We compare robust medians immediately before and after each
        # point rather than individual 10-ms samples.
        # --------------------------------------------------------------

        radius = max(
            2,
            int(round(0.040 * config.analysis_hz)),
        )

        for i in range(start + radius, end - radius):
            left = data.pitch_st[i - radius:i]
            right = data.pitch_st[i:i + radius]

            left_med = finite_median(left)
            right_med = finite_median(right)

            if not (
                np.isfinite(left_med)
                and np.isfinite(right_med)
            ):
                continue

            interval = right_med - left_med

            # This is only an event proposal aperture.
            # It is intentionally small enough to include expressive
            # transitions; later diagnostics decide what the event means.
            if abs(interval) < config.movement_step_st:
                continue

            events.append(
                WavEvent(
                    index=next_index,
                    time_s=float(data.time_s[i]),
                    left_pitch_st=float(left_med),
                    right_pitch_st=float(right_med),
                    interval_st=float(interval),
                    kind="PITCH_STEP",
                )
            )
            next_index += 1

    events.sort(key=lambda e: e.time_s)

    return [
        WavEvent(
            index=i,
            time_s=e.time_s,
            left_pitch_st=e.left_pitch_st,
            right_pitch_st=e.right_pitch_st,
            interval_st=e.interval_st,
            kind=e.kind,
        )
        for i, e in enumerate(events, start=1)
    ]


# ===========================================================================
# Global MIDI <-> WAV alignment
# ===========================================================================

def local_pitch_match_score(
    data: ECKFData,
    wav_time_s: float,
    midi_pitch: int,
    half_window_s: float = 0.120,
) -> float:
    """
    Robust pitch evidence around one proposed aligned MIDI onset.

    Returns 0 when there is no usable pitch.

    Score is intentionally broad because sung realization may contain
    scoop / glide / vibrato around the intended target.
    """

    lo = wav_time_s - half_window_s
    hi = wav_time_s + half_window_s

    mask = (
        (data.time_s >= lo)
        & (data.time_s <= hi)
        & data.corrected_valid
        & np.isfinite(data.pitch_st)
    )

    values = data.pitch_st[mask]

    if len(values) == 0:
        return 0.0

    errors = np.abs(values - float(midi_pitch))

    # Best-supported local approach to intended pitch.
    pitch_score = np.exp(
        -0.5 * (errors / 1.0) ** 2
    )

    # Median of strongest quarter avoids one lucky sample dominating.
    k = max(1, int(math.ceil(len(pitch_score) * 0.25)))
    strongest = np.partition(
        pitch_score,
        len(pitch_score) - k,
    )[-k:]

    return float(np.median(strongest))


def event_proximity_score(
    events: list[WavEvent],
    wav_time_s: float,
) -> float:
    """
    Soft event-time support.

    No event is REQUIRED to coincide with a MIDI onset.
    This merely helps estimate the single global synchronization offset.
    """

    if not events:
        return 0.0

    delta = min(
        abs(e.time_s - wav_time_s)
        for e in events
    )

    sigma_s = 0.150

    return float(
        math.exp(
            -0.5 * (delta / sigma_s) ** 2
        )
    )


def global_alignment_score(
    notes: tuple[MidiNote, ...],
    data: ECKFData,
    events: list[WavEvent],
    offset_s: float,
) -> float:
    """
    offset convention:

        WAV time = MIDI time + offset
    """

    scores = []

    for note in notes:
        wav_time = note.onset_s + offset_s

        if (
            wav_time < data.time_s[0]
            or wav_time > data.time_s[-1]
        ):
            continue

        pitch_support = local_pitch_match_score(
            data,
            wav_time,
            note.pitch,
        )

        event_support = event_proximity_score(
            events,
            wav_time,
        )

        # Pitch identity dominates.
        # Event timing is supporting evidence only.
        score = (
            0.80 * pitch_support
            + 0.20 * event_support
        )

        scores.append(score)

    if not scores:
        return -np.inf

    return float(np.median(scores))


def estimate_global_offset(
    midi: MidiData,
    data: ECKFData,
    events: list[WavEvent],
    config: Config,
) -> tuple[float, float]:
    offsets = np.arange(
        config.global_offset_min_s,
        config.global_offset_max_s
        + 0.5 * config.global_offset_step_s,
        config.global_offset_step_s,
    )

    scores = np.asarray(
        [
            global_alignment_score(
                midi.notes,
                data,
                events,
                float(offset),
            )
            for offset in offsets
        ],
        dtype=float,
    )

    if not np.any(np.isfinite(scores)):
        raise RuntimeError(
            "Could not estimate MIDI/WAV global offset."
        )

    best_index = int(np.nanargmax(scores))

    return (
        float(offsets[best_index]),
        float(scores[best_index]),
    )


# ===========================================================================
# Early / middle / late drift observation
# ===========================================================================

def estimate_section_offset(
    notes: tuple[MidiNote, ...],
    data: ECKFData,
    events: list[WavEvent],
    center_offset_s: float,
    radius_s: float = 0.300,
    step_s: float = 0.005,
) -> tuple[float, float]:
    offsets = np.arange(
        center_offset_s - radius_s,
        center_offset_s + radius_s + 0.5 * step_s,
        step_s,
    )

    best_offset = np.nan
    best_score = -np.inf

    for offset in offsets:
        score = global_alignment_score(
            notes,
            data,
            events,
            float(offset),
        )

        if score > best_score:
            best_score = score
            best_offset = float(offset)

    return best_offset, best_score


def section_offsets(
    midi: MidiData,
    data: ECKFData,
    events: list[WavEvent],
    global_offset_s: float,
) -> dict[str, tuple[float, float]]:
    notes = midi.notes

    if len(notes) < 6:
        return {}

    onset_times = np.asarray(
        [n.onset_s for n in notes],
        dtype=float,
    )

    q1, q2 = np.quantile(
        onset_times,
        [1.0 / 3.0, 2.0 / 3.0],
    )

    sections = {
        "EARLY": tuple(
            n for n in notes
            if n.onset_s <= q1
        ),
        "MIDDLE": tuple(
            n for n in notes
            if q1 < n.onset_s <= q2
        ),
        "LATE": tuple(
            n for n in notes
            if n.onset_s > q2
        ),
    }

    result = {}

    for name, subset in sections.items():
        result[name] = estimate_section_offset(
            subset,
            data,
            events,
            global_offset_s,
        )

    return result


# ===========================================================================
# Local trajectory analysis
# ===========================================================================

def nearest_index(
    time_s: np.ndarray,
    target_s: float,
) -> int:
    return int(
        np.argmin(
            np.abs(time_s - target_s)
        )
    )


def nearest_wav_event(
    events: list[WavEvent],
    target_s: float,
    max_distance_s: float,
) -> Optional[WavEvent]:
    candidates = [
        e for e in events
        if abs(e.time_s - target_s) <= max_distance_s
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda e: abs(e.time_s - target_s),
    )


def nearest_pitch_target_sample(
    data: ECKFData,
    target_time_s: float,
    midi_pitch: int,
    search_s: float,
) -> tuple[float, float]:
    mask = (
        (data.time_s >= target_time_s - search_s)
        & (data.time_s <= target_time_s + search_s)
        & data.corrected_valid
        & np.isfinite(data.pitch_st)
    )

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return np.nan, np.nan

    errors = np.abs(
        data.pitch_st[indices] - float(midi_pitch)
    )

    j = int(indices[int(np.argmin(errors))])

    return (
        float(data.time_s[j]),
        float(data.pitch_st[j]),
    )


def find_movement_start(
    data: ECKFData,
    onset_s: float,
    config: Config,
) -> float:
    """
    Observational estimate of when local movement toward/through the
    event begins.

    Searches backwards from onset within local_before_ms.

    This does NOT assert that movement belongs to the MIDI note.
    """

    before_s = config.local_before_ms / 1000.0

    mask = (
        (data.time_s >= onset_s - before_s)
        & (data.time_s <= onset_s)
        & data.corrected_valid
        & np.isfinite(data.pitch_st)
    )

    idx = np.flatnonzero(mask)

    if len(idx) < 3:
        return np.nan

    values = data.pitch_st[idx]
    diffs = np.abs(np.diff(values))

    moving = diffs >= config.movement_step_st

    if not np.any(moving):
        return np.nan

    # Find the final contiguous movement-containing zone leading toward
    # the onset. Small one-sample quiet gaps are tolerated.
    moving_idx = np.flatnonzero(moving)

    last = int(moving_idx[-1])
    start = last

    for k in range(last - 1, -1, -1):
        if moving[k]:
            start = k
            continue

        # tolerate one quiet step between movement samples
        if k - 1 >= 0 and moving[k - 1]:
            start = k
            continue

        break

    return float(
        data.time_s[idx[start]]
    )


def find_target_arrival(
    data: ECKFData,
    onset_s: float,
    midi_pitch: int,
    config: Config,
) -> float:
    before_s = config.local_before_ms / 1000.0
    after_s = config.local_after_ms / 1000.0

    mask = (
        (data.time_s >= onset_s - before_s)
        & (data.time_s <= onset_s + after_s)
        & data.corrected_valid
        & np.isfinite(data.pitch_st)
    )

    idx = np.flatnonzero(mask)

    if len(idx) == 0:
        return np.nan

    close = (
        np.abs(
            data.pitch_st[idx] - float(midi_pitch)
        )
        <= config.target_arrival_st
    )

    hits = np.flatnonzero(close)

    if len(hits) == 0:
        return np.nan

    return float(
        data.time_s[idx[int(hits[0])]]
    )


def find_target_settle(
    data: ECKFData,
    onset_s: float,
    midi_pitch: int,
    config: Config,
) -> float:
    before_s = config.local_before_ms / 1000.0
    after_s = config.local_after_ms / 1000.0

    mask = (
        (data.time_s >= onset_s - before_s)
        & (data.time_s <= onset_s + after_s)
    )

    idx = np.flatnonzero(mask)

    if len(idx) == 0:
        return np.nan

    settle_steps = max(
        1,
        int(
            round(
                config.target_settle_ms
                * config.analysis_hz
                / 1000.0
            )
        ),
    )

    for p in range(0, len(idx) - settle_steps + 1):
        window_idx = idx[p:p + settle_steps]

        if not np.all(
            data.corrected_valid[window_idx]
        ):
            continue

        values = data.pitch_st[window_idx]

        if not np.all(np.isfinite(values)):
            continue

        if np.all(
            np.abs(values - float(midi_pitch))
            <= config.target_arrival_st
        ):
            return float(
                data.time_s[window_idx[0]]
            )

    return np.nan


# ===========================================================================
# Harmonic / subharmonic observation
# ===========================================================================

HARMONIC_RELATIONS = (
    ("1/3", -19.0195500087),
    ("1/2", -12.0),
    ("2x", +12.0),
    ("3x", +19.0195500087),
)


def harmonic_relation(
    observed_pitch_st: float,
    intended_pitch_st: float,
    tolerance_st: float,
) -> str:
    if not (
        np.isfinite(observed_pitch_st)
        and np.isfinite(intended_pitch_st)
    ):
        return "-"

    interval = observed_pitch_st - intended_pitch_st

    best_name = "-"
    best_error = np.inf

    for name, expected in HARMONIC_RELATIONS:
        error = abs(interval - expected)

        if error < best_error:
            best_error = error
            best_name = name

    if best_error <= tolerance_st:
        return best_name

    return "-"


# ===========================================================================
# Per-note observation
# ===========================================================================

def observe_note(
    note: MidiNote,
    data: ECKFData,
    events: list[WavEvent],
    offset_s: float,
    config: Config,
) -> LocalObservation:
    onset = note.onset_s + offset_s

    nearest_event = nearest_wav_event(
        events,
        onset,
        config.event_search_ms / 1000.0,
    )

    if nearest_event is None:
        nearest_event_delta_ms = np.nan
    else:
        nearest_event_delta_ms = signed_ms(
            nearest_event.time_s - onset
        )

    local_before = config.local_before_ms / 1000.0
    local_after = config.local_after_ms / 1000.0

    local_mask = (
        (data.time_s >= onset - local_before)
        & (data.time_s <= onset + local_after)
    )

    local_indices = np.flatnonzero(local_mask)

    if len(local_indices):
        valid_fraction = float(
            np.mean(
                data.corrected_valid[local_indices]
            )
        )
    else:
        valid_fraction = 0.0

    onset_index = nearest_index(
        data.time_s,
        onset,
    )

    if (
        data.corrected_valid[onset_index]
        and np.isfinite(data.pitch_st[onset_index])
    ):
        pitch_at_onset = float(
            data.pitch_st[onset_index]
        )
        pitch_error = (
            pitch_at_onset - float(note.pitch)
        )
    else:
        pitch_at_onset = np.nan
        pitch_error = np.nan

    target_time, target_pitch = nearest_pitch_target_sample(
        data,
        onset,
        note.pitch,
        config.event_search_ms / 1000.0,
    )

    if np.isfinite(target_time):
        target_delta_ms = signed_ms(
            target_time - onset
        )
        target_error = (
            target_pitch - float(note.pitch)
        )
    else:
        target_delta_ms = np.nan
        target_error = np.nan

    movement_start = find_movement_start(
        data,
        onset,
        config,
    )

    movement_delta = (
        signed_ms(movement_start - onset)
        if np.isfinite(movement_start)
        else np.nan
    )

    arrival = find_target_arrival(
        data,
        onset,
        note.pitch,
        config,
    )

    arrival_delta = (
        signed_ms(arrival - onset)
        if np.isfinite(arrival)
        else np.nan
    )

    settle = find_target_settle(
        data,
        onset,
        note.pitch,
        config,
    )

    settle_delta = (
        signed_ms(settle - onset)
        if np.isfinite(settle)
        else np.nan
    )

    valid_local = (
        local_mask
        & data.corrected_valid
        & np.isfinite(data.pitch_st)
    )

    local_values = data.pitch_st[valid_local]

    if len(local_values):
        local_min = float(np.min(local_values))
        local_max = float(np.max(local_values))
        local_span = local_max - local_min
    else:
        local_min = np.nan
        local_max = np.nan
        local_span = np.nan

    harmonic = harmonic_relation(
        pitch_at_onset,
        float(note.pitch),
        config.harmonic_near_st,
    )

    return LocalObservation(
        midi_note=note,
        aligned_onset_s=onset,
        nearest_event=nearest_event,
        nearest_event_delta_ms=nearest_event_delta_ms,
        valid_fraction=valid_fraction,
        pitch_at_onset_st=pitch_at_onset,
        pitch_error_at_onset_st=pitch_error,
        nearest_target_time_s=target_time,
        nearest_target_delta_ms=target_delta_ms,
        nearest_target_pitch_st=target_pitch,
        nearest_target_pitch_error_st=target_error,
        movement_start_s=movement_start,
        movement_start_delta_ms=movement_delta,
        target_arrival_s=arrival,
        target_arrival_delta_ms=arrival_delta,
        target_settle_s=settle,
        target_settle_delta_ms=settle_delta,
        local_min_st=local_min,
        local_max_st=local_max,
        local_span_st=local_span,
        harmonic_relation=harmonic,
    )


# ===========================================================================
# Reverse lookup: WAV events vs MIDI
# ===========================================================================

def nearest_midi_onset(
    notes: tuple[MidiNote, ...],
    wav_time_s: float,
    offset_s: float,
) -> tuple[Optional[MidiNote], float]:
    if not notes:
        return None, np.nan

    best = min(
        notes,
        key=lambda n: abs(
            (n.onset_s + offset_s) - wav_time_s
        ),
    )

    delta_ms = signed_ms(
        wav_time_s
        - (best.onset_s + offset_s)
    )

    return best, delta_ms


# ===========================================================================
# Reporting
# ===========================================================================

def print_header(
    midi_path: Path,
    csv_path: Path,
    midi: MidiData,
    data: ECKFData,
    events: list[WavEvent],
    offset_s: float,
    offset_score: float,
    sections: dict[str, tuple[float, float]],
):
    print("=" * 120)
    print("MIDI <-> WAV EVENT CORRESPONDENCE DIAGNOSTIC V1")
    print("=" * 120)

    print(f"MIDI:                 {midi_path}")
    print(f"ECKF CSV:             {csv_path}")
    print(f"MIDI type:            {midi.midi_type}")
    print(f"Ticks per beat:       {midi.ticks_per_beat}")
    print(f"MIDI notes:           {len(midi.notes)}")
    print(f"ECKF rows:            {len(data.time_s)}")
    print(
        f"ECKF duration:        "
        f"{data.time_s[-1] - data.time_s[0]:.3f} s"
    )
    print(
        f"Corrected valid rows: "
        f"{int(np.sum(data.corrected_valid))}"
    )
    print(f"Primitive WAV events: {len(events)}")
    print()

    print("GLOBAL SYNCHRONIZATION")
    print("-" * 120)
    print(
        "Convention:            "
        "WAV_time = MIDI_time + offset"
    )
    print(
        f"Best global offset:    {offset_s:+.3f} s"
    )
    print(
        f"Alignment score:       {offset_score:.4f}"
    )

    for name in ("EARLY", "MIDDLE", "LATE"):
        if name not in sections:
            continue

        section_offset, score = sections[name]

        print(
            f"{name.title():<22}"
            f"{section_offset:+.3f} s"
            f"   score={score:.4f}"
        )

    if (
        "EARLY" in sections
        and "LATE" in sections
    ):
        drift = (
            sections["LATE"][0]
            - sections["EARLY"][0]
        )

        print(
            f"Late - early drift:    "
            f"{1000.0 * drift:+.1f} ms"
        )

    print()


def print_note_observation(
    obs: LocalObservation,
):
    note = obs.midi_note

    print(
        f"GT {note.index:03d}  "
        f"{midi_name(note.pitch):4s} ({note.pitch:3d})"
    )

    print(
        f"  MIDI onset:             "
        f"{note.onset_s:8.3f} s"
    )
    print(
        f"  aligned WAV onset:      "
        f"{obs.aligned_onset_s:8.3f} s"
    )
    print(
        f"  MIDI duration:          "
        f"{1000.0 * note.duration_s:8.1f} ms"
    )

    if obs.nearest_event is None:
        print(
            "  nearest WAV event:      -"
        )
    else:
        e = obs.nearest_event

        print(
            f"  nearest WAV event:      "
            f"{e.time_s:8.3f} s  "
            f"{e.kind:<14s}  "
            f"delta={fmt_signed(obs.nearest_event_delta_ms)} ms"
        )

        if np.isfinite(e.interval_st):
            print(
                f"    local pitch change:   "
                f"{e.left_pitch_st:6.2f}"
                f" -> {e.right_pitch_st:6.2f}"
                f"  ({e.interval_st:+.2f} st)"
            )

    print(
        f"  local valid coverage:   "
        f"{100.0 * obs.valid_fraction:8.1f} %"
    )

    print(
        f"  pitch at MIDI onset:    "
        f"{fmt_float(obs.pitch_at_onset_st):>8s} st"
        f"   error="
        f"{fmt_signed(obs.pitch_error_at_onset_st):>7s} st"
    )

    print(
        f"  nearest intended pitch: "
        f"{fmt_float(obs.nearest_target_pitch_st):>8s} st"
        f"   delta="
        f"{fmt_signed(obs.nearest_target_delta_ms):>7s} ms"
        f"   error="
        f"{fmt_signed(obs.nearest_target_pitch_error_st):>7s} st"
    )

    print(
        f"  movement begins:        "
        f"{fmt_float(obs.movement_start_s, 3):>8s} s"
        f"   delta="
        f"{fmt_signed(obs.movement_start_delta_ms):>7s} ms"
    )

    print(
        f"  target approached:      "
        f"{fmt_float(obs.target_arrival_s, 3):>8s} s"
        f"   delta="
        f"{fmt_signed(obs.target_arrival_delta_ms):>7s} ms"
    )

    print(
        f"  target settled:         "
        f"{fmt_float(obs.target_settle_s, 3):>8s} s"
        f"   delta="
        f"{fmt_signed(obs.target_settle_delta_ms):>7s} ms"
    )

    print(
        f"  local pitch range:      "
        f"{fmt_float(obs.local_min_st):>6s}"
        f" .. {fmt_float(obs.local_max_st):>6s} st"
        f"   span={fmt_float(obs.local_span_st)} st"
    )

    print(
        f"  harmonic relation:      "
        f"{obs.harmonic_relation}"
    )

    print()


def print_distribution(
    label: str,
    values: Iterable[float],
):
    arr = np.asarray(
        [
            x for x in values
            if np.isfinite(x)
        ],
        dtype=float,
    )

    if len(arr) == 0:
        print(f"{label:<28} no observations")
        return

    print(
        f"{label:<28}"
        f"n={len(arr):4d}  "
        f"median={np.median(arr):+7.1f} ms  "
        f"p10={np.percentile(arr, 10):+7.1f}  "
        f"p90={np.percentile(arr, 90):+7.1f}"
    )


def print_summary(
    observations: list[LocalObservation],
):
    print("=" * 120)
    print("LOCAL TIMING DISTRIBUTIONS")
    print("=" * 120)

    print_distribution(
        "Nearest WAV event",
        (
            o.nearest_event_delta_ms
            for o in observations
        ),
    )

    print_distribution(
        "Nearest intended pitch",
        (
            o.nearest_target_delta_ms
            for o in observations
        ),
    )

    print_distribution(
        "Movement start",
        (
            o.movement_start_delta_ms
            for o in observations
        ),
    )

    print_distribution(
        "Target approach",
        (
            o.target_arrival_delta_ms
            for o in observations
        ),
    )

    print_distribution(
        "Target settle",
        (
            o.target_settle_delta_ms
            for o in observations
        ),
    )

    print()

    pitch_errors = np.asarray(
        [
            abs(o.pitch_error_at_onset_st)
            for o in observations
            if np.isfinite(o.pitch_error_at_onset_st)
        ],
        dtype=float,
    )

    target_errors = np.asarray(
        [
            abs(o.nearest_target_pitch_error_st)
            for o in observations
            if np.isfinite(
                o.nearest_target_pitch_error_st
            )
        ],
        dtype=float,
    )

    print("=" * 120)
    print("PITCH CORRESPONDENCE")
    print("=" * 120)

    if len(pitch_errors):
        print(
            f"Pitch at nominal onset:"
            f"   n={len(pitch_errors)}"
            f"   median abs error="
            f"{np.median(pitch_errors):.3f} st"
        )

    if len(target_errors):
        print(
            f"Nearest local target:"
            f"      n={len(target_errors)}"
            f"   median abs error="
            f"{np.median(target_errors):.3f} st"
        )

    harmonic_counts = {}

    for obs in observations:
        if obs.harmonic_relation == "-":
            continue

        harmonic_counts[obs.harmonic_relation] = (
            harmonic_counts.get(
                obs.harmonic_relation,
                0,
            )
            + 1
        )

    if harmonic_counts:
        print(
            "Harmonic/subharmonic observations:"
        )

        for relation, count in sorted(
            harmonic_counts.items()
        ):
            print(
                f"  {relation:>4s}: {count}"
            )
    else:
        print(
            "Harmonic/subharmonic observations: none"
        )

    print()


def print_reverse_events(
    midi: MidiData,
    events: list[WavEvent],
    offset_s: float,
):
    print("=" * 120)
    print("WAV EVENTS -> NEAREST GT MIDI ONSET")
    print("=" * 120)

    print(
        "This section deliberately does NOT decide whether an unmatched "
        "WAV event is an ornament or an error."
    )
    print()

    print(
        f"{'EV':>4s}  "
        f"{'WAV TIME':>9s}  "
        f"{'KIND':<14s}  "
        f"{'GT':>5s}  "
        f"{'NOTE':>5s}  "
        f"{'DELTA MS':>10s}  "
        f"{'LOCAL ΔP':>9s}"
    )
    print("-" * 80)

    for event in events:
        note, delta_ms = nearest_midi_onset(
            midi.notes,
            event.time_s,
            offset_s,
        )

        if note is None:
            gt_index = "-"
            note_name = "-"
        else:
            gt_index = str(note.index)
            note_name = midi_name(note.pitch)

        interval = (
            f"{event.interval_st:+.2f}"
            if np.isfinite(event.interval_st)
            else "-"
        )

        print(
            f"{event.index:4d}  "
            f"{event.time_s:9.3f}  "
            f"{event.kind:<14s}  "
            f"{gt_index:>5s}  "
            f"{note_name:>5s}  "
            f"{fmt_signed(delta_ms):>10s}  "
            f"{interval:>9s}"
        )

    print()


# ===========================================================================
# CSV export
# ===========================================================================

def export_observations(
    path: Path,
    observations: list[LocalObservation],
):
    fieldnames = [
        "gt_index",
        "midi_pitch",
        "midi_note_name",
        "midi_onset_s",
        "aligned_wav_onset_s",
        "midi_duration_ms",

        "nearest_wav_event_time_s",
        "nearest_wav_event_kind",
        "nearest_wav_event_delta_ms",

        "local_valid_fraction",

        "pitch_at_onset_st",
        "pitch_error_at_onset_st",

        "nearest_target_time_s",
        "nearest_target_delta_ms",
        "nearest_target_pitch_st",
        "nearest_target_pitch_error_st",

        "movement_start_s",
        "movement_start_delta_ms",

        "target_arrival_s",
        "target_arrival_delta_ms",

        "target_settle_s",
        "target_settle_delta_ms",

        "local_min_st",
        "local_max_st",
        "local_span_st",

        "harmonic_relation",
    ]

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for obs in observations:
            note = obs.midi_note
            event = obs.nearest_event

            writer.writerow(
                {
                    "gt_index": note.index,
                    "midi_pitch": note.pitch,
                    "midi_note_name": midi_name(
                        note.pitch
                    ),
                    "midi_onset_s": note.onset_s,
                    "aligned_wav_onset_s":
                        obs.aligned_onset_s,
                    "midi_duration_ms":
                        1000.0 * note.duration_s,

                    "nearest_wav_event_time_s":
                        (
                            event.time_s
                            if event is not None
                            else ""
                        ),
                    "nearest_wav_event_kind":
                        (
                            event.kind
                            if event is not None
                            else ""
                        ),
                    "nearest_wav_event_delta_ms":
                        obs.nearest_event_delta_ms,

                    "local_valid_fraction":
                        obs.valid_fraction,

                    "pitch_at_onset_st":
                        obs.pitch_at_onset_st,
                    "pitch_error_at_onset_st":
                        obs.pitch_error_at_onset_st,

                    "nearest_target_time_s":
                        obs.nearest_target_time_s,
                    "nearest_target_delta_ms":
                        obs.nearest_target_delta_ms,
                    "nearest_target_pitch_st":
                        obs.nearest_target_pitch_st,
                    "nearest_target_pitch_error_st":
                        obs.nearest_target_pitch_error_st,

                    "movement_start_s":
                        obs.movement_start_s,
                    "movement_start_delta_ms":
                        obs.movement_start_delta_ms,

                    "target_arrival_s":
                        obs.target_arrival_s,
                    "target_arrival_delta_ms":
                        obs.target_arrival_delta_ms,

                    "target_settle_s":
                        obs.target_settle_s,
                    "target_settle_delta_ms":
                        obs.target_settle_delta_ms,

                    "local_min_st":
                        obs.local_min_st,
                    "local_max_st":
                        obs.local_max_st,
                    "local_span_st":
                        obs.local_span_st,

                    "harmonic_relation":
                        obs.harmonic_relation,
                }
            )


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare GT MIDI event timing against independently "
            "discovered WAV/ECKF pitch trajectories."
        )
    )

    parser.add_argument(
        "eckf_csv",
        type=Path,
        help="Production *.wav.eckf.csv file",
    )

    parser.add_argument(
        "ground_truth_midi",
        type=Path,
        help="Ground-truth MIDI",
    )

    parser.add_argument(
        "--offset-min",
        type=float,
        default=-15.0,
    )

    parser.add_argument(
        "--offset-max",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--offset-step",
        type=float,
        default=0.010,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    config = Config(
        global_offset_min_s=args.offset_min,
        global_offset_max_s=args.offset_max,
        global_offset_step_s=args.offset_step,
    )

    midi = load_midi_notes(
        args.ground_truth_midi
    )

    data = load_eckf_csv(
        args.eckf_csv
    )

    events = extract_wav_events(
        data,
        config,
    )

    offset_s, offset_score = estimate_global_offset(
        midi,
        data,
        events,
        config,
    )

    sections = section_offsets(
        midi,
        data,
        events,
        offset_s,
    )

    observations = [
        observe_note(
            note,
            data,
            events,
            offset_s,
            config,
        )
        for note in midi.notes
    ]

    print_header(
        args.ground_truth_midi,
        args.eckf_csv,
        midi,
        data,
        events,
        offset_s,
        offset_score,
        sections,
    )

    print("=" * 120)
    print("GT MIDI EVENTS -> WAV OBSERVATIONS")
    print("=" * 120)
    print()

    for obs in observations:
        print_note_observation(obs)

    print_summary(
        observations
    )

    print_reverse_events(
        midi,
        events,
        offset_s,
    )

    if args.output_csv is None:
        output_csv = (
            args.eckf_csv.parent
            / (
                args.eckf_csv.stem
                + ".midi_wav_correspondence.csv"
            )
        )
    else:
        output_csv = args.output_csv

    export_observations(
        output_csv,
        observations,
    )

    print("=" * 120)
    print(f"Observation CSV: {output_csv}")
    print("=" * 120)


if __name__ == "__main__":
    main()