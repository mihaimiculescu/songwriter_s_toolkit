from __future__ import annotations

"""Observational pitch-candidate field for the future adjudication bench.

This module does NOT change ECKF tracking, validity, target formation, gesture
structure, or interpretation candidates.  It re-opens each real offline frame
and records the acoustically measured pitch alternatives that were available
there, preserving enough evidence for the later range / interval / spectral /
temporal jurors and the conditional Harmonic Detective.

No candidate is promoted here.  The initializer's already-recorded winner is
only marked for provenance.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .harmonic_change import HarmonicChangeDetector
from .initialization_candidates import (
    _measured_period_candidates,
    _harmonic_support,
    _measured_amplitude_phase,
)
from .trajectory_resolver import vocal_transition_penalty


@dataclass(frozen=True)
class PitchCandidateEvidence:
    frame_index: int
    frame_start_sample: int
    time_s: float
    frame_real_sample_count: int
    acoustic_state_name: str
    source: str
    candidate_hz: float
    nearest_midi: int
    cents_from_nearest_midi: float
    measured_period_hz: float
    measured_period_lag: int
    measured_period_acf: float
    measured_period_cmndf: float
    spectral_harmonic_count: int
    spectral_harmonics: tuple[int, ...]
    measured_component_amplitude: float
    transition_penalty: float | None
    selected_by_initializer: bool
    initializer_selected_hz: float | None
    note_group_midi: int


def _midi_and_cents(hz: float) -> tuple[int, float]:
    midi_float = 69.0 + 12.0 * math.log2(hz / 440.0)
    midi = int(math.floor(midi_float + 0.5))
    center = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    cents = 1200.0 * math.log2(hz / center)
    return midi, cents


def _finite_positive(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def build_pitch_candidate_field(
    *,
    audio: np.ndarray,
    sample_rate: float,
    config,
    tracker_result,
    frame_acoustic_state_name_fn,
) -> tuple[PitchCandidateEvidence, ...]:
    """Enumerate acoustically measured candidate frequencies per real frame.

    Candidate construction intentionally mirrors the ingredients used by the
    offline initializer but does not re-run its dominance / ranking decision.
    Therefore this export keeps alternatives that a later tribunal may need to
    compare instead of silently discarding them.
    """

    y = np.asarray(audio, dtype=np.float64)
    block = int(config.block_size)
    detector = HarmonicChangeDetector(config)
    rows: list[PitchCandidateEvidence] = []

    starts = np.asarray(tracker_result.frame_start_samples, dtype=np.int64)
    real_counts = np.asarray(tracker_result.frame_real_sample_count, dtype=np.int64)
    states = np.asarray(tracker_result.frame_acoustic_state)
    selected = np.asarray(tracker_result.initialization_frequency_hz_per_frame, dtype=np.float64)

    previous_selected_hz: float | None = None
    previous_selected_time_s: float | None = None

    for fi, start in enumerate(starts):
        real_count = int(real_counts[fi])
        if real_count < block or start < 0 or start + block > len(y):
            continue

        state_name = frame_acoustic_state_name_fn(states[fi])
        # Candidate evidence only makes sense for frames the acoustic passport
        # considered voiced/tracked or voiced/unresolved.  Silence/unvoiced
        # remain explicitly absent rather than manufacturing alternatives.
        if state_name not in {"VOICED_TRACKED", "VOICED_UNRESOLVED"}:
            continue

        frame = y[start:start + block]
        measured = _measured_period_candidates(frame, sample_rate)
        if not measured:
            continue

        proposals: list[tuple[str, float]] = []
        # Spectral-spacing proposal is observational; failure simply means the
        # waveform-periodicity candidates remain available.
        try:
            proposal = detector.analyze(None, frame, sample_rate)
            hz = _finite_positive(proposal.f0_hz)
            if hz is not None:
                proposals.append(("spectral_spacing", hz))
        except Exception:
            pass
        proposals.extend(("waveform_periodicity", float(p[0])) for p in measured)

        # Deduplicate same source/frequency to avoid repeated rows caused by a
        # spectral proposal landing exactly on a measured period candidate.
        seen: set[tuple[str, int]] = set()
        selected_hz = _finite_positive(selected[fi])
        frame_time_s = float(start / sample_rate)

        for source, hz in proposals:
            if not (math.isfinite(hz) and 0 < hz < sample_rate / 2):
                continue
            matching = [
                p for p in measured
                if abs(1200.0 * math.log2(hz / p[0])) <= 50.0
            ]
            if not matching:
                continue
            representative = min(
                matching,
                key=lambda p: abs(1200.0 * math.log2(hz / p[0])),
            )
            # Use the measured-period frequency as the stable candidate identity
            # for waveform_periodicity; retain actual spectral proposal otherwise.
            candidate_hz = float(representative[0] if source == "waveform_periodicity" else hz)
            dedupe_key = (source, int(round(candidate_hz * 1000.0)))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            harmonics = _harmonic_support(detector, frame, sample_rate, candidate_hz)
            amplitude, _ = _measured_amplitude_phase(
                frame, sample_rate, candidate_hz, int(start)
            )
            midi, cents = _midi_and_cents(candidate_hz)

            transition_penalty = None
            if previous_selected_hz is not None and previous_selected_time_s is not None:
                elapsed_ms = max(0.0, (frame_time_s - previous_selected_time_s) * 1000.0)
                if elapsed_ms > 0:
                    transition_penalty = float(vocal_transition_penalty(
                        12.0 * math.log2(candidate_hz / previous_selected_hz),
                        elapsed_ms,
                    ))

            chosen = bool(
                selected_hz is not None
                and abs(1200.0 * math.log2(candidate_hz / selected_hz)) <= 1.0
            )
            rows.append(PitchCandidateEvidence(
                frame_index=int(fi),
                frame_start_sample=int(start),
                time_s=frame_time_s,
                frame_real_sample_count=real_count,
                acoustic_state_name=state_name,
                source=source,
                candidate_hz=candidate_hz,
                nearest_midi=midi,
                cents_from_nearest_midi=float(cents),
                measured_period_hz=float(representative[0]),
                measured_period_lag=int(representative[1]),
                measured_period_acf=float(representative[2]),
                measured_period_cmndf=float(representative[3]),
                spectral_harmonic_count=len(harmonics),
                spectral_harmonics=tuple(int(h) for h in harmonics),
                measured_component_amplitude=float(amplitude),
                transition_penalty=transition_penalty,
                selected_by_initializer=chosen,
                initializer_selected_hz=selected_hz,
                note_group_midi=midi,
            ))

        if selected_hz is not None:
            previous_selected_hz = selected_hz
            previous_selected_time_s = frame_time_s

    return tuple(rows)
