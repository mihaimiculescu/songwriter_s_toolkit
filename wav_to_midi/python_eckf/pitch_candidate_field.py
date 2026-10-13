from __future__ import annotations

"""Observational pitch-candidate field for the future adjudication bench.

R4 Stage 3 adds execution-only parallelism between independent real frames.
The acoustic formulas and row ordering are unchanged.  The only cross-frame
quantity here -- the previous selected pitch used by transition_penalty -- is
reapplied serially in original frame order after workers finish.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import math
import multiprocessing as mp
import time
from typing import Iterable

import numpy as np

from .harmonic_change import HarmonicChangeDetector
from .initialization_candidates import (
    _measured_period_candidates,
    _harmonic_support,
    _measured_amplitude_phase,
)
from .trajectory_resolver import vocal_transition_penalty
from .traffic_control import plan_pitch_candidate_field, PersistentTrafficController
from . import worker_state


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


def _build_pitch_candidate_field_serial(
    *,
    audio: np.ndarray,
    sample_rate: float,
    config,
    tracker_result,
    frame_acoustic_state_name_fn,
) -> tuple[PitchCandidateEvidence, ...]:
    """Frozen V3R9 observational implementation."""
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
        if state_name not in {"VOICED_TRACKED", "VOICED_UNRESOLVED"}:
            continue

        frame = y[start:start + block]
        measured = _measured_period_candidates(frame, sample_rate)
        if not measured:
            continue

        proposals: list[tuple[str, float]] = []
        try:
            proposal = detector.analyze(None, frame, sample_rate)
            hz = _finite_positive(proposal.f0_hz)
            if hz is not None:
                proposals.append(("spectral_spacing", hz))
        except Exception:
            pass
        proposals.extend(("waveform_periodicity", float(p[0])) for p in measured)

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


# Fork-only Stage 3 worker state.  Audio/config are inherited copy-on-write.
_FORK_PC_AUDIO: np.ndarray | None = None
_FORK_PC_SAMPLE_RATE: float | None = None
_FORK_PC_CONFIG = None
_FORK_PC_DETECTOR: HarmonicChangeDetector | None = None
_FORK_PC_THREAD_LIMITER = None


def _pc_worker_init() -> None:
    global _FORK_PC_DETECTOR, _FORK_PC_THREAD_LIMITER
    if _FORK_PC_CONFIG is None:
        raise RuntimeError("pitch-candidate worker started without config")
    _FORK_PC_DETECTOR = HarmonicChangeDetector(_FORK_PC_CONFIG)
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        _FORK_PC_THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _FORK_PC_THREAD_LIMITER = None


def _pc_frame_task(task):
    fi, start, real_count, state_name, selected_hz = task
    y = _FORK_PC_AUDIO
    sr = _FORK_PC_SAMPLE_RATE
    detector = _FORK_PC_DETECTOR
    config = _FORK_PC_CONFIG
    if y is None or sr is None or detector is None or config is None:
        # Stage 4 persistent crew uses the common fork-inherited state.
        y = worker_state.audio_original()
        sr = worker_state.sample_rate()
        detector = worker_state.detector()
        config = worker_state.config()

    block = int(config.block_size)
    frame = y[start:start + block]
    measured = _measured_period_candidates(frame, sr)
    if not measured:
        return fi, False, selected_hz, float(start / sr), ()

    proposals: list[tuple[str, float]] = []
    try:
        proposal = detector.analyze(None, frame, sr)
        hz = _finite_positive(proposal.f0_hz)
        if hz is not None:
            proposals.append(("spectral_spacing", hz))
    except Exception:
        pass
    proposals.extend(("waveform_periodicity", float(p[0])) for p in measured)

    seen: set[tuple[str, int]] = set()
    frame_time_s = float(start / sr)
    out: list[PitchCandidateEvidence] = []

    for source, hz in proposals:
        if not (math.isfinite(hz) and 0 < hz < sr / 2):
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
        candidate_hz = float(representative[0] if source == "waveform_periodicity" else hz)
        dedupe_key = (source, int(round(candidate_hz * 1000.0)))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        harmonics = _harmonic_support(detector, frame, sr, candidate_hz)
        amplitude, _ = _measured_amplitude_phase(frame, sr, candidate_hz, int(start))
        midi, cents = _midi_and_cents(candidate_hz)
        chosen = bool(
            selected_hz is not None
            and abs(1200.0 * math.log2(candidate_hz / selected_hz)) <= 1.0
        )
        out.append(PitchCandidateEvidence(
            frame_index=int(fi),
            frame_start_sample=int(start),
            time_s=frame_time_s,
            frame_real_sample_count=int(real_count),
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
            transition_penalty=None,
            selected_by_initializer=chosen,
            initializer_selected_hz=selected_hz,
            note_group_midi=midi,
        ))

    return fi, True, selected_hz, frame_time_s, tuple(out)


def _build_pitch_candidate_field_parallel(
    *,
    audio: np.ndarray,
    sample_rate: float,
    config,
    tracker_result,
    frame_acoustic_state_name_fn,
    runtime_resources,
    parallel_mode: str,
    report: bool,
    traffic_controller: PersistentTrafficController | None = None,
) -> tuple[PitchCandidateEvidence, ...]:
    global _FORK_PC_AUDIO, _FORK_PC_SAMPLE_RATE, _FORK_PC_CONFIG

    y = np.asarray(audio, dtype=np.float64)
    block = int(config.block_size)
    starts = np.asarray(tracker_result.frame_start_samples, dtype=np.int64)
    real_counts = np.asarray(tracker_result.frame_real_sample_count, dtype=np.int64)
    states = np.asarray(tracker_result.frame_acoustic_state)
    selected = np.asarray(tracker_result.initialization_frequency_hz_per_frame, dtype=np.float64)

    tasks = []
    for fi, start0 in enumerate(starts):
        start = int(start0)
        real_count = int(real_counts[fi])
        if real_count < block or start < 0 or start + block > len(y):
            continue
        state_name = frame_acoustic_state_name_fn(states[fi])
        if state_name not in {"VOICED_TRACKED", "VOICED_UNRESOLVED"}:
            continue
        tasks.append((fi, start, real_count, state_name, _finite_positive(selected[fi])))

    plan = plan_pitch_candidate_field(
        runtime_resources, len(tasks), mode=parallel_mode
    )
    started = time.perf_counter()

    if plan.effective_workers <= 1 or "fork" not in mp.get_all_start_methods():
        rows = _build_pitch_candidate_field_serial(
            audio=audio,
            sample_rate=sample_rate,
            config=config,
            tracker_result=tracker_result,
            frame_acoustic_state_name_fn=frame_acoustic_state_name_fn,
        )
        if report:
            print(
                "R4 traffic police: candidate workers=1/"
                f"{plan.host_worker_budget}; engine=python_serial; "
                f"elapsed={time.perf_counter() - started:.3f}s"
            )
        return rows

    _FORK_PC_AUDIO = y
    _FORK_PC_SAMPLE_RATE = float(sample_rate)
    _FORK_PC_CONFIG = config
    try:
        chunksize = max(1, len(tasks) // max(1, plan.task_count))
        if traffic_controller is not None and traffic_controller.available:
            bundles = list(traffic_controller.map(
                _pc_frame_task, tasks, chunksize=chunksize
            ))
            engine = "persistent_python_fork"
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=plan.effective_workers,
                mp_context=ctx,
                initializer=_pc_worker_init,
            ) as pool:
                bundles = list(pool.map(_pc_frame_task, tasks, chunksize=chunksize))
            engine = "python_fork"

        # Reapply the only cross-frame value serially in frozen frame order.
        rows: list[PitchCandidateEvidence] = []
        previous_selected_hz: float | None = None
        previous_selected_time_s: float | None = None
        for _fi, measured_nonempty, selected_hz, frame_time_s, frame_rows in bundles:
            for row in frame_rows:
                penalty = None
                if previous_selected_hz is not None and previous_selected_time_s is not None:
                    elapsed_ms = max(0.0, (frame_time_s - previous_selected_time_s) * 1000.0)
                    if elapsed_ms > 0:
                        penalty = float(vocal_transition_penalty(
                            12.0 * math.log2(row.candidate_hz / previous_selected_hz),
                            elapsed_ms,
                        ))
                rows.append(replace(row, transition_penalty=penalty))
            if measured_nonempty and selected_hz is not None:
                previous_selected_hz = selected_hz
                previous_selected_time_s = frame_time_s

        if report:
            print(
                "R4 traffic police: "
                f"candidate workers={plan.effective_workers}/{plan.host_worker_budget}; "
                f"RAM cap permits {plan.memory_worker_cap}; engine={engine}; "
                f"frames={len(tasks)}; elapsed={time.perf_counter() - started:.3f}s"
            )
        return tuple(rows)
    except Exception as exc:
        rows = _build_pitch_candidate_field_serial(
            audio=audio,
            sample_rate=sample_rate,
            config=config,
            tracker_result=tracker_result,
            frame_acoustic_state_name_fn=frame_acoustic_state_name_fn,
        )
        if report:
            print(
                "R4 traffic police: candidate engine=python_serial_fallback; "
                f"reason={type(exc).__name__}; elapsed={time.perf_counter() - started:.3f}s"
            )
        return rows
    finally:
        _FORK_PC_AUDIO = None
        _FORK_PC_SAMPLE_RATE = None
        _FORK_PC_CONFIG = None


def build_pitch_candidate_field(
    *,
    audio: np.ndarray,
    sample_rate: float,
    config,
    tracker_result,
    frame_acoustic_state_name_fn,
    runtime_resources=None,
    parallel_mode: str = "auto",
    report: bool = False,
    traffic_controller: PersistentTrafficController | None = None,
) -> tuple[PitchCandidateEvidence, ...]:
    """Enumerate acoustically measured candidate frequencies per real frame.

    With runtime_resources supplied, R4 Stage 3 may execute independent frame
    measurements in a bounded fork pool.  `parallel_mode=serial` is the exact
    frozen Stage-2 path for A/B regression.
    """
    if parallel_mode not in {"auto", "serial", "off"}:
        raise ValueError("candidate parallel mode must be auto, serial, or off")
    if runtime_resources is None or parallel_mode in {"serial", "off"}:
        started = time.perf_counter()
        rows = _build_pitch_candidate_field_serial(
            audio=audio,
            sample_rate=sample_rate,
            config=config,
            tracker_result=tracker_result,
            frame_acoustic_state_name_fn=frame_acoustic_state_name_fn,
        )
        if report:
            print(
                "R4 traffic police: candidate workers=1; engine=python_serial; "
                f"elapsed={time.perf_counter() - started:.3f}s"
            )
        return rows
    return _build_pitch_candidate_field_parallel(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
        tracker_result=tracker_result,
        frame_acoustic_state_name_fn=frame_acoustic_state_name_fn,
        runtime_resources=runtime_resources,
        parallel_mode=parallel_mode,
        report=report,
        traffic_controller=traffic_controller,
    )
