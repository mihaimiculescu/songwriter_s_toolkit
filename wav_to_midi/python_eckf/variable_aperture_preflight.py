from __future__ import annotations

"""R4 Stage 4 parallel acoustic preflight for variable-aperture reacquisition.

Workers only measure waveform evidence and tracker-boundary admissibility for
24/40/64/96-ms windows.  The V3R5 adjudicator remains serial and applies its
history-dependent transition prior only when the tracker actually requests a
reacquisition decision.
"""

from dataclasses import dataclass
import time
import numpy as np

from . import worker_state
from .traffic_control import (
    PersistentTrafficController,
    StageTrafficPlan,
    plan_variable_aperture_preflight,
)
from .variable_aperture_reacquisition import (
    APERTURES_MS,
    ApertureCandidate,
    _boundary_allows_aperture,
    _candidate_records_acoustic,
)
from .harmonic_change import HarmonicChangeDetector


@dataclass(frozen=True)
class VariableApertureFrameCache:
    frame_index: int
    start_sample: int
    aperture_rows: tuple[tuple[float, tuple[ApertureCandidate, ...]], ...]
    boundary_reason: str | None


@dataclass(frozen=True)
class VariableAperturePreflightCache:
    by_start_sample: dict[int, VariableApertureFrameCache]
    plan: StageTrafficPlan
    engine: str
    eligible_frames: int
    elapsed_seconds: float


def _measure_one(
    *,
    fi: int,
    start: int,
    y: np.ndarray,
    raw_y: np.ndarray,
    original_length: int,
    fs: float,
    config,
    silence_threshold: float,
    detector: HarmonicChangeDetector,
) -> VariableApertureFrameCache:
    block = int(config.block_size)
    rows: list[tuple[float, tuple[ApertureCandidate, ...]]] = []
    boundary_reason: str | None = None

    for aperture_ms in APERTURES_MS:
        width = max(3, int(round(float(aperture_ms) * fs / 1000.0)))
        end = start + width
        allowed, why = _boundary_allows_aperture(
            y=y,
            raw_y=raw_y,
            start=start,
            end=end,
            block=block,
            original_length=original_length,
            fs=fs,
            detector=detector,
            silence_threshold=silence_threshold,
            silence_flatness_threshold=config.silence_flatness_threshold,
        )
        if not allowed:
            boundary_reason = str(why)
            break
        acoustic = _candidate_records_acoustic(
            detector, y[start:end], fs, start
        )
        rows.append((float(aperture_ms), tuple(acoustic)))

    return VariableApertureFrameCache(
        frame_index=int(fi),
        start_sample=int(start),
        aperture_rows=tuple(rows),
        boundary_reason=boundary_reason,
    )


def _worker_task(task: tuple[int, int]) -> VariableApertureFrameCache:
    fi, start = task
    config = worker_state.config()
    return _measure_one(
        fi=int(fi),
        start=int(start),
        y=worker_state.tracker_normalized_padded(),
        raw_y=worker_state.tracker_raw_padded(),
        original_length=worker_state.original_length(),
        fs=worker_state.sample_rate(),
        config=config,
        silence_threshold=worker_state.silence_energy_threshold(),
        detector=worker_state.detector(),
    )


def build_variable_aperture_preflight(
    *,
    normalized_y: np.ndarray,
    raw_y: np.ndarray,
    original_length: int,
    sample_rate: float,
    config,
    silence_threshold: float,
    frame_periodicity,
    energy_silence: np.ndarray,
    runtime_resources,
    mode: str,
    traffic_controller: PersistentTrafficController | None = None,
    eligible_frame_indices: tuple[int, ...] | None = None,
) -> VariableAperturePreflightCache:
    if mode not in {"auto", "serial"}:
        raise ValueError("variable-aperture preflight mode must be auto or serial")

    nframes = len(raw_y) // int(config.block_size)
    tasks: list[tuple[int, int]] = []
    source_indices = (
        range(nframes) if eligible_frame_indices is None
        else eligible_frame_indices
    )
    for fi0 in source_indices:
        fi = int(fi0)
        if fi < 0 or fi >= nframes or bool(energy_silence[fi]):
            continue
        periodicity = frame_periodicity[fi]
        if periodicity is None or not bool(periodicity.voiced):
            continue
        tasks.append((fi, fi * int(config.block_size)))

    plan = plan_variable_aperture_preflight(
        runtime_resources, len(tasks), mode=mode
    )
    started = time.perf_counter()

    if (
        mode == "auto"
        and traffic_controller is not None
        and traffic_controller.available
        and plan.effective_workers > 1
        and tasks
    ):
        chunksize = max(1, len(tasks) // max(1, plan.task_count))
        rows = list(traffic_controller.map(
            _worker_task, tasks, chunksize=chunksize
        ))
        engine = "persistent_python_fork"
    else:
        detector = HarmonicChangeDetector(config)
        rows = [
            _measure_one(
                fi=fi,
                start=start,
                y=normalized_y,
                raw_y=raw_y,
                original_length=original_length,
                fs=sample_rate,
                config=config,
                silence_threshold=silence_threshold,
                detector=detector,
            )
            for fi, start in tasks
        ]
        engine = "python_serial"

    return VariableAperturePreflightCache(
        by_start_sample={int(row.start_sample): row for row in rows},
        plan=plan,
        engine=engine,
        eligible_frames=len(tasks),
        elapsed_seconds=float(time.perf_counter() - started),
    )
