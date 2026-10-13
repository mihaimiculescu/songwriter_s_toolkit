from __future__ import annotations

"""R4 Stage 6 exact preflight for initialization measured-period candidates.

Workers run the historical _measured_period_candidates() function unchanged.
Only predicted initialization frames are premeasured.  The serial tracker keeps
all transition priors, harmonic-family ranking, octave reconciliation, aperture
adjudication, and lookahead decisions.  Any cache miss falls back to the frozen
on-demand function.
"""

from dataclasses import dataclass
import time
import numpy as np

from . import worker_state
from .initialization_candidates import _measured_period_candidates
from .traffic_control import (
    PersistentTrafficController,
    StageTrafficPlan,
    plan_initialization_period_preflight,
)


@dataclass(frozen=True)
class InitializationPeriodFrame:
    frame_index: int
    start_sample: int
    measured: tuple[tuple[float, int, float, float], ...]


@dataclass(frozen=True)
class InitializationPeriodPreflightCache:
    by_start_sample: dict[int, InitializationPeriodFrame]
    plan: StageTrafficPlan
    engine: str
    eligible_frames: int
    elapsed_seconds: float


def _measure_one(fi: int, start: int, y: np.ndarray, fs: float, block: int):
    frame = y[start:start + block]
    measured = _measured_period_candidates(frame, fs)
    return InitializationPeriodFrame(
        frame_index=int(fi),
        start_sample=int(start),
        measured=tuple(measured),
    )


def _worker_task(task: tuple[int, int]) -> InitializationPeriodFrame:
    fi, start = task
    cfg = worker_state.config()
    return _measure_one(
        int(fi), int(start),
        worker_state.tracker_normalized_padded(),
        worker_state.sample_rate(),
        int(cfg.block_size),
    )


def build_initialization_period_preflight(
    *,
    normalized_y: np.ndarray,
    sample_rate: float,
    block_size: int,
    runtime_resources,
    mode: str,
    eligible_frame_indices: tuple[int, ...] | None,
    traffic_controller: PersistentTrafficController | None = None,
) -> InitializationPeriodPreflightCache:
    if mode not in {"auto", "serial"}:
        raise ValueError("initialization-period preflight mode must be auto or serial")

    nframes = len(normalized_y) // int(block_size)
    source = range(nframes) if eligible_frame_indices is None else eligible_frame_indices
    tasks = [
        (int(fi), int(fi) * int(block_size))
        for fi in source
        if 0 <= int(fi) < nframes
    ]
    plan = plan_initialization_period_preflight(
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
        rows = list(traffic_controller.map(_worker_task, tasks, chunksize=chunksize))
        engine = "persistent_python_fork"
    else:
        rows = [
            _measure_one(fi, start, normalized_y, sample_rate, int(block_size))
            for fi, start in tasks
        ]
        engine = "python_serial"

    return InitializationPeriodPreflightCache(
        by_start_sample={int(r.start_sample): r for r in rows},
        plan=plan,
        engine=engine,
        eligible_frames=len(tasks),
        elapsed_seconds=float(time.perf_counter() - started),
    )
