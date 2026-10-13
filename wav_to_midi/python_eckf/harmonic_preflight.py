from __future__ import annotations

"""R4 Stage 4 cache for pure HarmonicChangeDetector frame measurements."""

from dataclasses import dataclass
import time
import numpy as np

from . import worker_state
from .harmonic_change import HarmonicAnalysis, HarmonicChangeDetector
from .traffic_control import (
    PersistentTrafficController,
    StageTrafficPlan,
    plan_harmonic_preflight,
)


@dataclass(frozen=True)
class HarmonicFramePreflight:
    frame_index: int
    start_sample: int
    single: HarmonicAnalysis | None
    pair: HarmonicAnalysis | None


@dataclass(frozen=True)
class HarmonicPreflightCache:
    by_frame_index: dict[int, HarmonicFramePreflight]
    plan: StageTrafficPlan
    engine: str
    eligible_frames: int
    elapsed_seconds: float


def _safe_analyze(detector, prev, cur, fs):
    try:
        return detector.analyze(prev, cur, fs)
    except Exception:
        # A frame the serial tracker never reaches must not make preflight fail.
        # If the tracker later needs this frame it will fall back to its frozen
        # on-demand call, preserving historical failure/decision semantics.
        return None


def _measure_one(fi, start, y, fs, config, detector):
    block = int(config.block_size)
    cur = y[start:start + block]
    single = _safe_analyze(detector, None, cur, fs)
    pair = None
    if fi > 0:
        prev = y[start - block:start]
        pair = _safe_analyze(detector, prev, cur, fs)
    return HarmonicFramePreflight(int(fi), int(start), single, pair)


def _worker_task(task):
    fi, start = task
    return _measure_one(
        fi,
        start,
        worker_state.tracker_normalized_padded(),
        worker_state.sample_rate(),
        worker_state.config(),
        worker_state.detector(),
    )


def build_harmonic_preflight(
    *,
    normalized_y: np.ndarray,
    config,
    sample_rate: float,
    frame_periodicity,
    energy_silence: np.ndarray,
    runtime_resources,
    mode: str,
    traffic_controller: PersistentTrafficController | None = None,
) -> HarmonicPreflightCache:
    if mode not in {"auto", "serial"}:
        raise ValueError("harmonic preflight mode must be auto or serial")
    block = int(config.block_size)
    nframes = len(normalized_y) // block
    tasks=[]
    for fi in range(nframes):
        if bool(energy_silence[fi]):
            continue
        periodicity = frame_periodicity[fi]
        if periodicity is None or not bool(periodicity.voiced):
            continue
        tasks.append((fi, fi * block))

    plan=plan_harmonic_preflight(runtime_resources,len(tasks),mode=mode)
    started=time.perf_counter()
    if (
        mode == "auto"
        and traffic_controller is not None
        and traffic_controller.available
        and plan.effective_workers > 1
        and tasks
    ):
        chunksize=max(1,len(tasks)//max(1,plan.task_count))
        rows=list(traffic_controller.map(_worker_task,tasks,chunksize=chunksize))
        engine="persistent_python_fork"
    else:
        detector=HarmonicChangeDetector(config)
        rows=[_measure_one(fi,start,normalized_y,sample_rate,config,detector) for fi,start in tasks]
        engine="python_serial"

    return HarmonicPreflightCache(
        by_frame_index={r.frame_index:r for r in rows},
        plan=plan,
        engine=engine,
        eligible_frames=len(tasks),
        elapsed_seconds=float(time.perf_counter()-started),
    )


def predicted_initialization_frames(
    *,
    nframes: int,
    frame_periodicity,
    energy_silence: np.ndarray,
    harmonic_cache: HarmonicPreflightCache,
) -> tuple[int, ...]:
    """Predict the frozen initialization gate absent an initialization failure.

    Any unpredicted runtime initialization simply falls back to the original
    on-demand aperture path, so prediction can only affect speed, not logic.
    """
    out=[]
    previous_active=False
    harm_prev=0
    for fi in range(nframes):
        periodicity=frame_periodicity[fi]
        active=(
            not bool(energy_silence[fi])
            and periodicity is not None
            and bool(periodicity.voiced)
        )
        if not active:
            previous_active=False
            harm_prev=0
            continue
        row=harmonic_cache.by_frame_index.get(fi)
        if not previous_active:
            out.append(fi)
            harm_cur=0 if row is None or row.single is None else int(row.single.flag)
        else:
            harm_cur=0 if row is None or row.pair is None else int(row.pair.flag)
            if harm_prev == 0 and harm_cur == 1:
                out.append(fi)
        harm_prev=harm_cur
        previous_active=True
    return tuple(out)
