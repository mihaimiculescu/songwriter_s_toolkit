from __future__ import annotations

"""Behavior-preserving per-frame acoustic preflight for R4 Stage 2.

The expensive frozen periodicity function is parallelized *between frames*.
Each worker executes periodicity.assess_periodicity unchanged, so the numerical
path and tie behavior remain the V3R9 reference implementation.

On POSIX/Linux we use fork workers so the normalized WAV is inherited
copy-on-write instead of serialized once per worker.  Unsupported/failing hosts
fall back to the exact serial path.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import time
import numpy as np

from .periodicity import PeriodicityResult, assess_periodicity
from .silence import is_silent
from .traffic_control import StageTrafficPlan, PersistentTrafficController
from . import worker_state


@dataclass(frozen=True)
class FramePreflightCache:
    spectral_flatness: np.ndarray
    energy_db: np.ndarray
    original_silence: np.ndarray
    energy_silence: np.ndarray
    flatness_silence: np.ndarray
    periodicity: tuple[PeriodicityResult | None, ...]
    plan: StageTrafficPlan
    periodicity_engine: str
    silence_elapsed_seconds: float
    periodicity_elapsed_seconds: float
    elapsed_seconds: float


# Fork workers inherit these references copy-on-write.  They are deliberately
# module-private and populated only for the lifetime of one preflight pool.
_FORK_NORMALIZED_Y: np.ndarray | None = None
_FORK_SAMPLE_RATE: float | None = None
_FORK_BLOCK_SIZE: int | None = None
_FORK_THREAD_LIMITER = None


def _worker_init() -> None:
    """Keep each process single-threaded internally: workers are the parallelism."""
    global _FORK_THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        # Keep the controller alive for the whole child process.  This prevents
        # OpenBLAS/MKL/OpenMP from multiplying N process workers by M threads.
        _FORK_THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _FORK_THREAD_LIMITER = None


def _periodicity_chunk(indices: tuple[int, ...]):
    y = _FORK_NORMALIZED_Y
    sr = _FORK_SAMPLE_RATE
    block = _FORK_BLOCK_SIZE
    if y is None or sr is None or block is None:
        # Stage 4 persistent crew inherits the common worker state instead of
        # stage-private globals.  Stage-3 one-off pools still use the globals.
        y = worker_state.tracker_normalized_padded()
        sr = worker_state.sample_rate()
        block = int(worker_state.config().block_size)

    out = []
    for fi in indices:
        start = fi * block
        r = assess_periodicity(y[start:start + block], sr)
        out.append((
            fi,
            bool(r.voiced),
            float(r.acf_peak),
            float(r.cmndf_minimum),
            r.acf_frequency_hz,
            r.cmndf_frequency_hz,
            str(r.reason),
        ))
    return out


def _split_indices(indices: np.ndarray, ntasks: int) -> list[tuple[int, ...]]:
    if len(indices) == 0:
        return []
    ntasks = max(1, min(int(ntasks), len(indices)))
    return [
        tuple(int(v) for v in part)
        for part in np.array_split(indices, ntasks)
        if len(part)
    ]


def _parallel_periodicity_fork(
    normalized_y: np.ndarray,
    active_indices: np.ndarray,
    sample_rate: float,
    block_size: int,
    plan: StageTrafficPlan,
    traffic_controller: PersistentTrafficController | None = None,
) -> tuple[list[PeriodicityResult | None], str]:
    global _FORK_NORMALIZED_Y, _FORK_SAMPLE_RATE, _FORK_BLOCK_SIZE

    nframes = len(normalized_y) // block_size
    results: list[PeriodicityResult | None] = [None] * nframes
    if len(active_indices) == 0:
        return results, "python_fork"

    if "fork" not in mp.get_all_start_methods() or plan.effective_workers <= 1:
        for fi in active_indices:
            start = int(fi) * block_size
            results[int(fi)] = assess_periodicity(
                normalized_y[start:start + block_size], sample_rate
            )
        return results, "python_serial"

    # Globals are assigned before ProcessPoolExecutor creates its fork workers.
    # No worker argument contains the WAV itself.
    _FORK_NORMALIZED_Y = normalized_y
    _FORK_SAMPLE_RATE = float(sample_rate)
    _FORK_BLOCK_SIZE = int(block_size)
    chunks = _split_indices(active_indices, plan.task_count)

    try:
        if traffic_controller is not None and traffic_controller.available:
            mapped = traffic_controller.map(_periodicity_chunk, chunks, chunksize=1)
            engine = "persistent_python_fork"
        else:
            ctx = mp.get_context("fork")
            pool = ProcessPoolExecutor(
                max_workers=plan.effective_workers,
                mp_context=ctx,
                initializer=_worker_init,
            )
            mapped = pool.map(_periodicity_chunk, chunks)
            engine = "python_fork"

        try:
            # map yields in submitted order. Completion timing therefore never
            # leaks into the frozen serial musical logic.
            for chunk_result in mapped:
                for fi, voiced, acf, cmndf, af, cf, reason in chunk_result:
                    results[fi] = PeriodicityResult(
                        voiced=voiced,
                        acf_peak=acf,
                        cmndf_minimum=cmndf,
                        acf_frequency_hz=af,
                        cmndf_frequency_hz=cf,
                        reason=reason,
                    )
        finally:
            if traffic_controller is None or not traffic_controller.available:
                pool.shutdown(wait=True, cancel_futures=False)
        return results, engine
    except Exception:
        # Resource/pool failure must degrade performance, never correctness.
        for fi in active_indices:
            start = int(fi) * block_size
            results[int(fi)] = assess_periodicity(
                normalized_y[start:start + block_size], sample_rate
            )
        return results, "python_serial_fallback"
    finally:
        _FORK_NORMALIZED_Y = None
        _FORK_SAMPLE_RATE = None
        _FORK_BLOCK_SIZE = None


def build_frame_preflight(
    *,
    raw_y: np.ndarray,
    normalized_y: np.ndarray,
    sample_rate: float,
    block_size: int,
    silence_flatness_threshold: float,
    silence_energy_threshold: float,
    plan: StageTrafficPlan,
    mode: str,
    traffic_controller: PersistentTrafficController | None = None,
) -> FramePreflightCache:
    if mode not in {"auto", "serial"}:
        raise ValueError("preflight cache mode must be auto or serial")

    started = time.perf_counter()
    raw = np.asarray(raw_y, dtype=np.float64)
    norm = np.asarray(normalized_y, dtype=np.float64)
    if raw.ndim != 1 or norm.ndim != 1 or raw.shape != norm.shape:
        raise ValueError("preflight expects equally sized one-dimensional audio arrays")
    if block_size <= 0 or len(raw) % block_size:
        raise ValueError("preflight audio must be padded to an integer frame count")

    nframes = len(raw) // block_size
    spf = np.zeros(nframes, dtype=np.float64)
    energy = np.zeros(nframes, dtype=np.float64)
    original_silence = np.zeros(nframes, dtype=np.bool_)
    energy_silence = np.zeros(nframes, dtype=np.bool_)
    flatness_silence = np.zeros(nframes, dtype=np.bool_)

    # Spectral-flatness silence is comparatively cheap and runs through SciPy's
    # Welch implementation.  Keep this frozen path serial; the heavy ACF/CMNDF
    # work below is where the CPU budget pays off safely.
    silence_started = time.perf_counter()
    for fi in range(nframes):
        start = fi * block_size
        end = start + block_size
        silent, cur_spf, cur_energy = is_silent(
            raw[start:end],
            flatness_threshold=silence_flatness_threshold,
            energy_db_threshold=silence_energy_threshold,
        )
        spf[fi] = float(cur_spf)
        energy[fi] = float(cur_energy)
        original_silence[fi] = bool(silent)
        energy_silence[fi] = bool(cur_energy < silence_energy_threshold)
        flatness_silence[fi] = bool(cur_spf >= silence_flatness_threshold)
    silence_elapsed = time.perf_counter() - silence_started

    active_indices = np.flatnonzero(~energy_silence)
    periodicity_started = time.perf_counter()
    if mode == "serial":
        periodicity: list[PeriodicityResult | None] = [None] * nframes
        for fi in active_indices:
            start = int(fi) * block_size
            periodicity[int(fi)] = assess_periodicity(
                norm[start:start + block_size], sample_rate
            )
        engine = "python_serial"
    else:
        periodicity, engine = _parallel_periodicity_fork(
            norm, active_indices, sample_rate, block_size, plan,
            traffic_controller=traffic_controller,
        )
    periodicity_elapsed = time.perf_counter() - periodicity_started

    return FramePreflightCache(
        spectral_flatness=spf,
        energy_db=energy,
        original_silence=original_silence,
        energy_silence=energy_silence,
        flatness_silence=flatness_silence,
        periodicity=tuple(periodicity),
        plan=plan,
        periodicity_engine=engine,
        silence_elapsed_seconds=float(silence_elapsed),
        periodicity_elapsed_seconds=float(periodicity_elapsed),
        elapsed_seconds=float(time.perf_counter() - started),
    )
