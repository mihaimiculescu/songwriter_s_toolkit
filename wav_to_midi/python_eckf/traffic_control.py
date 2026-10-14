from __future__ import annotations

"""R4 execution traffic police.

This module owns concurrency policy for parallel-safe WAV->MIDI stages. Musical
logic must never depend on the policy chosen here.

Rules:
  * the host-wide budget comes from runtime_resources.py;
  * a stage may use fewer workers than the host budget, never more;
  * memory gets a vote before CPU count does;
  * each process worker is kept single-threaded internally, preventing nested
    BLAS/OpenMP oversubscription;
  * on Linux/POSIX immutable audio is inherited copy-on-write by fork workers
    instead of being copied once per worker;
  * any pool failure must degrade performance, never correctness.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp


from .runtime_resources import RuntimeResources
from .worker_state import worker_initializer

_MIB = 1024 * 1024


@dataclass(frozen=True)
class StageTrafficPlan:
    stage: str
    host_worker_budget: int
    effective_workers: int
    memory_worker_cap: int
    memory_budget_bytes: int
    fixed_reserve_bytes: int
    per_worker_reserve_bytes: int
    task_count: int
    nested_native_threads: int = 1

    @property
    def memory_budget_mib(self) -> float:
        return self.memory_budget_bytes / _MIB

    @property
    def fixed_reserve_mib(self) -> float:
        return self.fixed_reserve_bytes / _MIB

    @property
    def per_worker_reserve_mib(self) -> float:
        return self.per_worker_reserve_bytes / _MIB


def _plan_parallel_stage(
    resources: RuntimeResources,
    item_count: int,
    *,
    stage: str,
    mode: str,
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 4,
) -> StageTrafficPlan:
    if mode not in {"auto", "serial", "off"}:
        raise ValueError(f"{stage} mode must be auto, serial, or off")
    if item_count < 0:
        raise ValueError("item_count must be >= 0")

    budget = max(1, int(resources.memory_budget_bytes))
    fixed = min(256 * _MIB, max(1, budget // 2))
    per_worker = max(1, int(per_worker_reserve_mib) * _MIB)
    scratch = max(0, budget - fixed)
    memory_cap = max(1, scratch // per_worker) if scratch > 0 else 1

    if mode in {"off", "serial"} or item_count <= 1:
        workers = 1
    else:
        workers = min(
            max(1, int(resources.worker_budget)),
            max(1, int(memory_cap)),
            max(1, int(item_count)),
        )

    tasks = 0 if item_count == 0 else min(
        int(item_count),
        max(1, workers * max(1, int(max_tasks_per_worker))),
    )

    return StageTrafficPlan(
        stage=stage,
        host_worker_budget=int(resources.worker_budget),
        effective_workers=int(workers),
        memory_worker_cap=int(memory_cap),
        memory_budget_bytes=budget,
        fixed_reserve_bytes=int(fixed),
        per_worker_reserve_bytes=int(per_worker),
        task_count=int(tasks),
        nested_native_threads=1,
    )


def plan_frame_preflight(
    resources: RuntimeResources,
    nframes: int,
    *,
    mode: str = "auto",
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 4,
) -> StageTrafficPlan:
    """Plan the independent per-frame acoustic preflight."""
    return _plan_parallel_stage(
        resources,
        nframes,
        stage="frame_acoustic_preflight",
        mode=mode,
        per_worker_reserve_mib=per_worker_reserve_mib,
        max_tasks_per_worker=max_tasks_per_worker,
    )


def plan_pitch_candidate_field(
    resources: RuntimeResources,
    nframes: int,
    *,
    mode: str = "auto",
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 4,
) -> StageTrafficPlan:
    """Plan observational per-frame pitch-candidate measurements."""
    return _plan_parallel_stage(
        resources,
        nframes,
        stage="pitch_candidate_field",
        mode=mode,
        per_worker_reserve_mib=per_worker_reserve_mib,
        max_tasks_per_worker=max_tasks_per_worker,
    )


def plan_juror_bench(
    resources: RuntimeResources,
    npairs: int,
    *,
    mode: str = "auto",
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 8,
) -> StageTrafficPlan:
    """Plan independent juror pair acoustic measurements."""
    return _plan_parallel_stage(
        resources,
        npairs,
        stage="juror_pair_measurement",
        mode=mode,
        per_worker_reserve_mib=per_worker_reserve_mib,
        max_tasks_per_worker=max_tasks_per_worker,
    )


class PersistentTrafficController:
    """One bounded fork crew reused by every parallel-safe R4 stage.

    Stage 3 created and destroyed three separate 24-process pools.  Stage 4
    keeps one crew alive from acoustic preflight through juror measurement.
    The pool size is capped globally by the same conservative memory reserve
    used by per-stage planning.
    """

    def __init__(
        self,
        resources: RuntimeResources,
        *,
        per_worker_reserve_mib: int = 256,
    ) -> None:
        self.resources = resources
        budget = max(1, int(resources.memory_budget_bytes))
        fixed = min(256 * _MIB, max(1, budget // 2))
        per_worker = max(1, int(per_worker_reserve_mib) * _MIB)
        scratch = max(0, budget - fixed)
        memory_cap = max(1, scratch // per_worker) if scratch > 0 else 1
        self.memory_worker_cap = int(memory_cap)
        self.max_workers = int(min(
            max(1, int(resources.worker_budget)),
            max(1, int(memory_cap)),
        ))
        self.per_worker_reserve_mib = int(per_worker_reserve_mib)
        self._pool: ProcessPoolExecutor | None = None
        self._engine = "not_started"

    @property
    def available(self) -> bool:
        return self.max_workers > 1 and "fork" in mp.get_all_start_methods()

    @property
    def engine(self) -> str:
        return self._engine

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is not None:
            return self._pool
        if not self.available:
            raise RuntimeError("persistent fork crew unavailable")
        ctx = mp.get_context("fork")
        self._pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=ctx,
            initializer=worker_initializer,
        )
        self._engine = "persistent_python_fork"
        return self._pool

    def map(self, fn, iterable, *, chunksize: int = 1):
        pool = self._ensure_pool()
        return pool.map(fn, iterable, chunksize=max(1, int(chunksize)))

    def shutdown(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
        if self._engine != "not_started":
            self._engine = "stopped"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


def plan_variable_aperture_preflight(
    resources: RuntimeResources,
    nframes: int,
    *,
    mode: str = "auto",
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 4,
) -> StageTrafficPlan:
    """Plan independent variable-aperture acoustic measurements."""
    return _plan_parallel_stage(
        resources,
        nframes,
        stage="variable_aperture_preflight",
        mode=mode,
        per_worker_reserve_mib=per_worker_reserve_mib,
        max_tasks_per_worker=max_tasks_per_worker,
    )


def plan_harmonic_preflight(
    resources: RuntimeResources,
    nframes: int,
    *,
    mode: str = "auto",
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 4,
) -> StageTrafficPlan:
    """Plan independent harmonic detector frame measurements."""
    return _plan_parallel_stage(
        resources,
        nframes,
        stage="harmonic_preflight",
        mode=mode,
        per_worker_reserve_mib=per_worker_reserve_mib,
        max_tasks_per_worker=max_tasks_per_worker,
    )


def plan_initialization_period_preflight(
    resources: RuntimeResources,
    nframes: int,
    *,
    mode: str = "auto",
    per_worker_reserve_mib: int = 256,
    max_tasks_per_worker: int = 4,
) -> StageTrafficPlan:
    """Plan frozen measured-period initialization measurements."""
    return _plan_parallel_stage(
        resources,
        nframes,
        stage="initialization_period_preflight",
        mode=mode,
        per_worker_reserve_mib=per_worker_reserve_mib,
        max_tasks_per_worker=max_tasks_per_worker,
    )
