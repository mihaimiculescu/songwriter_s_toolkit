from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

_MIB = 1024 * 1024


@dataclass(frozen=True)
class RuntimeResources:
    logical_cpus: int
    worker_budget: int
    available_ram_bytes: int | None
    memory_budget_bytes: int
    cpu_detection: str
    ram_detection: str

    @property
    def available_ram_mib(self) -> float | None:
        if self.available_ram_bytes is None:
            return None
        return self.available_ram_bytes / _MIB

    @property
    def memory_budget_mib(self) -> float:
        return self.memory_budget_bytes / _MIB


def _logical_cpu_count() -> tuple[int, str]:
    # Respect container/cgroup/CPU-affinity restrictions when the OS exposes them.
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity)), "sched_getaffinity"
    except (AttributeError, OSError):
        pass

    count = os.cpu_count()
    if count is not None and count > 0:
        return int(count), "os.cpu_count"

    return 4, "fallback_4"


def _available_ram_linux_proc() -> int | None:
    path = Path("/proc/meminfo")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) >= 2:
                    kib = int(fields[1])
                    if kib > 0:
                        return kib * 1024
    except (OSError, ValueError):
        return None
    return None


def _available_ram() -> tuple[int | None, str]:
    # psutil is deliberately optional.  Do not make the converter depend on it.
    try:
        import psutil  # type: ignore

        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available, "psutil"
    except Exception:
        pass

    available = _available_ram_linux_proc()
    if available is not None:
        return available, "/proc/meminfo"

    # POSIX fallback: available physical pages.  Some platforms do not expose
    # SC_AVPHYS_PAGES; failures deliberately fall through to the conservative
    # fixed-memory fallback below.
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available = pages * page_size
        if available > 0:
            return available, "sysconf"
    except (AttributeError, OSError, ValueError):
        pass

    return None, "fallback"


def detect_runtime_resources(
    *,
    cpu_fraction: float = 0.75,
    memory_fraction: float = 0.40,
    workers_override: int | None = None,
    memory_budget_mib_override: int | None = None,
) -> RuntimeResources:
    """Detect a conservative host budget for the WAV->MIDI worker.

    Detection is intentionally cheap enough to run once per conversion.  The
    values are *budgets*, not a promise that every stage will create that many
    threads.  Serial stages remain serial; parallel stages must stay below the
    returned worker and memory ceilings.
    """
    if not (0.0 < cpu_fraction <= 1.0):
        raise ValueError("cpu_fraction must be in (0, 1]")
    if not (0.0 < memory_fraction <= 1.0):
        raise ValueError("memory_fraction must be in (0, 1]")

    logical, cpu_detection = _logical_cpu_count()
    available_ram, ram_detection = _available_ram()

    if workers_override is None:
        workers = max(1, int(math.floor(logical * cpu_fraction)))
    else:
        if workers_override <= 0:
            raise ValueError("workers_override must be > 0")
        workers = min(int(workers_override), logical)

    if memory_budget_mib_override is not None:
        if memory_budget_mib_override <= 0:
            raise ValueError("memory_budget_mib_override must be > 0")
        memory_budget = int(memory_budget_mib_override) * _MIB
    elif available_ram is not None:
        # Use at most 40% of currently available memory.  This leaves substantial
        # headroom for the GUI, model workers, OS cache, and other toolkit engines.
        memory_budget = max(64 * _MIB, int(available_ram * memory_fraction))
        memory_budget = min(memory_budget, available_ram)
    else:
        # RAM detection failure must not turn into aggressive allocation.
        memory_budget = 512 * _MIB

    return RuntimeResources(
        logical_cpus=logical,
        worker_budget=workers,
        available_ram_bytes=available_ram,
        memory_budget_bytes=memory_budget,
        cpu_detection=cpu_detection,
        ram_detection=ram_detection,
    )
