from __future__ import annotations

import numpy as np

from .kalman_accel import resolve_kalman_engine, run_kalman_block
from .runtime_resources import detect_runtime_resources


def main() -> None:
    resources = detect_runtime_resources()
    engine = resolve_kalman_engine("auto")

    print(
        "R4 resources: "
        f"logical CPUs={resources.logical_cpus}; "
        f"worker budget={resources.worker_budget}; "
        f"memory budget={resources.memory_budget_mib:.0f} MiB"
    )
    print(f"R4 Kalman engine: {engine}")

    if engine != "numba":
        print("Numba is unavailable; nothing to precompile.")
        return

    sr = 48000.0
    ts = 1.0 / sr
    f0 = 220.0
    amplitude = 0.25
    phase = 0.0
    n_matlab = 1
    x_last = np.array(
        [
            np.exp(1j * 2.0 * np.pi * f0 * ts),
            amplitude * np.exp(1j * (2.0 * np.pi * f0 * n_matlab * ts + phase)),
            amplitude * np.exp(-1j * (2.0 * np.pi * f0 * n_matlab * ts + phase)),
        ],
        dtype=np.complex128,
    ).reshape(3, 1)
    p_last = np.zeros((3, 3), dtype=np.complex128)
    t = np.arange(2048, dtype=np.float64) / sr
    y = 0.25 * np.sin(2.0 * np.pi * f0 * t)

    run_kalman_block(
        y,
        p_last,
        x_last,
        ts=ts,
        c=7.0,
        absolute_start_sample=0,
        engine="numba",
    )
    print("R4 cached Kalman kernel is ready.")


if __name__ == "__main__":
    main()
