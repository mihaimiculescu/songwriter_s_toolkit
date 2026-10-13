from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - installations without Numba
    njit = None


@dataclass(frozen=True)
class KalmanBlockResult:
    f0_hz: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    fundamental: np.ndarray
    p_last: np.ndarray
    x_last: np.ndarray
    k_last: np.ndarray


def numba_available() -> bool:
    return njit is not None


def resolve_kalman_engine(requested: str = "auto") -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "python", "numba"}:
        raise ValueError("kalman engine must be one of: auto, python, numba")
    if requested == "python":
        return "python"
    if requested == "numba":
        if njit is None:
            raise RuntimeError(
                "Numba engine requested but numba is not installed. "
                "Install the R4 requirements or use --kalman-engine python."
            )
        return "numba"
    return "numba" if njit is not None else "python"


def _reference_block(
    y_frame: np.ndarray,
    p_last: np.ndarray,
    x_last: np.ndarray,
    ts: float,
    c: float,
    absolute_start_sample: int,
):
    """Literal block form of the frozen V3R9 sample recurrence."""
    y_frame = np.asarray(y_frame, dtype=np.float64)
    P_last = np.asarray(p_last, dtype=np.complex128).copy()
    x_last = np.asarray(x_last, dtype=np.complex128).reshape(3, 1).copy()

    H = np.array([[0.0, 0.5, 0.5]], dtype=np.complex128)
    K = np.zeros((3, 1), dtype=np.complex128)

    f0 = np.zeros(len(y_frame), dtype=np.float64)
    amp = np.zeros(len(y_frame), dtype=np.float64)
    phase = np.zeros(len(y_frame), dtype=np.float64)
    x_est = np.zeros(len(y_frame), dtype=np.complex128)

    for k in range(len(y_frame)):
        # These statements intentionally preserve the frozen equations and
        # high-level operation order.  Do not add fastmath/reassociation.
        numerator = P_last @ H.conj().T
        denominator = (H @ P_last @ H.conj().T)[0, 0] + 1.0
        K = numerator / denominator

        P = P_last - K @ H @ P_last

        sample = y_frame[k]
        predicted_measurement = (H @ x_last)[0, 0]
        innovation = sample - predicted_measurement
        x = x_last + K * innovation

        x1, x2, x3 = x[:, 0]
        x_next = np.array(
            [x1, x1 * x2, x3 / x1],
            dtype=np.complex128,
        ).reshape(3, 1)

        F = np.array(
            [
                [1.0, 0.0, 0.0],
                [x2, x1, 0.0],
                [-x3 / (x1 ** 2), 0.0, 1.0 / x1],
            ],
            dtype=np.complex128,
        )

        innovation_after_update = abs(sample - (H @ x)[0, 0])
        q = 10.0 ** (-(c - innovation_after_update))
        P_next = F @ P @ F.conj().T + q * np.eye(3, dtype=np.complex128)

        f0[k] = abs(np.log(x1) / (1j * ts * 2.0 * np.pi))
        amp[k] = abs(x2)
        if amp[k] > 0.0:
            n_matlab = absolute_start_sample + k + 1
            phase_complex = -1j * (
                np.log(x2 / amp[k])
                - (2.0 * np.pi * f0[k] * ts * n_matlab)
            )
            phase[k] = abs(phase_complex)
        x_est[k] = (H @ x)[0, 0]

        P_last = P_next
        x_last = x_next

    return f0, amp, phase, x_est, P_last, x_last, K


if njit is not None:
    @njit(cache=True, fastmath=False, error_model="numpy")
    def _numba_block(
        y_frame,
        p_last,
        x_last,
        ts,
        c,
        absolute_start_sample,
    ):
        # Numba-specific spelling of the same frozen equations.  Arrays are
        # allocated explicitly instead of from heterogeneous Python literals;
        # this avoids a Numba typing ambiguity without changing the recurrence.
        P_last = p_last.copy()
        x_last = x_last.copy()

        H = np.empty((1, 3), dtype=np.complex128)
        H[0, 0] = 0.0 + 0.0j
        H[0, 1] = 0.5 + 0.0j
        H[0, 2] = 0.5 + 0.0j
        Hh = H.conj().T
        K = np.zeros((3, 1), dtype=np.complex128)

        count = len(y_frame)
        f0 = np.zeros(count, dtype=np.float64)
        amp = np.zeros(count, dtype=np.float64)
        phase = np.zeros(count, dtype=np.float64)
        x_est = np.zeros(count, dtype=np.complex128)

        for k in range(count):
            numerator = P_last @ Hh
            denominator = (H @ P_last @ Hh)[0, 0] + 1.0
            K = numerator / denominator

            P = P_last - K @ H @ P_last

            sample = y_frame[k]
            predicted_measurement = (H @ x_last)[0, 0]
            innovation = sample - predicted_measurement
            x = x_last + K * innovation

            x1 = x[0, 0]
            x2 = x[1, 0]
            x3 = x[2, 0]

            x_next = np.empty((3, 1), dtype=np.complex128)
            x_next[0, 0] = x1
            x_next[1, 0] = x1 * x2
            x_next[2, 0] = x3 / x1

            F = np.zeros((3, 3), dtype=np.complex128)
            F[0, 0] = 1.0 + 0.0j
            F[1, 0] = x2
            F[1, 1] = x1
            F[2, 0] = -x3 / (x1 ** 2)
            F[2, 2] = 1.0 / x1

            innovation_after_update = abs(sample - (H @ x)[0, 0])
            q = 10.0 ** (-(c - innovation_after_update))
            P_next = F @ P @ F.conj().T + q * np.eye(3, dtype=np.complex128)

            f0[k] = abs(np.log(x1) / (1j * ts * 2.0 * np.pi))
            amp[k] = abs(x2)
            if amp[k] > 0.0:
                n_matlab = absolute_start_sample + k + 1
                phase_complex = -1j * (
                    np.log(x2 / amp[k])
                    - (2.0 * np.pi * f0[k] * ts * n_matlab)
                )
                phase[k] = abs(phase_complex)
            x_est[k] = (H @ x)[0, 0]

            P_last = P_next
            x_last = x_next

        return f0, amp, phase, x_est, P_last, x_last, K
else:
    _numba_block = None

def run_kalman_block(
    y_frame: np.ndarray,
    p_last: np.ndarray,
    x_last: np.ndarray,
    *,
    ts: float,
    c: float,
    absolute_start_sample: int,
    engine: str = "auto",
) -> KalmanBlockResult:
    """Run one sequential ECKF block with Python or cached Numba execution.

    The recurrence itself is intentionally never parallelized.  R4 compiles
    the serial numerical spine; later independent acoustic-evidence passes may
    use the separately detected worker budget.
    """
    resolved = resolve_kalman_engine(engine)
    fn = _numba_block if resolved == "numba" else _reference_block
    f0, amp, phase, x_est, p_out, x_out, k_out = fn(
        np.asarray(y_frame, dtype=np.float64),
        np.asarray(p_last, dtype=np.complex128),
        np.asarray(x_last, dtype=np.complex128).reshape(3, 1),
        float(ts),
        float(c),
        int(absolute_start_sample),
    )
    return KalmanBlockResult(
        f0_hz=f0,
        amplitude=amp,
        phase=phase,
        fundamental=x_est,
        p_last=p_out,
        x_last=x_out,
        k_last=k_out,
    )
