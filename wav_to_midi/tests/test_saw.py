"""
Diagnostic equivalent of eckf_pitch_final/test/test_saw.m.

Known instantaneous F0:

    f(t) = 440 + 25*cos(2*pi*5*t)

so ground truth spans 415..465 Hz.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.signal import sawtooth

from python_eckf import ECKFConfig, track_pitch


def awgn_measured(x: np.ndarray, snr_db: float, rng) -> np.ndarray:
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    noise = rng.normal(
        0.0,
        np.sqrt(noise_power),
        size=x.shape,
    )

    return x + noise


def calculate_metrics(est, truth):
    valid = np.isfinite(est) & (est > 0.0)

    e = est[valid] - truth[valid]

    return {
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "std": float(np.std(e, ddof=1)),
        "count": int(np.sum(valid)),
    }


def best_lag(
    est,
    truth,
    fs,
    max_lag_ms=50.0,
    trim_ms=100.0,
):
    """
    Find the temporal shift of EST that minimizes MAE against truth.

    Positive lag means the ECKF estimate occurs later than the truth.
    """

    max_lag = int(round(max_lag_ms * 1e-3 * fs))
    trim = int(round(trim_ms * 1e-3 * fs))

    start = trim
    stop = min(len(est), len(truth)) - trim

    est0 = est[start:stop]
    truth0 = truth[start:stop]

    best = None

    for lag in range(-max_lag, max_lag + 1):

        if lag > 0:
            # estimate is delayed
            e = est0[lag:]
            t = truth0[:-lag]

        elif lag < 0:
            # estimate is early
            e = est0[:lag]
            t = truth0[-lag:]

        else:
            e = est0
            t = truth0

        valid = (
            np.isfinite(e)
            & (e > 0.0)
            & np.isfinite(t)
        )

        if np.sum(valid) < 100:
            continue

        mae = float(
            np.mean(
                np.abs(
                    e[valid] - t[valid]
                )
            )
        )

        if best is None or mae < best["mae"]:
            best = {
                "lag_samples": lag,
                "lag_ms": 1000.0 * lag / fs,
                "mae": mae,
            }

    return best


def run_one(label, signal, truth, fs, cfg):
    result = track_pitch(signal, fs, cfg)

    est = result.f0_hz[:len(truth)]

    fit = fit_vibrato(
        est,
        fs,
    )

    metrics = calculate_metrics(est, truth)
    lag = best_lag(est, truth, fs)

    print(
        f"{label:>6}  "
        f"MAE={metrics['mae']:8.4f} Hz  "
        f"bias={metrics['bias']:8.4f} Hz  "
        f"std={metrics['std']:8.4f} Hz  "
        f"voiced={metrics['count']:6d}  "
        f"best_lag={lag['lag_ms']:8.3f} ms  "
        f"aligned_MAE={lag['mae']:8.4f} Hz"
        f"fit_center={fit['offset']:8.3f} Hz  "
        f"fit_amp={fit['amplitude']:7.3f} Hz  "
        f"fit_phase={fit['phase_deg']:7.2f} deg  "
        f"fit_lag={fit['lag_ms']:7.3f} ms"
    )

def fit_vibrato(est, fs, fm=5.0, trim_ms=150.0):
    trim = int(round(trim_ms * 1e-3 * fs))

    indices = np.arange(len(est), dtype=np.int64)

    keep = slice(
        trim,
        len(est) - trim,
    )

    est2 = est[keep]
    idx2 = indices[keep]

    valid = (
        np.isfinite(est2)
        & (est2 > 0.0)
    )

    est2 = est2[valid]
    t = idx2[valid] / fs

    w = 2.0 * np.pi * fm

    X = np.column_stack([
        np.ones_like(t),
        np.cos(w * t),
        np.sin(w * t),
    ])

    coeff, _, _, _ = np.linalg.lstsq(
        X,
        est2,
        rcond=None,
    )

    offset = coeff[0]
    a = coeff[1]
    b = coeff[2]

    amplitude = np.sqrt(
        a*a + b*b
    )

    phase_rad = np.arctan2(
        b,
        a,
    )

    phase_deg = np.degrees(
        phase_rad
    )

    lag_ms = (
        phase_rad
        / (2.0 * np.pi * fm)
        * 1000.0
    )

    return {
        "offset": float(offset),
        "amplitude": float(amplitude),
        "phase_deg": float(phase_deg),
        "lag_ms": float(lag_ms),
    }

def main():
    fs = 44100
    duration = 2.0

    t = (
        np.arange(
            int(fs * duration),
            dtype=np.float64,
        )
        / fs
    )

    f0 = 440.0
    fm = 5.0
    am = 5.0

    phase = (
        2.0 * np.pi * f0 * t
        + am * np.sin(
            2.0 * np.pi * fm * t
        )
    )

    x = sawtooth(phase)

    truth = (
        f0
        + am * fm
        * np.cos(
            2.0 * np.pi * fm * t
        )
    )

    print()
    print("=== ECKF SAW DIAGNOSTIC ===")
    print(
        f"Ground truth: {truth.min():.3f} .. "
        f"{truth.max():.3f} Hz"
    )
    print()

    for mode in ("matlab", "offline"):

        print()
        print("=" * 80)
        print(f"MODE: {mode}")
        print("=" * 80)

        cfg = ECKFConfig(
            block_size=1024,
            c=10.0,
            num_buf_to_wait=3,
            npeaks=5,
            nsemitones=2.0,
            mode=mode,
            vocal_floor_hz=60.0,
        )

        run_one(
            "clean",
            x,
            truth,
            fs,
            cfg,
        )

        for i, snr_db in enumerate(
            [5, 10, 15, 20, 25]
        ):
            rng = np.random.default_rng(1000 + i)

            xn = awgn_measured(
                x,
                snr_db,
                rng,
            )

            run_one(
                f"{snr_db}dB",
                xn,
                truth,
                fs,
                cfg,
            )

if __name__ == "__main__":
    main()