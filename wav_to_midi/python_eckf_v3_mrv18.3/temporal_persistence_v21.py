from __future__ import annotations

"""Exact V21-style shifted temporal persistence measurement.

This is a direct port of the read-only V21 temporal probe geometry into the
active V2 juror bench.  It measures candidate periodicity directly from the
original waveform at 24/40/64 ms apertures, shifted by -8/0/+8 ms around the
pair timestamp.  The historical ACF support threshold and persistence labels
are preserved.

It does *not* select F0 by itself.  It only supplies the temporal juror with
pairwise acoustic persistence evidence.
"""

from dataclasses import dataclass
import math
import numpy as np

WIDTHS_MS = (24, 40, 64)
SHIFTS_MS = (-8.0, 0.0, 8.0)
ACF_SUPPORT_THRESHOLD = 0.72
MIN_CYCLES = 3.0
PERSISTENT_MIN_SUPPORT = 6
NONPERSISTENT_MAX_SUPPORT = 2
MIN_PAIRED_MEASUREMENTS = 3


@dataclass(frozen=True)
class V21TemporalResult:
    low_hz: float
    high_hz: float
    paired_measurements: int
    low_supported_shift_count: int
    high_supported_shift_count: int
    low_acf_median: float | None
    high_acf_median: float | None
    pattern: str
    low_component: float
    high_component: float


def _segment(audio: np.ndarray, sr: int, center_s: float, width_ms: float):
    a = round((center_s - width_ms / 2000.0) * sr)
    b = round((center_s + width_ms / 2000.0) * sr)
    if a < 0 or b > len(audio) or b - a < 64:
        return None
    return audio[a:b]


def _coherence(x, sr: int, hz: float):
    """Historical V21 overlap-normalized lag coherence."""
    if x is None or hz <= 0:
        return None
    z = np.asarray(x, dtype=np.float64)
    z = z - z.mean()
    n = len(z)
    cycles = n * hz / sr
    if cycles < MIN_CYCLES:
        return None
    if float(z @ z) < 1e-18:
        return None
    lo = max(2, round(sr / hz * 0.975))
    hi = min(n // 2, round(sr / hz * 1.025))
    if hi < lo:
        return None
    best = -float("inf")
    for lag in range(lo, hi + 1):
        a, b = z[:-lag], z[lag:]
        den = math.sqrt(float(a @ a) * float(b @ b))
        score = float(a @ b) / den if den > 1e-18 else 0.0
        if score > best:
            best = score
    return float(best)


def measure_v21_temporal_pair(
    audio: np.ndarray,
    sample_rate: int,
    time_s: float,
    hz_a: float,
    hz_b: float,
) -> V21TemporalResult:
    low_hz, high_hz = sorted((float(hz_a), float(hz_b)))
    low_acfs: list[float] = []
    high_acfs: list[float] = []

    for width_ms in WIDTHS_MS:
        for shift_ms in SHIFTS_MS:
            x = _segment(
                audio,
                sample_rate,
                float(time_s) + shift_ms / 1000.0,
                width_ms,
            )
            low = _coherence(x, sample_rate, low_hz)
            high = _coherence(x, sample_rate, high_hz)
            # V21 only counted a shifted view when both candidates were testable.
            if low is not None and high is not None:
                low_acfs.append(low)
                high_acfs.append(high)

    paired = len(low_acfs)
    low_support = sum(v >= ACF_SUPPORT_THRESHOLD for v in low_acfs)
    high_support = sum(v >= ACF_SUPPORT_THRESHOLD for v in high_acfs)

    if paired < MIN_PAIRED_MEASUREMENTS:
        pattern = "insufficient_paired_shift_measurements"
        low_component = high_component = 0.0
    elif low_support >= PERSISTENT_MIN_SUPPORT and high_support >= PERSISTENT_MIN_SUPPORT:
        pattern = "both_integer_periods_shift_persistent"
        low_component = high_component = 0.0
    elif high_support >= PERSISTENT_MIN_SUPPORT and low_support <= NONPERSISTENT_MAX_SUPPORT:
        pattern = "higher_period_persistent_lower_not"
        low_component, high_component = -1.0, 1.0
    elif low_support >= PERSISTENT_MIN_SUPPORT and high_support <= NONPERSISTENT_MAX_SUPPORT:
        pattern = "lower_period_persistent_higher_not"
        low_component, high_component = 1.0, -1.0
    else:
        pattern = "temporal_support_mixed_or_weak"
        low_component = high_component = 0.0

    return V21TemporalResult(
        low_hz=low_hz,
        high_hz=high_hz,
        paired_measurements=paired,
        low_supported_shift_count=int(low_support),
        high_supported_shift_count=int(high_support),
        low_acf_median=(float(np.median(low_acfs)) if low_acfs else None),
        high_acf_median=(float(np.median(high_acfs)) if high_acfs else None),
        pattern=pattern,
        low_component=low_component,
        high_component=high_component,
    )

@dataclass(frozen=True)
class V21CandidateShiftACF:
    hz: float
    testable_views: int
    supported_views: int
    acf_median: float | None
    acf_min: float | None
    acf_max: float | None


def measure_v21_candidate_shift_acf(audio, sample_rate, time_s, hz):
    """Direct V21 shifted-window ACF field for one candidate."""
    vals = []
    for width_ms in WIDTHS_MS:
        for shift_ms in SHIFTS_MS:
            x = _segment(
                np.asarray(audio, dtype=np.float64), int(sample_rate),
                float(time_s) + shift_ms / 1000.0, width_ms,
            )
            v = _coherence(x, int(sample_rate), float(hz))
            if v is not None:
                vals.append(float(v))
    return V21CandidateShiftACF(
        hz=float(hz),
        testable_views=len(vals),
        supported_views=sum(v >= ACF_SUPPORT_THRESHOLD for v in vals),
        acf_median=(float(np.median(vals)) if vals else None),
        acf_min=(float(min(vals)) if vals else None),
        acf_max=(float(max(vals)) if vals else None),
    )
