"""Independent, waveform-based periodicity assessment.

No pitch correction, interpolation, continuity prior, or amplitude threshold.
A rejected frame has no F0.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PeriodicityResult:
    voiced: bool
    acf_peak: float
    cmndf_minimum: float
    acf_frequency_hz: float | None
    cmndf_frequency_hz: float | None
    reason: str


def assess_periodicity(
    frame: np.ndarray,
    sample_rate: float,
    *,
    min_f0_hz: float = 70.0,
    max_f0_hz: float = 1000.0,
    min_acf: float = 0.70,
    max_cmndf: float = 0.25,
    max_disagreement_cents: float = 50.0,
) -> PeriodicityResult:
    """Assess whether a frame supports a defensible periodic frequency.

    Thresholds are provisional and must be validated on additional material.
    They do not modify the audio or replace the proposed F0.
    """
    x = np.asarray(frame, dtype=np.float64)

    if x.ndim != 1 or x.size < 3:
        raise ValueError("Expected a one-dimensional audio frame")

    if not np.all(np.isfinite(x)):
        raise ValueError("Frame contains NaN or Inf")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if not (0 < min_f0_hz < max_f0_hz):
        raise ValueError("Invalid F0 range")

    x = x - np.mean(x)

    min_lag = max(1, int(np.floor(sample_rate / max_f0_hz)))
    max_lag = int(np.ceil(sample_rate / min_f0_hz))

    if max_lag >= len(x):
        raise ValueError(
            "Frame too short for requested minimum F0"
        )

    acf = np.full(max_lag + 1, np.nan)
    difference = np.zeros(max_lag + 1)

    for lag in range(1, max_lag + 1):
        left = x[:-lag]
        right = x[lag:]

        energy = np.sqrt(
            np.dot(left, left) * np.dot(right, right)
        )

        if energy > 0:
            acf[lag] = np.dot(left, right) / energy

        delta = left - right
        difference[lag] = np.dot(delta, delta)

    cumulative = np.cumsum(difference[1:])
    cmndf = np.ones(max_lag + 1)

    for lag in range(1, max_lag + 1):
        if cumulative[lag - 1] > 0:
            cmndf[lag] = (
                difference[lag]
                * lag
                / cumulative[lag - 1]
            )

    search = slice(min_lag, max_lag + 1)

    acf_values = acf[search]
    cmndf_values = cmndf[search]

    if not np.any(np.isfinite(acf_values)):
        return PeriodicityResult(
            False, 0.0, 1.0, None, None,
            "no_valid_autocorrelation",
        )

    acf_lag = min_lag + int(np.nanargmax(acf_values))
    cmndf_lag = min_lag + int(np.argmin(cmndf_values))

    acf_peak = float(acf[acf_lag])
    cmndf_minimum = float(cmndf[cmndf_lag])

    acf_frequency = sample_rate / acf_lag
    cmndf_frequency = sample_rate / cmndf_lag

    disagreement = abs(
        1200.0 * np.log2(
            acf_frequency / cmndf_frequency
        )
    )

    reasons = []

    if acf_peak < min_acf:
        reasons.append("weak_autocorrelation")

    if cmndf_minimum > max_cmndf:
        reasons.append("weak_difference_periodicity")

    if disagreement > max_disagreement_cents:
        reasons.append("period_disagreement")

    return PeriodicityResult(
        voiced=not reasons,
        acf_peak=acf_peak,
        cmndf_minimum=cmndf_minimum,
        acf_frequency_hz=float(acf_frequency),
        cmndf_frequency_hz=float(cmndf_frequency),
        reason="accepted" if not reasons else ",".join(reasons),
    )