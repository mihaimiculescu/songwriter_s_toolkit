from __future__ import annotations

"""Fork-inherited immutable worker state for R4 Stage 4.

The parent configures this module exactly once after tracker audio has been
prepared and before the persistent process pool is first started.  Linux fork
workers then inherit the arrays copy-on-write.  No musical decision state is
stored here.
"""

import numpy as np

from .harmonic_change import HarmonicChangeDetector

_AUDIO_ORIGINAL: np.ndarray | None = None
_TRACKER_RAW_PADDED: np.ndarray | None = None
_TRACKER_NORMALIZED_PADDED: np.ndarray | None = None
_SAMPLE_RATE: float | None = None
_CONFIG = None
_ORIGINAL_LENGTH: int | None = None
_SILENCE_ENERGY_THRESHOLD: float | None = None
_DETECTOR: HarmonicChangeDetector | None = None
_THREAD_LIMITER = None


def configure_parent_state(
    *,
    audio_original: np.ndarray,
    tracker_raw_padded: np.ndarray,
    tracker_normalized_padded: np.ndarray,
    sample_rate: float,
    config,
    original_length: int,
    silence_energy_threshold: float,
) -> None:
    """Publish immutable conversion inputs before the first worker fork."""
    global _AUDIO_ORIGINAL, _TRACKER_RAW_PADDED, _TRACKER_NORMALIZED_PADDED
    global _SAMPLE_RATE, _CONFIG, _ORIGINAL_LENGTH, _SILENCE_ENERGY_THRESHOLD
    global _DETECTOR

    _AUDIO_ORIGINAL = np.asarray(audio_original, dtype=np.float64)
    _TRACKER_RAW_PADDED = np.asarray(tracker_raw_padded, dtype=np.float64)
    _TRACKER_NORMALIZED_PADDED = np.asarray(tracker_normalized_padded, dtype=np.float64)
    _SAMPLE_RATE = float(sample_rate)
    _CONFIG = config
    _ORIGINAL_LENGTH = int(original_length)
    _SILENCE_ENERGY_THRESHOLD = float(silence_energy_threshold)
    _DETECTOR = None


def worker_initializer() -> None:
    """Initialize one persistent worker without creating nested CPU armies."""
    global _THREAD_LIMITER, _DETECTOR
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        _THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _THREAD_LIMITER = None

    if _CONFIG is not None:
        _DETECTOR = HarmonicChangeDetector(_CONFIG)


def audio_original() -> np.ndarray:
    if _AUDIO_ORIGINAL is None:
        raise RuntimeError("R4 worker audio_original was not configured before fork")
    return _AUDIO_ORIGINAL


def tracker_raw_padded() -> np.ndarray:
    if _TRACKER_RAW_PADDED is None:
        raise RuntimeError("R4 worker raw tracker audio was not configured before fork")
    return _TRACKER_RAW_PADDED


def tracker_normalized_padded() -> np.ndarray:
    if _TRACKER_NORMALIZED_PADDED is None:
        raise RuntimeError("R4 worker normalized tracker audio was not configured before fork")
    return _TRACKER_NORMALIZED_PADDED


def sample_rate() -> float:
    if _SAMPLE_RATE is None:
        raise RuntimeError("R4 worker sample rate was not configured before fork")
    return float(_SAMPLE_RATE)


def config():
    if _CONFIG is None:
        raise RuntimeError("R4 worker config was not configured before fork")
    return _CONFIG


def original_length() -> int:
    if _ORIGINAL_LENGTH is None:
        raise RuntimeError("R4 worker original length was not configured before fork")
    return int(_ORIGINAL_LENGTH)


def silence_energy_threshold() -> float:
    if _SILENCE_ENERGY_THRESHOLD is None:
        raise RuntimeError("R4 worker silence threshold was not configured before fork")
    return float(_SILENCE_ENERGY_THRESHOLD)


def detector() -> HarmonicChangeDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = HarmonicChangeDetector(config())
    return _DETECTOR
