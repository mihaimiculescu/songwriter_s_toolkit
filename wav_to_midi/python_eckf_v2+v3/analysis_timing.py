from __future__ import annotations

from dataclasses import dataclass
import math


# Production default from V3R7 onward.  This controls only the OFFLINE
# interpretation trajectory.  The ECKF tracker itself remains full
# audio-sample resolution.
DEFAULT_OFFLINE_ANALYSIS_HZ: float = 135.0


@dataclass(frozen=True)
class OfflineAnalysisGrid:
    requested_hz: float
    actual_hz: float
    dt_s: float
    hop_samples: int
    sample_rate_hz: int


def resolve_offline_analysis_grid(
    sample_rate_hz: int,
    requested_hz: float = DEFAULT_OFFLINE_ANALYSIS_HZ,
) -> OfflineAnalysisGrid:
    """Resolve a requested trajectory rate onto an integer audio-sample hop.

    The actual rate, not the nominal request, is the authoritative value passed
    downstream.  For example, 135 Hz at 44.1 kHz becomes a 327-sample hop and
    therefore 134.862385... Hz.
    """
    sr = int(sample_rate_hz)
    hz = float(requested_hz)
    if sr <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if not math.isfinite(hz) or hz <= 0.0:
        raise ValueError("requested_hz must be finite and > 0")

    hop = max(1, int(round(sr / hz)))
    actual = float(sr) / float(hop)
    return OfflineAnalysisGrid(
        requested_hz=hz,
        actual_hz=actual,
        dt_s=1.0 / actual,
        hop_samples=hop,
        sample_rate_hz=sr,
    )
