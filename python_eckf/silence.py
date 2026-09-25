from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .matlab_compat import matlab_pwelch_default


@dataclass(frozen=True)
class SilenceCalibration:
    mode: str
    threshold: float
    method: str
    block_size: int
    frame_ms: float
    min_silence_ms: float
    min_silence_frames: int
    search_quantile_percent: float | None
    candidate_energy_ceiling: float | None
    long_run_count: int
    selected_frame_count: int
    selected_duration_s: float
    selected_energy_median: float | None
    selected_energy_p95: float | None
    headroom_stat_units: float


def _energy_statistic(frame: np.ndarray) -> float:
    """Historical MATLAB energy statistic: 20*log10(sum(frame**2))."""
    frame = np.asarray(frame, dtype=np.float64).reshape(-1)
    sumsq = float(np.sum(frame * frame))
    return -np.inf if sumsq <= 0.0 else float(20.0 * np.log10(sumsq))


def is_silent(
    x: np.ndarray,
    flatness_threshold: float = 0.45,
    energy_db_threshold: float = -50.0,
):
    """
    Translation of eckf_pitch_final/is_silent.m.

    MATLAB source:
        [psdw,~] = pwelch(x);
        energy = 20*log10(sum(x.^2));
        spectral_flatness = geomean(psdw)/mean(psdw);
        silent = spectral_flatness >= 0.45 | energy < threshold;

    IMPORTANT: energy_db is the literal MATLAB 20*log10(sum of squares)
    statistic, NOT RMS dBFS.  Offline V2 can now calibrate its threshold
    once per recording; MATLAB/fixed mode can still use the historical -50.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    psd = matlab_pwelch_default(x)

    energy_db = _energy_statistic(x)

    if psd.size == 0:
        spectral_flatness = 0.0
    else:
        mean_psd = float(np.mean(psd))
        if mean_psd <= 0.0:
            spectral_flatness = 0.0
        elif np.any(psd <= 0.0):
            spectral_flatness = 0.0
        else:
            spectral_flatness = float(np.exp(np.mean(np.log(psd))) / mean_psd)

    silent = bool(
        spectral_flatness >= flatness_threshold
        or energy_db < energy_db_threshold
    )
    return silent, spectral_flatness, energy_db


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start,end) runs of True values."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def calibrate_silence_threshold(
    audio: np.ndarray,
    sample_rate: float,
    block_size: int,
    *,
    min_silence_ms: float = 500.0,
    search_quantiles: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0),
    headroom_stat_units: float = 6.0,
) -> SilenceCalibration:
    """Estimate one energy threshold from genuinely long quiet regions.

    This intentionally stays simple and recording-level:

    1. Measure the historical MATLAB energy statistic on real ECKF frames.
    2. Starting with the quietest 10% of finite frames, look for contiguous
       runs lasting at least ``min_silence_ms``.  Relax only as far as the
       quietest 40% if necessary.
    3. Treat those long quiet runs as the recording's noise-floor sample.
    4. Threshold = 95th percentile of their finite energies + a small
       headroom.  ``+6`` on the historical 20*log10(sum-of-squares) statistic
       corresponds to about +3 dB on an ordinary amplitude-dB scale.

    No pitch result, GroundTruth MIDI, fixed -50 cutoff, or future adjudicator
    verdict is used to estimate this threshold.
    """
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or block_size <= 0:
        raise ValueError("sample_rate and block_size must be positive")

    nframes = len(y) // block_size
    frame_ms = 1000.0 * block_size / sample_rate
    min_frames = max(1, int(math.ceil(min_silence_ms / frame_ms)))

    if nframes <= 0:
        # Empty/too-short input: there is no evidence from which to calibrate.
        # Keep a deterministic extremely-low threshold rather than borrowing
        # the historical -50 constant.
        return SilenceCalibration(
            mode="adaptive", threshold=-300.0, method="no_full_frames",
            block_size=block_size, frame_ms=frame_ms,
            min_silence_ms=min_silence_ms, min_silence_frames=min_frames,
            search_quantile_percent=None, candidate_energy_ceiling=None,
            long_run_count=0, selected_frame_count=0, selected_duration_s=0.0,
            selected_energy_median=None, selected_energy_p95=None,
            headroom_stat_units=headroom_stat_units,
        )

    frames = y[: nframes * block_size].reshape(nframes, block_size)
    sumsq = np.sum(frames * frames, axis=1)
    energies = np.full(nframes, -np.inf, dtype=np.float64)
    positive = sumsq > 0.0
    energies[positive] = 20.0 * np.log10(sumsq[positive])

    finite = energies[np.isfinite(energies)]
    if finite.size == 0:
        return SilenceCalibration(
            mode="adaptive", threshold=-300.0, method="digital_silence",
            block_size=block_size, frame_ms=frame_ms,
            min_silence_ms=min_silence_ms, min_silence_frames=min_frames,
            search_quantile_percent=None, candidate_energy_ceiling=None,
            long_run_count=1, selected_frame_count=nframes,
            selected_duration_s=nframes * block_size / sample_rate,
            selected_energy_median=None, selected_energy_p95=None,
            headroom_stat_units=headroom_stat_units,
        )

    chosen_q = None
    chosen_ceiling = None
    chosen_runs: list[tuple[int, int]] = []
    for q in search_quantiles:
        ceiling = float(np.percentile(finite, q))
        quiet = energies <= ceiling
        long_runs = [r for r in _true_runs(quiet) if (r[1] - r[0]) >= min_frames]
        if long_runs:
            chosen_q = float(q)
            chosen_ceiling = ceiling
            chosen_runs = long_runs
            break

    method = "long_quiet_runs"
    if not chosen_runs:
        # Rare fallback: if the recording genuinely contains no 500 ms quiet
        # plateau, use the longest run found at the loosest search quantile.
        # This remains per-file and is reported explicitly in the audit.
        chosen_q = float(search_quantiles[-1])
        chosen_ceiling = float(np.percentile(finite, chosen_q))
        all_runs = _true_runs(energies <= chosen_ceiling)
        if all_runs:
            longest = max(all_runs, key=lambda r: r[1] - r[0])
            chosen_runs = [longest]
            method = "longest_quiet_run_fallback"
        else:
            chosen_runs = []
            method = "quiet_percentile_fallback"

    if chosen_runs:
        selected_idx = np.concatenate([
            np.arange(start, end, dtype=np.int64) for start, end in chosen_runs
        ])
        selected = energies[selected_idx]
    else:
        selected = energies[energies <= chosen_ceiling]
        selected_idx = np.flatnonzero(energies <= chosen_ceiling)

    selected_finite = selected[np.isfinite(selected)]
    if selected_finite.size:
        median = float(np.median(selected_finite))
        p95 = float(np.percentile(selected_finite, 95.0))
        threshold = p95 + float(headroom_stat_units)
    else:
        # Selected region is exact digital zero. Any measurable non-zero frame
        # should be above the silence floor.
        median = None
        p95 = None
        threshold = -300.0

    return SilenceCalibration(
        mode="adaptive",
        threshold=float(threshold),
        method=method,
        block_size=block_size,
        frame_ms=frame_ms,
        min_silence_ms=float(min_silence_ms),
        min_silence_frames=min_frames,
        search_quantile_percent=chosen_q,
        candidate_energy_ceiling=chosen_ceiling,
        long_run_count=len(chosen_runs),
        selected_frame_count=int(len(selected_idx)),
        selected_duration_s=float(len(selected_idx) * block_size / sample_rate),
        selected_energy_median=median,
        selected_energy_p95=p95,
        headroom_stat_units=float(headroom_stat_units),
    )


def resolve_silence_calibration(config, audio=None, sample_rate=None) -> SilenceCalibration:
    """Resolve one silence calibration per recording."""
    frame_ms = 1000.0 * config.block_size / sample_rate if sample_rate else float("nan")
    if config.silence_mode == "fixed":
        return SilenceCalibration(
            mode="fixed",
            threshold=float(config.silence_energy_db_threshold),
            method="fixed_historical_threshold",
            block_size=config.block_size,
            frame_ms=frame_ms,
            min_silence_ms=0.0,
            min_silence_frames=0,
            search_quantile_percent=None,
            candidate_energy_ceiling=None,
            long_run_count=0,
            selected_frame_count=0,
            selected_duration_s=0.0,
            selected_energy_median=None,
            selected_energy_p95=None,
            headroom_stat_units=0.0,
        )
    if config.silence_mode == "adaptive":
        if audio is None or sample_rate is None:
            raise ValueError("adaptive silence calibration requires audio and sample_rate")
        return calibrate_silence_threshold(
            audio, float(sample_rate), int(config.block_size)
        )
    raise ValueError(f"Unknown silence_mode: {config.silence_mode!r}")


def resolve_silence_energy_threshold(config, audio=None, sample_rate=None) -> float:
    """Backward-compatible scalar helper."""
    return float(resolve_silence_calibration(config, audio, sample_rate).threshold)
