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
    raw_calculated_threshold: float | None
    minimum_allowed_threshold: float
    minimum_threshold_applied: bool
    digital_mute_frame_count: int
    digital_mute_duration_s: float
    trustworthy_noise_floor: bool
    trustworthy_run_count: int
    rejected_nonstationary_run_count: int
    stationarity_max_spread_db: float
    stationarity_max_abs_slope_db_per_s: float
    trust_failure_reason: str | None


def _energy_statistic(frame: np.ndarray) -> float:
    """Historical MATLAB energy statistic: 20*log10(sum(frame**2))."""
    frame = np.asarray(frame, dtype=np.float64).reshape(-1)
    sumsq = float(np.sum(frame * frame))
    return -np.inf if sumsq <= 0.0 else float(20.0 * np.log10(sumsq))


def _rms_dbfs_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    out = np.full(len(rms), -np.inf, dtype=np.float64)
    pos = rms > 0.0
    out[pos] = 20.0 * np.log10(rms[pos])
    return out


def is_silent(
    x: np.ndarray,
    flatness_threshold: float = 0.45,
    energy_db_threshold: float = -13.7734,
):
    """
    Translation of eckf_pitch_final/is_silent.m.

    MATLAB source:
        [psdw,~] = pwelch(x);
        energy = 20*log10(sum(x.^2));
        spectral_flatness = geomean(psdw)/mean(psdw);
        silent = spectral_flatness >= 0.45 | energy < threshold;

    IMPORTANT: energy_db is the literal MATLAB 20*log10(sum of squares)
    statistic, NOT RMS dBFS.  In the offline V2 path the default/fallback
    threshold is -13.7734, approximately -40 dBFS RMS for a 2048-sample frame.
    MATLAB compatibility mode remains historical unless explicitly overridden.
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


def _linear_slope_db_per_s(values_db: np.ndarray, frame_seconds: float) -> float:
    values_db = np.asarray(values_db, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values_db)
    values_db = values_db[finite]
    if values_db.size < 2 or frame_seconds <= 0:
        return 0.0
    t = np.arange(values_db.size, dtype=np.float64) * frame_seconds
    t -= np.mean(t)
    y = values_db - np.mean(values_db)
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(t, y) / denom)


def _stationary_run_interior(
    run: tuple[int, int],
    rms_dbfs: np.ndarray,
    *,
    frame_seconds: float,
    edge_trim_ms: float,
    min_interior_ms: float,
    max_spread_db: float,
    max_abs_slope_db_per_s: float,
) -> tuple[bool, tuple[int, int] | None, float | None, float | None]:
    """Return whether a candidate run looks like stationary noise-floor silence.

    The test is intentionally simple.  Reverb tails are typically directional
    decays with a wide level spread; a usable noise-floor region should be
    comparatively stationary.  We trim the boundaries first so breath/noise or
    note attacks immediately adjacent to a pause do not contaminate the test.
    """
    start, end = run
    trim = max(0, int(math.ceil((edge_trim_ms / 1000.0) / frame_seconds)))
    istart = start + trim
    iend = end - trim
    min_frames = max(2, int(math.ceil((min_interior_ms / 1000.0) / frame_seconds)))
    if iend - istart < min_frames:
        return False, None, None, None

    vals = np.asarray(rms_dbfs[istart:iend], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < min_frames:
        return False, None, None, None

    p10, p90 = np.percentile(vals, [10.0, 90.0])
    spread = float(p90 - p10)
    slope = _linear_slope_db_per_s(vals, frame_seconds)
    ok = bool(spread <= max_spread_db and abs(slope) <= max_abs_slope_db_per_s)
    return ok, (istart, iend), spread, slope


def calibrate_silence_threshold(
    audio: np.ndarray,
    sample_rate: float,
    block_size: int,
    *,
    min_silence_ms: float = 500.0,
    search_quantiles: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0),
    headroom_stat_units: float = 6.0,
    minimum_allowed_threshold: float = -13.7734,
    digital_mute_rms_dbfs: float = -145.0,
    stationarity_edge_trim_ms: float = 100.0,
    stationarity_min_interior_ms: float = 250.0,
    stationarity_max_spread_db: float = 6.0,
    stationarity_max_abs_slope_db_per_s: float = 6.0,
) -> SilenceCalibration:
    """Estimate one per-file threshold from trustworthy long silence regions.

    V4 rules:
      * digital mute is <= -145 dBFS RMS and is excluded from calibration;
      * the normal offline fallback/default is -13.7734 on the historical
        statistic (~-40 dBFS RMS for 2048-sample frames);
      * an adaptive threshold is accepted only when at least one sufficiently
        long quiet run contains a stationary interior.  A reverb decay is not
        considered a trustworthy noise-floor measurement merely because it is
        quiet and long.

    Trustworthiness is deliberately uncomplicated: after trimming 100 ms from
    each edge, the interior must last at least 250 ms, have a P90-P10 RMS-level
    spread <= 6 dB, and an absolute linear level slope <= 6 dB/s.  If no such
    run exists, the calibrator falls back to the per-file default/fallback.
    """
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or block_size <= 0:
        raise ValueError("sample_rate and block_size must be positive")

    nframes = len(y) // block_size
    frame_ms = 1000.0 * block_size / sample_rate
    frame_seconds = block_size / sample_rate
    min_frames = max(1, int(math.ceil(min_silence_ms / frame_ms)))

    def make_result(*, threshold, method, chosen_q=None, chosen_ceiling=None,
                    long_runs=(), selected_idx=None, median=None, p95=None,
                    raw_threshold=None, mute_count=0, trustworthy=False,
                    trustworthy_count=0, rejected_nonstationary=0,
                    trust_failure_reason=None):
        selected_count = 0 if selected_idx is None else int(len(selected_idx))
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
            long_run_count=len(long_runs),
            selected_frame_count=selected_count,
            selected_duration_s=float(selected_count * block_size / sample_rate),
            selected_energy_median=median,
            selected_energy_p95=p95,
            headroom_stat_units=float(headroom_stat_units),
            raw_calculated_threshold=raw_threshold,
            minimum_allowed_threshold=float(minimum_allowed_threshold),
            minimum_threshold_applied=(raw_threshold is None or raw_threshold < minimum_allowed_threshold),
            digital_mute_frame_count=int(mute_count),
            digital_mute_duration_s=float(mute_count * block_size / sample_rate),
            trustworthy_noise_floor=bool(trustworthy),
            trustworthy_run_count=int(trustworthy_count),
            rejected_nonstationary_run_count=int(rejected_nonstationary),
            stationarity_max_spread_db=float(stationarity_max_spread_db),
            stationarity_max_abs_slope_db_per_s=float(stationarity_max_abs_slope_db_per_s),
            trust_failure_reason=trust_failure_reason,
        )

    if nframes <= 0:
        return make_result(
            threshold=minimum_allowed_threshold,
            method="fallback_no_full_frames",
            trust_failure_reason="no_full_frames",
        )

    frames = y[: nframes * block_size].reshape(nframes, block_size)
    rms_dbfs = _rms_dbfs_frames(frames)
    mute_rms_amplitude = 10.0 ** (float(digital_mute_rms_dbfs) / 20.0)
    rms_linear = np.sqrt(np.mean(frames * frames, axis=1))
    digitally_muted = rms_linear <= mute_rms_amplitude
    mute_count = int(np.sum(digitally_muted))

    sumsq = np.sum(frames * frames, axis=1)
    energies = np.full(nframes, -np.inf, dtype=np.float64)
    positive = sumsq > 0.0
    energies[positive] = 20.0 * np.log10(sumsq[positive])

    calibration_mask = np.isfinite(energies) & ~digitally_muted
    finite = energies[calibration_mask]
    if finite.size == 0:
        return make_result(
            threshold=minimum_allowed_threshold,
            method="fallback_all_digital_mute",
            mute_count=mute_count,
            trust_failure_reason="all_frames_digital_mute_or_nonfinite",
        )

    chosen_q = None
    chosen_ceiling = None
    candidate_runs: list[tuple[int, int]] = []
    for q in search_quantiles:
        ceiling = float(np.percentile(finite, q))
        quiet = calibration_mask & (energies <= ceiling)
        long_runs = [r for r in _true_runs(quiet) if (r[1] - r[0]) >= min_frames]
        if long_runs:
            chosen_q = float(q)
            chosen_ceiling = ceiling
            candidate_runs = long_runs
            break

    if not candidate_runs:
        return make_result(
            threshold=minimum_allowed_threshold,
            method="fallback_no_long_nonmuted_quiet_run",
            chosen_q=float(search_quantiles[-1]),
            chosen_ceiling=float(np.percentile(finite, search_quantiles[-1])),
            mute_count=mute_count,
            trust_failure_reason="no_long_nonmuted_quiet_run",
        )

    trusted_interiors: list[tuple[int, int]] = []
    rejected_nonstationary = 0
    for run in candidate_runs:
        ok, interior, _, _ = _stationary_run_interior(
            run,
            rms_dbfs,
            frame_seconds=frame_seconds,
            edge_trim_ms=stationarity_edge_trim_ms,
            min_interior_ms=stationarity_min_interior_ms,
            max_spread_db=stationarity_max_spread_db,
            max_abs_slope_db_per_s=stationarity_max_abs_slope_db_per_s,
        )
        if ok and interior is not None:
            trusted_interiors.append(interior)
        else:
            rejected_nonstationary += 1

    if not trusted_interiors:
        return make_result(
            threshold=minimum_allowed_threshold,
            method="fallback_no_trustworthy_noise_floor_silence",
            chosen_q=chosen_q,
            chosen_ceiling=chosen_ceiling,
            long_runs=candidate_runs,
            mute_count=mute_count,
            trustworthy=False,
            trustworthy_count=0,
            rejected_nonstationary=rejected_nonstationary,
            trust_failure_reason="long_quiet_regions_not_stationary",
        )

    selected_idx = np.concatenate([
        np.arange(start, end, dtype=np.int64) for start, end in trusted_interiors
    ])
    selected = energies[selected_idx]
    selected_finite = selected[np.isfinite(selected)]
    if not selected_finite.size:
        return make_result(
            threshold=minimum_allowed_threshold,
            method="fallback_trusted_region_empty",
            chosen_q=chosen_q,
            chosen_ceiling=chosen_ceiling,
            long_runs=candidate_runs,
            selected_idx=selected_idx,
            mute_count=mute_count,
            trustworthy=False,
            trustworthy_count=len(trusted_interiors),
            rejected_nonstationary=rejected_nonstationary,
            trust_failure_reason="trusted_region_nonfinite",
        )

    median = float(np.median(selected_finite))
    p95 = float(np.percentile(selected_finite, 95.0))
    raw_threshold = p95 + float(headroom_stat_units)
    threshold = max(float(raw_threshold), float(minimum_allowed_threshold))
    method = "stationary_noise_floor_runs"
    if threshold != raw_threshold:
        method += "+default_floor_clamp"

    return make_result(
        threshold=threshold,
        method=method,
        chosen_q=chosen_q,
        chosen_ceiling=chosen_ceiling,
        long_runs=candidate_runs,
        selected_idx=selected_idx,
        median=median,
        p95=p95,
        raw_threshold=raw_threshold,
        mute_count=mute_count,
        trustworthy=True,
        trustworthy_count=len(trusted_interiors),
        rejected_nonstationary=rejected_nonstationary,
        trust_failure_reason=None,
    )


def resolve_silence_calibration(config, audio=None, sample_rate=None) -> SilenceCalibration:
    """Resolve one silence calibration per recording."""
    frame_ms = 1000.0 * config.block_size / sample_rate if sample_rate else float("nan")
    if config.silence_mode == "fixed":
        return SilenceCalibration(
            mode="fixed",
            threshold=float(config.silence_energy_db_threshold),
            method="fixed_threshold",
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
            raw_calculated_threshold=float(config.silence_energy_db_threshold),
            minimum_allowed_threshold=float(config.silence_energy_db_threshold),
            minimum_threshold_applied=False,
            digital_mute_frame_count=0,
            digital_mute_duration_s=0.0,
            trustworthy_noise_floor=False,
            trustworthy_run_count=0,
            rejected_nonstationary_run_count=0,
            stationarity_max_spread_db=6.0,
            stationarity_max_abs_slope_db_per_s=6.0,
            trust_failure_reason="fixed_mode_not_calibrated",
        )
    if config.silence_mode == "adaptive":
        if audio is None or sample_rate is None:
            raise ValueError("adaptive silence calibration requires audio and sample_rate")
        return calibrate_silence_threshold(
            audio,
            float(sample_rate),
            int(config.block_size),
            minimum_allowed_threshold=float(config.silence_energy_db_threshold),
        )
    raise ValueError(f"Unknown silence_mode: {config.silence_mode!r}")


def resolve_silence_energy_threshold(config, audio=None, sample_rate=None) -> float:
    """Backward-compatible scalar helper."""
    return float(resolve_silence_calibration(config, audio, sample_rate).threshold)
