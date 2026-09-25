from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


_EPS = 1e-12


@dataclass(frozen=True)
class ExpressiveAmplitudeConfig:
    """
    Observational amplitude-expression lane.

    This module intentionally does NOT decide MIDI CC messages, note velocity,
    tremolo/vibrato labels, or ornament classes.  It preserves enough
    gain-robust amplitude information for those later decisions.

    The raw dBFS envelope is retained for provenance.  All expressive residuals
    are expressed relative to a slower local baseline, so multiplying the whole
    file by a constant gain does not change the expressive contour (apart from
    numerical dust and clipping).
    """

    analysis_hz: float = 100.0

    # RMS analysis around each 100 Hz sample.  20 ms keeps attack timing fairly
    # local while still integrating enough waveform samples for a stable RMS.
    rms_window_ms: float = 20.0

    # Light smoothing of the RMS envelope before expressive measurements.
    shape_window_ms: float = 50.0
    shape_polyorder: int = 2

    # Slow running-median reference representing local note/phrase loudness.
    # Subtracting this baseline in dB makes the residual invariant to a global
    # gain change.
    baseline_window_ms: float = 250.0

    # Observational modulation window.  Long enough to contain a few cycles of
    # typical vocal amplitude modulation, but no label is inferred from it.
    modulation_window_ms: float = 600.0

    # Search range for local amplitude-modulation rate.  This is evidence only,
    # not a classifier.  It spans slow pulsation through fast vocal tremolo.
    modulation_rate_min_hz: float = 2.0
    modulation_rate_max_hz: float = 12.0


@dataclass(frozen=True)
class ExpressiveAmplitudeResult:
    raw_rms_linear: np.ndarray
    raw_rms_dbfs: np.ndarray
    shape_rms_dbfs: np.ndarray
    baseline_rms_dbfs: np.ndarray
    residual_db: np.ndarray

    local_depth_peak_to_peak_db: np.ndarray
    local_depth_rms_db: np.ndarray
    local_modulation_rate_hz: np.ndarray
    local_modulation_periodicity: np.ndarray
    local_pitch_amplitude_correlation: np.ndarray

    trusted_pitch: np.ndarray
    analysis_hz: float


def _odd_steps(ms: float, hz: float, minimum: int = 3) -> int:
    steps = max(minimum, int(round(ms * hz / 1000.0)))
    if steps % 2 == 0:
        steps += 1
    return steps


def _centered_rms(audio: np.ndarray, sample_idx: np.ndarray, sr: int, window_ms: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    n = len(audio)

    width = max(1, int(round(window_ms * sr / 1000.0)))
    half_left = width // 2
    half_right = width - half_left

    sq = np.square(audio, dtype=np.float64)
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(sq, out=prefix[1:])

    out = np.empty(len(sample_idx), dtype=np.float64)
    for i, center in enumerate(sample_idx):
        left = max(0, int(center) - half_left)
        right = min(n, int(center) + half_right)
        count = max(1, right - left)
        power = (prefix[right] - prefix[left]) / float(count)
        out[i] = np.sqrt(max(power, 0.0))
    return out


def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    half = window // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        vals = x[left:right]
        vals = vals[np.isfinite(vals)]
        out[i] = float(np.median(vals)) if len(vals) else np.nan
    return out


def _smooth_db(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    minimum = polyorder + 1
    if minimum % 2 == 0:
        minimum += 1
    if w < minimum:
        return x.copy()
    return savgol_filter(x, window_length=w, polyorder=polyorder, mode="interp")


def _local_modulation_metrics(
    residual_db: np.ndarray,
    trusted_pitch: np.ndarray,
    pitch_residual_st: np.ndarray,
    analysis_hz: float,
    config: ExpressiveAmplitudeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local depth/rate/periodicity/correlation evidence per sample."""

    residual_db = np.asarray(residual_db, dtype=np.float64)
    trusted_pitch = np.asarray(trusted_pitch, dtype=bool)
    pitch_residual_st = np.asarray(pitch_residual_st, dtype=np.float64)

    n = len(residual_db)
    p2p = np.full(n, np.nan, dtype=np.float64)
    rms = np.full(n, np.nan, dtype=np.float64)
    rate = np.full(n, np.nan, dtype=np.float64)
    periodicity = np.full(n, np.nan, dtype=np.float64)
    corr = np.full(n, np.nan, dtype=np.float64)

    win = _odd_steps(config.modulation_window_ms, analysis_hz, minimum=5)
    half = win // 2

    min_lag = max(1, int(np.floor(analysis_hz / config.modulation_rate_max_hz)))
    max_lag = max(min_lag + 1, int(np.ceil(analysis_hz / config.modulation_rate_min_hz)))

    for i in range(n):
        if not trusted_pitch[i]:
            continue

        left = max(0, i - half)
        right = min(n, i + half + 1)

        local_mask = trusted_pitch[left:right] & np.isfinite(residual_db[left:right])
        if np.count_nonzero(local_mask) < 5:
            continue

        a = residual_db[left:right][local_mask]
        a = a - np.median(a)

        p2p[i] = float(np.max(a) - np.min(a))
        rms[i] = float(np.sqrt(np.mean(a * a)))

        # ACF-based local periodicity.  This is deliberately a diagnostic
        # measurement only; it does not assert "amplitude vibrato".
        denom = float(np.dot(a, a))
        if denom > _EPS:
            best_corr = -np.inf
            best_lag = None
            # Use contiguous local values for ACF so lag has a real time meaning.
            full = residual_db[left:right].copy()
            full_valid = trusted_pitch[left:right] & np.isfinite(full)
            if np.count_nonzero(full_valid) >= 5:
                # Fill internal missing samples only for this local diagnostic by
                # linear interpolation; never exported as signal data.
                idx = np.arange(len(full), dtype=np.float64)
                good = np.flatnonzero(full_valid)
                if len(good) >= 2:
                    filled = np.interp(idx, good.astype(float), full[good])
                    filled -= np.mean(filled)
                    energy = float(np.dot(filled, filled))
                    if energy > _EPS:
                        lag_scores = []
                        for lag in range(min_lag, min(max_lag, len(filled) - 2) + 1):
                            x = filled[:-lag]
                            y = filled[lag:]
                            d = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
                            if d <= _EPS:
                                continue
                            c = float(np.dot(x, y) / d)
                            lag_scores.append((lag, c))

                        if lag_scores:
                            best_corr = max(c for _, c in lag_scores)
                            # Numerical tie handling matters for gain invariance:
                            # a pure gain change can move correlations by ~1e-15.
                            # Treat effectively equal maxima as one plateau and
                            # deterministically choose the smallest lag.
                            tied_lags = [
                                lag for lag, c in lag_scores
                                if c >= best_corr - 1e-10
                            ]
                            best_lag = min(tied_lags)
                            periodicity[i] = best_corr
                            rate[i] = float(analysis_hz / best_lag)

        # Pitch/amplitude residual correlation: tells later code whether both
        # modulations move together, oppositely, or independently.
        p = pitch_residual_st[left:right]
        pair = trusted_pitch[left:right] & np.isfinite(p) & np.isfinite(residual_db[left:right])
        if np.count_nonzero(pair) >= 5:
            aa = residual_db[left:right][pair]
            pp = p[pair]
            if np.std(aa) > _EPS and np.std(pp) > _EPS:
                corr[i] = float(np.corrcoef(aa, pp)[0, 1])

    return p2p, rms, rate, periodicity, corr


def analyse_expressive_amplitude(
    audio: np.ndarray,
    sample_rate: int,
    sample_idx: np.ndarray,
    trusted_pitch: np.ndarray,
    pitch_residual_st: np.ndarray,
    analysis_hz: float,
    config: ExpressiveAmplitudeConfig | None = None,
) -> ExpressiveAmplitudeResult:
    """
    Build the amplitude-expression lane at the same 100 Hz timeline used by
    the pitch expressive lane.

    Global gain invariance:
      raw_rms_dbfs changes under file gain (intentionally preserved), while
      residual_db, modulation depth, rate, periodicity and pitch/amplitude
      correlation are based on local relative dynamics and therefore should
      remain stable under a pure gain change.
    """

    if config is None:
        config = ExpressiveAmplitudeConfig(analysis_hz=float(analysis_hz))
    elif not np.isclose(config.analysis_hz, analysis_hz):
        raise ValueError("config.analysis_hz does not match analysis_hz")

    trusted_pitch = np.asarray(trusted_pitch, dtype=bool)
    pitch_residual_st = np.asarray(pitch_residual_st, dtype=np.float64)
    sample_idx = np.asarray(sample_idx, dtype=np.int64)

    if len(trusted_pitch) != len(sample_idx) or len(pitch_residual_st) != len(sample_idx):
        raise ValueError("sample_idx, trusted_pitch and pitch_residual_st must have identical length")

    raw_linear = _centered_rms(audio, sample_idx, sample_rate, config.rms_window_ms)
    raw_dbfs = 20.0 * np.log10(np.maximum(raw_linear, _EPS))

    shape_window = _odd_steps(config.shape_window_ms, analysis_hz)
    shape_dbfs = _smooth_db(raw_dbfs, shape_window, config.shape_polyorder)

    baseline_window = _odd_steps(config.baseline_window_ms, analysis_hz)
    baseline_dbfs = _running_median(shape_dbfs, baseline_window)
    residual_db = shape_dbfs - baseline_dbfs

    p2p, rms, rate, periodicity, corr = _local_modulation_metrics(
        residual_db=residual_db,
        trusted_pitch=trusted_pitch,
        pitch_residual_st=pitch_residual_st,
        analysis_hz=analysis_hz,
        config=config,
    )

    return ExpressiveAmplitudeResult(
        raw_rms_linear=raw_linear,
        raw_rms_dbfs=raw_dbfs,
        shape_rms_dbfs=shape_dbfs,
        baseline_rms_dbfs=baseline_dbfs,
        residual_db=residual_db,
        local_depth_peak_to_peak_db=p2p,
        local_depth_rms_db=rms,
        local_modulation_rate_hz=rate,
        local_modulation_periodicity=periodicity,
        local_pitch_amplitude_correlation=corr,
        trusted_pitch=trusted_pitch.copy(),
        analysis_hz=float(analysis_hz),
    )
