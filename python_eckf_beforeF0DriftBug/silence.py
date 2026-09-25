from __future__ import annotations

import numpy as np

from .matlab_compat import matlab_pwelch_default


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
        silent = spectral_flatness >= 0.45 | energy < -50;
    IMPORTANT: energy_db is the literal MATLAB 20*log10(sum of squares)
    statistic, NOT RMS dBFS. Its -50 cutoff depends on frame length.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    psd = matlab_pwelch_default(x)

    sumsq = float(np.sum(x * x))
    energy_db = -np.inf if sumsq <= 0.0 else 20.0 * np.log10(sumsq)

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


def resolve_silence_energy_threshold(config, audio=None, sample_rate=None) -> float:
    """Resolve the energy threshold once per recording.

    Today only fixed mode is implemented; do not silently fall back when
    adaptive mode is requested. audio/sample_rate are reserved for a future
    recording-level calibration implementation and are currently ignored.
    This threshold uses the ORIGINAL 20*log10(sum(frame**2)) scale, not dBFS.
    """
    if config.silence_mode == "fixed":
        return float(config.silence_energy_db_threshold)
    if config.silence_mode == "adaptive":
        raise NotImplementedError("Adaptive silence calibration has not been implemented or validated")
    raise ValueError(f"Unknown silence_mode: {config.silence_mode!r}")
