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
