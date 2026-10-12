from __future__ import annotations

import math
import numpy as np
from scipy import signal


def nextpow2_exponent(x: int) -> int:
    """MATLAB nextpow2(x): exponent p such that 2**p >= abs(x)."""
    x = int(abs(x))
    if x <= 1:
        return 0
    return int(math.ceil(math.log2(x)))


def matlab_round(x):
    """MATLAB-style round: ties away from zero."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    if np.ndim(x) == 0:
        return float(out)
    return out


def matlab_mode_smallest(values):
    """
    MATLAB numeric mode behavior needed here:
    if several values have the same maximum count, choose the smallest.
    """
    v = np.asarray(values)
    if v.size == 0:
        raise ValueError("mode of empty data")
    uniq, counts = np.unique(v, return_counts=True)
    max_count = counts.max()
    return uniq[counts == max_count].min()


def matlab_blackman(length: int) -> np.ndarray:
    """
    MATLAB blackman(L) defaults to the symmetric form.
    scipy.signal.windows.blackman(..., sym=True) matches that convention.
    """
    return signal.windows.blackman(int(length), sym=True).astype(np.float64)


def matlab_hamming(length: int) -> np.ndarray:
    """MATLAB hamming(L) is symmetric by default."""
    return signal.windows.hamming(int(length), sym=True).astype(np.float64)


def matlab_pwelch_default(x: np.ndarray) -> np.ndarray:
    """
    Reproduce pwelch(x) defaults documented by MathWorks:

      nsc = floor(N/4.5)
      Hamming(nsc)
      overlap = floor(nsc/2)
      nfft = max(256, 2**nextpow2(nsc))

    MATLAB pwelch does not perform SciPy's default constant detrend, so
    detrend=False is explicit.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    nsc = max(1, int(math.floor(n / 4.5)))
    nov = int(math.floor(nsc / 2))
    nfft = max(256, 2 ** nextpow2_exponent(nsc))
    win = matlab_hamming(nsc)

    _, pxx = signal.welch(
        x,
        fs=2.0 * np.pi,   # scale is irrelevant to spectral flatness ratio
        window=win,
        nperseg=nsc,
        noverlap=nov,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    return np.asarray(pxx, dtype=np.float64)


def matlab_hist_counts_centers(x, nbins: int):
    """
    Compatibility helper for old MATLAB:
        [n,c] = hist(x, nbins)

    Centers are equally spaced between min(x) and max(x).  Bin boundaries are
    midpoints between adjacent centers; the outer bins extend to infinity.
    """
    data = np.asarray(x, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return np.zeros(nbins, dtype=int), np.full(nbins, np.nan)

    lo = float(data.min())
    hi = float(data.max())

    if lo == hi:
        centers = np.full(nbins, lo, dtype=np.float64)
        counts = np.zeros(nbins, dtype=int)
        # Old hist effectively puts identical data in a central bin.
        idx = (nbins - 1) // 2
        counts[idx] = data.size
        return counts, centers

    centers = np.linspace(lo, hi, nbins, dtype=np.float64)
    edges = np.empty(nbins + 1, dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    counts, _ = np.histogram(data, bins=edges)
    return counts.astype(int), centers


def complex_min_matlab_like(z):
    """
    Approximate MATLAB min() ordering for complex values:
    smallest magnitude, then phase as tie breaker.

    For the ECKF reset gate this mostly matters after initialization; the
    first K is exactly zero.
    """
    a = np.asarray(z).reshape(-1)
    if a.size == 0:
        raise ValueError("min of empty array")
    mags = np.abs(a)
    minmag = mags.min()
    candidates = a[np.isclose(mags, minmag, rtol=0.0, atol=0.0)]
    if candidates.size == 1:
        return candidates[0]
    phases = np.angle(candidates)
    return candidates[np.argmin(phases)]
