from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks

from .config import ECKFConfig
from .interpolation import parabolic_interpolation
from .matlab_compat import (
    matlab_blackman,
    matlab_hist_counts_centers,
    matlab_mode_smallest,
    matlab_round,
    nextpow2_exponent,
)


@dataclass
class HarmonicAnalysis:
    flag: int
    f0_hz: float
    amplitude: float
    phase: float
    peak_frequencies_hz: np.ndarray


class HarmonicChangeDetector:
    """
    Translation of harmonic_change_detector.m.

    Unlike MATLAB's single persistent cache, this object rebuilds cached FFT
    geometry whenever (fs, frame_length, mode, vocal_floor_hz) changes.
    """

    def __init__(self, config: ECKFConfig):
        self.config = config
        self._cache_key = None
        self._nfft = None
        self._fbins = None
        self._win = None
        self._crop_start = None
        self._exp_win = None

    def _prepare(self, frame_len: int, fs: float):
        key = (
            int(frame_len),
            float(fs),
            self.config.mode,
            float(self.config.vocal_floor_hz),
        )
        if key == self._cache_key:
            return

        nfft = 2 ** nextpow2_exponent(4 * (frame_len + 1))
        win = matlab_blackman(frame_len)

        if self.config.mode == "matlab":
            # Literal source behavior:
            #   fbins = linspace(-fs/2, fs/2, nfft)
            #   nbins_below50 = round(50/(fs/2*nfft))
            #   crop starts at MATLAB index nfft/2 + nbins_below50.
            #
            # The expression produces zero at normal audio rates.  Because
            # MATLAB is 1-based, nfft/2 maps to Python nfft/2 - 1.
            fbins_full = np.linspace(-fs / 2.0, fs / 2.0, nfft)
            nbins_below50 = int(matlab_round(50.0 / ((fs / 2.0) * nfft)))
            crop_start = nfft // 2 - 1 + nbins_below50
            crop_start = max(0, crop_start)
            fbins = fbins_full[crop_start:]
        else:
            # Vocal-only corrected behavior.  Use true FFT-bin coordinates
            # and reject everything below the configured vocal floor.
            fbins_full = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
            indices = np.flatnonzero(fbins_full >= self.config.vocal_floor_hz)
            if indices.size == 0:
                raise ValueError(
                    f"vocal_floor_hz={self.config.vocal_floor_hz} exceeds FFT range"
                )
            crop_start = int(indices[0])
            fbins = fbins_full[crop_start:]

        # Literal repository Poisson window alpha=5.
        m = len(fbins)
        if m <= 1:
            exp_win = np.ones(m, dtype=np.float64)
        else:
            exp_win = np.exp(
                -0.5 * 5.0 * np.arange(m, dtype=np.float64) / (m - 1)
            )

        self._cache_key = key
        self._nfft = nfft
        self._fbins = fbins
        self._win = win
        self._crop_start = crop_start
        self._exp_win = exp_win

    def _frame_spectrum(self, x: np.ndarray, fs: float):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self._prepare(x.size, fs)

        xw = (x - np.mean(x)) * self._win
        X_full = np.fft.fftshift(np.fft.fft(xw, self._nfft))
        X = X_full[self._crop_start:]
        mag = (np.abs(X) / np.mean(self._win)) * self._exp_win
        phase = np.angle(X)
        return X, mag, phase

    def _select_peaks(self, mag: np.ndarray):
        peak_pos, _ = find_peaks(mag)
        if peak_pos.size < self.config.npeaks:
            raise RuntimeError(
                f"Only {peak_pos.size} spectral peaks found; "
                f"{self.config.npeaks} required"
            )

        pvals = mag[peak_pos]

        # MATLAB:
        # [~,ind] = sort(pval,'descend');
        # ind = sort(ind(1:npeaks));    % back into frequency order
        strongest = np.argsort(-pvals, kind="stable")[: self.config.npeaks]
        strongest = np.sort(strongest)

        selected_pos = peak_pos[strongest]
        selected_vals = pvals[strongest]
        selected_freqs = self._fbins[selected_pos]
        return selected_pos, selected_vals, selected_freqs

    def analyze(
        self,
        xprev: np.ndarray | None,
        xcur: np.ndarray,
        fs: float,
    ) -> HarmonicAnalysis:
        _, mag_cur, phase_cur = self._frame_spectrum(xcur, fs)
        pos_cur, vals_cur, freqs_cur = self._select_peaks(mag_cur)

        mpos = int(pos_cur[0])
        if mpos <= 0 or mpos >= len(mag_cur) - 1:
            raise RuntimeError("Selected first peak has no interpolation neighbors")

        amp, _ = parabolic_interpolation(
            mag_cur[mpos - 1],
            mag_cur[mpos],
            mag_cur[mpos + 1],
        )
        amp = float(np.real(amp) / self._nfft)

        spacings = matlab_round(np.diff(freqs_cur))
        f0_est = float(matlab_mode_smallest(spacings))

        phase, _ = parabolic_interpolation(
            phase_cur[mpos - 1],
            phase_cur[mpos],
            phase_cur[mpos + 1],
        )
        phase = float(np.real(phase))

        flag = 0

        if xprev is not None and len(xprev) > 0:
            _, mag_prev, _ = self._frame_spectrum(xprev, fs)
            _, _, freqs_prev = self._select_peaks(mag_prev)

            deviation = np.abs(np.diff(freqs_cur) - np.diff(freqs_prev))
            counts, centers = matlab_hist_counts_centers(deviation, 100)

            # MATLAB max returns first maximum.
            ix = int(np.argmax(counts))
            representative_delta_hz = float(centers[ix])

            denom = f0_est + representative_delta_hz
            if f0_est > 0.0 and denom > 0.0:
                cent_dev = 1200.0 * np.log2(f0_est / denom)
                if abs(cent_dev) >= self.config.nsemitones * 100.0:
                    flag = 1

        return HarmonicAnalysis(
            flag=flag,
            f0_hz=f0_est,
            amplitude=amp,
            phase=phase,
            peak_frequencies_hz=np.asarray(freqs_cur, dtype=np.float64),
        )
