"""Waveform-only offline ECKF initialization candidate assessment.

The *global* ACF/YIN optimum can be a multiple of the true period (e.g.
100 Hz for a 400 Hz tone). Inspect shorter measured period minima before
considering that optimum. Neither the MIDI nor the preceding F0 creates
candidates. Existing transition prior only breaks acoustic ties.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from .trajectory_resolver import vocal_transition_penalty


@dataclass(frozen=True)
class InitializationChoice:
    frequency_hz: float | None
    amplitude: float | None
    phase: float | None
    source: str
    reason: str
    spectral_harmonics: int
    transition_penalty: float | None


def _harmonic_support(detector, frame, fs, frequency):
    _, mag, _ = detector._frame_spectrum(frame, fs)
    positions, _ = find_peaks(mag)
    frequencies = detector._fbins[positions]
    resolution = fs / detector._nfft
    supported = []
    for harmonic in range(1, 13):
        target = frequency * harmonic
        if target >= fs / 2:
            break
        width = max(resolution * 1.5, target * (2 ** (35 / 1200) - 1))
        if np.any(np.abs(frequencies - target) <= width):
            supported.append(harmonic)
    return tuple(supported)


def _measured_amplitude_phase(frame, fs, frequency, start_sample):
    x = np.asarray(frame, dtype=np.float64)
    t = (np.arange(len(x), dtype=np.float64) + start_sample + 1) / fs
    angle = 2 * np.pi * frequency * t
    design = np.column_stack((np.cos(angle), np.sin(angle), np.ones(len(x))))
    coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
    cosine, sine = map(float, coefficients[:2])
    return float(np.hypot(cosine, sine)), float(np.arctan2(-sine, cosine))


def _measured_period_candidates(frame, fs, min_hz=70., max_hz=1000.,
                                min_acf=.70, max_cmndf=.25):
    """Return measured local period minima, not integer divisions of an F0.

    Uses the same normalized lag ACF and CMNDF definitions and acceptance
    bounds as periodicity.py. No frequency is inferred from MIDI/context.
    """
    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)
    first = max(1, int(np.floor(fs / max_hz)))
    last = min(len(x) - 1, int(np.ceil(fs / min_hz)))
    difference = np.zeros(last + 1, dtype=np.float64)
    acf = np.full(last + 1, np.nan)
    for lag in range(1, last + 1):
        a, b = x[:-lag], x[lag:]
        energy = np.sqrt(np.dot(a, a) * np.dot(b, b))
        if energy > 0:
            acf[lag] = np.dot(a, b) / energy
        difference[lag] = np.dot(a - b, a - b)
    cumulative = np.cumsum(difference[1:])
    cmndf = np.ones(last + 1, dtype=np.float64)
    for lag in range(1, last + 1):
        if cumulative[lag - 1] > 0:
            cmndf[lag] = difference[lag] * lag / cumulative[lag - 1]

    # Peaks in ACF, not arbitrary lags: a candidate must represent an
    # observed local periodic maximum and independently low CMNDF.
    peaks, _ = find_peaks(np.nan_to_num(acf, nan=-1.))
    peaks = peaks[(peaks >= first) & (peaks <= last)]
    if last > first and np.isfinite(acf[last]) and acf[last] > acf[last - 1]:
        peaks = np.append(peaks, last)
    result = []
    for lag in peaks:
        if acf[lag] >= min_acf and cmndf[lag] <= max_cmndf:
            result.append((float(fs / lag), int(lag), float(acf[lag]),
                           float(cmndf[lag])))
    return result


def choose_initialization(detector, frame, fs, start_sample,
                          proposal, periodicity, previous_hz=None,
                          elapsed_ms=None, max_disagreement_cents=50.0):
    if not periodicity.voiced:
        return InitializationChoice(None, None, None, 'none', 'unvoiced', 0, None)

    measured = _measured_period_candidates(frame, fs)
    if not measured:
        return InitializationChoice(None, None, None, 'none',
                                    'no_measured_period_candidate', 0, None)
    candidates = []
    for source, hz in [('spectral_spacing', float(proposal.f0_hz))] + [
            ('waveform_periodicity', p[0]) for p in measured]:
        if not np.isfinite(hz) or hz <= 0 or hz >= fs / 2:
            continue
        # A proposed spectral F0 must agree with a *measured* local period.
        matching = [p for p in measured if
                    abs(1200 * np.log2(hz / p[0])) <= max_disagreement_cents]
        if not matching:
            continue
        harmonics = _harmonic_support(detector, frame, fs, hz)
        # Require multiple distinct harmonics including at least one of
        # 1..3. This prevents a 400-Hz waveform's fourth/eighth harmonics
        # from masquerading as 100 Hz. Ambiguous high-only spectra stay
        # unresolved rather than fabricating a low fundamental.
        if len(harmonics) < 2 or not any(h <= 3 for h in harmonics):
            continue
        penalty = None
        if (previous_hz is not None and np.isfinite(previous_hz)
                and previous_hz > 0 and elapsed_ms is not None):
            penalty = float(vocal_transition_penalty(
                12 * np.log2(hz / previous_hz), elapsed_ms))
        candidates.append((source, hz, harmonics, penalty,
                           min(abs(1200 * np.log2(hz / p[0])) for p in matching)))

    if not candidates:
        return InitializationChoice(None, None, None, 'none',
                                    'no_acoustically_supported_candidate', 0, None)
    # Prefer the primitive measured period (shortest lag / highest Hz)
    # with low-harmonic evidence; penalty then orders genuinely comparable
    # candidates. Never select a new frequency based on continuity alone.
    candidates.sort(key=lambda c: (
        min(c[2]), -len(c[2]),
        float('inf') if c[3] is None else c[3],
        c[4], 0 if c[0] == 'spectral_spacing' else 1))
    source, hz, harmonics, penalty, _ = candidates[0]
    if source == 'spectral_spacing':
        amplitude, phase = float(proposal.amplitude), float(proposal.phase)
    else:
        amplitude, phase = _measured_amplitude_phase(frame, fs, hz, start_sample)
    if not (np.isfinite(amplitude) and amplitude > 0 and np.isfinite(phase)):
        return InitializationChoice(None, None, None, 'none',
                                    'invalid_measured_oscillator', len(harmonics), penalty)
    return InitializationChoice(hz, amplitude, phase, source, 'accepted',
                                len(harmonics), penalty)
