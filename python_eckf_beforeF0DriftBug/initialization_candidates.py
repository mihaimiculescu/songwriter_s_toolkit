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
        # Peak-grid matches alone can be shared by submultiples. Retain
        # high-only families for *comparison*, not unconditional acceptance.
        if len(harmonics) < 2:
            continue
        penalty = None
        if (previous_hz is not None and np.isfinite(previous_hz)
                and previous_hz > 0 and elapsed_ms is not None):
            penalty = float(vocal_transition_penalty(
                12 * np.log2(hz / previous_hz), elapsed_ms))
        # A measured single-sinusoid component at the proposed fundamental
        # provides independent evidence not supplied by counting harmonics.
        # In particular, a 201-Hz grid can borrow peaks belonging to a
        # genuinely ~401-Hz family without having significant 201-Hz energy.
        amplitude, phase = _measured_amplitude_phase(frame, fs, hz, start_sample)
        if not (np.isfinite(amplitude) and amplitude > 0 and np.isfinite(phase)):
            continue
        representative = min(matching, key=lambda p: abs(1200 * np.log2(hz / p[0])))
        candidates.append((source, hz, harmonics, penalty,
                           min(abs(1200 * np.log2(hz / p[0])) for p in matching),
                           amplitude, phase, representative))

    if not candidates:
        return InitializationChoice(None, None, None, 'none',
                                    'no_acoustically_supported_candidate', 0, None)
    # Compare directly *measured* periods. A later multiple-lag ACF peak
    # can be a submultiple illusion: require a separately measured, stronger
    # shorter period AND substantially stronger energy at that frequency.
    # This is not pitch multiplication or continuation from previous_hz.
    def dominates(higher, lower):
        _, high_hz, _, _, _, high_amp, _, high_period = higher
        _, low_hz, _, _, _, low_amp, _, low_period = lower
        if high_hz <= low_hz * 1.4:
            return False
        # Compare integer-related period families only; do not suppress a
        # genuine lower voice just because an unrelated upper pitch exists.
        ratio = high_hz / low_hz
        if not (2 <= round(ratio) <= 5 and
                abs(1200 * np.log2(ratio / round(ratio))) <= max_disagreement_cents):
            return False
        clear_period_advantage = (high_period[2] >= low_period[2] + .04 and
                                  high_period[3] <= low_period[3] - .03)
        overwhelming_fundamental_advantage = (high_amp >= 8.0 * low_amp and
                                              high_period[2] >= low_period[2] - .02 and
                                              high_period[3] <= low_period[3] + .02)
        return (high_amp >= 3.0 * low_amp and clear_period_advantage
                or overwhelming_fundamental_advantage)

    candidates = [candidate for candidate in candidates
                  if not any(dominates(other, candidate) for other in candidates
                             if other is not candidate)]
    # A high-only harmonic family is acceptable only when its fundamental
    # component is independently strong relative to other measured periods.
    # If all evidence fails this condition, defer rather than manufacture F0.
    floor = max((candidate[5] for candidate in candidates), default=0.0)
    candidates = [candidate for candidate in candidates
                  if any(h <= 3 for h in candidate[2]) or
                  (candidate[5] >= .25 * floor and candidate[7][2] >= .80
                   and candidate[7][3] <= .20)]
    if not candidates:
        return InitializationChoice(None, None, None, 'none',
                                    'ambiguous_harmonic_family', 0, None)

    a = float(periodicity.acf_frequency_hz)
    b = float(periodicity.cmndf_frequency_hz)
    reliable_reference = (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0
                          and float(periodicity.acf_peak) >= .90
                          and float(periodicity.cmndf_minimum) <= .10
                          and abs(1200 * np.log2(a / b)) <= max_disagreement_cents)

    def ranking(candidate):
        source, hz, harmonics, penalty, disagreement, amp, phase, measured_period = candidate
        reference_mismatch = (abs(1200 * np.log2(hz / a))
                              if reliable_reference else 0.0)
        # Acoustic periodicity and measured fundamental strength precede
        # incidental peak counts. Prior affects ranking only after evidence.
        return (-measured_period[2], measured_period[3],
                -amp, 0.0 if penalty is None else penalty,
                reference_mismatch, -len(harmonics), disagreement,
                0 if source == 'spectral_spacing' else 1)

    candidates.sort(key=ranking)
    source, hz, harmonics, penalty, _, amplitude, phase, _ = candidates[0]
    return InitializationChoice(hz, amplitude, phase, source, 'accepted',
                                len(harmonics), penalty)
