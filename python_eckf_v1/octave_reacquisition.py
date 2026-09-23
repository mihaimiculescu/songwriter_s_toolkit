"""Independent-WAV-evidence check for ECKF octave-locked initialization.

Only correct a candidate after TWO waveform periodicity estimators agree, and
spectral peaks independently support the measured oscillator. Does not use a
previous-pitch prior to *create* a frequency; never simply halves/doubles F0.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .initialization_candidates import (_measured_period_candidates,
                                        _harmonic_support, _measured_amplitude_phase,
                                        InitializationChoice)
from .trajectory_resolver import vocal_transition_penalty

@dataclass(frozen=True)
class OctaveEvidence:
    state: str
    reason: str
    measured_hz: float | None
    harmonics: tuple[int, ...]
    transition_penalty: float | None
    amplitude: float | None = None
    phase: float | None = None


def inspect_octave_disagreement(detector, frame, sample_rate, start_sample,
                                proposed_hz, periodicity, previous_hz=None,
                                elapsed_ms=None) -> OctaveEvidence:
    """Return keep / replace / unresolved; caller owns all state changes.

    This is deliberately limited to approximately octave-related conflicts.
    It does not adjudicate an unrelated interval or determine note boundaries.
    """
    keep = lambda reason: OctaveEvidence('keep', reason, None, (), None)
    if not periodicity.voiced or proposed_hz is None:
        return keep('not_applicable')
    a = float(periodicity.acf_frequency_hz)
    b = float(periodicity.cmndf_frequency_hz)
    p = float(proposed_hz)
    if not all(np.isfinite(v) and v > 0 for v in (a, b, p)):
        return keep('missing_frequency')
    # ACF AND CMNDF must both indicate clean, mutually consistent periodicity.
    if (float(periodicity.acf_peak) < .90 or
            float(periodicity.cmndf_minimum) > .10 or
            abs(1200 * np.log2(a / b)) > 50):
        return keep('acoustic_evidence_insufficient')
    interval = 12 * np.log2(p / a)
    if abs(abs(interval) - 12.0) > 1.0:
        return keep('not_an_octave_conflict')
    # Exact measured local ACF peak: the global optimum by itself is not enough.
    measured = _measured_period_candidates(frame, sample_rate)
    matches = [v for v in measured if abs(1200*np.log2(v[0]/a)) <= 50]
    if not matches:
        return OctaveEvidence('unresolved', 'no_local_measured_period', a, (), None)
    harmonics = _harmonic_support(detector, frame, sample_rate, a)
    if len(harmonics) < 2 or not any(h <= 3 for h in harmonics):
        return OctaveEvidence('unresolved', 'insufficient_spectral_support', a,
                              harmonics, None)
    amplitude, phase = _measured_amplitude_phase(frame, sample_rate, a, start_sample)
    if not (np.isfinite(amplitude) and amplitude > 0 and np.isfinite(phase)):
        return OctaveEvidence('unresolved', 'invalid_oscillator_measurement',
                              a, harmonics, None)
    penalty = None
    if (previous_hz is not None and np.isfinite(previous_hz) and
            previous_hz > 0 and elapsed_ms is not None):
        penalty = float(vocal_transition_penalty(
            12*np.log2(a/previous_hz), elapsed_ms))
    return OctaveEvidence('replace', 'independent_waveform_and_spectral_octave_evidence',
                          a, harmonics, penalty, amplitude, phase)


def reconcile_initialization(detector, frame, fs, start, choice,
                             periodicity, previous_hz=None, elapsed_ms=None):
    """Keep normal choice unless there is independent octave-conflict evidence."""
    evidence = inspect_octave_disagreement(
        detector, frame, fs, start, choice.frequency_hz, periodicity,
        previous_hz, elapsed_ms)
    if evidence.state == 'keep':
        return choice, evidence
    if evidence.state == 'unresolved':
        return InitializationChoice(None, None, None, 'none', evidence.reason,
                                    len(evidence.harmonics), evidence.transition_penalty), evidence
    return InitializationChoice(evidence.measured_hz, evidence.amplitude,
                                evidence.phase, 'measured_octave_evidence',
                                evidence.reason, len(evidence.harmonics),
                                evidence.transition_penalty), evidence
