"""Lossless voicing/pitch status contract for downstream analysis.

A missing pitch is NOT synonymous with unvoiced. Consumers must not infer
note-off/note-on from VOICED_UNRESOLVED. This module does not synthesize F0.
"""
from __future__ import annotations

from enum import IntEnum
import numpy as np


class PitchStatus(IntEnum):
    UNVOICED = 0
    VOICED_VALID = 1
    VOICED_UNRESOLVED = 2


def validate_pitch_status(f0_hz: np.ndarray, status: np.ndarray) -> None:
    """Fail closed if the numeric pitch and status disagree."""
    f0 = np.asarray(f0_hz)
    s = np.asarray(status)
    if f0.shape != s.shape or f0.ndim != 1:
        raise ValueError('f0_hz and pitch_status must be same-size 1D arrays')
    if not np.all(np.isin(s, [int(x) for x in PitchStatus])):
        raise ValueError('Unknown pitch status')
    valid = s == PitchStatus.VOICED_VALID
    has_f0 = np.isfinite(f0) & (f0 > 0)
    if not np.array_equal(valid, has_f0):
        raise ValueError('Only VOICED_VALID may have a positive finite F0')


def validity_masks(f0_hz: np.ndarray, status: np.ndarray):
    """Return three masks; never coerce unresolved voice to unvoiced.

    Existing validity analysis can process VOICED_VALID pitches while its
    calling pipeline must separately carry the unresolved mask to MIDI export.
    """
    validate_pitch_status(f0_hz, status)
    s = np.asarray(status)
    return (
        s == PitchStatus.VOICED_VALID,
        s == PitchStatus.VOICED_UNRESOLVED,
        s == PitchStatus.UNVOICED,
    )


def note_boundary_permission(status: np.ndarray, left: int, right: int) -> str:
    """Conservative status-only MIDI boundary assessment, NOT a MIDI converter.

    No status alone establishes a new note onset (rearticulation needs separate
    evidence). Unresolved means defer, not off, not hold pitch, not split.
    """
    s = np.asarray(status)
    if s.ndim != 1 or not 0 <= left < right <= len(s):
        raise ValueError('Invalid status interval')
    segment = s[left:right]
    if np.any(segment == PitchStatus.VOICED_UNRESOLVED):
        return 'DEFER_UNRESOLVED'
    if np.all(segment == PitchStatus.UNVOICED):
        return 'UNVOICED_INTERVAL'
    return 'REQUIRES_ACOUSTIC_NOTE_BOUNDARY_EVIDENCE'
