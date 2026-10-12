"""Typed frame-level acoustic evidence for offline ECKF V2.

This module deliberately separates acoustic state from pitch identity.
A frame may be silent, unvoiced, voiced-but-unresolved, or voiced-and-tracked.
Downstream code must not infer MIDI rests from these states alone.
"""
from __future__ import annotations

from enum import IntEnum
import numpy as np


class FrameAcousticState(IntEnum):
    NOT_PROCESSED = 0
    LOW_ENERGY_SILENCE = 1
    UNVOICED = 2
    VOICED_UNRESOLVED = 3
    VOICED_TRACKED = 4


FRAME_ACOUSTIC_STATE_NAMES = {
    int(FrameAcousticState.NOT_PROCESSED): "NOT_PROCESSED",
    int(FrameAcousticState.LOW_ENERGY_SILENCE): "LOW_ENERGY_SILENCE",
    int(FrameAcousticState.UNVOICED): "UNVOICED",
    int(FrameAcousticState.VOICED_UNRESOLVED): "VOICED_UNRESOLVED",
    int(FrameAcousticState.VOICED_TRACKED): "VOICED_TRACKED",
}


def frame_acoustic_state_name(value: int) -> str:
    try:
        return FRAME_ACOUSTIC_STATE_NAMES[int(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown frame acoustic state: {value!r}") from exc


def validate_frame_acoustic_states(states: np.ndarray) -> None:
    arr = np.asarray(states)
    allowed = np.asarray(list(FRAME_ACOUSTIC_STATE_NAMES), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("frame acoustic states must be 1-D")
    if not np.all(np.isin(arr, allowed)):
        raise ValueError("Unknown frame acoustic state")


def validate_frame_evidence_contract(
    states: np.ndarray,
    energy_silence: np.ndarray,
    periodicity_voiced: np.ndarray,
) -> None:
    """Validate the V2 frame-state contract without inferring MIDI semantics.

    LOW_ENERGY_SILENCE must come from the authoritative low-energy veto.
    UNVOICED means periodicity rejected.
    VOICED_UNRESOLVED and VOICED_TRACKED both require periodic evidence.
    """
    st = np.asarray(states)
    es = np.asarray(energy_silence, dtype=bool)
    pv = np.asarray(periodicity_voiced, dtype=bool)
    if st.ndim != 1 or es.shape != st.shape or pv.shape != st.shape:
        raise ValueError("frame evidence arrays must be same-size 1-D arrays")
    validate_frame_acoustic_states(st)

    silence = st == int(FrameAcousticState.LOW_ENERGY_SILENCE)
    if np.any(silence & ~es):
        raise ValueError("LOW_ENERGY_SILENCE frame without energy-silence evidence")

    unvoiced = st == int(FrameAcousticState.UNVOICED)
    if np.any(unvoiced & pv):
        raise ValueError("UNVOICED frame cannot have accepted periodicity")

    voiced_state = np.isin(
        st,
        [int(FrameAcousticState.VOICED_UNRESOLVED), int(FrameAcousticState.VOICED_TRACKED)],
    )
    if np.any(voiced_state & ~pv):
        raise ValueError("voiced frame state requires accepted periodicity")
