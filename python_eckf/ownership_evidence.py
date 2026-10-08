from __future__ import annotations

"""Shared pitch-ownership evidence accessors for V3R0.

V3R0 is a behavior-preserving refactor of bidirectional V19.  This module does
NOT introduce a new adjudicator and does NOT change any threshold.  It merely
centralizes facts that were previously reimplemented in several downstream
modules so later releases can preserve provenance instead of silently
recomputing/forgetting earlier decisions.

The core invariants are intentionally identical to V19:
  * range admission means finite range_confidence > 0;
  * acoustic admission is the same candidate-field test used by the Harmonic
    Detective;
  * interval transition cost delegates to the existing cubic interval-juror
    penalty implementation in juror_evidence.
"""

from dataclasses import dataclass
import math

from .juror_evidence import _juror_quadratic_transition_penalty


@dataclass(frozen=True)
class OwnershipEvidenceSnapshot:
    """Read-only provenance container; no scoring or authority is implied."""
    frame_index: int | None
    group_id: str | None
    frequency_hz: float | None
    midi_float: float | None
    range_admitted: bool | None
    acoustically_admissible: bool | None
    interval_penalty: float | None
    source: str
    reason: str = ""


def midi_float(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def cents_distance(a_hz: float, b_hz: float) -> float:
    return abs(1200.0 * math.log2(float(a_hz) / float(b_hz)))


def range_admitted(row) -> bool:
    rc = getattr(row, "range_confidence", None)
    return rc is not None and math.isfinite(float(rc)) and float(rc) > 0.0


def acoustically_admissible(row) -> bool:
    """Exact bidirectional-V19 Harmonic-Detective admissibility predicate."""
    if not range_admitted(row):
        return False
    acf = getattr(row, "acf_median", None)
    return acf is not None and math.isfinite(float(acf)) and float(acf) >= 0.72


def interval_transition_penalty(
    hz: float,
    time_s: float,
    ref_hz: float,
    ref_time_s: float,
) -> float:
    interval = abs(midi_float(float(hz)) - midi_float(float(ref_hz)))
    available_ms = max(abs(float(time_s) - float(ref_time_s)) * 1000.0, 1.0)
    return float(_juror_quadratic_transition_penalty(interval, available_ms))


def snapshot_from_row(row, *, interval_penalty: float | None = None,
                      source: str = "candidate_field", reason: str = "") -> OwnershipEvidenceSnapshot:
    hz = getattr(row, "representative_hz", None)
    hz = None if hz is None else float(hz)
    return OwnershipEvidenceSnapshot(
        frame_index=(None if getattr(row, "frame_index", None) is None else int(row.frame_index)),
        group_id=getattr(row, "group_id", None),
        frequency_hz=hz,
        midi_float=(None if hz is None or not math.isfinite(hz) or hz <= 0 else midi_float(hz)),
        range_admitted=range_admitted(row),
        acoustically_admissible=acoustically_admissible(row),
        interval_penalty=interval_penalty,
        source=source,
        reason=reason,
    )
