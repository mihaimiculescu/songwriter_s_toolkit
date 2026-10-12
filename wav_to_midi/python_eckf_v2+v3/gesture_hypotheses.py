from __future__ import annotations

"""V2 interpretation-candidate substrate.

This module deliberately does *not* choose an ornament or MIDI rendering.
It translates already-frozen structural gesture segments into a lossless set
of possible interpretation families plus the evidence needed later to
adjudicate them.

The goal is architectural: later MIDI work must be able to choose between
separate notes, pitch bend, legato-connected notes, and amplitude expression
without reverse-engineering information that an earlier stage discarded.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GestureInterpretationCandidate:
    object_index: int
    segment_index: int

    transition_region_indices: tuple[int, ...]
    stable_region_indices: tuple[int, ...]
    internal_link_indices: tuple[int, ...]
    unresolved_link_indices: tuple[int, ...]

    start_index: int
    end_index: int
    duration_ms: float

    n_transitions: int
    n_stable_targets: int
    n_join_links: int
    n_unresolved_links: int

    # Topology witnesses.  These are not musical labels.
    has_return_topology: bool
    has_same_direction_topology: bool
    has_multi_target_chain: bool
    has_unresolved_structure: bool

    # Pitch-expression witnesses.
    pitch_shape_span_st: float
    pitch_center_span_st: float
    pitch_residual_rms_st: float
    pitch_periodic_evidence_available: bool
    pitch_cycle_rate_median_hz: float
    pitch_periodicity_median: float

    # Amplitude-expression witnesses.
    amplitude_evidence_available: bool
    amp_residual_rms_db: float
    amp_residual_peak_to_peak_db: float
    amp_modulation_rate_median_hz: float
    amp_modulation_periodicity_median: float
    pitch_amplitude_correlation_median: float

    # Candidate families deliberately remain plural.  No winner is selected.
    discrete_note_sequence_candidate: bool
    continuous_pitch_motion_candidate: bool
    returning_ornament_candidate: bool
    oscillatory_pitch_candidate: bool
    legato_chain_candidate: bool
    amplitude_modulation_candidate: bool

    candidate_family_count: int
    candidate_families: tuple[str, ...]


def _finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _median_or_nan(values: Iterable[float]) -> float:
    arr = _finite_values(values)
    return float(np.median(arr)) if len(arr) else float("nan")


def _rms_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


def build_interpretation_candidates(
    *,
    structure,
    interpretation,
    smoothing,
    gesture_features,
    amplitude_expression,
    analysis_hz: float,
) -> tuple[GestureInterpretationCandidate, ...]:
    """Build multi-hypothesis interpretation records from structural segments.

    No tunable musical threshold is introduced here.  Candidate-family flags
    are based on structural availability/topology and whether a corresponding
    expressive measurement actually exists.  Later adjudication/classification
    may decide which family is musically appropriate.
    """

    region_map = {i: r for i, r in enumerate(interpretation.regions)}
    feature_map = {g.region_index: g for g in gesture_features.gestures}
    out: list[GestureInterpretationCandidate] = []

    for obj in structure.objects:
        link_by_index = {d.link_index: d for d in obj.link_decisions}

        for seg in obj.segments:
            all_regions = sorted(set(seg.transition_region_indices + seg.stable_region_indices))
            if all_regions:
                start = min(region_map[i].start_index for i in all_regions)
                end = max(region_map[i].end_index for i in all_regions)
            else:
                start = end = 0

            shape = np.asarray(smoothing.shape_pitch_st[start:end], dtype=np.float64)
            center = np.asarray(smoothing.center_pitch_st[start:end], dtype=np.float64)
            residual = np.asarray(smoothing.residual_pitch_st[start:end], dtype=np.float64)
            shape_f = shape[np.isfinite(shape)]
            center_f = center[np.isfinite(center)]

            shape_span = float(np.max(shape_f) - np.min(shape_f)) if len(shape_f) else float("nan")
            center_span = float(np.max(center_f) - np.min(center_f)) if len(center_f) else float("nan")
            residual_rms = _rms_or_nan(residual)

            transition_features = [
                feature_map[i] for i in seg.transition_region_indices if i in feature_map
            ]
            cycle_rates = [g.residual_cycle_rate_hz for g in transition_features]
            periodicities = [g.residual_periodicity for g in transition_features]
            auto_periodicities = [g.residual_autocorr_peak for g in transition_features]
            cycle_counts = [g.residual_full_cycles_est for g in transition_features]

            cycle_rate_med = _median_or_nan(cycle_rates)
            periodicity_med = _median_or_nan(periodicities + auto_periodicities)
            pitch_periodic_available = bool(
                np.isfinite(cycle_rate_med)
                or np.isfinite(periodicity_med)
                or any(np.isfinite(v) and v > 0 for v in cycle_counts)
            )

            links = [link_by_index[i] for i in seg.internal_link_indices if i in link_by_index]
            n_join = sum(d.decision.name == "JOIN" for d in links)
            has_return = any(
                d.v2.direction_reversal
                or d.v2.returns_toward_previous
                or d.v2.returns_toward_next
                for d in links
            )
            has_same_direction = any(d.v2.same_direction for d in links)

            amp_residual = np.asarray(
                amplitude_expression.residual_db[start:end], dtype=np.float64
            )
            amp_rate = np.asarray(
                amplitude_expression.local_modulation_rate_hz[start:end], dtype=np.float64
            )
            amp_periodicity = np.asarray(
                amplitude_expression.local_modulation_periodicity[start:end], dtype=np.float64
            )
            amp_corr = np.asarray(
                amplitude_expression.local_pitch_amplitude_correlation[start:end], dtype=np.float64
            )

            amp_residual_f = amp_residual[np.isfinite(amp_residual)]
            amp_rms = _rms_or_nan(amp_residual_f)
            amp_p2p = (
                float(np.max(amp_residual_f) - np.min(amp_residual_f))
                if len(amp_residual_f) else float("nan")
            )
            amp_rate_med = _median_or_nan(amp_rate)
            amp_periodicity_med = _median_or_nan(amp_periodicity)
            amp_corr_med = _median_or_nan(amp_corr)
            amp_available = bool(
                np.isfinite(amp_rms)
                and (np.isfinite(amp_rate_med) or np.isfinite(amp_periodicity_med))
            )

            n_stable = len(seg.stable_region_indices)
            n_trans = len(seg.transition_region_indices)
            has_multi = n_stable >= 2
            has_unresolved = bool(seg.unresolved_link_indices)

            # These flags identify *possible downstream representations* only.
            # They intentionally overlap; later stages must adjudicate them.
            discrete = bool(n_stable >= 1)
            continuous = bool(n_trans >= 1 and np.isfinite(center_span))
            returning = bool(has_return)
            oscillatory = bool(pitch_periodic_available)
            legato = bool(has_multi and n_trans >= 1 and (n_join > 0 or has_unresolved))
            amp_mod = bool(amp_available)

            families = []
            if discrete:
                families.append("DISCRETE_NOTE_SEQUENCE")
            if continuous:
                families.append("CONTINUOUS_PITCH_MOTION")
            if returning:
                families.append("RETURNING_ORNAMENT")
            if oscillatory:
                families.append("OSCILLATORY_PITCH")
            if legato:
                families.append("LEGATO_CHAIN")
            if amp_mod:
                families.append("AMPLITUDE_MODULATION")

            out.append(
                GestureInterpretationCandidate(
                    object_index=seg.object_index,
                    segment_index=seg.segment_index,
                    transition_region_indices=tuple(seg.transition_region_indices),
                    stable_region_indices=tuple(seg.stable_region_indices),
                    internal_link_indices=tuple(seg.internal_link_indices),
                    unresolved_link_indices=tuple(seg.unresolved_link_indices),
                    start_index=int(start),
                    end_index=int(end),
                    duration_ms=float((end - start) * 1000.0 / analysis_hz),
                    n_transitions=n_trans,
                    n_stable_targets=n_stable,
                    n_join_links=int(n_join),
                    n_unresolved_links=len(seg.unresolved_link_indices),
                    has_return_topology=has_return,
                    has_same_direction_topology=has_same_direction,
                    has_multi_target_chain=has_multi,
                    has_unresolved_structure=has_unresolved,
                    pitch_shape_span_st=shape_span,
                    pitch_center_span_st=center_span,
                    pitch_residual_rms_st=residual_rms,
                    pitch_periodic_evidence_available=pitch_periodic_available,
                    pitch_cycle_rate_median_hz=cycle_rate_med,
                    pitch_periodicity_median=periodicity_med,
                    amplitude_evidence_available=amp_available,
                    amp_residual_rms_db=amp_rms,
                    amp_residual_peak_to_peak_db=amp_p2p,
                    amp_modulation_rate_median_hz=amp_rate_med,
                    amp_modulation_periodicity_median=amp_periodicity_med,
                    pitch_amplitude_correlation_median=amp_corr_med,
                    discrete_note_sequence_candidate=discrete,
                    continuous_pitch_motion_candidate=continuous,
                    returning_ornament_candidate=returning,
                    oscillatory_pitch_candidate=oscillatory,
                    legato_chain_candidate=legato,
                    amplitude_modulation_candidate=amp_mod,
                    candidate_family_count=len(families),
                    candidate_families=tuple(families),
                )
            )

    return tuple(out)
