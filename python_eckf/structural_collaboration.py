from __future__ import annotations

"""Structural collaboration v4 for residual frame adjudications.

No new acoustic juror is introduced here.  This pass runs only after the
existing jury + Harmonic Detective path has abstained and uses already-frozen
structural information more faithfully.

V4 retains the frame-span ownership fix and refines residual temporal
classification by collaborating with the already-computed expressive pitch,
amplitude, and target-formation evidence. No new juror is introduced:

1. Frame-span ownership
   Structural ownership is based on overlap with the *entire physical frame
   interval* [frame_start, frame_end), not on a single timestamp at the frame's
   left edge.  The region with the largest temporal overlap owns the frame.

2. Residual temporal ownership
   If no stable-target / covered-transition region owns the frame, the pass
   inspects the already-frozen 100 Hz trusted trajectory samples that physically
   fall inside that frame.  It does not estimate a new F0 and introduces no new
   threshold.  It only describes where trusted musical material exists:

     - trusted material at the left edge only  -> left_release
     - trusted material at the right edge only -> right_attack
     - no trusted material                     -> nonmusical
     - trusted interior island                 -> short_target_candidate
     - trusted material spanning both edges    -> leave unresolved

Boundary/gesture/nonmusical results intentionally carry no synthetic whole-frame
pitch winner.  They are successful temporal interpretations, not replacement
pitch estimates.
"""

from dataclasses import dataclass
import math
import numpy as np

from .harmonic_detective import FinalAdjudicationVerdict
from .trajectory_interpreter import RegionKind


@dataclass(frozen=True)
class StructuralCollaborationVerdict:
    frame_index: int
    time_s: float
    applied: bool
    frame_start_s: float
    frame_end_s: float
    trajectory_region_index: int | None
    trajectory_region_kind: str
    trajectory_overlap_ms: float
    gesture_object_index: int | None
    gesture_object_kind: str
    gesture_overlap_ms: float
    trusted_point_count: int
    trusted_run_start_s: float | None
    trusted_run_end_s: float | None
    structural_reference_st: float | None
    structural_reference_hz: float | None
    structural_reference_midi: int | None
    action: str
    reason: str


def _st_to_hz(pitch_st: float) -> float:
    return float(440.0 * (2.0 ** ((float(pitch_st) - 69.0) / 12.0)))


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def _best_overlapping_region(interpretation, start_s: float, end_s: float):
    candidates = []
    for region in interpretation.regions:
        ov = _overlap_s(start_s, end_s, region.start_time_s, region.end_time_s)
        if ov > 0.0:
            candidates.append((ov, region))
    if not candidates:
        return None, 0.0
    # Largest physical overlap wins.  Stable target wins only as a deterministic
    # tie-break when overlap is exactly equal; there is no overlap threshold.
    ov, region = max(
        candidates,
        key=lambda item: (
            item[0],
            1 if item[1].kind is RegionKind.STABLE_TARGET else 0,
            -int(item[1].index),
        ),
    )
    return region, float(ov)


def _best_overlapping_gesture_object(gesture_objects, start_s: float, end_s: float):
    if gesture_objects is None:
        return None, 0.0
    candidates = []
    for obj in gesture_objects.objects:
        ov = _overlap_s(start_s, end_s, obj.start_time_s, obj.end_time_s)
        if ov > 0.0:
            candidates.append((ov, obj))
    if not candidates:
        return None, 0.0
    # Largest overlap first, then narrowest constructed gesture, then index.
    ov, obj = max(
        candidates,
        key=lambda item: (
            item[0],
            -float(item[1].duration_ms),
            -int(item[1].object_index),
        ),
    )
    return obj, float(ov)


def _frame_bounds_by_index(pitch_candidates):
    """Recover real physical frame spans from the already-exported candidate rows."""
    result = {}
    for row in pitch_candidates:
        fi = int(row.frame_index)
        if fi not in result:
            start = int(row.frame_start_sample)
            count = int(row.frame_real_sample_count)
            result[fi] = (start, count)
    return result


def _trusted_points_inside_frame(
    *,
    interpretation,
    sample_times_s: np.ndarray,
    start_s: float,
    end_s: float,
):
    times = np.asarray(sample_times_s, dtype=np.float64)
    valid = np.asarray(interpretation.valid, dtype=bool)
    if times.shape != valid.shape:
        raise ValueError("sample_times_s must align with interpretation.valid")
    mask = (times >= float(start_s)) & (times < float(end_s)) & valid
    ix = np.flatnonzero(mask)
    return ix, times


def _single_contiguous_run(ix: np.ndarray) -> tuple[int, int] | None:
    if ix.size == 0:
        return None
    # The residual ownership lane stays conservative: if trusted points inside
    # one physical frame are fragmented into multiple runs, do not manufacture
    # an interpretation.
    if ix.size > 1 and np.any(np.diff(ix) != 1):
        return None
    return int(ix[0]), int(ix[-1])



def _bic_prefers_linear(values: np.ndarray) -> bool:
    """Standard BIC model choice: linear trend versus constant pitch.

    This is not a tuned musical threshold.  It asks whether the extra slope
    parameter earns its complexity penalty on the already-computed shape lane.
    """
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 3:
        return False
    x = np.arange(n, dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    mean = float(np.mean(y))
    sse0 = float(np.sum((y - mean) ** 2))
    A = np.column_stack((np.ones(n, dtype=np.float64), x))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sse1 = float(np.sum(resid * resid))
    bic0 = n * math.log(max(sse0 / n, eps)) + math.log(n)      # k=1
    bic1 = n * math.log(max(sse1 / n, eps)) + 2.0 * math.log(n) # k=2
    return bool(bic1 < bic0)


def _context_median(arr: np.ndarray, start: int, end: int) -> float | None:
    vals = np.asarray(arr[max(0, start):min(len(arr), end)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return None if vals.size == 0 else float(np.median(vals))


def _energy_context_direction(shape_rms_dbfs: np.ndarray, first_ix: int, last_ix: int) -> str:
    """Return attack/release/neutral from existing 100 Hz amplitude context.

    The comparison is relational only: the in-frame median is compared with
    equal-size immediately adjacent contexts.  No dB delta threshold is added.
    """
    n = max(1, int(last_ix - first_ix + 1))
    cur = _context_median(shape_rms_dbfs, first_ix, last_ix + 1)
    left = _context_median(shape_rms_dbfs, first_ix - n, first_ix)
    right = _context_median(shape_rms_dbfs, last_ix + 1, last_ix + 1 + n)
    if cur is None:
        return "neutral"
    # Attack: energy approaches/enters the frame from a weaker left context.
    # Release: energy leaves the frame into a weaker right context.
    gain_from_left = None if left is None else cur - left
    loss_to_right = None if right is None else cur - right
    if gain_from_left is not None and loss_to_right is not None:
        if gain_from_left > 0.0 and gain_from_left >= loss_to_right:
            return "attack"
        if loss_to_right > 0.0 and loss_to_right > gain_from_left:
            return "release"
    elif gain_from_left is not None and gain_from_left > 0.0:
        return "attack"
    elif loss_to_right is not None and loss_to_right > 0.0:
        return "release"
    return "neutral"

def apply_structural_collaboration(
    final_adjudication,
    interpretation,
    gesture_objects,
    *,
    pitch_candidates,
    sample_times_s,
    sample_rate: float,
    expressive=None,
    amplitude_expression=None,
):
    """Resolve residual abstentions using already-frozen structural evidence.

    Existing jury and Harmonic Detective winners are immutable.
    Returns ``(audit_rows, revised_final_adjudication)``.
    """

    audits = []
    revised = []
    frame_bounds = _frame_bounds_by_index(pitch_candidates)
    sr = float(sample_rate)

    for final in final_adjudication:
        fi = int(final.frame_index)
        t = float(final.time_s)

        if final.status != "unresolved":
            revised.append(final)
            continue

        bounds = frame_bounds.get(fi)
        if bounds is None:
            # No voiced candidate row means we cannot reconstruct a trustworthy
            # physical frame span here; preserve the abstention exactly.
            audits.append(StructuralCollaborationVerdict(
                fi, t, False, t, t,
                None, "", 0.0,
                None, "", 0.0,
                0, None, None,
                None, None, None,
                "none", "physical_frame_span_unavailable",
            ))
            revised.append(final)
            continue

        start_sample, real_count = bounds
        start_s = float(start_sample / sr)
        end_s = float((start_sample + real_count) / sr)

        region, region_ov = _best_overlapping_region(interpretation, start_s, end_s)
        obj, obj_ov = _best_overlapping_gesture_object(gesture_objects, start_s, end_s)

        region_index = None if region is None else int(region.index)
        region_kind = "" if region is None else region.kind.name
        object_index = None if obj is None else int(obj.object_index)
        object_kind = "" if obj is None else obj.kind.name

        trusted_ix, sample_times = _trusted_points_inside_frame(
            interpretation=interpretation,
            sample_times_s=np.asarray(sample_times_s, dtype=np.float64),
            start_s=start_s,
            end_s=end_s,
        )
        trusted_run = _single_contiguous_run(trusted_ix)
        trusted_count = int(trusted_ix.size)
        trusted_start_s = None if trusted_run is None else float(sample_times[trusted_run[0]])
        # End is the next 100 Hz sample boundary conceptually.  For the audit we
        # report the last trusted sample's timestamp; no new duration threshold
        # is inferred from it.
        trusted_end_s = None if trusted_run is None else float(sample_times[trusted_run[1]])

        # ------------------------------------------------------------
        # Criterion 1: dominant overlap is a frozen stable target.
        # ------------------------------------------------------------
        if region is not None and region.kind is RegionKind.STABLE_TARGET:
            target_st = float(region.median_pitch_st)
            if math.isfinite(target_st):
                target_hz = _st_to_hz(target_st)
                target_midi = int(round(target_st))
                audits.append(StructuralCollaborationVerdict(
                    fi, t, True, start_s, end_s,
                    region_index, region_kind, region_ov * 1000.0,
                    object_index, object_kind, obj_ov * 1000.0,
                    trusted_count, trusted_start_s, trusted_end_s,
                    target_st, target_hz, target_midi,
                    "stable_target_authority",
                    "largest_frame_span_overlap_is_frozen_stable_target",
                ))
                revised.append(FinalAdjudicationVerdict(
                    fi, t,
                    "stable_target",
                    "resolved",
                    f"stable_target:{region_index}",
                    target_midi,
                    target_hz,
                    final.original_jury_status,
                    final.original_abstention_category,
                ))
                continue

        # ------------------------------------------------------------
        # Criterion 2: dominant overlap is a frozen transition and the
        # physical frame overlaps an already-existing gesture object.
        # ------------------------------------------------------------
        if region is not None and region.kind is RegionKind.TRANSITION and obj is not None:
            audits.append(StructuralCollaborationVerdict(
                fi, t, True, start_s, end_s,
                region_index, region_kind, region_ov * 1000.0,
                object_index, object_kind, obj_ov * 1000.0,
                trusted_count, trusted_start_s, trusted_end_s,
                None, None, None,
                "gesture_absorb",
                "largest_frame_span_overlap_is_frozen_transition_covered_by_gesture",
            ))
            revised.append(FinalAdjudicationVerdict(
                fi, t,
                "gesture_structure",
                "resolved_gesture",
                None, None, None,
                final.original_jury_status,
                final.original_abstention_category,
            ))
            continue

        # ------------------------------------------------------------
        # Criterion 3 (v4): residual sub-frame collaboration.
        #
        # The existing corrected-valid topology is combined with three lanes
        # that have already been computed upstream:
        #   * expressive shape pitch,
        #   * expressive RMS shape,
        #   * target-formation core/stability evidence.
        # No new candidate, confidence score, or tuned musical threshold is
        # introduced. Previously resolved jury/HD verdicts remain immutable.
        # ------------------------------------------------------------
        valid = np.asarray(interpretation.valid, dtype=bool)
        times = np.asarray(sample_times_s, dtype=np.float64)
        inside_ix = np.flatnonzero((times >= start_s) & (times < end_s))

        if trusted_count == 0:
            audits.append(StructuralCollaborationVerdict(
                fi, t, True, start_s, end_s,
                region_index, region_kind, region_ov * 1000.0,
                object_index, object_kind, obj_ov * 1000.0,
                0, None, None,
                None, None, None,
                "nonmusical_absorb",
                "no_trusted_trajectory_points_inside_physical_frame",
            ))
            revised.append(FinalAdjudicationVerdict(
                fi, t, "temporal_structure", "resolved_nonmusical",
                None, None, None,
                final.original_jury_status, final.original_abstention_category,
            ))
            continue

        runs = []
        if trusted_ix.size:
            split_at = np.flatnonzero(np.diff(trusted_ix) != 1) + 1
            for chunk in np.split(trusted_ix, split_at):
                if chunk.size:
                    runs.append((int(chunk[0]), int(chunk[-1])))

        # Fragmentation is itself a successful temporal interpretation.  It
        # must not be coerced into one whole-frame note.
        if len(runs) != 1:
            audits.append(StructuralCollaborationVerdict(
                fi, t, True, start_s, end_s,
                region_index, region_kind, region_ov * 1000.0,
                object_index, object_kind, obj_ov * 1000.0,
                trusted_count, trusted_start_s, trusted_end_s,
                None, None, None,
                "fragmented_voiced_material",
                "multiple_disconnected_trusted_runs_inside_physical_frame",
            ))
            revised.append(FinalAdjudicationVerdict(
                fi, t, "temporal_structure", "resolved_fragmented_voiced",
                None, None, None,
                final.original_jury_status, final.original_abstention_category,
            ))
            continue

        run_first, run_last = runs[0]
        global_first = run_first
        while global_first > 0 and valid[global_first - 1]:
            global_first -= 1
        global_last = run_last
        while global_last + 1 < valid.size and valid[global_last + 1]:
            global_last += 1

        first_frame_ix = int(inside_ix[0]) if inside_ix.size else run_first
        last_frame_ix = int(inside_ix[-1]) if inside_ix.size else run_last
        continues_left = global_first < first_frame_ix
        continues_right = global_last > last_frame_ix
        audit_run_start_s = float(times[global_first])
        audit_run_end_s = float(times[global_last])

        # Existing expressive pitch shape: ask whether a linear motion model is
        # preferred to a constant model by standard BIC.
        motion = False
        if expressive is not None:
            shape = np.asarray(expressive.shape_pitch_st, dtype=np.float64)
            motion = _bic_prefers_linear(shape[run_first:run_last + 1])

        # Existing target-formation evidence: a locally coherent core may be
        # real even when the duration rule correctly refused to freeze a full
        # stable target.
        core = np.asarray(interpretation.variable_core_candidate, dtype=bool)
        local_pass = np.asarray(interpretation.local_stability_pass, dtype=bool)
        core_present = bool(np.any(core[run_first:run_last + 1]))
        locally_stable_present = bool(np.any(local_pass[run_first:run_last + 1]))

        amp_direction = "neutral"
        if amplitude_expression is not None:
            amp_direction = _energy_context_direction(
                np.asarray(amplitude_expression.shape_rms_dbfs, dtype=np.float64),
                run_first, run_last,
            )

        # Explicit one-sided topology has first authority: this is true
        # ownership information, not a score.
        if continues_left and not continues_right:
            action = "left_release"
            status = "resolved_boundary_left"
            reason = "trusted_run_continues_left_only;existing_subframe_evidence_consistent_with_release_ownership"
        elif continues_right and not continues_left:
            # A locally coherent but duration-insufficient island is retained
            # as a short target when the pitch lane itself does not prefer
            # continuous motion.  Otherwise it is an attack/transition.
            if core_present and not motion and amp_direction == "neutral":
                action = "short_target_candidate"
                status = "resolved_short_target_candidate"
                reason = "right_continuing_run_has_existing_local_target_core_without_motion_or_energy_boundary_direction"
            else:
                action = "right_attack"
                status = "resolved_boundary_right"
                reason = "trusted_run_continues_right_only;subframe_pitch_amplitude_context_assigns_attack_ownership"
        elif continues_left and continues_right:
            if motion:
                action = "continuous_motion"
                status = "resolved_continuous_motion"
                reason = "through_run_and_bic_prefers_linear_existing_pitch_shape"
            elif amp_direction == "attack":
                action = "right_attack"
                status = "resolved_boundary_right"
                reason = "through_run_but_existing_amplitude_context_is_attack_dominant"
            elif amp_direction == "release":
                action = "left_release"
                status = "resolved_boundary_left"
                reason = "through_run_but_existing_amplitude_context_is_release_dominant"
            elif core_present or locally_stable_present:
                action = "short_target_candidate"
                status = "resolved_short_target_candidate"
                reason = "through_run_is_locally_target_like_without_motion_or_boundary_energy_direction"
            else:
                action = "continuous_voiced_context"
                status = "resolved_continuous_motion"
                reason = "through_run_has_no_target_core;preserved_as_continuous_context"
        else:
            # Isolated trusted island.  Existing amplitude direction can still
            # tell us that the island is the fading/rising periodic part of a
            # mixed phonetic boundary.  A coherent core with no such direction
            # remains a short-target candidate.  No core + no motion is treated
            # as nonmusical voiced residue rather than a synthetic note.
            if motion:
                action = "continuous_motion"
                status = "resolved_continuous_motion"
                reason = "isolated_run_but_bic_prefers_linear_existing_pitch_shape"
            elif amp_direction == "release":
                action = "left_release"
                status = "resolved_boundary_left"
                reason = "isolated_run_with_existing_amplitude_release_direction"
            elif amp_direction == "attack":
                action = "right_attack"
                status = "resolved_boundary_right"
                reason = "isolated_run_with_existing_amplitude_attack_direction"
            elif core_present or locally_stable_present:
                action = "short_target_candidate"
                status = "resolved_short_target_candidate"
                reason = "isolated_run_has_existing_local_target_core"
            else:
                action = "nonmusical_absorb"
                status = "resolved_nonmusical"
                reason = "isolated_run_has_no_existing_target_core_motion_or_boundary_direction"

        audits.append(StructuralCollaborationVerdict(
            fi, t, True, start_s, end_s,
            region_index, region_kind, region_ov * 1000.0,
            object_index, object_kind, obj_ov * 1000.0,
            trusted_count, audit_run_start_s, audit_run_end_s,
            None, None, None,
            action, reason,
        ))
        revised.append(FinalAdjudicationVerdict(
            fi, t, "temporal_structure", status,
            None, None, None,
            final.original_jury_status, final.original_abstention_category,
        ))
        continue

    return tuple(audits), tuple(revised)
