from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .frame_evidence import FrameAcousticState


@dataclass(frozen=True)
class PrimaryReconciliationV3R7Config:
    """Conservative V3R7 reconciliation of ECKF ownership with primary periodicity.

    This is a tracker/interpretation repair, not a MIDI stage.  It may only use
    acoustic periodicity evidence, the existing corrected-valid trajectory and
    the file-level vocal range.  It never consumes MIDI, attack GT or song/time
    exceptions.
    """

    min_acf_peak: float = 0.88
    max_cmndf_minimum: float = 0.12
    family_agreement_cents: float = 60.0
    integer_family_tolerance_cents: float = 60.0
    range_margin_semitones: float = 0.25
    context_frames: int = 4
    context_time_s: float = 0.18
    max_context_distance_semitones: float = 3.5
    integer_choice_margin_semitones: float = 4.0
    max_adjacent_primary_step_semitones: float = 4.0
    trusted_context_tolerance_semitones: float = 1.5
    replacement_min_disagreement_semitones: float = 3.5

    # Existing-valid tracker drift is more dangerous to overwrite than rejected
    # pitch is to recover.  V3R7 therefore adjudicates the original local
    # trajectory around each disagreement.  The context is expressed in
    # physical time; it is not tied to a 100-Hz grid.
    trajectory_context_time_s: float = 0.080
    trajectory_neighbors_per_side: int = 4
    trajectory_preference_margin_semitones: float = 0.75
    trajectory_forward_min_advantage_semitones: float = 1.25

    # A frame-wide primary estimate that has just jumped to a new family must
    # not immediately overwrite the first few trajectory samples of that frame.
    # The physical transition may be occurring inside the ~46 ms primary frame.
    primary_jump_guard_threshold_semitones: float = 3.5
    primary_jump_guard_time_s: float = 0.020

    # Persistent multi-primary disagreement remains an independent authority,
    # but it is no longer the universal gate.  A locally coherent original
    # trajectory can veto replacement even when adjacent primary frames agree.
    replacement_persistence_neighbor_frames: int = 1
    replacement_persistence_min_disagreement_time_s: float = 0.018

    # Separate narrow certificate for a short false-valid island: one or two
    # accepted interpretation-grid samples, bounded by rejected samples, may be replaced by
    # overwhelming same-frame primary evidence when that primary family also
    # agrees with trusted pitch on BOTH sides.
    isolated_false_valid_max_samples: int = 2
    isolated_false_valid_min_acf_peak: float = 0.95
    isolated_false_valid_max_cmndf_minimum: float = 0.05
    isolated_false_valid_context_time_s: float = 0.20
    isolated_false_valid_context_tolerance_semitones: float = 2.0


@dataclass(frozen=True)
class PrimaryReconciliationV3R7Row:
    frame_index: int
    frame_start_time_s: float
    acoustic_state: int
    acf_peak: float
    cmndf_minimum: float
    acf_frequency_hz: float
    cmndf_frequency_hz: float
    candidate_hz: Optional[float]
    candidate_source: str
    coherent_support: bool
    trusted_context_support: bool
    trajectory_candidate_support_count: int
    trajectory_current_support_count: int
    trajectory_inconclusive_count: int
    trajectory_jump_guard_count: int
    promoted_sample_count: int
    replaced_sample_count: int
    action: str
    reason: str


def _midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def _cents(a: float, b: float) -> float:
    return abs(1200.0 * math.log2(float(a) / float(b)))


def _in_range(hz: float, low_midi: Optional[float], high_midi: Optional[float], margin: float) -> bool:
    if not np.isfinite(hz) or hz <= 0.0:
        return False
    if low_midi is None or high_midi is None:
        return True
    m = _midi(hz)
    return (float(low_midi) - margin) <= m <= (float(high_midi) + margin)


def _integer_related(a: float, b: float, tol_cents: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
        return False
    hi, lo = max(float(a), float(b)), min(float(a), float(b))
    ratio = hi / lo
    for n in (2.0, 3.0, 4.0):
        if abs(1200.0 * math.log2(ratio / n)) <= tol_cents:
            return True
    return False


def _valid_island_bounds(valid: np.ndarray, index: int) -> tuple[int, int]:
    """Return inclusive bounds of the corrected-valid island containing index."""
    n = len(valid)
    lo = int(index)
    hi = int(index)
    while lo > 0 and bool(valid[lo - 1]):
        lo -= 1
    while hi + 1 < n and bool(valid[hi + 1]):
        hi += 1
    return lo, hi


def _nearest_external_valid_hz(
    *,
    start: int,
    stop: int,
    sample_times_s: np.ndarray,
    valid: np.ndarray,
    clean_f0_hz: np.ndarray,
    max_dt_s: float,
) -> tuple[Optional[float], Optional[float]]:
    """Nearest trusted pitch before and after [start, stop], within max_dt_s."""
    left: Optional[float] = None
    right: Optional[float] = None

    for j in range(start - 1, -1, -1):
        if float(sample_times_s[start] - sample_times_s[j]) > max_dt_s:
            break
        if bool(valid[j]) and np.isfinite(clean_f0_hz[j]) and clean_f0_hz[j] > 0.0:
            left = float(clean_f0_hz[j])
            break

    for j in range(stop + 1, len(valid)):
        if float(sample_times_s[j] - sample_times_s[stop]) > max_dt_s:
            break
        if bool(valid[j]) and np.isfinite(clean_f0_hz[j]) and clean_f0_hz[j] > 0.0:
            right = float(clean_f0_hz[j])
            break

    return left, right




def _side_trajectory_reference_midi(
    *,
    index: int,
    direction: int,
    sample_times_s: np.ndarray,
    valid: np.ndarray,
    clean_f0_hz: np.ndarray,
    exclude_mask: np.ndarray,
    max_dt_s: float,
    max_neighbors: int,
) -> Optional[float]:
    """Median trusted pitch on one side of an existing-valid disagreement.

    Samples already identified as part of the same large-disagreement micro-run
    are excluded.  This prevents a two-sample tracker glitch from voting for
    itself.  The search radius is physical time, so the rule is invariant to
    the offline interpretation rate.
    """
    refs: list[float] = []
    t0 = float(sample_times_s[index])
    j = int(index) + int(direction)
    n = len(valid)
    while 0 <= j < n:
        if abs(float(sample_times_s[j]) - t0) > float(max_dt_s):
            break
        if (
            not bool(exclude_mask[j])
            and bool(valid[j])
            and np.isfinite(clean_f0_hz[j])
            and clean_f0_hz[j] > 0.0
        ):
            refs.append(_midi(float(clean_f0_hz[j])))
            if len(refs) >= int(max_neighbors):
                break
        j += int(direction)
    if not refs:
        return None
    return float(np.median(np.asarray(refs, dtype=np.float64)))


def _side_preference(
    ref_midi: Optional[float],
    current_midi: float,
    candidate_midi: float,
    margin_st: float,
) -> tuple[str, float]:
    """Return current/candidate/neutral/none plus winner advantage in st."""
    if ref_midi is None or not np.isfinite(ref_midi):
        return "none", 0.0
    d_cur = abs(float(ref_midi) - float(current_midi))
    d_cand = abs(float(ref_midi) - float(candidate_midi))
    if d_cand + float(margin_st) < d_cur:
        return "candidate", float(d_cur - d_cand)
    if d_cur + float(margin_st) < d_cand:
        return "current", float(d_cand - d_cur)
    return "neutral", float(abs(d_cur - d_cand))


def _trajectory_adjudication(
    *,
    index: int,
    candidate_hz: float,
    sample_times_s: np.ndarray,
    valid: np.ndarray,
    clean_f0_hz: np.ndarray,
    disagreement_mask: np.ndarray,
    cfg: PrimaryReconciliationV3R7Config,
) -> tuple[str, Optional[float], Optional[float]]:
    """Adjudicate an already-valid ECKF disagreement from local trajectory shape.

    The rule is directional rather than a generic smoothing prior:

    * future support for the existing ECKF path protects a real excursion;
    * a current->candidate transition with strong future support can repair
      tracker lag into a newly established family;
    * bilateral candidate support can repair an isolated tracker spike;
    * candidate support only on the left is insufficient, because phrase-end
      falls/releases are legitimate and must not be flattened.

    Persistent multi-primary drift and the isolated-false-valid certificate are
    separate authorities handled by the caller.
    """
    cur_hz = float(clean_f0_hz[index])
    if not (np.isfinite(cur_hz) and cur_hz > 0.0):
        return "inconclusive", None, None
    cur_m = _midi(cur_hz)
    cand_m = _midi(float(candidate_hz))

    left_ref = _side_trajectory_reference_midi(
        index=index, direction=-1, sample_times_s=sample_times_s,
        valid=valid, clean_f0_hz=clean_f0_hz, exclude_mask=disagreement_mask,
        max_dt_s=cfg.trajectory_context_time_s,
        max_neighbors=cfg.trajectory_neighbors_per_side,
    )
    right_ref = _side_trajectory_reference_midi(
        index=index, direction=+1, sample_times_s=sample_times_s,
        valid=valid, clean_f0_hz=clean_f0_hz, exclude_mask=disagreement_mask,
        max_dt_s=cfg.trajectory_context_time_s,
        max_neighbors=cfg.trajectory_neighbors_per_side,
    )
    lp, la = _side_preference(
        left_ref, cur_m, cand_m, cfg.trajectory_preference_margin_semitones
    )
    rp, ra = _side_preference(
        right_ref, cur_m, cand_m, cfg.trajectory_preference_margin_semitones
    )

    # Future support for the current path is the strongest evidence that the
    # ECKF sample belongs to real ongoing motion rather than tracker drift.
    if rp == "current":
        return "current", left_ref, right_ref

    # Bilateral support for the primary family is a classic isolated tracker
    # spike signature.
    if lp == "candidate" and rp == "candidate":
        return "candidate", left_ref, right_ref

    # A lagging transition often has old/current support on the left and the
    # new primary family on the right.  Require the forward side to win by a
    # useful margin; this is deliberately asymmetric so a legitimate outgoing
    # phrase-end fall is not pulled backward by the previous pitch family.
    if rp == "candidate" and lp in {"current", "neutral", "none"}:
        if ra >= cfg.trajectory_forward_min_advantage_semitones:
            return "candidate", left_ref, right_ref

    # Candidate on the left with no confirming future trajectory is not enough
    # to overwrite an accepted ECKF sample.
    if lp == "current":
        return "current", left_ref, right_ref

    return "inconclusive", left_ref, right_ref


def _nearest_context_midi(
    frame_i: int,
    frame_candidates: np.ndarray,
    frame_times: np.ndarray,
    sample_times_s: np.ndarray,
    valid: np.ndarray,
    clean_f0_hz: np.ndarray,
    cfg: PrimaryReconciliationV3R7Config,
) -> list[float]:
    refs: list[float] = []
    lo = max(0, frame_i - cfg.context_frames)
    hi = min(len(frame_candidates), frame_i + cfg.context_frames + 1)
    for k in range(lo, hi):
        if k == frame_i:
            continue
        hz = frame_candidates[k]
        if np.isfinite(hz) and hz > 0.0:
            refs.append(_midi(float(hz)))

    t0 = float(frame_times[frame_i])
    mask = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(clean_f0_hz)
        & (clean_f0_hz > 0.0)
        & (sample_times_s >= t0 - cfg.context_time_s)
        & (sample_times_s <= t0 + cfg.context_time_s)
    )
    if np.any(mask):
        refs.extend([_midi(float(x)) for x in clean_f0_hz[mask]])
    return refs


def apply_primary_reconciliation_v3r7(
    *,
    offline_validity,
    sample_times_s: np.ndarray,
    sample_frame_index: np.ndarray,
    frame_start_times_s: np.ndarray,
    frame_acoustic_state: np.ndarray,
    periodicity_acf_peak_per_frame: np.ndarray,
    periodicity_cmndf_minimum_per_frame: np.ndarray,
    periodicity_acf_frequency_hz_per_frame: np.ndarray,
    periodicity_cmndf_frequency_hz_per_frame: np.ndarray,
    range_zero_low_midi: Optional[float],
    range_zero_high_midi: Optional[float],
    config: PrimaryReconciliationV3R7Config = PrimaryReconciliationV3R7Config(),
) -> list[PrimaryReconciliationV3R7Row]:
    """Repair tracker ownership when primary periodicity is stronger than ECKF state.

    Three deliberately narrow actions are permitted:

    * PRIMARY_EVIDENCE_RECOVERED: promote rejected samples when a strong,
      range-valid primary family is temporally coherent.
    * PRIMARY_TRACKER_DRIFT_REPLACED: replace a corrected-valid ECKF value when
      either the disagreement is sustained in physical time across primary
      frames or the immutable local ECKF trajectory itself favors the primary
      family. Integer-family disputes remain the responsibility of the existing
      V3R3/V3R4/V3R5 register machinery.
    * PRIMARY_ISOLATED_FALSE_VALID_REPLACED: a separate narrow certificate for
      a 1-2 sample accepted island bounded by rejected samples, when overwhelming
      same-frame primary evidence agrees with trusted pitch on both sides.

    LOW_ENERGY_SILENCE is never promoted.
    """

    sample_times_s = np.asarray(sample_times_s, dtype=np.float64)
    sample_frame_index = np.asarray(sample_frame_index, dtype=np.int64)
    frame_start_times_s = np.asarray(frame_start_times_s, dtype=np.float64)
    # Keep a writable reference to the authoritative frame passport.
    # V3R7 can promote a previously unresolved/rejected frame to tracked
    # ownership; when it does, the acoustic-state passport must be reconciled
    # with that decision as well.  Using dtype=np.int64 here would silently
    # create a copy for the tracker's int16 state array, leaving downstream
    # trajectory validation with stale non-VOICED_TRACKED states.
    frame_state_ref = np.asarray(frame_acoustic_state)
    states = np.asarray(frame_state_ref, dtype=np.int64)
    acf = np.asarray(periodicity_acf_peak_per_frame, dtype=np.float64)
    cmn = np.asarray(periodicity_cmndf_minimum_per_frame, dtype=np.float64)
    fa = np.asarray(periodicity_acf_frequency_hz_per_frame, dtype=np.float64)
    fc = np.asarray(periodicity_cmndf_frequency_hz_per_frame, dtype=np.float64)

    n_frames = len(frame_start_times_s)
    if not (len(states) == len(acf) == len(cmn) == len(fa) == len(fc) == n_frames):
        raise ValueError("V3R7 primary reconciliation frame arrays must have equal length")

    # Work from the state after V3R5 adjudication feedback.
    valid = offline_validity.valid
    clean = offline_validity.clean_f0_hz

    # All replacement certificates are judged against the V3R5-authoritative
    # state that entered this stage, not against samples V3R7 has already
    # modified earlier in the loop.  This prevents one V3R7 decision from
    # manufacturing evidence for another.
    original_valid = np.asarray(valid, dtype=bool).copy()
    original_clean = np.asarray(clean, dtype=np.float64).copy()

    candidate = np.full(n_frames, np.nan, dtype=np.float64)
    source = np.full(n_frames, "", dtype=object)
    disputed = np.zeros(n_frames, dtype=bool)

    # Pass 1: strong, directly measurable primary candidates.
    for i in range(n_frames):
        if states[i] == int(FrameAcousticState.LOW_ENERGY_SILENCE):
            continue
        if not (np.isfinite(acf[i]) and np.isfinite(cmn[i])):
            continue
        if acf[i] < config.min_acf_peak or cmn[i] > config.max_cmndf_minimum:
            continue

        vals = []
        for hz in (fa[i], fc[i]):
            if _in_range(hz, range_zero_low_midi, range_zero_high_midi, config.range_margin_semitones):
                vals.append(float(hz))
        if not vals:
            continue
        if len(vals) == 1:
            candidate[i] = vals[0]
            # Preserve whether the rejected alternate estimator is an
            # integer-related out-of-range family.  In that case the file-level
            # vocal range has independently resolved the register dispute.
            other = fc[i] if abs(float(vals[0]) - float(fa[i])) < 1e-9 else fa[i]
            if (
                np.isfinite(other) and other > 0.0
                and _integer_related(vals[0], float(other), config.integer_family_tolerance_cents)
            ):
                source[i] = "range_resolved_integer_family"
            else:
                source[i] = "single_in_range_primary"
            continue

        if _cents(vals[0], vals[1]) <= config.family_agreement_cents:
            # geometric mean preserves multiplicative pitch symmetry
            candidate[i] = math.sqrt(vals[0] * vals[1])
            source[i] = "acf_cmndf_agreement"
        elif _integer_related(vals[0], vals[1], config.integer_family_tolerance_cents):
            disputed[i] = True
            source[i] = "integer_family_dispute"

    # Pass 2: resolve only very clear integer-family disputes from independent
    # temporal context.  This is evidence preservation, not a new octave juror.
    for i in np.flatnonzero(disputed):
        opts = [x for x in (fa[i], fc[i]) if _in_range(x, range_zero_low_midi, range_zero_high_midi, config.range_margin_semitones)]
        if not opts:
            continue
        refs = _nearest_context_midi(
            int(i), candidate, frame_start_times_s,
            sample_times_s, valid, clean, config,
        )
        if not refs:
            continue
        ref = float(np.median(np.asarray(refs, dtype=np.float64)))
        d = np.asarray([abs(_midi(float(x)) - ref) for x in opts], dtype=np.float64)
        order = np.argsort(d)
        best = int(order[0])
        second = float(d[order[1]]) if len(order) > 1 else float("inf")
        if d[best] <= config.max_context_distance_semitones and (second - d[best]) >= config.integer_choice_margin_semitones:
            candidate[i] = float(opts[best])
            source[i] = "integer_family_context_resolved"

    # Pass 3: require temporal support.  A strong candidate may be supported by
    # another nearby primary frame OR by trusted corrected-valid pitch in the
    # same neighborhood.  Isolated spectral accidents do not gain ownership.
    coherent = np.zeros(n_frames, dtype=bool)
    trusted_support = np.zeros(n_frames, dtype=bool)
    cmidi = np.full(n_frames, np.nan, dtype=np.float64)
    finite = np.isfinite(candidate) & (candidate > 0.0)
    cmidi[finite] = 69.0 + 12.0 * np.log2(candidate[finite] / 440.0)

    for i in np.flatnonzero(finite):
        mi = cmidi[i]
        for k in range(max(0, i - 2), min(n_frames, i + 3)):
            if k == i or not np.isfinite(cmidi[k]):
                continue
            if abs(cmidi[k] - mi) <= config.max_adjacent_primary_step_semitones:
                coherent[i] = True
                break

        t0 = frame_start_times_s[i]
        sm = (
            np.asarray(valid, dtype=bool)
            & np.isfinite(clean)
            & (clean > 0.0)
            & (sample_times_s >= t0 - 0.12)
            & (sample_times_s <= t0 + 0.16)
        )
        if np.any(sm):
            trusted_midi = 69.0 + 12.0 * np.log2(clean[sm] / 440.0)
            if np.nanmin(np.abs(trusted_midi - mi)) <= config.trusted_context_tolerance_semitones:
                trusted_support[i] = True

    qualified = finite & (coherent | trusted_support)

    # ------------------------------------------------------------------
    # Existing-valid replacement authority
    # ------------------------------------------------------------------
    # V3R7 separates two questions that HF2 conflated:
    #
    #   (1) Is the primary disagreement genuinely sustained in physical time?
    #   (2) If it is only a micro-disagreement, does the original 100/135/333-Hz
    #       trajectory itself support the ECKF sample as real vocal motion?
    #
    # The persistence lane remains useful for long tracker-state drift (Damelo
    # and similar cases), but a frame only counts as persistently disagreeing
    # when the disagreement occupies a minimum physical duration.  This prevents
    # one 7-10 ms sample in each of two adjacent primary frames from masquerading
    # as sustained drift.
    sample_dt_s = (
        float(np.median(np.diff(sample_times_s)))
        if len(sample_times_s) >= 2 else 0.0
    )
    if not np.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        sample_dt_s = 0.0

    frame_disagreement_masks: list[np.ndarray] = [
        np.zeros(len(original_valid), dtype=bool) for _ in range(n_frames)
    ]
    frame_disagreement_duration_s = np.zeros(n_frames, dtype=np.float64)
    frame_has_drift_disagreement = np.zeros(n_frames, dtype=bool)

    for i in np.flatnonzero(qualified):
        cand = float(candidate[i])
        mask = frame_disagreement_masks[i]
        for j in np.flatnonzero(sample_frame_index == i):
            if not bool(original_valid[j]):
                continue
            cur = float(original_clean[j]) if np.isfinite(original_clean[j]) else float("nan")
            if not np.isfinite(cur) or cur <= 0.0:
                continue
            disagreement_st = abs(12.0 * math.log2(cand / cur))
            if disagreement_st < config.replacement_min_disagreement_semitones:
                continue
            if _integer_related(cand, cur, config.integer_family_tolerance_cents):
                continue
            mask[j] = True

        ix = np.flatnonzero(mask)
        if len(ix):
            frame_has_drift_disagreement[i] = True
            # Persistence is about sustained physical time, not the number of
            # scattered outliers in a frame.  Measure only the longest
            # contiguous disagreement run on the interpretation grid.
            longest = 1
            run = 1
            for a, b in zip(ix[:-1], ix[1:]):
                if int(b) == int(a) + 1:
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 1
            frame_disagreement_duration_s[i] = (
                float(longest) * sample_dt_s if sample_dt_s > 0.0 else 0.0
            )

    persistent_drift = np.zeros(n_frames, dtype=bool)
    radius = max(1, int(config.replacement_persistence_neighbor_frames))
    min_persist_s = float(config.replacement_persistence_min_disagreement_time_s)
    for i in np.flatnonzero(frame_has_drift_disagreement):
        if frame_disagreement_duration_s[i] + 1e-12 < min_persist_s:
            continue
        mi = cmidi[i]
        for k in range(max(0, i - radius), min(n_frames, i + radius + 1)):
            if k == i or not frame_has_drift_disagreement[k] or not np.isfinite(cmidi[k]):
                continue
            if frame_disagreement_duration_s[k] + 1e-12 < min_persist_s:
                continue
            if abs(float(cmidi[k]) - float(mi)) <= config.max_adjacent_primary_step_semitones:
                persistent_drift[i] = True
                break

    # Sample-level trajectory adjudication is evaluated against the immutable
    # V3R5/HF2 input snapshot.  No V3R7 correction may vote for another.
    trajectory_status = np.full(len(original_valid), "not_evaluated", dtype=object)
    trajectory_left_ref_midi = np.full(len(original_valid), np.nan, dtype=np.float64)
    trajectory_right_ref_midi = np.full(len(original_valid), np.nan, dtype=np.float64)

    for i in np.flatnonzero(frame_has_drift_disagreement):
        cand = float(candidate[i])
        disagreement_mask = frame_disagreement_masks[i]
        for j in np.flatnonzero(disagreement_mask):
            status, left_ref, right_ref = _trajectory_adjudication(
                index=int(j), candidate_hz=cand,
                sample_times_s=sample_times_s, valid=original_valid,
                clean_f0_hz=original_clean, disagreement_mask=disagreement_mask,
                cfg=config,
            )
            trajectory_status[j] = status
            if left_ref is not None:
                trajectory_left_ref_midi[j] = float(left_ref)
            if right_ref is not None:
                trajectory_right_ref_midi[j] = float(right_ref)

    # Guard the leading edge of a frame whose primary family itself just made a
    # large jump.  A frame-wide periodicity estimate is not sample-accurate
    # evidence that the new family already owned the very first trajectory point.
    # This protects genuine in-frame glides/transitions without weakening
    # sustained drift or the isolated-false-valid certificate.
    primary_jump_guarded_sample = np.zeros(len(original_valid), dtype=bool)
    for i in range(1, n_frames):
        if not (qualified[i] and qualified[i - 1] and np.isfinite(cmidi[i]) and np.isfinite(cmidi[i - 1])):
            continue
        jump_st = abs(float(cmidi[i]) - float(cmidi[i - 1]))
        if jump_st < config.primary_jump_guard_threshold_semitones:
            continue
        frame_t0 = float(frame_start_times_s[i])
        for j in np.flatnonzero(sample_frame_index == i):
            dt = float(sample_times_s[j] - frame_t0)
            if -1e-9 <= dt <= config.primary_jump_guard_time_s + 1e-12:
                primary_jump_guarded_sample[j] = True

    # Narrow exception to generic trajectory adjudication: repair a short false-valid
    # island when it is topologically isolated (1-2 accepted interpretation-grid samples,
    # rejected on both sides), overwhelming same-frame primary evidence points
    # elsewhere, and that primary family agrees with trusted pitch on BOTH
    # sides.  Across the frozen 12-song V3R5 corpus this certificate is designed
    # for cases such as Comomelo ~50.24 and Bodies ~116.83-116.84, not for
    # continuous expressive motion.
    isolated_false_valid_sample = np.zeros(len(original_valid), dtype=bool)
    visited_islands: set[tuple[int, int]] = set()
    for j in np.flatnonzero(original_valid):
        lo, hi = _valid_island_bounds(original_valid, int(j))
        key = (lo, hi)
        if key in visited_islands:
            continue
        visited_islands.add(key)
        island_len = hi - lo + 1
        if island_len > int(config.isolated_false_valid_max_samples):
            continue
        # Must be truly isolated, not merely at file boundary or inside a
        # longer valid phrase.
        if lo <= 0 or hi + 1 >= len(original_valid):
            continue
        if bool(original_valid[lo - 1]) or bool(original_valid[hi + 1]):
            continue

        fis = {int(sample_frame_index[q]) for q in range(lo, hi + 1)}
        if len(fis) != 1:
            continue
        fi = next(iter(fis))
        if fi < 0 or fi >= n_frames or not qualified[fi]:
            continue
        if acf[fi] < config.isolated_false_valid_min_acf_peak:
            continue
        if cmn[fi] > config.isolated_false_valid_max_cmndf_minimum:
            continue
        if not trusted_support[fi]:
            continue

        cand = float(candidate[fi])
        island_disagrees = True
        for q in range(lo, hi + 1):
            cur = float(original_clean[q]) if np.isfinite(original_clean[q]) else float("nan")
            if not np.isfinite(cur) or cur <= 0.0:
                island_disagrees = False
                break
            disagreement_st = abs(12.0 * math.log2(cand / cur))
            if disagreement_st < config.replacement_min_disagreement_semitones:
                island_disagrees = False
                break
            if _integer_related(cand, cur, config.integer_family_tolerance_cents):
                island_disagrees = False
                break
        if not island_disagrees:
            continue

        left_hz, right_hz = _nearest_external_valid_hz(
            start=lo, stop=hi,
            sample_times_s=sample_times_s,
            valid=original_valid, clean_f0_hz=original_clean,
            max_dt_s=config.isolated_false_valid_context_time_s,
        )
        if left_hz is None or right_hz is None:
            continue
        cand_m = _midi(cand)
        tol = config.isolated_false_valid_context_tolerance_semitones
        if abs(cand_m - _midi(left_hz)) > tol:
            continue
        if abs(cand_m - _midi(right_hz)) > tol:
            continue

        isolated_false_valid_sample[lo:hi + 1] = True

    # V3R7 keeps rejected-pitch recovery exactly as conservative as V3R6/HF2.  Generic UNVOICED
    # frames are not promoted merely because a periodic estimator found a peak.
    # Promotion authority is limited to:
    #   (a) tracker-state/range failures (BELOW_VOCAL_FLOOR etc.) backed by a
    #       sustained primary family, or
    #   (b) an integer-family dispute that independent temporal context resolves
    #       decisively (the classic ACF-vs-CMNDF octave disagreement).
    sustained_primary = np.zeros(n_frames, dtype=bool)
    for i in np.flatnonzero(qualified):
        for k in (i - 1, i + 1):
            if 0 <= k < n_frames and qualified[k] and np.isfinite(cmidi[k]):
                if abs(cmidi[k] - cmidi[i]) <= config.max_adjacent_primary_step_semitones:
                    sustained_primary[i] = True
                    break

    rows: list[PrimaryReconciliationV3R7Row] = []
    for i in range(n_frames):
        promoted = 0
        replaced = 0
        reason = "NO_STRONG_PRIMARY_CANDIDATE"
        action = "NONE"

        if finite[i] and not qualified[i]:
            reason = "PRIMARY_CANDIDATE_LACKS_TEMPORAL_SUPPORT"
        elif qualified[i]:
            reason = "QUALIFIED_PRIMARY_EVIDENCE"
            idx = np.flatnonzero(sample_frame_index == i)
            for j in idx:
                # Never convert authoritative energy silence into pitch.
                if states[i] == int(FrameAcousticState.LOW_ENERGY_SILENCE):
                    continue
                cand = float(candidate[i])
                if not bool(valid[j]):
                    original_reason = str(offline_validity.reason[j])
                    ordinary_tracker_failure = original_reason in {
                        "BELOW_VOCAL_FLOOR",
                        "RANGE_ZERO_VETO",
                        "LOCK_LOST_DISCONTINUITY",
                        "ADJUDICATION_REACQUISITION_ISLAND_REJECTED",
                        "INTERVAL_READJUDICATION_ABSTAIN",
                    }
                    dispute_rescue = (
                        source[i] in {
                            "integer_family_context_resolved",
                            "range_resolved_integer_family",
                        }
                        and acf[i] >= 0.95
                        and cmn[i] <= 0.05
                        and trusted_support[i]
                    )
                    if not (
                        (ordinary_tracker_failure and sustained_primary[i])
                        or dispute_rescue
                    ):
                        continue
                    valid[j] = True
                    clean[j] = cand
                    offline_validity.reason[j] = "PRIMARY_EVIDENCE_RECOVERED_V3R7"
                    offline_validity.correction_reason[j] = "PRIMARY_EVIDENCE_RECOVERED_V3R7"

                    # Reconcile the frame passport with the newly authoritative
                    # pitch ownership.  The trajectory interpreter deliberately
                    # rejects valid=True on any non-VOICED_TRACKED frame, so a
                    # primary-evidence promotion is incomplete unless the frame
                    # state is promoted too.  LOW_ENERGY_SILENCE was already
                    # excluded above and can never reach this branch.
                    frame_state_ref[i] = int(FrameAcousticState.VOICED_TRACKED)
                    states[i] = int(FrameAcousticState.VOICED_TRACKED)

                    promoted += 1
                    continue

                cur = float(clean[j]) if np.isfinite(clean[j]) else float("nan")
                if not np.isfinite(cur) or cur <= 0.0:
                    continue
                # Existing-valid replacement has three independent authorities:
                #
                # 1) isolated false-valid island: overwhelming primary evidence
                #    plus trusted pitch on both sides;
                # 2) persistent drift: large disagreement occupies real physical
                #    time in adjacent primary frames;
                # 3) local trajectory adjudication: the original trusted path
                #    itself points forward/bilaterally toward the primary family
                #    instead of supporting the disputed ECKF excursion.
                #
                # A coherent outgoing/V-shaped ECKF excursion is preserved even
                # when a coarse primary frame prefers another family.
                disagreement_st = abs(12.0 * math.log2(cand / cur))
                if disagreement_st < config.replacement_min_disagreement_semitones:
                    continue
                if _integer_related(cand, cur, config.integer_family_tolerance_cents):
                    continue

                if bool(isolated_false_valid_sample[j]):
                    clean[j] = cand
                    offline_validity.reason[j] = "PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R7"
                    offline_validity.correction_reason[j] = "PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R7"
                    replaced += 1
                    continue

                status = str(trajectory_status[j])

                # Sustained disagreement in real physical time remains the
                # strongest generic drift authority.  The duration gate above
                # is what distinguishes Damelo-like tracker drift from a single
                # expressive 7-10 ms excursion in each of two adjacent frames.
                if persistent_drift[i]:
                    clean[j] = cand
                    offline_validity.reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R7_PERSISTENT"
                    offline_validity.correction_reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R7_PERSISTENT"
                    replaced += 1
                    continue

                # Without sustained drift authority, a strong future vote for
                # the original ECKF path protects real fast dips, mordents and
                # phrase-release motion.
                if status == "current":
                    continue

                if status != "candidate":
                    continue

                if bool(primary_jump_guarded_sample[j]):
                    continue

                clean[j] = cand
                offline_validity.reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R7_TRAJECTORY"
                offline_validity.correction_reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R7_TRAJECTORY"
                replaced += 1

            if promoted and replaced:
                action = "PROMOTE_AND_REPLACE"
            elif promoted:
                action = "PROMOTE_REJECTED"
            elif replaced:
                action = "REPLACE_TRACKER_DRIFT"
            else:
                action = "EVIDENCE_CONFIRMED_NO_CHANGE"

        frame_idx = np.flatnonzero(sample_frame_index == i)
        traj_cand_count = int(np.count_nonzero(trajectory_status[frame_idx] == "candidate"))
        traj_cur_count = int(np.count_nonzero(trajectory_status[frame_idx] == "current"))
        traj_inc_count = int(np.count_nonzero(
            np.isin(trajectory_status[frame_idx], ["inconclusive", "not_evaluated"])
            & frame_disagreement_masks[i][frame_idx]
        ))
        traj_guard_count = int(np.count_nonzero(
            primary_jump_guarded_sample[frame_idx] & frame_disagreement_masks[i][frame_idx]
        ))

        rows.append(PrimaryReconciliationV3R7Row(
            frame_index=int(i),
            frame_start_time_s=float(frame_start_times_s[i]),
            acoustic_state=int(states[i]),
            acf_peak=float(acf[i]) if np.isfinite(acf[i]) else float("nan"),
            cmndf_minimum=float(cmn[i]) if np.isfinite(cmn[i]) else float("nan"),
            acf_frequency_hz=float(fa[i]) if np.isfinite(fa[i]) else float("nan"),
            cmndf_frequency_hz=float(fc[i]) if np.isfinite(fc[i]) else float("nan"),
            candidate_hz=float(candidate[i]) if np.isfinite(candidate[i]) else None,
            candidate_source=str(source[i]),
            coherent_support=bool(coherent[i]),
            trusted_context_support=bool(trusted_support[i]),
            trajectory_candidate_support_count=traj_cand_count,
            trajectory_current_support_count=traj_cur_count,
            trajectory_inconclusive_count=traj_inc_count,
            trajectory_jump_guard_count=traj_guard_count,
            promoted_sample_count=int(promoted),
            replaced_sample_count=int(replaced),
            action=action,
            reason=reason,
        ))

    return rows
