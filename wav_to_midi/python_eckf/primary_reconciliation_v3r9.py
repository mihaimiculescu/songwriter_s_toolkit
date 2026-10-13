from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .frame_evidence import FrameAcousticState


@dataclass(frozen=True)
class PrimaryReconciliationV3R9Config:
    """Conservative V3R9 reconciliation of ECKF ownership with primary periodicity.

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
    # pitch is to recover.  V3R9 therefore adjudicates the original local
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

    # Persistent disagreement is measured as one CONTINUOUS physical-time run
    # across interpretation samples and primary-frame boundaries.  This avoids
    # V3R7's accidental dependence on where a ~46 ms primary frame happened to
    # start/end.  At the 135-Hz production grid, 18 ms requires at least three
    # consecutive disagreeing samples; a two-sample expressive micro-excursion
    # remains below the persistence authority.
    replacement_persistence_min_disagreement_time_s: float = 0.018

    # Hysteresis for the shoulders of an ALREADY-ESTABLISHED persistent drift
    # run.  Entry remains strict at 3.5 st; once persistence has been proved,
    # the run may continue through adjacent samples down to 2.5 st while the
    # qualified primary family remains continuous.  This repairs threshold
    # clipping without making a 2.5-st micro-excursion capable of starting a
    # replacement run by itself.
    replacement_hysteresis_exit_disagreement_semitones: float = 2.5

    # Narrow register-outlier certificate for an already-valid low-register
    # sample.
    # If ACF and CMNDF overwhelmingly agree on one primary family, the accepted
    # sample is an integer-related register outlier, and trusted pitch on BOTH
    # sides agrees with the primary family, repair the lone register glitch.
    # This remains separate from generic non-integer tracker-drift replacement.
    embedded_register_min_acf_peak: float = 0.95
    embedded_register_max_cmndf_minimum: float = 0.05
    embedded_register_context_time_s: float = 0.050
    embedded_register_context_tolerance_semitones: float = 1.5
    embedded_register_primary_context_frames: int = 2

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
class PrimaryReconciliationV3R9Row:
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
    continuous_persistence_count: int
    hysteresis_shoulder_count: int
    embedded_register_outlier_count: int
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
    cfg: PrimaryReconciliationV3R9Config,
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


def _continuous_persistence_mask(
    *,
    sample_times_s: np.ndarray,
    disagreement_mask: np.ndarray,
    sample_candidate_midi: np.ndarray,
    min_duration_s: float,
    max_candidate_step_semitones: float,
) -> np.ndarray:
    """Mark disagreement samples belonging to one sustained physical-time run.

    A run may cross any number of primary-frame boundaries.  It remains one run
    only while interpretation samples are consecutive and the primary candidate
    family itself stays locally compatible.  This removes V3R7's frame-boundary
    butterfly: real drift is not split merely because 46-ms primary windows split
    it, while a one/two-sample expressive excursion still lacks persistence.
    """
    times = np.asarray(sample_times_s, dtype=np.float64)
    mask = np.asarray(disagreement_mask, dtype=bool)
    cand = np.asarray(sample_candidate_midi, dtype=np.float64)
    out = np.zeros(len(mask), dtype=bool)
    if len(mask) == 0:
        return out

    if len(times) >= 2:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 0.0
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.0

    ix = np.flatnonzero(mask & np.isfinite(cand))
    if len(ix) == 0:
        return out

    run_start = int(ix[0])
    run_end = int(ix[0])

    def commit(a: int, b: int) -> None:
        duration = float(times[b] - times[a]) + dt if dt > 0.0 else 0.0
        if duration + 1e-12 >= float(min_duration_s):
            out[a:b + 1] = mask[a:b + 1]

    prev = int(ix[0])
    for raw in ix[1:]:
        j = int(raw)
        contiguous = j == prev + 1
        family_continuous = (
            contiguous
            and np.isfinite(cand[prev])
            and np.isfinite(cand[j])
            and abs(float(cand[j]) - float(cand[prev]))
                <= float(max_candidate_step_semitones)
        )
        if family_continuous:
            run_end = j
        else:
            commit(run_start, run_end)
            run_start = run_end = j
        prev = j
    commit(run_start, run_end)
    return out



def _extend_persistent_drift_hysteresis(
    *,
    core_mask: np.ndarray,
    original_valid: np.ndarray,
    sample_candidate_midi: np.ndarray,
    sample_disagreement_st: np.ndarray,
    exit_threshold_semitones: float,
    max_candidate_step_semitones: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend established drift runs through their lower-disagreement shoulders.

    This is true hysteresis: a shoulder can NEVER create a drift run.  A run
    must first satisfy the strict >=3.5-st entry threshold and physical-time
    persistence test.  Only then may directly adjacent samples be absorbed while
    the disagreement remains >= the lower exit threshold and the qualified
    primary candidate family remains continuous.

    Returns (full_hysteretic_mask, shoulder_only_mask).
    """
    core = np.asarray(core_mask, dtype=bool)
    valid = np.asarray(original_valid, dtype=bool)
    cand = np.asarray(sample_candidate_midi, dtype=np.float64)
    disagreement = np.asarray(sample_disagreement_st, dtype=np.float64)
    out = core.copy()
    shoulder = np.zeros(len(core), dtype=bool)

    ix = np.flatnonzero(core)
    if len(ix) == 0:
        return out, shoulder

    runs: list[tuple[int, int]] = []
    start = prev = int(ix[0])
    for raw in ix[1:]:
        j = int(raw)
        if j == prev + 1:
            prev = j
        else:
            runs.append((start, prev))
            start = prev = j
    runs.append((start, prev))

    def may_extend(j: int, neighbor: int) -> bool:
        if j < 0 or j >= len(core):
            return False
        if not bool(valid[j]):
            return False
        if not (np.isfinite(cand[j]) and np.isfinite(cand[neighbor])):
            return False
        if not np.isfinite(disagreement[j]):
            return False
        if float(disagreement[j]) + 1e-12 < float(exit_threshold_semitones):
            return False
        if abs(float(cand[j]) - float(cand[neighbor])) > float(max_candidate_step_semitones):
            return False
        return True

    for start, stop in runs:
        neighbor = start
        j = start - 1
        while may_extend(j, neighbor):
            if not out[j]:
                out[j] = True
                shoulder[j] = True
            neighbor = j
            j -= 1

        neighbor = stop
        j = stop + 1
        while may_extend(j, neighbor):
            if not out[j]:
                out[j] = True
                shoulder[j] = True
            neighbor = j
            j += 1

    return out, shoulder


def _nearest_strong_primary_context_midi(
    *,
    frame_i: int,
    direction: int,
    candidate_midi: np.ndarray,
    strong_primary: np.ndarray,
    max_frames: int,
) -> Optional[float]:
    """Nearest overwhelming ACF/CMNDF-agreeing primary family on one side."""
    n = len(candidate_midi)
    for step in range(1, int(max_frames) + 1):
        k = int(frame_i) + int(direction) * step
        if k < 0 or k >= n:
            break
        if bool(strong_primary[k]) and np.isfinite(candidate_midi[k]):
            return float(candidate_midi[k])
    return None


def _nearest_context_midi(
    frame_i: int,
    frame_candidates: np.ndarray,
    frame_times: np.ndarray,
    sample_times_s: np.ndarray,
    valid: np.ndarray,
    clean_f0_hz: np.ndarray,
    cfg: PrimaryReconciliationV3R9Config,
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


def apply_primary_reconciliation_v3r9(
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
    config: PrimaryReconciliationV3R9Config = PrimaryReconciliationV3R9Config(),
) -> list[PrimaryReconciliationV3R9Row]:
    """Repair tracker ownership when primary periodicity is stronger than ECKF state.

    Four deliberately narrow actions are permitted:

    * PRIMARY_EVIDENCE_RECOVERED: promote rejected samples when a strong,
      range-valid primary family is temporally coherent.
    * PRIMARY_TRACKER_DRIFT_REPLACED: replace a corrected-valid ECKF value when
      either the disagreement is sustained continuously in physical time across
      primary-frame boundaries or the immutable local ECKF trajectory itself
      favors the primary family.
    * PRIMARY_EMBEDDED_REGISTER_OUTLIER_REPLACED: repair a lone already-valid
      integer-family register outlier only when overwhelming ACF/CMNDF evidence
      and trusted pitch on both sides agree on the primary family.
    * PRIMARY_ISOLATED_FALSE_VALID_REPLACED: a separate narrow certificate for
      a 1-2 sample accepted island bounded by rejected samples, when overwhelming
      same-frame primary evidence agrees with trusted pitch on both sides.

    LOW_ENERGY_SILENCE is never promoted.
    """

    sample_times_s = np.asarray(sample_times_s, dtype=np.float64)
    sample_frame_index = np.asarray(sample_frame_index, dtype=np.int64)
    frame_start_times_s = np.asarray(frame_start_times_s, dtype=np.float64)
    # Keep a writable reference to the authoritative frame passport.
    # V3R9 can promote a previously unresolved/rejected frame to tracked
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
        raise ValueError("V3R9 primary reconciliation frame arrays must have equal length")

    # Work from the state after V3R5 adjudication feedback.
    valid = offline_validity.valid
    clean = offline_validity.clean_f0_hz

    # All replacement certificates are judged against the V3R5-authoritative
    # state that entered this stage, not against samples V3R9 has already
    # modified earlier in the loop.  This prevents one V3R9 decision from
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
    # V3R9 separates two questions that HF2 conflated:
    #
    #   (1) Is the primary disagreement genuinely sustained in physical time?
    #   (2) If it is only a micro-disagreement, does the original 100/135/333-Hz
    #       trajectory itself support the ECKF sample as real vocal motion?
    #
    # The persistence lane remains useful for long tracker-state drift (Damelo
    # and similar cases), but persistence is measured across the continuous
    # interpretation trajectory rather than separately inside each primary frame.
    # A short V-shaped/mordent excursion therefore stays below the duration gate,
    # while a real drift crossing an arbitrary frame boundary remains one run.
    frame_disagreement_masks: list[np.ndarray] = [
        np.zeros(len(original_valid), dtype=bool) for _ in range(n_frames)
    ]
    frame_has_drift_disagreement = np.zeros(n_frames, dtype=bool)
    all_drift_disagreement = np.zeros(len(original_valid), dtype=bool)
    # candidate/disagreement arrays cover every generic non-integer existing-valid
    # sample in a qualified frame, not only >=3.5-st entry points.  The wider
    # arrays are used ONLY for hysteretic continuation after a strict persistent
    # core has already been established.
    sample_candidate_midi = np.full(len(original_valid), np.nan, dtype=np.float64)
    sample_disagreement_st = np.full(len(original_valid), np.nan, dtype=np.float64)

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
            # Generic drift replacement deliberately leaves integer-family
            # disputes to the register machinery / embedded-register certificate.
            if _integer_related(cand, cur, config.integer_family_tolerance_cents):
                continue

            sample_candidate_midi[j] = float(cmidi[i])
            sample_disagreement_st[j] = float(disagreement_st)

            # Strict ENTRY threshold.  Samples below this value cannot create a
            # drift run; they may only be absorbed later as hysteretic shoulders
            # of a run that already proved physical-time persistence.
            if disagreement_st < config.replacement_min_disagreement_semitones:
                continue
            mask[j] = True
            all_drift_disagreement[j] = True

        if np.any(mask):
            frame_has_drift_disagreement[i] = True

    # V3R9 surgery: persistence is one continuous physical-time property, not
    # two independent per-frame properties.  Runs may cross ~46 ms primary
    # frame boundaries as long as the primary family remains compatible.
    continuous_persistent_drift = _continuous_persistence_mask(
        sample_times_s=sample_times_s,
        disagreement_mask=all_drift_disagreement,
        sample_candidate_midi=sample_candidate_midi,
        min_duration_s=config.replacement_persistence_min_disagreement_time_s,
        max_candidate_step_semitones=config.max_adjacent_primary_step_semitones,
    )

    # V3R9 final CLI squeeze: once a strict >=3.5-st run has proved persistence,
    # let it retain ownership through contiguous >=2.5-st shoulders.  This is
    # intentionally one-way hysteresis: shoulder samples cannot start a run.
    hysteretic_persistent_drift, hysteresis_shoulder_sample = _extend_persistent_drift_hysteresis(
        core_mask=continuous_persistent_drift,
        original_valid=original_valid,
        sample_candidate_midi=sample_candidate_midi,
        sample_disagreement_st=sample_disagreement_st,
        exit_threshold_semitones=config.replacement_hysteresis_exit_disagreement_semitones,
        max_candidate_step_semitones=config.max_adjacent_primary_step_semitones,
    )

    # Sample-level trajectory adjudication is evaluated against the immutable
    # V3R5/HF2 input snapshot.  No V3R9 correction may vote for another.
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

    # V3R9 embedded-register surgery.  V3R8 tried to certify the lone
    # Bodies-style low-register glitch from trusted corrected-valid pitch on
    # both sides.  That can fail because some neighboring low-family samples are
    # not corrected until the later register-readjudication stage.  The primary
    # evidence needed to expose the glitch already exists BEFORE that stage, so
    # V3R9 adds an independent primary-context certificate.
    #
    # Safety is intentionally asymmetric: this certificate may only repair an
    # embedded LOW tracker register upward to an overwhelming higher primary
    # family.  Corpus inspection showed that overwhelming lower primary families
    # are a common subharmonic pattern and must not be allowed to drag valid high
    # ECKF ownership downward through this lane.
    embedded_register_outlier_sample = np.zeros(len(original_valid), dtype=bool)
    embedded_primary_strong = (
        finite
        & (source == "acf_cmndf_agreement")
        & (acf >= config.embedded_register_min_acf_peak)
        & (cmn <= config.embedded_register_max_cmndf_minimum)
    )

    for j in np.flatnonzero(original_valid):
        fi = int(sample_frame_index[j])
        if fi < 0 or fi >= n_frames or not bool(embedded_primary_strong[fi]):
            continue

        cur = float(original_clean[j]) if np.isfinite(original_clean[j]) else float("nan")
        cand = float(candidate[fi]) if np.isfinite(candidate[fi]) else float("nan")
        if not (np.isfinite(cur) and cur > 0.0 and np.isfinite(cand) and cand > 0.0):
            continue

        # Upward-only embedded register rescue.  A downward raw-primary octave is
        # far more likely to be subharmonic evidence than proof that an accepted
        # high tracker sample is wrong.
        if cand <= cur:
            continue

        disagreement_st = abs(12.0 * math.log2(cand / cur))
        if disagreement_st < config.replacement_min_disagreement_semitones:
            continue
        if not _integer_related(cand, cur, config.integer_family_tolerance_cents):
            continue

        cand_m = _midi(cand)
        tol = float(config.embedded_register_context_tolerance_semitones)

        # Certificate A: already-trusted corrected-valid pitch on both sides.
        left_hz, right_hz = _nearest_external_valid_hz(
            start=int(j), stop=int(j),
            sample_times_s=sample_times_s,
            valid=original_valid, clean_f0_hz=original_clean,
            max_dt_s=config.embedded_register_context_time_s,
        )
        trusted_context_ok = (
            left_hz is not None and right_hz is not None
            and abs(cand_m - _midi(left_hz)) <= tol
            and abs(cand_m - _midi(right_hz)) <= tol
        )

        # Certificate B: overwhelming ACF/CMNDF-agreeing primary families on
        # both neighboring sides.  This exists before later register
        # readjudication and fixes the V3R8 ordering blind spot.
        left_primary_m = _nearest_strong_primary_context_midi(
            frame_i=fi, direction=-1,
            candidate_midi=cmidi,
            strong_primary=embedded_primary_strong,
            max_frames=config.embedded_register_primary_context_frames,
        )
        right_primary_m = _nearest_strong_primary_context_midi(
            frame_i=fi, direction=+1,
            candidate_midi=cmidi,
            strong_primary=embedded_primary_strong,
            max_frames=config.embedded_register_primary_context_frames,
        )
        primary_context_ok = (
            left_primary_m is not None and right_primary_m is not None
            and abs(cand_m - float(left_primary_m)) <= tol
            and abs(cand_m - float(right_primary_m)) <= tol
        )

        if not (trusted_context_ok or primary_context_ok):
            continue

        embedded_register_outlier_sample[j] = True

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

    # V3R9 keeps rejected-pitch recovery exactly as conservative as V3R6/HF2.  Generic UNVOICED
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

    rows: list[PrimaryReconciliationV3R9Row] = []
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
                    offline_validity.reason[j] = "PRIMARY_EVIDENCE_RECOVERED_V3R9"
                    offline_validity.correction_reason[j] = "PRIMARY_EVIDENCE_RECOVERED_V3R9"

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
                # Existing-valid replacement has four independent authorities:
                #
                # 1) embedded register outlier: a lone integer-family glitch
                #    bracketed by trusted pitch supporting the primary family;
                # 2) isolated false-valid island: overwhelming primary evidence
                #    plus trusted pitch on both sides;
                # 3) continuous persistent drift: large disagreement occupies
                #    enough real physical time even across primary-frame boundaries;
                # 4) local trajectory adjudication: the original trusted path
                #    itself points forward/bilaterally toward the primary family
                #    instead of supporting the disputed ECKF excursion.
                #
                # A coherent outgoing/V-shaped ECKF excursion is preserved even
                # when a coarse primary frame prefers another family.
                disagreement_st = abs(12.0 * math.log2(cand / cur))
                # Normal replacement authority still enters at >=3.5 st.  The
                # sole exception is a hysteretic shoulder that belongs to an
                # already-established persistent run; by construction such a
                # shoulder cannot create a run on its own.
                if (
                    disagreement_st < config.replacement_min_disagreement_semitones
                    and not bool(hysteresis_shoulder_sample[j])
                ):
                    continue

                # Register-family disagreements remain outside generic drift
                # replacement, except for the very narrow embedded one-sample
                # outlier certificate above.
                if bool(embedded_register_outlier_sample[j]):
                    clean[j] = cand
                    offline_validity.reason[j] = "PRIMARY_EMBEDDED_REGISTER_OUTLIER_REPLACED_V3R9"
                    offline_validity.correction_reason[j] = "PRIMARY_EMBEDDED_REGISTER_OUTLIER_REPLACED_V3R9"
                    replaced += 1
                    continue

                if _integer_related(cand, cur, config.integer_family_tolerance_cents):
                    continue

                if bool(isolated_false_valid_sample[j]):
                    clean[j] = cand
                    offline_validity.reason[j] = "PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R9"
                    offline_validity.correction_reason[j] = "PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R9"
                    replaced += 1
                    continue

                status = str(trajectory_status[j])

                # Sustained disagreement in real physical time remains the
                # strongest generic drift authority.  V3R9 measures that run
                # continuously across primary-frame boundaries, so a boundary
                # can no longer manufacture a false "short" disagreement.
                if bool(hysteretic_persistent_drift[j]):
                    # Persistence removes the arbitrary primary-frame boundary,
                    # and hysteresis prevents the established run from being
                    # clipped at a 3.5-st threshold shoulder.  Neither authority
                    # overrules the leading-edge primary-family-jump protection.
                    if bool(primary_jump_guarded_sample[j]):
                        continue
                    clean[j] = cand
                    if bool(hysteresis_shoulder_sample[j]):
                        repl_reason = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R9_HYSTERESIS"
                    else:
                        repl_reason = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R9_CONTINUOUS"
                    offline_validity.reason[j] = repl_reason
                    offline_validity.correction_reason[j] = repl_reason
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
                offline_validity.reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R9_TRAJECTORY"
                offline_validity.correction_reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R9_TRAJECTORY"
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
        continuous_persist_count = int(np.count_nonzero(
            continuous_persistent_drift[frame_idx]
        ))
        hysteresis_shoulder_count = int(np.count_nonzero(
            hysteresis_shoulder_sample[frame_idx]
        ))
        embedded_register_count = int(np.count_nonzero(
            embedded_register_outlier_sample[frame_idx]
        ))

        rows.append(PrimaryReconciliationV3R9Row(
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
            continuous_persistence_count=continuous_persist_count,
            hysteresis_shoulder_count=hysteresis_shoulder_count,
            embedded_register_outlier_count=embedded_register_count,
            promoted_sample_count=int(promoted),
            replaced_sample_count=int(replaced),
            action=action,
            reason=reason,
        ))

    return rows
