from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .frame_evidence import FrameAcousticState


@dataclass(frozen=True)
class PrimaryReconciliationConfig:
    """Conservative V3R6 reconciliation of ECKF ownership with primary periodicity.

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
    # pitch is to recover.  Generic drift replacement therefore requires the
    # disagreement itself to persist into an adjacent primary frame.
    replacement_persistence_neighbor_frames: int = 1
    # Separate narrow certificate for a short false-valid island: one or two
    # accepted 100-Hz samples, bounded by rejected samples, may be replaced by
    # overwhelming same-frame primary evidence when that primary family also
    # agrees with trusted pitch on BOTH sides.
    isolated_false_valid_max_samples: int = 2
    isolated_false_valid_min_acf_peak: float = 0.95
    isolated_false_valid_max_cmndf_minimum: float = 0.05
    isolated_false_valid_context_time_s: float = 0.20
    isolated_false_valid_context_tolerance_semitones: float = 2.0


@dataclass(frozen=True)
class PrimaryReconciliationRow:
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


def _nearest_context_midi(
    frame_i: int,
    frame_candidates: np.ndarray,
    frame_times: np.ndarray,
    sample_times_s: np.ndarray,
    valid: np.ndarray,
    clean_f0_hz: np.ndarray,
    cfg: PrimaryReconciliationConfig,
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


def apply_primary_reconciliation_v3r6(
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
    config: PrimaryReconciliationConfig = PrimaryReconciliationConfig(),
) -> list[PrimaryReconciliationRow]:
    """Repair tracker ownership when primary periodicity is stronger than ECKF state.

    Two deliberately narrow actions are permitted:

    * PRIMARY_EVIDENCE_RECOVERED: promote rejected samples when a strong,
      range-valid primary family is temporally coherent.
    * PRIMARY_TRACKER_DRIFT_REPLACED: replace a corrected-valid ECKF value only
      when the large non-integer disagreement itself persists across multiple
      primary frames.  Integer-family disputes remain the responsibility of
      the existing V3R3/V3R4/V3R5 register machinery.
    * PRIMARY_ISOLATED_FALSE_VALID_REPLACED: a separate narrow certificate for
      a 1-2 sample accepted island bounded by rejected samples, when overwhelming
      same-frame primary evidence agrees with trusted pitch on both sides.

    LOW_ENERGY_SILENCE is never promoted.
    """

    sample_times_s = np.asarray(sample_times_s, dtype=np.float64)
    sample_frame_index = np.asarray(sample_frame_index, dtype=np.int64)
    frame_start_times_s = np.asarray(frame_start_times_s, dtype=np.float64)
    # Keep a writable reference to the authoritative frame passport.
    # V3R6 can promote a previously unresolved/rejected frame to tracked
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
        raise ValueError("V3R6 primary reconciliation frame arrays must have equal length")

    # Work from the state after V3R5 adjudication feedback.
    valid = offline_validity.valid
    clean = offline_validity.clean_f0_hz

    # All replacement certificates are judged against the V3R5-authoritative
    # state that entered this stage, not against samples V3R6 has already
    # modified earlier in the loop.  This prevents one V3R6 decision from
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
    # A primary family merely existing in adjacent frames is not enough to
    # overwrite an already-valid ECKF trajectory.  Fast vocal motion can be
    # real inside a ~46-ms primary frame.  Generic drift replacement now
    # requires the *large non-integer disagreement itself* to persist into an
    # adjacent primary frame.  This preserves sustained drift repairs (e.g.
    # Damelo) while blocking one-frame / one-sample flattening of mordents,
    # glissandi and melismas.
    frame_has_drift_disagreement = np.zeros(n_frames, dtype=bool)
    for i in np.flatnonzero(qualified):
        cand = float(candidate[i])
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
            frame_has_drift_disagreement[i] = True
            break

    persistent_drift = np.zeros(n_frames, dtype=bool)
    radius = max(1, int(config.replacement_persistence_neighbor_frames))
    for i in np.flatnonzero(frame_has_drift_disagreement):
        mi = cmidi[i]
        for k in range(max(0, i - radius), min(n_frames, i + radius + 1)):
            if k == i or not frame_has_drift_disagreement[k] or not np.isfinite(cmidi[k]):
                continue
            if abs(float(cmidi[k]) - float(mi)) <= config.max_adjacent_primary_step_semitones:
                persistent_drift[i] = True
                break

    # Narrow exception to the multi-frame rule: repair a short false-valid
    # island when it is topologically isolated (1-2 accepted 100-Hz samples,
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

    # V3R6 intentionally keeps the first release narrow.  Generic UNVOICED
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

    rows: list[PrimaryReconciliationRow] = []
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
                    offline_validity.reason[j] = "PRIMARY_EVIDENCE_RECOVERED_V3R6"
                    offline_validity.correction_reason[j] = "PRIMARY_EVIDENCE_RECOVERED_V3R6"

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
                # Existing-valid replacement has two authorities only:
                #
                # 1) generic tracker drift: the large non-integer disagreement
                #    itself persists across multiple primary frames;
                # 2) isolated false-valid island: a 1-2 sample accepted island
                #    is bounded by rejected samples and overwhelming primary
                #    evidence agrees with trusted pitch on both sides.
                #
                # Mere sustained primary periodicity is NOT sufficient.
                disagreement_st = abs(12.0 * math.log2(cand / cur))
                if disagreement_st < config.replacement_min_disagreement_semitones:
                    continue
                if _integer_related(cand, cur, config.integer_family_tolerance_cents):
                    continue

                if bool(isolated_false_valid_sample[j]):
                    clean[j] = cand
                    offline_validity.reason[j] = "PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R6"
                    offline_validity.correction_reason[j] = "PRIMARY_ISOLATED_FALSE_VALID_REPLACED_V3R6"
                    replaced += 1
                    continue

                if not persistent_drift[i]:
                    continue

                clean[j] = cand
                offline_validity.reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R6_PERSISTENT"
                offline_validity.correction_reason[j] = "PRIMARY_TRACKER_DRIFT_REPLACED_V3R6_PERSISTENT"
                replaced += 1

            if promoted and replaced:
                action = "PROMOTE_AND_REPLACE"
            elif promoted:
                action = "PROMOTE_REJECTED"
            elif replaced:
                action = "REPLACE_TRACKER_DRIFT"
            else:
                action = "EVIDENCE_CONFIRMED_NO_CHANGE"

        rows.append(PrimaryReconciliationRow(
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
            promoted_sample_count=int(promoted),
            replaced_sample_count=int(replaced),
            action=action,
            reason=reason,
        ))

    return rows
