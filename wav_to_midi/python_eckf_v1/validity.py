from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .trajectory_resolver import vocal_transition_penalty


@dataclass
class PitchValidityConfig:
    analysis_hz: float = 100.0

    min_vocal_hz: float = 120.0
    max_vocal_hz: float = 1500.0

    # Maximum instantaneous movement between adjacent accepted
    # 10 ms observations.
    max_step_semitones: float = 4.0

    # Tiny interruptions shorter than this can be bridged when the
    # trajectories on both sides agree.
    bridge_gap_ms: float = 30.0

    # A newly appearing voiced island must persist at least this long
    # to be accepted on its own.
    min_island_ms: float = 50.0

    # Context examined on each side of a gap/island.
    context_ms: float = 120.0

    # Pitch agreement required for bridging a short interruption.
    bridge_pitch_tolerance_semitones: float = 3.0


@dataclass
class PitchValidityResult:
    sample_index: np.ndarray
    time_s: np.ndarray
    f0_hz: np.ndarray

    valid: np.ndarray
    reason: np.ndarray

    cleaned_f0_hz: np.ndarray


@dataclass(frozen=True)
class OfflineValidityCorrectionConfig:
    """
    Second-pass, offline-only correction.

    This pass does NOT replace analyse_pitch_validity().
    It consumes its result and may promote selected rejected raw-F0
    samples using past/future trajectory evidence.
    """

    # Broad working pitch range used by the current validity machinery.
    min_vocal_hz: float = 120.0
    max_vocal_hz: float = 1500.0

    # Local trajectory continuity.
    #
    # This is deliberately an ADJACENT-step condition only.
    # There is NO maximum total trajectory span.
    max_adjacent_step_semitones: float = 4.0

    # Robust accepted context on each side.
    context_ms: float = 120.0

    # Minimum coherent trajectory length.
    min_subrun_ms: float = 30.0

    # Excursion / return candidate.
    ornament_max_ms: float = 250.0
    ornament_min_span_semitones: float = 1.5

    movement_epsilon_semitones: float = 0.05

    # Diagnostic property of the enclosing rejected episode.
    # It does NOT by itself forbid rescuing a coherent suffix/prefix.
    catastrophic_step_semitones: float = 8.0

    # V3 observational thresholds.
    rescue_edge_support_min: float = 0.60
    ornament_score_min: float = 0.65


@dataclass
class OfflineValidityCorrectionResult:
    """
    Result of the second, offline correction pass.

    first_pass_valid:
        Immutable copy of the original causal/future-confirmed mask.

    valid:
        Corrected validity mask.

    first_pass_reason:
        Original first-pass reason strings, unchanged.

    reason:
        Final reason strings. Rescued rows receive explicit labels.

    correction_reason:
        What THIS pass did:
            UNCHANGED_VALID
            UNCHANGED_REJECTED
            BACKWARD_RESCUED
            FORWARD_RESCUED
            ORNAMENT_RESCUED
            REJECTED_INTERNAL
            TOO_SHORT_TO_INTERPRET

    clean_f0_hz:
        raw F0 where final valid=True, NaN elsewhere.
        No interpolation is performed.
    """

    first_pass_valid: np.ndarray
    valid: np.ndarray

    first_pass_reason: np.ndarray
    reason: np.ndarray
    correction_reason: np.ndarray

    clean_f0_hz: np.ndarray



def hz_to_midi_float(f0_hz: np.ndarray) -> np.ndarray:

    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    out = np.full_like(
        f0_hz,
        np.nan,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(f0_hz)
        & (f0_hz > 0.0)
    )

    out[mask] = (
        69.0
        + 12.0
        * np.log2(
            f0_hz[mask] / 440.0
        )
    )

    return out


def _runs(mask: np.ndarray):
    """
    Yield contiguous True runs as (start, end),
    where end is exclusive.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    n = len(mask)

    i = 0

    while i < n:

        if not mask[i]:
            i += 1
            continue

        start = i

        while (
            i < n
            and mask[i]
        ):
            i += 1

        yield start, i


def _false_runs(mask: np.ndarray):
    """
    Yield contiguous False runs.
    """

    yield from _runs(~mask)


def _median_finite(x: np.ndarray) -> float:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return np.nan

    return float(
        np.median(x)
    )

def analyse_pitch_validity(
    f0_hz: np.ndarray,
    sample_rate: float,
    config: PitchValidityConfig | None = None,
) -> PitchValidityResult:

    if config is None:
        config = PitchValidityConfig()

    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    if f0_hz.ndim != 1:
        raise ValueError(
            "f0_hz must be 1-D"
        )

    sample_rate = float(
        sample_rate
    )

    analysis_step = max(
        1,
        int(
            round(
                sample_rate
                / config.analysis_hz
            )
        ),
    )

    sample_index = np.arange(
        0,
        len(f0_hz),
        analysis_step,
        dtype=np.int64,
    )

    sampled_f0 = (
        f0_hz[sample_index]
    )

    time_s = (
        sample_index.astype(np.float64)
        / sample_rate
    )

    midi = hz_to_midi_float(
        sampled_f0
    )

    n = len(sampled_f0)

    valid = np.zeros(
        n,
        dtype=bool,
    )

    reason = np.full(
        n,
        "INVALID",
        dtype=object,
    )

    # ------------------------------------------------------------
    # Hard physical/domain plausibility.
    # ------------------------------------------------------------

    candidate = (
        np.isfinite(sampled_f0)
        & (sampled_f0 >= config.min_vocal_hz)
        & (sampled_f0 <= config.max_vocal_hz)
    )

    reason[
        ~np.isfinite(sampled_f0)
        | (sampled_f0 <= 0.0)
    ] = "NO_F0"

    reason[
        np.isfinite(sampled_f0)
        & (sampled_f0 > 0.0)
        & (sampled_f0 < config.min_vocal_hz)
    ] = "BELOW_VOCAL_FLOOR"

    reason[
        np.isfinite(sampled_f0)
        & (sampled_f0 > config.max_vocal_hz)
    ] = "ABOVE_VOCAL_CEILING"

    # ------------------------------------------------------------
    # Offline parameters.
    # ------------------------------------------------------------

    # Once lock is lost, require this much coherent FUTURE trajectory
    # before reacquiring.
    reacquire_ms = 120.0

    reacquire_steps = max(
        2,
        int(
            round(
                reacquire_ms
                * config.analysis_hz
                / 1000.0
            )
        ),
    )

    # Maximum adjacent motion allowed inside a proposed trajectory.
    max_step = (
        config.max_step_semitones
    )

    # The proposed future window also needs reasonable overall
    # coherence. It may glide; it does NOT have to remain on one note.
    #
    # This is deliberately much larger than vibrato, because genuine
    # glissandi must survive.
 
    # ------------------------------------------------------------
    # Helper: does FUTURE evidence support a real trajectory beginning
    # at i?
    # ------------------------------------------------------------

    def future_support(i: int) -> bool:

        end = min(
            n,
            i + reacquire_steps,
        )

        if (
            end - i
            < reacquire_steps
        ):
            return False

        window_candidate = (
            candidate[i:end]
        )

        if not np.all(
            window_candidate
        ):
            return False

        pitches = midi[i:end]

        if not np.all(
            np.isfinite(pitches)
        ):
            return False

        steps = np.abs(
            np.diff(pitches)
        )

        if (
            steps.size
            and np.max(steps) > max_step
        ):
            return False

        return True

    # ------------------------------------------------------------
    # State machine.
    #
    # LOCKED:
    #   accept coherent continuation.
    #
    # LOST:
    #   do NOT accept random above-floor pitches.
    #   Require sustained future support first.
    # ------------------------------------------------------------

    locked = False

    last_valid_pitch = np.nan

    i = 0

    while i < n:

        if not candidate[i]:

            locked = False

            # Existing hard-failure reason already set.
            i += 1
            continue

        current = midi[i]

        # --------------------------------------------------------
        # We do not currently have lock.
        # Look FORWARD before accepting this as a real reacquisition.
        # --------------------------------------------------------

        if not locked:

            if not future_support(i):

                reason[i] = (
                    "WAITING_FOR_REACQUISITION"
                )

                i += 1
                continue

            # Future confirms this trajectory.
            #
            # Accept the whole confirmation window retroactively.
            end = min(
                n,
                i + reacquire_steps,
            )

            valid[i:end] = True

            reason[i:end] = (
                "REACQUIRED_FUTURE_CONFIRMED"
            )

            locked = True

            last_valid_pitch = float(
                midi[end - 1]
            )

            i = end

            continue

        # --------------------------------------------------------
        # Already locked.
        # --------------------------------------------------------

        step = abs(
            current
            - last_valid_pitch
        )

        if (
            step <= max_step
        ):

            valid[i] = True

            reason[i] = (
                "VALID_CONTINUOUS"
            )

            last_valid_pitch = float(
                current
            )

            i += 1
            continue

        # --------------------------------------------------------
        # Catastrophic discontinuity.
        #
        # DO NOT let this observation become a new anchor.
        # Lock is now lost.
        # --------------------------------------------------------

        reason[i] = (
            "LOCK_LOST_DISCONTINUITY"
        )

        locked = False
        last_valid_pitch = np.nan

        i += 1

    # ------------------------------------------------------------
    # Cleaned F0.
    #
    # We never interpolate here.
    # Invalid means invalid.
    # ------------------------------------------------------------

    cleaned_f0 = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    cleaned_f0[valid] = (
        sampled_f0[valid]
    )

    return PitchValidityResult(
        sample_index=sample_index,
        time_s=time_s,
        f0_hz=sampled_f0,
        valid=valid,
        reason=reason,
        cleaned_f0_hz=cleaned_f0,
    )

# =====================================================================
# Offline bidirectional validity correction
# =====================================================================

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _validity_hz_to_midi(
    f0_hz: np.ndarray,
) -> np.ndarray:
    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    midi = np.full(
        f0_hz.shape,
        np.nan,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(f0_hz)
        & (f0_hz > 0.0)
    )

    midi[mask] = (
        69.0
        + 12.0
        * np.log2(
            f0_hz[mask] / 440.0
        )
    )

    return midi


def _true_runs(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Return half-open [start, end) runs where mask is True.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.size == 0:
        return []

    padded = np.concatenate(
        (
            np.array(
                [False],
                dtype=bool,
            ),
            mask,
            np.array(
                [False],
                dtype=bool,
            ),
        )
    )

    changes = np.diff(
        padded.astype(
            np.int8
        )
    )

    starts = np.flatnonzero(
        changes == 1
    )

    ends = np.flatnonzero(
        changes == -1
    )

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def _median_finite(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.median(values)
    )


def _vocal_support(
    interval_st: float,
    elapsed_ms: float,
) -> float:
    """
    Convert the existing SOFT vocal interval/time prior into support.

        penalty 0 -> support 1
        increasing penalty -> decreasing support

    No new hard physiological interval limit is introduced here.
    """

    if (
        not np.isfinite(interval_st)
        or not np.isfinite(elapsed_ms)
        or elapsed_ms < 0.0
    ):
        return 0.0

    penalty = vocal_transition_penalty(
        interval_st,
        elapsed_ms,
    )

    return float(
        np.exp(
            -float(penalty)
        )
    )


def _hard_pitch_ok(
    f0_hz: np.ndarray,
    config: OfflineValidityCorrectionConfig,
) -> np.ndarray:
    return (
        np.isfinite(f0_hz)
        & (
            f0_hz
            >= config.min_vocal_hz
        )
        & (
            f0_hz
            <= config.max_vocal_hz
        )
    )


# ---------------------------------------------------------------------
# Nearest accepted samples
# ---------------------------------------------------------------------


def _nearest_valid_left(
    midi: np.ndarray,
    valid: np.ndarray,
    index: int,
) -> int | None:
    i = index - 1

    while i >= 0:
        if (
            valid[i]
            and np.isfinite(
                midi[i]
            )
        ):
            return i

        i -= 1

    return None


def _nearest_valid_right(
    midi: np.ndarray,
    valid: np.ndarray,
    index: int,
) -> int | None:
    n = len(midi)

    i = index

    while i < n:
        if (
            valid[i]
            and np.isfinite(
                midi[i]
            )
        ):
            return i

        i += 1

    return None


def _left_edge_evidence(
    midi: np.ndarray,
    valid: np.ndarray,
    subrun_start: int,
    analysis_hz: float,
) -> tuple[
    int | None,
    float,
    float,
]:
    """
    Returns:
        accepted_index
        interval_semitones
        support
    """

    anchor = _nearest_valid_left(
        midi,
        valid,
        subrun_start,
    )

    if anchor is None:
        return (
            None,
            np.nan,
            0.0,
        )

    interval = float(
        midi[subrun_start]
        - midi[anchor]
    )

    elapsed_ms = float(
        (
            subrun_start
            - anchor
        )
        / analysis_hz
        * 1000.0
    )

    support = _vocal_support(
        interval,
        elapsed_ms,
    )

    return (
        anchor,
        interval,
        support,
    )


def _right_edge_evidence(
    midi: np.ndarray,
    valid: np.ndarray,
    subrun_end: int,
    analysis_hz: float,
) -> tuple[
    int | None,
    float,
    float,
]:
    """
    subrun_end is exclusive.

    Returns:
        accepted_index
        interval_semitones
        support
    """

    last = subrun_end - 1

    anchor = _nearest_valid_right(
        midi,
        valid,
        subrun_end,
    )

    if anchor is None:
        return (
            None,
            np.nan,
            0.0,
        )

    interval = float(
        midi[anchor]
        - midi[last]
    )

    elapsed_ms = float(
        (
            anchor
            - last
        )
        / analysis_hz
        * 1000.0
    )

    support = _vocal_support(
        interval,
        elapsed_ms,
    )

    return (
        anchor,
        interval,
        support,
    )


# ---------------------------------------------------------------------
# Robust accepted context
# ---------------------------------------------------------------------


def _accepted_context(
    midi: np.ndarray,
    valid: np.ndarray,
    boundary: int,
    side: str,
    analysis_hz: float,
    config: OfflineValidityCorrectionConfig,
) -> float:
    n = len(midi)

    steps = max(
        1,
        int(
            round(
                config.context_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    if side == "left":
        lo = max(
            0,
            boundary - steps,
        )

        hi = boundary

    elif side == "right":
        lo = boundary

        hi = min(
            n,
            boundary + steps,
        )

    else:
        raise ValueError(
            f"Unknown side: {side}"
        )

    values = midi[
        lo:hi
    ]

    mask = (
        valid[lo:hi]
        & np.isfinite(values)
    )

    return _median_finite(
        values[mask]
    )


def _context_pitch_support(
    interval_st: float,
    config: OfflineValidityCorrectionConfig,
) -> float:
    if not np.isfinite(
        interval_st
    ):
        return 0.0

    return _vocal_support(
        interval_st,
        config.context_ms,
    )


def _context_return_support(
    left_context: float,
    right_context: float,
) -> float:
    if (
        not np.isfinite(
            left_context
        )
        or not np.isfinite(
            right_context
        )
    ):
        return 0.0

    difference = abs(
        right_context
        - left_context
    )

    sigma_st = 2.0

    return float(
        np.exp(
            -0.5
            * (
                difference
                / sigma_st
            ) ** 2
        )
    )


# ---------------------------------------------------------------------
# Local trajectory geometry
# ---------------------------------------------------------------------


def _trajectory_metrics(
    pitch: np.ndarray,
    config: OfflineValidityCorrectionConfig,
) -> dict | None:
    pitch = np.asarray(
        pitch,
        dtype=np.float64,
    )

    pitch = pitch[
        np.isfinite(pitch)
    ]

    if pitch.size == 0:
        return None

    if pitch.size == 1:
        return {
            "span": 0.0,
            "endpoint": 0.0,
            "max_step": 0.0,
            "return_ratio": 1.0,
            "monotonic_fraction": 1.0,
        }

    d = np.diff(
        pitch
    )

    abs_d = np.abs(
        d
    )

    span = float(
        np.max(pitch)
        - np.min(pitch)
    )

    endpoint = float(
        pitch[-1]
        - pitch[0]
    )

    max_step = float(
        np.max(abs_d)
    )

    return_ratio = (
        abs(endpoint) / span
        if span > 1e-12
        else 1.0
    )

    moving = d[
        np.abs(d)
        >= config.movement_epsilon_semitones
    ]

    if moving.size == 0:
        monotonic_fraction = 1.0

    else:
        if endpoint > 0.0:
            preferred = (
                moving > 0.0
            )

        elif endpoint < 0.0:
            preferred = (
                moving < 0.0
            )

        else:
            positive_count = int(
                np.sum(
                    moving > 0.0
                )
            )

            negative_count = int(
                np.sum(
                    moving < 0.0
                )
            )

            preferred = (
                moving > 0.0
                if positive_count
                >= negative_count
                else moving < 0.0
            )

        monotonic_fraction = float(
            np.mean(
                preferred
            )
        )

    return {
        "span": span,
        "endpoint": endpoint,
        "max_step": max_step,
        "return_ratio": float(
            return_ratio
        ),
        "monotonic_fraction": float(
            monotonic_fraction
        ),
    }


# ---------------------------------------------------------------------
# Split a rejected episode into coherent raw trajectories
# ---------------------------------------------------------------------


def _coherent_rejected_subruns(
    f0_hz: np.ndarray,
    midi: np.ndarray,
    episode_start: int,
    episode_end: int,
    config: OfflineValidityCorrectionConfig,
) -> list[tuple[int, int]]:
    """
    Break on:
        invalid/out-of-working-range pitch
        non-finite MIDI pitch
        adjacent movement > max_adjacent_step_semitones

    IMPORTANT:
        There is NO total-span restriction.
    """

    hard_ok = _hard_pitch_ok(
        f0_hz,
        config,
    )

    runs: list[
        tuple[int, int]
    ] = []

    run_start: int | None = None
    previous: int | None = None

    for i in range(
        episode_start,
        episode_end,
    ):
        if (
            not hard_ok[i]
            or not np.isfinite(
                midi[i]
            )
        ):
            if run_start is not None:
                runs.append(
                    (
                        run_start,
                        i,
                    )
                )

            run_start = None
            previous = None
            continue

        if run_start is None:
            run_start = i
            previous = i
            continue

        step = abs(
            midi[i]
            - midi[previous]
        )

        if (
            not np.isfinite(step)
            or step
            > config.max_adjacent_step_semitones
        ):
            runs.append(
                (
                    run_start,
                    i,
                )
            )

            run_start = i

        previous = i

    if run_start is not None:
        runs.append(
            (
                run_start,
                episode_end,
            )
        )

    return runs


# ---------------------------------------------------------------------
# Ornament evidence
# ---------------------------------------------------------------------


def _offline_ornament_score(
    metrics: dict,
    duration_ms: float,
    touches_left: bool,
    touches_right: bool,
    left_edge_support: float,
    right_edge_support: float,
    left_context: float,
    right_context: float,
    left_context_support: float,
    right_context_support: float,
    config: OfflineValidityCorrectionConfig,
) -> float:
    """
    V3 excursion-and-return evidence.

    A floating fragment buried in rejected garbage is NOT an ornament:
    it must touch at least one rejected-episode boundary.
    """

    if not (
        touches_left
        or touches_right
    ):
        return 0.0

    if (
        duration_ms
        > config.ornament_max_ms
    ):
        return 0.0

    if (
        metrics["span"]
        < config.ornament_min_span_semitones
    ):
        return 0.0

    if (
        metrics["max_step"]
        > config.max_adjacent_step_semitones
    ):
        return 0.0

    return_score = float(
        np.exp(
            -2.0
            * metrics[
                "return_ratio"
            ]
        )
    )

    context_return = (
        _context_return_support(
            left_context,
            right_context,
        )
    )

    edge_support = max(
        left_edge_support,
        right_edge_support,
    )

    context_connection = max(
        left_context_support,
        right_context_support,
    )

    return float(
        0.55 * return_score
        + 0.20 * context_return
        + 0.15 * edge_support
        + 0.10 * context_connection
    )


# ---------------------------------------------------------------------
# Main correction pass
# ---------------------------------------------------------------------


def apply_offline_validity_correction(
    f0_hz: np.ndarray,
    first_pass,
    sample_rate: float,
    config: OfflineValidityCorrectionConfig | None = None,
) -> OfflineValidityCorrectionResult:
    """
    Apply an OFFLINE bidirectional correction to the existing first-pass
    validity result.

    Parameters
    ----------
    f0_hz:
        Raw analysis-rate ECKF F0 trajectory.

    first_pass:
        Result returned by analyse_pitch_validity().

        Expected fields:
            first_pass.valid
            first_pass.reason

        The object itself is NEVER modified.

    sample_rate:
        Analysis trajectory rate, e.g. 100.0 Hz for the current exported
        RATATA diagnostic trajectory.

        This is NOT necessarily the WAV sample rate.

    config:
        Optional OfflineValidityCorrectionConfig.

    Returns
    -------
    OfflineValidityCorrectionResult

    Important
    ---------
    * Existing first-pass validity is preserved.
    * Only False -> True promotions are possible.
    * No existing True sample is invalidated.
    * Rescued samples use their ORIGINAL raw ECKF F0.
    * No interpolation is performed.
    """

    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    if f0_hz.ndim != 1:
        raise ValueError(
            "f0_hz must be a 1-D trajectory"
        )

    analysis_hz = float(
        sample_rate
    )

    if (
        not np.isfinite(
            analysis_hz
        )
        or analysis_hz <= 0.0
    ):
        raise ValueError(
            "sample_rate must be the positive "
            "analysis trajectory rate"
        )

    if config is None:
        config = (
            OfflineValidityCorrectionConfig()
        )

    first_valid = np.asarray(
        first_pass.valid,
        dtype=bool,
    )

    if (
        first_valid.shape
        != f0_hz.shape
    ):
        raise ValueError(
            "first_pass.valid must have the same "
            "shape as f0_hz"
        )

    # -------------------------------------------------------------
    # Preserve original first-pass reasons.
    # -------------------------------------------------------------

    if hasattr(
        first_pass,
        "reason",
    ):
        first_reason = np.asarray(
            first_pass.reason,
            dtype=object,
        ).copy()

    elif hasattr(
        first_pass,
        "reasons",
    ):
        # Defensive compatibility in case the current dataclass happens
        # to use the plural field name.
        first_reason = np.asarray(
            first_pass.reasons,
            dtype=object,
        ).copy()

    else:
        first_reason = np.full(
            f0_hz.shape,
            "",
            dtype=object,
        )

    if (
        first_reason.shape
        != f0_hz.shape
    ):
        raise ValueError(
            "first-pass reason array must have the "
            "same shape as f0_hz"
        )

    corrected_valid = (
        first_valid.copy()
    )

    final_reason = (
        first_reason.copy()
    )

    correction_reason = np.where(
        first_valid,
        "UNCHANGED_VALID",
        "UNCHANGED_REJECTED",
    ).astype(
        object
    )

    midi = _validity_hz_to_midi(
        f0_hz
    )

    # IMPORTANT:
    # Rejected episodes are based on the ORIGINAL first-pass mask.
    #
    # We do not let an early rescue alter episode segmentation while
    # processing later candidates.
    rejected_episodes = _true_runs(
        ~first_valid
    )

    # Integer-step minimum removes the floating 29.999999 ms issue
    # observed in V3.
    min_subrun_steps = max(
        1,
        int(
            np.ceil(
                config.min_subrun_ms
                * analysis_hz
                / 1000.0
                - 1e-9
            )
        ),
    )

    for (
        episode_start,
        episode_end,
    ) in rejected_episodes:

        subruns = (
            _coherent_rejected_subruns(
                f0_hz=f0_hz,
                midi=midi,
                episode_start=episode_start,
                episode_end=episode_end,
                config=config,
            )
        )

        if not subruns:
            continue

        left_context = (
            _accepted_context(
                midi=midi,
                valid=first_valid,
                boundary=episode_start,
                side="left",
                analysis_hz=analysis_hz,
                config=config,
            )
        )

        right_context = (
            _accepted_context(
                midi=midi,
                valid=first_valid,
                boundary=episode_end,
                side="right",
                analysis_hz=analysis_hz,
                config=config,
            )
        )

        for (
            subrun_start,
            subrun_end,
        ) in subruns:

            n_steps = (
                subrun_end
                - subrun_start
            )

            # -----------------------------------------------------
            # Too short to establish a trajectory.
            # -----------------------------------------------------

            if (
                n_steps
                < min_subrun_steps
            ):
                correction_reason[
                    subrun_start:subrun_end
                ] = (
                    "TOO_SHORT_TO_INTERPRET"
                )

                continue

            pitch = midi[
                subrun_start:subrun_end
            ]

            metrics = (
                _trajectory_metrics(
                    pitch,
                    config,
                )
            )

            if metrics is None:
                continue

            duration_ms = float(
                n_steps
                / analysis_hz
                * 1000.0
            )

            touches_left = (
                subrun_start
                == episode_start
            )

            touches_right = (
                subrun_end
                == episode_end
            )

            (
                _left_anchor,
                _left_interval,
                left_edge_support,
            ) = _left_edge_evidence(
                midi=midi,
                valid=first_valid,
                subrun_start=subrun_start,
                analysis_hz=analysis_hz,
            )

            (
                _right_anchor,
                _right_interval,
                right_edge_support,
            ) = _right_edge_evidence(
                midi=midi,
                valid=first_valid,
                subrun_end=subrun_end,
                analysis_hz=analysis_hz,
            )

            first_pitch = float(
                pitch[0]
            )

            last_pitch = float(
                pitch[-1]
            )

            left_context_interval = (
                first_pitch
                - left_context
                if np.isfinite(
                    left_context
                )
                else np.nan
            )

            right_context_interval = (
                right_context
                - last_pitch
                if np.isfinite(
                    right_context
                )
                else np.nan
            )

            left_context_support = (
                _context_pitch_support(
                    left_context_interval,
                    config,
                )
                if np.isfinite(
                    left_context_interval
                )
                else 0.0
            )

            right_context_support = (
                _context_pitch_support(
                    right_context_interval,
                    config,
                )
                if np.isfinite(
                    right_context_interval
                )
                else 0.0
            )

            ornament = (
                _offline_ornament_score(
                    metrics=metrics,
                    duration_ms=duration_ms,
                    touches_left=touches_left,
                    touches_right=touches_right,
                    left_edge_support=left_edge_support,
                    right_edge_support=right_edge_support,
                    left_context=left_context,
                    right_context=right_context,
                    left_context_support=left_context_support,
                    right_context_support=right_context_support,
                    config=config,
                )
            )

            # =====================================================
            # 1. Ornament rescue
            #
            # Checked first because an excursion-and-return trajectory
            # has a different musical interpretation from a mere
            # boundary extension.
            # =====================================================

            if (
                ornament
                >= config.ornament_score_min
            ):
                corrected_valid[
                    subrun_start:subrun_end
                ] = True

                final_reason[
                    subrun_start:subrun_end
                ] = "ORNAMENT_RESCUED"

                correction_reason[
                    subrun_start:subrun_end
                ] = "ORNAMENT_RESCUED"

                continue

            # =====================================================
            # 2. Backward rescue from trusted future evidence
            # =====================================================

            if (
                touches_right
                and right_edge_support
                >= config.rescue_edge_support_min
            ):
                corrected_valid[
                    subrun_start:subrun_end
                ] = True

                final_reason[
                    subrun_start:subrun_end
                ] = "BACKWARD_RESCUED"

                correction_reason[
                    subrun_start:subrun_end
                ] = "BACKWARD_RESCUED"

                continue

            # =====================================================
            # 3. Forward rescue from trusted past evidence
            # =====================================================

            if (
                touches_left
                and left_edge_support
                >= config.rescue_edge_support_min
            ):
                corrected_valid[
                    subrun_start:subrun_end
                ] = True

                final_reason[
                    subrun_start:subrun_end
                ] = "FORWARD_RESCUED"

                correction_reason[
                    subrun_start:subrun_end
                ] = "FORWARD_RESCUED"

                continue

            # -----------------------------------------------------
            # Coherent raw pitch existed but was not supported well
            # enough to become valid.
            # -----------------------------------------------------

            correction_reason[
                subrun_start:subrun_end
            ] = "REJECTED_INTERNAL"

    # -------------------------------------------------------------
    # Never synthesize/interpolate F0 here.
    #
    # Valid -> exact raw ECKF F0
    # Invalid -> NaN
    # -------------------------------------------------------------

    clean_f0_hz = np.full(
        f0_hz.shape,
        np.nan,
        dtype=np.float64,
    )

    keep = (
        corrected_valid
        & np.isfinite(f0_hz)
        & (f0_hz > 0.0)
    )

    clean_f0_hz[
        keep
    ] = f0_hz[
        keep
    ]

    return OfflineValidityCorrectionResult(
        first_pass_valid=(
            first_valid.copy()
        ),
        valid=corrected_valid,
        first_pass_reason=(
            first_reason
        ),
        reason=final_reason,
        correction_reason=(
            correction_reason
        ),
        clean_f0_hz=clean_f0_hz,
    )