from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------
# Public classifications
# ---------------------------------------------------------------------


class TrajectoryClass(str, Enum):
    CONTINUATION = "CONTINUATION"
    PLAUSIBLE_NEW_TRAJECTORY = "PLAUSIBLE_NEW_TRAJECTORY"
    SMOOTH_GLIDE = "SMOOTH_GLIDE"

    PROBABLE_2X_HARMONIC_LOCK = "PROBABLE_2X_HARMONIC_LOCK"
    PROBABLE_3X_HARMONIC_LOCK = "PROBABLE_3X_HARMONIC_LOCK"
    PROBABLE_1_2_SUBHARMONIC_LOCK = "PROBABLE_1_2_SUBHARMONIC_LOCK"
    PROBABLE_1_3_SUBHARMONIC_LOCK = "PROBABLE_1_3_SUBHARMONIC_LOCK"

    CHAOTIC_TRACKING_FAILURE = "CHAOTIC_TRACKING_FAILURE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class ResolverConfig:
    analysis_hz: float = 100.0

    # Trusted context inspected on each side.
    context_ms: float = 200.0

    # Ignore a few samples immediately beside the boundary when taking
    # the context median. This makes the context less sensitive to the
    # transition itself.
    context_guard_ms: float = 20.0

    # Continuous harmonic-evidence width.
    #
    # This is NOT a pass/fail tolerance.
    # An error of 0 st gives evidence ~= 1.
    # Around 1 st still carries substantial evidence.
    # Several semitones away contributes very little.
    harmonic_sigma_st: float = 1.0

    # Continuous evidence that the trusted trajectory before and after
    # the island belongs to the same pitch neighbourhood.
    context_sigma_st: float = 1.5

    # Do not print a "nearest harmonic" label once the relationship is
    # so remote that the label becomes diagnostically misleading.
    harmonic_label_min_evidence: float = 0.10

    # Classification thresholds operate on COMBINED continuous
    # evidence, not on a hard semitone tolerance.
    harmonic_pair_min_evidence: float = 0.40
    context_min_evidence: float = 0.35
    harmonic_lock_score_min: float = 1.25

    # Internal trajectory properties.
    chaotic_step_st: float = 7.0
    smooth_step_st: float = 2.0

    # A glide does not have to be perfectly monotonic because real
    # singing contains vibrato and small reversals.
    glide_efficiency_min: float = 0.55
    glide_monotonic_fraction_min: float = 0.70


@dataclass
class TrajectoryResolution:
    start: int
    end: int

    duration_ms: float

    median_pitch: float
    left_pitch: float
    right_pitch: float

    entry_interval_st: float
    exit_interval_st: float
    left_right_interval_st: float

    span_st: float
    endpoint_change_st: float
    path_length_st: float
    efficiency: float
    monotonic_fraction: float
    max_step_st: float

    entry_vocal_penalty: float
    exit_vocal_penalty: float

    harmonic_relation: str | None
    harmonic_error_st: float

    left_harmonic_evidence: float
    right_harmonic_evidence: float
    harmonic_pair_evidence: float
    context_evidence: float

    sandwich_score: float

    classification: TrajectoryClass
    confidence: float


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def hz_to_midi(f0_hz):
    f0_hz = np.asarray(f0_hz, dtype=np.float64)

    result = np.full(
        f0_hz.shape,
        np.nan,
        dtype=np.float64,
    )

    good = (
        np.isfinite(f0_hz)
        & (f0_hz > 0.0)
    )

    result[good] = (
        69.0
        + 12.0 * np.log2(f0_hz[good] / 440.0)
    )

    return result


def _median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.median(values))


def true_runs(mask):
    mask = np.asarray(mask, dtype=bool)

    runs = []
    i = 0

    while i < len(mask):
        if not mask[i]:
            i += 1
            continue

        start = i

        while i < len(mask) and mask[i]:
            i += 1

        runs.append((start, i))

    return runs


# ---------------------------------------------------------------------
# Human-vocal interval prior
# ---------------------------------------------------------------------


# These are SOFT prior anchors, not physiological limits.
#
# interval semitones -> approximate time at which the interval stops
# being intrinsically suspicious as a discrete vocal transition.
#
# Important:
#
#   thirds AND BELOW receive essentially no penalty.
#
# Large intervals remain possible. 24 st deliberately remains in the
# tail rather than being declared impossible.
_VOCAL_TIME_ANCHORS = np.asarray(
    [
        [4.0,    0.0],
        [7.0,  100.0],
        [9.0,  175.0],
        [12.0, 300.0],
        [15.5, 475.0],
        [19.0, 700.0],
        [24.0, 1050.0],
        [30.0, 1500.0],
    ],
    dtype=np.float64,
)


def vocal_required_time_ms(interval_st: float) -> float:
    """
    Soft timing prior for a DISCRETE vocal leap.

    This must never be interpreted as a hard physical limit.
    """

    interval = abs(float(interval_st))

    if interval <= 4.0:
        return 0.0

    x = _VOCAL_TIME_ANCHORS[:, 0]
    y = _VOCAL_TIME_ANCHORS[:, 1]

    if interval <= x[-1]:
        return float(np.interp(interval, x, y))

    # Beyond our deliberately generous 30-st anchor, continue the
    # final slope rather than declaring anything mathematically
    # impossible.
    slope = (
        (y[-1] - y[-2])
        / (x[-1] - x[-2])
    )

    return float(
        y[-1]
        + (interval - x[-1]) * slope
    )


def vocal_transition_penalty(
    interval_st: float,
    available_ms: float,
) -> float:
    """
    Soft penalty only.

    0 means that interval/time combination is unsurprising according
    to this prior. Increasing values mean increasingly demanding.

    A smooth continuous glide is handled separately and must not be
    rejected merely because its endpoints are far apart.
    """

    interval = abs(float(interval_st))

    if not np.isfinite(interval):
        return 0.0

    if interval <= 4.0:
        return 0.0

    required = vocal_required_time_ms(interval)

    available = max(
        float(available_ms),
        1.0,
    )

    if available >= required:
        return 0.0

    shortage = (
        required - available
    ) / max(required, 1.0)

    # Size weighting begins gently above a third.
    size_weight = (
        0.5
        + 0.10 * (interval - 4.0)
    )

    return float(
        shortage * size_weight
    )


# ---------------------------------------------------------------------
# Harmonic relationships
# ---------------------------------------------------------------------


_HARMONIC_RELATIONS = (
    ("1/3", -19.0195500087),
    ("1/2", -12.0),
    ("2x", +12.0),
    ("3x", +19.0195500087),
)


def nearest_harmonic(interval_st: float):
    if not np.isfinite(interval_st):
        return None, np.inf

    best_name = None
    best_error = np.inf

    for name, target in _HARMONIC_RELATIONS:
        error = abs(float(interval_st) - target)

        if error < best_error:
            best_name = name
            best_error = error

    return best_name, float(best_error)

def gaussian_evidence(
    error: float,
    sigma: float,
) -> float:
    """
    Convert a distance/error into continuous evidence in [0, 1].

    error = 0       -> 1.0
    error ~ sigma   -> ~0.61
    error ~ 2sigma  -> ~0.14

    There is deliberately no binary semitone cutoff here.
    """

    if not np.isfinite(error):
        return 0.0

    sigma = max(float(sigma), 1e-9)
    z = abs(float(error)) / sigma

    return float(
        np.exp(-0.5 * z * z)
    )


def harmonic_evidence(
    interval_st: float,
    config: ResolverConfig,
):
    """
    Return:

        relation
        error_st
        evidence

    The relation is suppressed when the nearest harmonic is so remote
    that naming it would be misleading.
    """

    relation, error = nearest_harmonic(
        interval_st
    )

    evidence = gaussian_evidence(
        error,
        config.harmonic_sigma_st,
    )

    if evidence < config.harmonic_label_min_evidence:
        relation = None

    return (
        relation,
        float(error),
        float(evidence),
    )


def same_pitch_context_evidence(
    interval_st: float,
    config: ResolverConfig,
) -> float:
    """
    Continuous evidence that left and right trusted contexts belong to
    approximately the same pitch neighbourhood.
    """

    if not np.isfinite(interval_st):
        return 0.0

    return gaussian_evidence(
        interval_st,
        config.context_sigma_st,
    )

def harmonic_class(name):
    if name == "2x":
        return TrajectoryClass.PROBABLE_2X_HARMONIC_LOCK

    if name == "3x":
        return TrajectoryClass.PROBABLE_3X_HARMONIC_LOCK

    if name == "1/2":
        return TrajectoryClass.PROBABLE_1_2_SUBHARMONIC_LOCK

    if name == "1/3":
        return TrajectoryClass.PROBABLE_1_3_SUBHARMONIC_LOCK

    return TrajectoryClass.AMBIGUOUS


# ---------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------


def _trusted_context_pitch(
    midi,
    valid,
    boundary,
    side,
    config,
):
    n = len(midi)

    context_steps = max(
        1,
        int(
            round(
                config.context_ms
                * config.analysis_hz
                / 1000.0
            )
        ),
    )

    guard_steps = max(
        0,
        int(
            round(
                config.context_guard_ms
                * config.analysis_hz
                / 1000.0
            )
        ),
    )

    if side == "left":
        hi = max(0, boundary - guard_steps)
        lo = max(0, hi - context_steps)

    elif side == "right":
        lo = min(n, boundary + guard_steps)
        hi = min(n, lo + context_steps)

    else:
        raise ValueError(side)

    mask = (
        valid[lo:hi]
        & np.isfinite(midi[lo:hi])
    )

    return _median(
        midi[lo:hi][mask]
    )


# ---------------------------------------------------------------------
# Internal trajectory
# ---------------------------------------------------------------------


def _trajectory_metrics(pitch):
    p = np.asarray(pitch, dtype=np.float64)
    p = p[np.isfinite(p)]

    if p.size == 0:
        return None

    if p.size == 1:
        return {
            "span": 0.0,
            "endpoint": 0.0,
            "path": 0.0,
            "efficiency": 1.0,
            "monotonic_fraction": 1.0,
            "max_step": 0.0,
        }

    d = np.diff(p)
    ad = np.abs(d)

    span = float(
        np.max(p) - np.min(p)
    )

    endpoint = float(
        p[-1] - p[0]
    )

    path = float(
        np.sum(ad)
    )

    direct = abs(endpoint)

    efficiency = (
        direct / path
        if path > 1e-12
        else 1.0
    )

    moving = d[np.abs(d) > 1e-6]

    if moving.size == 0:
        monotonic_fraction = 1.0

    elif endpoint > 0:
        monotonic_fraction = float(
            np.mean(moving > 0)
        )

    elif endpoint < 0:
        monotonic_fraction = float(
            np.mean(moving < 0)
        )

    else:
        monotonic_fraction = 0.5

    return {
        "span": span,
        "endpoint": endpoint,
        "path": path,
        "efficiency": float(efficiency),
        "monotonic_fraction": monotonic_fraction,
        "max_step": float(np.max(ad)),
    }


# ---------------------------------------------------------------------
# Island resolution
# ---------------------------------------------------------------------


def resolve_island(
    midi,
    valid,
    start,
    end,
    config,
):
    """
    OBSERVATIONAL resolver.

    It classifies the island but DOES NOT modify pitch or validity.
    """

    island = midi[start:end]

    metrics = _trajectory_metrics(island)

    if metrics is None:
        return None

    median_pitch = _median(island)

    left = _trusted_context_pitch(
        midi=midi,
        valid=valid,
        boundary=start,
        side="left",
        config=config,
    )

    right = _trusted_context_pitch(
        midi=midi,
        valid=valid,
        boundary=end,
        side="right",
        config=config,
    )

    entry = (
        median_pitch - left
        if np.isfinite(left)
        else np.nan
    )

    exit_interval = (
        right - median_pitch
        if np.isfinite(right)
        else np.nan
    )

    left_right = (
        right - left
        if np.isfinite(left) and np.isfinite(right)
        else np.nan
    )

    duration_ms = (
        (end - start)
        / config.analysis_hz
        * 1000.0
    )

    # -------------------------------------------------------------
    # Human-vocal transition prior.
    #
    # For now the island duration is the conservative available-time
    # estimate. This is intentionally only one score component.
    # -------------------------------------------------------------

    entry_penalty = vocal_transition_penalty(
        entry,
        duration_ms,
    )

    exit_penalty = vocal_transition_penalty(
        exit_interval,
        duration_ms,
    )

    # -------------------------------------------------------------
    # Continuous harmonic evidence from BOTH sides.
    #
    # We compare the island against the trusted trajectory before it,
    # and independently against the trusted trajectory after it.
    #
    # No binary "within 0.75 st" gate exists anymore.
    # -------------------------------------------------------------

    (
        left_relation,
        left_error,
        left_harmonic_evidence,
    ) = harmonic_evidence(
        entry,
        config,
    )

    # exit_interval is:
    #
    #     right - island
    #
    # For harmonic-family comparison we want:
    #
    #     island - right
    #
    island_vs_right = (
        median_pitch - right
        if np.isfinite(right)
        else np.nan
    )

    (
        right_relation,
        right_error,
        right_harmonic_evidence,
    ) = harmonic_evidence(
        island_vs_right,
        config,
    )

    # Strong harmonic-lock evidence requires the SAME harmonic family
    # on both sides.
    same_harmonic_family = (
        left_relation is not None
        and right_relation is not None
        and left_relation == right_relation
    )

    if same_harmonic_family:
        relation = left_relation

        # Geometric mean:
        #
        # both sides matter; one excellent side cannot completely hide
        # a terrible opposite side.
        harmonic_pair_evidence = float(
            np.sqrt(
                left_harmonic_evidence
                * right_harmonic_evidence
            )
        )

        harmonic_error = float(
            0.5
            * (
                left_error
                + right_error
            )
        )

    else:
        relation = None
        harmonic_pair_evidence = 0.0

        # Still retain the numerically nearest individual error for
        # diagnostics, but do NOT attach a harmonic-family label.
        harmonic_error = float(
            min(
                left_error,
                right_error,
            )
        )

    # -------------------------------------------------------------
    # Continuous LEFT ~= RIGHT evidence.
    #
    # Again: no binary "within N semitones" gate.
    # -------------------------------------------------------------

    context_evidence = (
        same_pitch_context_evidence(
            left_right,
            config,
        )
    )

    # -------------------------------------------------------------
    # Sandwich score.
    #
    # The core pattern is:
    #
    #     trusted A
    #         ↓
    #     harmonic-looking island B
    #         ↓
    #     trusted C
    #
    # where:
    #
    #     A ~= C
    #     and
    #     B is the SAME harmonic family relative to both.
    #
    # The two strongest ingredients are therefore:
    #
    #     harmonic_pair_evidence
    #     context_evidence
    #
    # Vocal-transition difficulty contributes supporting evidence,
    # but cannot by itself create a harmonic-lock classification.
    # -------------------------------------------------------------

    sandwich_core = (
        harmonic_pair_evidence
        * context_evidence
    )

    transition_support = float(
        1.0
        - np.exp(
            -0.5
            * (
                entry_penalty
                + exit_penalty
            )
        )
    )

    sandwich_score = float(
        2.0 * sandwich_core
        + 0.50
        * sandwich_core
        * transition_support
    )

    # -------------------------------------------------------------
    # Classification.
    # -------------------------------------------------------------

    if metrics["max_step"] >= config.chaotic_step_st:
        classification = (
            TrajectoryClass.CHAOTIC_TRACKING_FAILURE
        )

        confidence = min(
            1.0,
            0.65
            + 0.05
            * (
                metrics["max_step"]
                - config.chaotic_step_st
            ),
        )

    elif (
        same_harmonic_family
        and harmonic_pair_evidence
        >= config.harmonic_pair_min_evidence
        and context_evidence
        >= config.context_min_evidence
        and sandwich_score
        >= config.harmonic_lock_score_min
    ):
        classification = harmonic_class(
            relation
        )

        confidence = float(
            np.clip(
                0.50
                + 0.20 * harmonic_pair_evidence
                + 0.15 * context_evidence
                + 0.10 * min(
                    sandwich_score,
                    2.0,
                ),
                0.0,
                1.0,
            )
        )

    else:
        smooth_glide = (
            metrics["span"] >= 4.0
            and metrics["max_step"] <= config.smooth_step_st
            and (
                metrics["efficiency"]
                >= config.glide_efficiency_min
            )
            and (
                metrics["monotonic_fraction"]
                >= config.glide_monotonic_fraction_min
            )
        )

        if smooth_glide:
            classification = (
                TrajectoryClass.SMOOTH_GLIDE
            )

            confidence = 0.75

        elif (
            np.isfinite(left)
            and abs(entry) <= 4.0
        ):
            classification = (
                TrajectoryClass.CONTINUATION
            )

            confidence = 0.70

        elif (
            max(
                entry_penalty,
                exit_penalty,
            )
            < 0.75
        ):
            classification = (
                TrajectoryClass.PLAUSIBLE_NEW_TRAJECTORY
            )

            confidence = 0.60

        else:
            classification = (
                TrajectoryClass.AMBIGUOUS
            )

            confidence = 0.40

    return TrajectoryResolution(
        start=start,
        end=end,

        duration_ms=float(duration_ms),

        median_pitch=float(median_pitch),
        left_pitch=float(left),
        right_pitch=float(right),

        entry_interval_st=float(entry),
        exit_interval_st=float(exit_interval),
        left_right_interval_st=float(left_right),

        span_st=float(metrics["span"]),
        endpoint_change_st=float(metrics["endpoint"]),
        path_length_st=float(metrics["path"]),
        efficiency=float(metrics["efficiency"]),
        monotonic_fraction=float(
            metrics["monotonic_fraction"]
        ),
        max_step_st=float(metrics["max_step"]),

        entry_vocal_penalty=float(entry_penalty),
        exit_vocal_penalty=float(exit_penalty),

        harmonic_relation=relation,
        harmonic_error_st=float(harmonic_error),

        left_harmonic_evidence=float(
            left_harmonic_evidence
        ),
        right_harmonic_evidence=float(
            right_harmonic_evidence
        ),
        harmonic_pair_evidence=float(
            harmonic_pair_evidence
        ),
        context_evidence=float(
            context_evidence
        ),

        sandwich_score=float(
            sandwich_score
        ),
        
        classification=classification,
        confidence=float(confidence),
    )


def resolve_valid_runs(
    f0_hz,
    valid,
    config,
):
    """
    Resolve every contiguous currently-valid trajectory.

    IMPORTANT:
    observational only; no samples are changed.
    """

    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    midi = hz_to_midi(
        f0_hz
    )

    runs = true_runs(
        valid & np.isfinite(midi)
    )

    resolutions = []

    for start, end in runs:
        result = resolve_island(
            midi=midi,
            valid=valid,
            start=start,
            end=end,
            config=config,
        )

        if result is not None:
            resolutions.append(result)

    return resolutions