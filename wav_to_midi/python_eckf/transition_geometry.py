from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trajectory_interpreter import (
    RegionKind,
    TrajectoryInterpretation,
)
from .expressive_smoothing import (
    ExpressiveSmoothingResult,
)

@dataclass(frozen=True)
class TransitionGeometryConfig:
    """
    Purely observational geometry configuration.

    No musical gesture classification happens here.
    """

    # Ignore microscopic frame-to-frame motion when measuring direction.
    movement_epsilon_st: float = 0.05

    # A sample is considered "near" a target inside this radius.
    target_near_st: float = 0.35


@dataclass(frozen=True)
class TransitionGeometry:
    """
    Geometry of one TRANSITION region.

    All pitch quantities are in semitones on the absolute MIDI-pitch
    coordinate used by trajectory_interpreter.py.

    No values here imply MIDI quantization.
    """

    region_index: int

    start_index: int
    end_index: int

    start_time_s: float
    end_time_s: float
    duration_ms: float

    n_samples: int

    # ------------------------------------------------------------
    # Neighboring stable targets
    # ------------------------------------------------------------

    source_target_st: float
    destination_target_st: float

    has_source: bool
    has_destination: bool

    target_interval_st: float

    # ------------------------------------------------------------
    # Raw corrected-F0 trajectory
    # ------------------------------------------------------------

    start_pitch_st: float
    end_pitch_st: float
    median_pitch_st: float

    min_pitch_st: float
    max_pitch_st: float
    span_st: float

    # ------------------------------------------------------------
    # Boundary relationships
    # ------------------------------------------------------------

    departure_from_source_st: float
    arrival_error_destination_st: float

    # ------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------

    net_movement_st: float
    path_length_st: float
    path_efficiency: float

    upward_path_st: float
    downward_path_st: float

    upward_fraction: float
    downward_fraction: float

    dominant_direction: int
    monotonic_fraction: float
    direction_reversals: int

    max_adjacent_step_st: float
    median_adjacent_step_st: float

    # ------------------------------------------------------------
    # Target corridor
    # ------------------------------------------------------------

    corridor_low_st: float
    corridor_high_st: float

    overshoot_above_st: float
    undershoot_below_st: float

    # ------------------------------------------------------------
    # Target proximity / return evidence
    # ------------------------------------------------------------

    min_distance_source_st: float
    min_distance_destination_st: float

    fraction_near_source: float
    fraction_near_destination: float

    leaves_source_st: float
    returns_toward_source_st: float

    return_evidence: float

@dataclass(frozen=True)
class TransitionGeometryComparison:
    """
    Same transition measured on:

      raw:
          corrected unsmoothed ECKF pitch

      shape:
          lightly bidirectionally smoothed pitch

    The target centers are identical in both measurements.
    """

    raw: TransitionGeometry
    shape: TransitionGeometry


@dataclass(frozen=True)
class TransitionGeometryComparisonResult:
    transitions: tuple[
        TransitionGeometryComparison,
        ...
    ]

    analysis_hz: float

@dataclass(frozen=True)
class TransitionGeometryResult:
    transitions: tuple[TransitionGeometry, ...]
    analysis_hz: float


def _safe_fraction(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0

    return float(
        numerator / denominator
    )


def _motion_geometry(
    pitch: np.ndarray,
    movement_epsilon_st: float,
) -> dict[str, float | int]:
    """
    Describe raw trajectory motion without interpreting its musical cause.
    """

    p = np.asarray(
        pitch,
        dtype=np.float64,
    )

    if len(p) <= 1:
        return {
            "net_movement_st": 0.0,
            "path_length_st": 0.0,
            "path_efficiency": 1.0,
            "upward_path_st": 0.0,
            "downward_path_st": 0.0,
            "upward_fraction": 0.0,
            "downward_fraction": 0.0,
            "dominant_direction": 0,
            "monotonic_fraction": 1.0,
            "direction_reversals": 0,
            "max_adjacent_step_st": 0.0,
            "median_adjacent_step_st": 0.0,
        }

    diff = np.diff(p)
    abs_diff = np.abs(diff)

    net = float(
        p[-1] - p[0]
    )

    path = float(
        np.sum(abs_diff)
    )

    upward_path = float(
        np.sum(
            diff[diff > 0.0]
        )
    )

    downward_path = float(
        np.sum(
            -diff[diff < 0.0]
        )
    )

    efficiency = (
        abs(net) / path
        if path > 0.0
        else 1.0
    )

    meaningful = (
        abs_diff
        >= movement_epsilon_st
    )

    meaningful_diff = diff[
        meaningful
    ]

    if len(meaningful_diff) == 0:
        upward_fraction = 0.0
        downward_fraction = 0.0
        dominant_direction = 0
        monotonic_fraction = 1.0
        reversals = 0

    else:
        signs = np.sign(
            meaningful_diff
        )

        n_up = int(
            np.sum(signs > 0)
        )

        n_down = int(
            np.sum(signs < 0)
        )

        total = len(signs)

        upward_fraction = (
            n_up / total
        )

        downward_fraction = (
            n_down / total
        )

        if n_up > n_down:
            dominant_direction = 1
        elif n_down > n_up:
            dominant_direction = -1
        else:
            dominant_direction = 0

        monotonic_fraction = (
            max(n_up, n_down)
            / total
        )

        if len(signs) <= 1:
            reversals = 0
        else:
            reversals = int(
                np.sum(
                    signs[1:]
                    != signs[:-1]
                )
            )

    return {
        "net_movement_st": net,
        "path_length_st": path,
        "path_efficiency": float(
            efficiency
        ),
        "upward_path_st": upward_path,
        "downward_path_st": downward_path,
        "upward_fraction": float(
            upward_fraction
        ),
        "downward_fraction": float(
            downward_fraction
        ),
        "dominant_direction": int(
            dominant_direction
        ),
        "monotonic_fraction": float(
            monotonic_fraction
        ),
        "direction_reversals": int(
            reversals
        ),
        "max_adjacent_step_st": float(
            np.max(abs_diff)
        ),
        "median_adjacent_step_st": float(
            np.median(abs_diff)
        ),
    }


def _target_corridor(
    pitch: np.ndarray,
    source: float,
    destination: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Measure excursion outside the interval defined by the two targets.

    If only one target exists, the corridor collapses to that target.
    If neither exists, corridor metrics are NaN.
    """

    p = np.asarray(
        pitch,
        dtype=np.float64,
    )

    source_ok = np.isfinite(
        source
    )

    destination_ok = np.isfinite(
        destination
    )

    if (
        source_ok
        and destination_ok
    ):
        low = float(
            min(
                source,
                destination,
            )
        )

        high = float(
            max(
                source,
                destination,
            )
        )

    elif source_ok:
        low = float(source)
        high = float(source)

    elif destination_ok:
        low = float(destination)
        high = float(destination)

    else:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    overshoot = max(
        0.0,
        float(
            np.max(p) - high
        ),
    )

    undershoot = max(
        0.0,
        float(
            low - np.min(p)
        ),
    )

    return (
        low,
        high,
        overshoot,
        undershoot,
    )


def _target_proximity(
    pitch: np.ndarray,
    target: float,
    near_st: float,
) -> tuple[
    float,
    float,
]:
    """
    Return:
        minimum absolute distance to target,
        fraction of samples lying near target.
    """

    if not np.isfinite(
        target
    ):
        return (
            np.nan,
            np.nan,
        )

    distance = np.abs(
        pitch - target
    )

    return (
        float(
            np.min(distance)
        ),
        float(
            np.mean(
                distance <= near_st
            )
        ),
    )


def _source_return_geometry(
    pitch: np.ndarray,
    source: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Quantify leave-and-return geometry relative to the source target.

    leaves_source_st:
        maximum distance reached from source.

    returns_toward_source_st:
        how much of that excursion has been recovered by the final sample.

    return_evidence:
        normalized 0..1 measure:

            0 = finishes at maximum excursion
            1 = returns completely to source

    This is deliberately geometric only.  High return evidence does NOT
    mean "mordent", "vibrato", "turn", etc.
    """

    if (
        not np.isfinite(source)
        or len(pitch) == 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    distance = np.abs(
        pitch - source
    )

    max_distance = float(
        np.max(distance)
    )

    final_distance = float(
        distance[-1]
    )

    if max_distance <= 1e-12:
        return (
            0.0,
            0.0,
            0.0,
        )

    recovered = max(
        0.0,
        max_distance
        - final_distance,
    )

    evidence = np.clip(
        recovered
        / max_distance,
        0.0,
        1.0,
    )

    return (
        max_distance,
        float(recovered),
        float(evidence),
    )

def _measure_transition(
    region,
    pitch_st: np.ndarray,
    config: TransitionGeometryConfig,
) -> TransitionGeometry:
    """
    Measure one already-segmented TRANSITION using the supplied pitch
    representation.

    Stable target centers always come from the frozen target
    segmentation. Only the trajectory being geometrically measured
    changes.
    """

    start = int(
        region.start_index
    )

    end = int(
        region.end_index
    )

    pitch = np.asarray(
        pitch_st[start:end],
        dtype=np.float64,
    )

    if len(pitch) == 0:
        raise RuntimeError(
            "Cannot measure an empty TRANSITION"
        )

    if not np.all(
        np.isfinite(pitch)
    ):
        raise RuntimeError(
            "TRANSITION contains non-finite pitch: "
            f"region={region.index}, "
            f"range={region.start_time_s:.3f}-"
            f"{region.end_time_s:.3f}s"
        )

    source = float(
        region.previous_target_st
    )

    destination = float(
        region.next_target_st
    )

    has_source = bool(
        np.isfinite(source)
    )

    has_destination = bool(
        np.isfinite(destination)
    )

    if (
        has_source
        and has_destination
    ):
        target_interval = float(
            destination - source
        )
    else:
        target_interval = np.nan

    start_pitch = float(
        pitch[0]
    )

    end_pitch = float(
        pitch[-1]
    )

    if has_source:
        departure = float(
            start_pitch - source
        )
    else:
        departure = np.nan

    if has_destination:
        arrival_error = float(
            end_pitch - destination
        )
    else:
        arrival_error = np.nan

    motion = _motion_geometry(
        pitch,
        config.movement_epsilon_st,
    )

    (
        corridor_low,
        corridor_high,
        overshoot,
        undershoot,
    ) = _target_corridor(
        pitch,
        source,
        destination,
    )

    (
        min_source_distance,
        fraction_near_source,
    ) = _target_proximity(
        pitch,
        source,
        config.target_near_st,
    )

    (
        min_destination_distance,
        fraction_near_destination,
    ) = _target_proximity(
        pitch,
        destination,
        config.target_near_st,
    )

    (
        leaves_source,
        returns_toward_source,
        return_evidence,
    ) = _source_return_geometry(
        pitch,
        source,
    )

    return TransitionGeometry(
        region_index=int(
            region.index
        ),

        start_index=start,
        end_index=end,

        start_time_s=float(
            region.start_time_s
        ),

        end_time_s=float(
            region.end_time_s
        ),

        duration_ms=float(
            region.duration_ms
        ),

        n_samples=int(
            end - start
        ),

        source_target_st=source,
        destination_target_st=(
            destination
        ),

        has_source=has_source,
        has_destination=(
            has_destination
        ),

        target_interval_st=(
            target_interval
        ),

        start_pitch_st=(
            start_pitch
        ),

        end_pitch_st=(
            end_pitch
        ),

        median_pitch_st=float(
            np.median(pitch)
        ),

        min_pitch_st=float(
            np.min(pitch)
        ),

        max_pitch_st=float(
            np.max(pitch)
        ),

        span_st=float(
            np.max(pitch)
            - np.min(pitch)
        ),

        departure_from_source_st=(
            departure
        ),

        arrival_error_destination_st=(
            arrival_error
        ),

        net_movement_st=float(
            motion[
                "net_movement_st"
            ]
        ),

        path_length_st=float(
            motion[
                "path_length_st"
            ]
        ),

        path_efficiency=float(
            motion[
                "path_efficiency"
            ]
        ),

        upward_path_st=float(
            motion[
                "upward_path_st"
            ]
        ),

        downward_path_st=float(
            motion[
                "downward_path_st"
            ]
        ),

        upward_fraction=float(
            motion[
                "upward_fraction"
            ]
        ),

        downward_fraction=float(
            motion[
                "downward_fraction"
            ]
        ),

        dominant_direction=int(
            motion[
                "dominant_direction"
            ]
        ),

        monotonic_fraction=float(
            motion[
                "monotonic_fraction"
            ]
        ),

        direction_reversals=int(
            motion[
                "direction_reversals"
            ]
        ),

        max_adjacent_step_st=float(
            motion[
                "max_adjacent_step_st"
            ]
        ),

        median_adjacent_step_st=float(
            motion[
                "median_adjacent_step_st"
            ]
        ),

        corridor_low_st=float(
            corridor_low
        ),

        corridor_high_st=float(
            corridor_high
        ),

        overshoot_above_st=float(
            overshoot
        ),

        undershoot_below_st=float(
            undershoot
        ),

        min_distance_source_st=float(
            min_source_distance
        ),

        min_distance_destination_st=float(
            min_destination_distance
        ),

        fraction_near_source=float(
            fraction_near_source
        ),

        fraction_near_destination=float(
            fraction_near_destination
        ),

        leaves_source_st=float(
            leaves_source
        ),

        returns_toward_source_st=float(
            returns_toward_source
        ),

        return_evidence=float(
            return_evidence
        ),
    )

def analyse_transition_geometry(
    interpretation: TrajectoryInterpretation,
    config: TransitionGeometryConfig | None = None,
) -> TransitionGeometryResult:

    if config is None:
        config = (
            TransitionGeometryConfig()
        )

    raw_pitch = np.asarray(
        interpretation.pitch_st,
        dtype=np.float64,
    )

    transitions = []

    for region in interpretation.regions:

        if (
            region.kind
            is not RegionKind.TRANSITION
        ):
            continue

        transitions.append(
            _measure_transition(
                region=region,
                pitch_st=raw_pitch,
                config=config,
            )
        )

    return TransitionGeometryResult(
        transitions=tuple(
            transitions
        ),
        analysis_hz=float(
            interpretation.analysis_hz
        ),
    )

def compare_transition_geometry(
    interpretation: TrajectoryInterpretation,
    smoothing: ExpressiveSmoothingResult,
    config: TransitionGeometryConfig | None = None,
) -> TransitionGeometryComparisonResult:

    if config is None:
        config = (
            TransitionGeometryConfig()
        )

    if not np.isclose(
        interpretation.analysis_hz,
        smoothing.analysis_hz,
    ):
        raise ValueError(
            "Interpreter and smoothing analysis rates differ"
        )

    if (
        len(interpretation.pitch_st)
        != len(smoothing.shape_pitch_st)
    ):
        raise ValueError(
            "Interpreter and smoothing trajectories "
            "have different lengths"
        )

    raw_pitch = np.asarray(
        interpretation.pitch_st,
        dtype=np.float64,
    )

    shape_pitch = np.asarray(
        smoothing.shape_pitch_st,
        dtype=np.float64,
    )

    comparisons = []

    for region in interpretation.regions:

        if (
            region.kind
            is not RegionKind.TRANSITION
        ):
            continue

        raw = _measure_transition(
            region=region,
            pitch_st=raw_pitch,
            config=config,
        )

        shape = _measure_transition(
            region=region,
            pitch_st=shape_pitch,
            config=config,
        )

        comparisons.append(
            TransitionGeometryComparison(
                raw=raw,
                shape=shape,
            )
        )

    return (
        TransitionGeometryComparisonResult(
            transitions=tuple(
                comparisons
            ),
            analysis_hz=float(
                interpretation.analysis_hz
            ),
        )
    )