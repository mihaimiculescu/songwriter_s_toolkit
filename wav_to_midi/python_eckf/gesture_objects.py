from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .trajectory_interpreter import (
    PitchRegion,
    RegionKind,
    TrajectoryInterpretation,
)

from .gesture_features import (
    GestureFeatureResult,
    GestureFeatures,
)


class GestureObjectKind(Enum):
    """
    Structural object type only.

    These names describe how frozen trajectory regions are grouped.
    They are NOT musical gesture labels.
    """

    SINGLE_TRANSITION = auto()
    TRANSITION_TARGET_TRANSITION = auto()
    TRANSITION_CHAIN = auto()


@dataclass(frozen=True)
class GestureObjectConfig:
    """
    Observational grouping configuration.

    short_target_max_ms:
        A stable target at or below this duration is eligible to act
        as an interior target connecting two neighboring transitions.

        This does NOT mean that such a target is an ornament.
        It merely makes the larger trajectory available for later
        interpretation.

    max_chain_transitions:
        Safety limit for how many transitions may belong to one
        candidate chain.

    require_same_valid_island:
        Validity holes remain walls.
    """

    short_target_max_ms: float = 250.0
    max_chain_transitions: int = 8
    require_same_valid_island: bool = True

@dataclass(frozen=True)
class GestureObjectLink:
    """
    Geometry of one frozen T-S-T connection inside a multi-transition
    candidate object.

    This is observational only.

    A link corresponds to:

        left TRANSITION
            ↓
        STABLE_TARGET
            ↓
        right TRANSITION

    No musical interpretation is assigned here.
    """

    link_index: int

    left_transition_region_index: int
    stable_region_index: int
    right_transition_region_index: int

    stable_target_st: float
    stable_duration_ms: float

    previous_target_st: float
    next_target_st: float

    entry_interval_st: float
    exit_interval_st: float

    outer_interval_st: float

    entry_abs_interval_st: float
    exit_abs_interval_st: float
    outer_abs_interval_st: float

    stable_from_previous_st: float
    next_from_stable_st: float

    direction_in: int
    direction_out: int

    same_direction: bool
    direction_reversal: bool

    outer_targets_available: bool

    stable_to_previous_distance_st: float
    stable_to_next_distance_st: float

    returns_toward_previous: bool
    returns_toward_next: bool

@dataclass(frozen=True)
class GestureObject:
    """
    One candidate multi-region trajectory object.

    All indices refer to the already-frozen TrajectoryInterpretation.

    Nothing here changes the original segmentation.
    """

    object_index: int
    kind: GestureObjectKind

    start_region_index: int
    end_region_index: int

    region_indices: tuple[int, ...]
    transition_region_indices: tuple[int, ...]
    stable_region_indices: tuple[int, ...]

    transition_feature_indices: tuple[int, ...]
    links: tuple[GestureObjectLink, ...]

    start_index: int
    end_index: int

    start_time_s: float
    end_time_s: float
    duration_ms: float

    n_regions: int
    n_transitions: int
    n_stable_targets: int

    # ------------------------------------------------------------
    # Boundary target context
    # ------------------------------------------------------------

    source_target_st: float
    destination_target_st: float

    has_source_target: bool
    has_destination_target: bool

    boundary_interval_st: float

    # ------------------------------------------------------------
    # Interior stable-target information
    # ------------------------------------------------------------

    interior_targets_st: tuple[float, ...]
    interior_target_count: int

    interior_target_min_st: float
    interior_target_max_st: float
    interior_target_span_st: float

    interior_target_total_ms: float
    interior_target_max_duration_ms: float

    # ------------------------------------------------------------
    # Transition feature summaries
    # ------------------------------------------------------------

    transition_total_duration_ms: float

    shape_path_total_st: float
    center_path_total_st: float

    residual_rms_weighted_st: float
    residual_abs_max_st: float

    source_return_max: float

    residual_zero_crossings_total: int
    residual_peak_count_total: int
    residual_trough_count_total: int

    autocorr_support_max: float
    autocorr_cycles_max: float

    # ------------------------------------------------------------
    # Frozen-region topology
    # ------------------------------------------------------------

    begins_with_transition: bool
    ends_with_transition: bool

    contains_short_interior_target: bool

    returns_near_source_st: float


@dataclass(frozen=True)
class GestureObjectResult:
    """
    Candidate objects plus invariant counts from the frozen input.
    """

    objects: tuple[GestureObject, ...]

    analysis_hz: float

    frozen_region_count: int
    frozen_transition_count: int

    represented_transition_region_indices: tuple[int, ...]

def _signed_difference(
    destination: float,
    source: float,
) -> float:

    if not (
        np.isfinite(source)
        and np.isfinite(destination)
    ):
        return np.nan

    return float(
        destination - source
    )


def _abs_or_nan(
    value: float,
) -> float:

    if not np.isfinite(value):
        return np.nan

    return float(
        abs(value)
    )


def _direction(
    interval_st: float,
    epsilon_st: float = 0.05,
) -> int:
    """
    Return:
        -1 = downward
         0 = effectively flat / unavailable
        +1 = upward

    This is geometry, not a musical label.
    """

    if not np.isfinite(
        interval_st
    ):
        return 0

    if interval_st > epsilon_st:
        return 1

    if interval_st < -epsilon_st:
        return -1

    return 0

def _construct_object_links(
    *,
    region_indices: tuple[int, ...],
    regions,
) -> tuple[GestureObjectLink, ...]:
    """
    Construct every exact T-S-T connector contained in one candidate
    object's frozen region sequence.

    Original regions are never modified.
    """

    links = []

    if len(region_indices) < 3:
        return tuple()

    for position in range(
        1,
        len(region_indices) - 1,
    ):

        stable_index = (
            region_indices[position]
        )

        stable_region = (
            regions[stable_index]
        )

        if (
            stable_region.kind
            is not RegionKind.STABLE_TARGET
        ):
            continue

        left_index = (
            region_indices[
                position - 1
            ]
        )

        right_index = (
            region_indices[
                position + 1
            ]
        )

        left_region = (
            regions[left_index]
        )

        right_region = (
            regions[right_index]
        )

        if (
            left_region.kind
            is not RegionKind.TRANSITION
        ):
            continue

        if (
            right_region.kind
            is not RegionKind.TRANSITION
        ):
            continue

        stable_target = (
            _region_target_st(
                stable_region
            )
        )

        previous_target = (
            _finite_or_nan(
                stable_region.previous_target_st
            )
        )

        next_target = (
            _finite_or_nan(
                stable_region.next_target_st
            )
        )

        entry_interval = (
            _signed_difference(
                stable_target,
                previous_target,
            )
        )

        exit_interval = (
            _signed_difference(
                next_target,
                stable_target,
            )
        )

        outer_interval = (
            _signed_difference(
                next_target,
                previous_target,
            )
        )

        direction_in = (
            _direction(
                entry_interval
            )
        )

        direction_out = (
            _direction(
                exit_interval
            )
        )

        outer_targets_available = bool(
            np.isfinite(
                previous_target
            )
            and np.isfinite(
                next_target
            )
        )

        same_direction = bool(
            direction_in != 0
            and direction_out != 0
            and direction_in
            == direction_out
        )

        direction_reversal = bool(
            direction_in != 0
            and direction_out != 0
            and direction_in
            == -direction_out
        )

        stable_to_previous = (
            _abs_or_nan(
                entry_interval
            )
        )

        stable_to_next = (
            _abs_or_nan(
                exit_interval
            )
        )

        returns_toward_previous = bool(
            outer_targets_available
            and np.isfinite(
                stable_to_previous
            )
            and np.isfinite(
                outer_interval
            )
            and abs(
                outer_interval
            )
            < stable_to_previous
        )

        returns_toward_next = bool(
            outer_targets_available
            and np.isfinite(
                stable_to_next
            )
            and np.isfinite(
                outer_interval
            )
            and abs(
                outer_interval
            )
            < stable_to_next
        )

        links.append(
            GestureObjectLink(
                link_index=len(
                    links
                ),

                left_transition_region_index=(
                    left_index
                ),
                stable_region_index=(
                    stable_index
                ),
                right_transition_region_index=(
                    right_index
                ),

                stable_target_st=float(
                    stable_target
                ),
                stable_duration_ms=float(
                    stable_region.duration_ms
                ),

                previous_target_st=float(
                    previous_target
                ),
                next_target_st=float(
                    next_target
                ),

                entry_interval_st=float(
                    entry_interval
                ),
                exit_interval_st=float(
                    exit_interval
                ),
                outer_interval_st=float(
                    outer_interval
                ),

                entry_abs_interval_st=(
                    _abs_or_nan(
                        entry_interval
                    )
                ),
                exit_abs_interval_st=(
                    _abs_or_nan(
                        exit_interval
                    )
                ),
                outer_abs_interval_st=(
                    _abs_or_nan(
                        outer_interval
                    )
                ),

                stable_from_previous_st=float(
                    stable_to_previous
                ),
                next_from_stable_st=float(
                    stable_to_next
                ),

                direction_in=(
                    direction_in
                ),
                direction_out=(
                    direction_out
                ),

                same_direction=(
                    same_direction
                ),
                direction_reversal=(
                    direction_reversal
                ),

                outer_targets_available=(
                    outer_targets_available
                ),

                stable_to_previous_distance_st=float(
                    stable_to_previous
                ),
                stable_to_next_distance_st=float(
                    stable_to_next
                ),

                returns_toward_previous=(
                    returns_toward_previous
                ),
                returns_toward_next=(
                    returns_toward_next
                ),
            )
        )

    return tuple(
        links
    )


def _safe_float(
    value,
) -> float:

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return np.nan

    return result


def _finite_or_nan(
    value,
) -> float:

    value = _safe_float(
        value
    )

    if np.isfinite(
        value
    ):
        return value

    return np.nan


def _region_duration_ms(
    region: PitchRegion,
    analysis_hz: float,
) -> float:
    """
    Prefer the frozen region's own duration if available.

    Fall back to index support only if necessary.
    """

    duration = _finite_or_nan(
        getattr(
            region,
            "duration_ms",
            np.nan,
        )
    )

    if (
        np.isfinite(
            duration
        )
        and duration >= 0.0
    ):
        return duration

    n = max(
        0,
        int(
            region.end_index
        )
        - int(
            region.start_index
        ),
    )

    return float(
        n
        * 1000.0
        / analysis_hz
    )


def _region_target_st(
    region: PitchRegion,
) -> float:
    """
    Extract the frozen structural stable-target center.

    Use median_pitch_st from PitchRegion.
    Do not substitute raw_median_pitch_st: object construction
    should use the frozen structural representation.
    """

    return _finite_or_nan(
        region.median_pitch_st
    )


def _is_short_stable_target(
    region: PitchRegion,
    analysis_hz: float,
    config: GestureObjectConfig,
) -> bool:

    if (
        region.kind
        is not RegionKind.STABLE_TARGET
    ):
        return False

    duration_ms = (
        _region_duration_ms(
            region,
            analysis_hz,
        )
    )

    return bool(
        duration_ms
        <= config.short_target_max_ms
    )


def _build_transition_feature_map(
    features: GestureFeatureResult,
) -> dict[int, GestureFeatures]:
    """
    Map frozen transition-region index -> its GestureFeatures.

    GestureFeatures.region_index is the ordinal transition index in
    the V1/V2 feature extractor, not necessarily the PitchRegion list
    index. Therefore we do NOT assume they are interchangeable here.

    The actual mapping is reconstructed by transition order later.
    """

    return {
        int(
            feature.region_index
        ): feature
        for feature in features.gestures
    }


def _transition_regions(
    interpretation: TrajectoryInterpretation,
) -> list[tuple[int, PitchRegion]]:

    return [
        (
            region_index,
            region,
        )
        for (
            region_index,
            region,
        ) in enumerate(
            interpretation.regions
        )
        if (
            region.kind
            is RegionKind.TRANSITION
        )
    ]


def _transition_feature_by_region(
    interpretation: TrajectoryInterpretation,
    features: GestureFeatureResult,
) -> dict[int, GestureFeatures]:
    """
    Associate each frozen TRANSITION PitchRegion with its feature
    object by order.

    This preserves the invariant established by Gesture Features V2:

        frozen transitions in == feature objects out
    """

    transitions = (
        _transition_regions(
            interpretation
        )
    )

    feature_list = list(
        features.gestures
    )

    if (
        len(transitions)
        != len(feature_list)
    ):
        raise ValueError(
            "transition count does not match gesture feature count: "
            f"{len(transitions)} != {len(feature_list)}"
        )

    result = {}

    for (
        (
            frozen_region_index,
            _,
        ),
        feature,
    ) in zip(
        transitions,
        feature_list,
    ):
        result[
            frozen_region_index
        ] = feature

    return result


def _same_valid_island(
    left_region: PitchRegion,
    right_region: PitchRegion,
    interpretation: TrajectoryInterpretation,
) -> bool:
    """
    A validity hole is a wall.

    Regions that are directly adjacent in interpretation.regions are
    normally already within one trusted island, but this explicit
    check protects the grouping layer from future representation
    changes.
    """

    left_end = int(
        left_region.end_index
    )

    right_start = int(
        right_region.start_index
    )

    if right_start < left_end:
        return True

    valid = np.asarray(
        interpretation.valid,
        dtype=bool,
    ) if hasattr(
        interpretation,
        "valid",
    ) else None

    if valid is None:
        # The frozen interpreter itself already partitions at validity
        # walls. Do not manufacture additional assumptions.
        return (
            right_start
            == left_end
        )

    start = max(
        0,
        left_end,
    )

    end = min(
        len(valid),
        right_start,
    )

    if end <= start:
        return True

    return bool(
        np.all(
            valid[
                start:end
            ]
        )
    )


def _can_connect_through_stable(
    regions: tuple[PitchRegion, ...] | list[PitchRegion],
    left_transition_index: int,
    stable_index: int,
    right_transition_index: int,
    interpretation: TrajectoryInterpretation,
    analysis_hz: float,
    config: GestureObjectConfig,
) -> bool:

    if not (
        0
        <= left_transition_index
        < stable_index
        < right_transition_index
        < len(regions)
    ):
        return False

    left = regions[
        left_transition_index
    ]

    middle = regions[
        stable_index
    ]

    right = regions[
        right_transition_index
    ]

    if (
        left.kind
        is not RegionKind.TRANSITION
    ):
        return False

    if (
        middle.kind
        is not RegionKind.STABLE_TARGET
    ):
        return False

    if (
        right.kind
        is not RegionKind.TRANSITION
    ):
        return False

    # Exact T-S-T adjacency only.
    if (
        stable_index
        != left_transition_index + 1
    ):
        return False

    if (
        right_transition_index
        != stable_index + 1
    ):
        return False

    if not _is_short_stable_target(
        middle,
        analysis_hz,
        config,
    ):
        return False

    if (
        config.require_same_valid_island
        and not _same_valid_island(
            left,
            middle,
            interpretation,
        )
    ):
        return False

    if (
        config.require_same_valid_island
        and not _same_valid_island(
            middle,
            right,
            interpretation,
        )
    ):
        return False

    return True


def _candidate_chains(
    interpretation: TrajectoryInterpretation,
    config: GestureObjectConfig,
) -> list[tuple[int, ...]]:
    """
    Construct maximal candidate chains of the form:

        T
        T-S-T
        T-S-T-S-T
        ...

    where each S is an already-frozen short STABLE_TARGET.

    Every transition remains represented even when it does not belong
    to a multi-transition chain.

    No region is modified.
    """

    regions = tuple(
        interpretation.regions
    )

    analysis_hz = float(
        interpretation.analysis_hz
    )

    transition_indices = [
        i
        for (
            i,
            region,
        ) in enumerate(
            regions
        )
        if (
            region.kind
            is RegionKind.TRANSITION
        )
    ]

    consumed = set()
    chains = []

    for transition_index in transition_indices:

        if transition_index in consumed:
            continue

        chain = [
            transition_index
        ]

        current_transition = (
            transition_index
        )

        while True:

            if (
                len(
                    [
                        i
                        for i in chain
                        if (
                            regions[i].kind
                            is RegionKind.TRANSITION
                        )
                    ]
                )
                >= config.max_chain_transitions
            ):
                break

            stable_index = (
                current_transition + 1
            )

            next_transition = (
                current_transition + 2
            )

            if (
                next_transition
                >= len(regions)
            ):
                break

            if not _can_connect_through_stable(
                regions=regions,
                left_transition_index=current_transition,
                stable_index=stable_index,
                right_transition_index=next_transition,
                interpretation=interpretation,
                analysis_hz=analysis_hz,
                config=config,
            ):
                break

            chain.extend(
                [
                    stable_index,
                    next_transition,
                ]
            )

            current_transition = (
                next_transition
            )

        chains.append(
            tuple(
                chain
            )
        )

        for index in chain:
            if (
                regions[index].kind
                is RegionKind.TRANSITION
            ):
                consumed.add(
                    index
                )

    return chains


def _weighted_rms(
    features: list[GestureFeatures],
) -> float:

    if not features:
        return np.nan

    weights = np.asarray(
        [
            max(
                1,
                int(
                    feature.n_samples
                ),
            )
            for feature in features
        ],
        dtype=np.float64,
    )

    rms = np.asarray(
        [
            feature.residual_rms_st
            for feature in features
        ],
        dtype=np.float64,
    )

    finite = (
        np.isfinite(
            weights
        )
        & np.isfinite(
            rms
        )
    )

    if not np.any(
        finite
    ):
        return np.nan

    weights = weights[
        finite
    ]

    rms = rms[
        finite
    ]

    denominator = float(
        np.sum(
            weights
        )
    )

    if denominator <= 0.0:
        return np.nan

    return float(
        np.sqrt(
            np.sum(
                weights
                * rms
                * rms
            )
            / denominator
        )
    )


def _nanmax_or_nan(
    values,
) -> float:

    values = np.asarray(
        list(
            values
        ),
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(finite) == 0:
        return np.nan

    return float(
        np.max(
            finite
        )
    )


def _nansum_or_zero(
    values,
) -> float:

    values = np.asarray(
        list(
            values
        ),
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(finite) == 0:
        return 0.0

    return float(
        np.sum(
            finite
        )
    )


def _boundary_targets(
    transition_features: list[GestureFeatures],
) -> tuple[
    float,
    float,
    bool,
    bool,
    float,
]:

    if not transition_features:
        return (
            np.nan,
            np.nan,
            False,
            False,
            np.nan,
        )

    first = transition_features[
        0
    ]

    last = transition_features[
        -1
    ]

    source = _finite_or_nan(
        first.source_target_st
    )

    destination = _finite_or_nan(
        last.destination_target_st
    )

    has_source = bool(
        np.isfinite(
            source
        )
    )

    has_destination = bool(
        np.isfinite(
            destination
        )
    )

    if (
        has_source
        and has_destination
    ):
        interval = float(
            destination
            - source
        )
    else:
        interval = np.nan

    return (
        source,
        destination,
        has_source,
        has_destination,
        interval,
    )


def _return_near_source(
    source_target_st: float,
    destination_target_st: float,
) -> float:
    """
    Absolute source/destination separation.

    Small values mean the complete candidate object returns near its
    starting target.

    This is geometry only. It is NOT a mordent/trill/etc. score.
    """

    if not (
        np.isfinite(
            source_target_st
        )
        and np.isfinite(
            destination_target_st
        )
    ):
        return np.nan

    return float(
        abs(
            destination_target_st
            - source_target_st
        )
    )


def construct_gesture_objects(
    interpretation: TrajectoryInterpretation,
    features: GestureFeatureResult,
    config: GestureObjectConfig | None = None,
) -> GestureObjectResult:
    """
    Construct candidate multi-region objects from the already-frozen
    trajectory segmentation.

    GUARANTEES
    ----------
    - does not alter TrajectoryInterpretation;
    - does not alter GestureFeatureResult;
    - does not move region boundaries;
    - does not change validity;
    - does not assign musical gesture labels;
    - every frozen transition is represented exactly once by the
      maximal object partition produced here.
    """

    if config is None:
        config = GestureObjectConfig()

    if config.short_target_max_ms <= 0.0:
        raise ValueError(
            "short_target_max_ms must be > 0"
        )

    if config.max_chain_transitions < 1:
        raise ValueError(
            "max_chain_transitions must be >= 1"
        )

    analysis_hz = float(
        interpretation.analysis_hz
    )

    if not np.isclose(
        analysis_hz,
        features.analysis_hz,
    ):
        raise ValueError(
            "interpretation and gesture feature "
            "analysis rates differ"
        )

    regions = tuple(
        interpretation.regions
    )

    transition_feature_by_region = (
        _transition_feature_by_region(
            interpretation,
            features,
        )
    )

    frozen_transition_indices = tuple(
        index
        for (
            index,
            region,
        ) in enumerate(
            regions
        )
        if (
            region.kind
            is RegionKind.TRANSITION
        )
    )

    chains = _candidate_chains(
        interpretation,
        config,
    )

    objects = []

    represented_transitions = []

    for object_zero_index, chain in enumerate(
        chains
    ):

        chain_regions = [
            regions[index]
            for index in chain
        ]

        transition_indices = tuple(
            index
            for index in chain
            if (
                regions[index].kind
                is RegionKind.TRANSITION
            )
        )

        stable_indices = tuple(
            index
            for index in chain
            if (
                regions[index].kind
                is RegionKind.STABLE_TARGET
            )
        )

        transition_features = [
            transition_feature_by_region[
                index
            ]
            for index in transition_indices
        ]

        represented_transitions.extend(
            transition_indices
        )

        if len(
            transition_indices
        ) == 1:

            kind = (
                GestureObjectKind.SINGLE_TRANSITION
            )

        elif (
            len(
                transition_indices
            ) == 2
            and len(
                stable_indices
            ) == 1
        ):

            kind = (
                GestureObjectKind.TRANSITION_TARGET_TRANSITION
            )

        else:

            kind = (
                GestureObjectKind.TRANSITION_CHAIN
            )

        first_region = (
            chain_regions[
                0
            ]
        )

        last_region = (
            chain_regions[
                -1
            ]
        )

        start_index = int(
            first_region.start_index
        )

        end_index = int(
            last_region.end_index
        )

        start_time_s = float(
            first_region.start_time_s
        )

        end_time_s = float(
            last_region.end_time_s
        )

        duration_ms = float(
            max(
                0.0,
                (
                    end_index
                    - start_index
                )
                * 1000.0
                / analysis_hz,
            )
        )

        interior_target_values = [
            _region_target_st(
                regions[index]
            )
            for index in stable_indices
        ]

        interior_target_values = [
            value
            for value in interior_target_values
            if np.isfinite(
                value
            )
        ]

        interior_targets_st = tuple(
            float(value)
            for value in interior_target_values
        )

        interior_durations = [
            _region_duration_ms(
                regions[index],
                analysis_hz,
            )
            for index in stable_indices
        ]

        if interior_target_values:

            interior_min = float(
                np.min(
                    interior_target_values
                )
            )

            interior_max = float(
                np.max(
                    interior_target_values
                )
            )

            interior_span = float(
                interior_max
                - interior_min
            )

        else:

            interior_min = np.nan
            interior_max = np.nan
            interior_span = np.nan

        if interior_durations:

            interior_total_ms = float(
                np.sum(
                    interior_durations
                )
            )

            interior_max_duration_ms = float(
                np.max(
                    interior_durations
                )
            )

        else:

            interior_total_ms = 0.0
            interior_max_duration_ms = 0.0

        (
            source_target,
            destination_target,
            has_source,
            has_destination,
            boundary_interval,
        ) = _boundary_targets(
            transition_features
        )

        transition_total_duration_ms = (
            _nansum_or_zero(
                feature.duration_ms
                for feature
                in transition_features
            )
        )

        shape_path_total = (
            _nansum_or_zero(
                feature.shape_path_st
                for feature
                in transition_features
            )
        )

        center_path_total = (
            _nansum_or_zero(
                feature.center_path_st
                for feature
                in transition_features
            )
        )

        residual_rms_weighted = (
            _weighted_rms(
                transition_features
            )
        )

        residual_abs_max = (
            _nanmax_or_nan(
                feature.residual_abs_max_st
                for feature
                in transition_features
            )
        )

        source_return_max = (
            _nanmax_or_nan(
                feature.source_return_fraction
                for feature
                in transition_features
            )
        )

        residual_zero_crossings_total = int(
            sum(
                feature.residual_zero_crossings
                for feature
                in transition_features
            )
        )

        residual_peak_count_total = int(
            sum(
                feature.residual_peak_count
                for feature
                in transition_features
            )
        )

        residual_trough_count_total = int(
            sum(
                feature.residual_trough_count
                for feature
                in transition_features
            )
        )

        autocorr_support_max = (
            _nanmax_or_nan(
                feature.residual_autocorr_support
                for feature
                in transition_features
            )
        )

        autocorr_cycles_max = (
            _nanmax_or_nan(
                feature.residual_autocorr_cycles_supported
                for feature
                in transition_features
            )
        )

        links = _construct_object_links(
            region_indices=tuple(
                int(index)
                for index in chain
            ),
            regions=regions,
        )

        expected_links = max(
            0,
            len(transition_indices) - 1,
        )

        if len(links) != expected_links:
            raise RuntimeError(
                "Gesture-object link invariant failed: "
                f"object contains "
                f"{len(transition_indices)} transitions "
                f"but produced {len(links)} links; "
                f"expected {expected_links}. "
                f"regions={tuple(chain)}"
            )

        objects.append(
            GestureObject(
                object_index=(
                    object_zero_index
                    + 1
                ),
                kind=kind,

                start_region_index=int(
                    chain[0]
                ),
                end_region_index=int(
                    chain[-1]
                ),

                region_indices=tuple(
                    int(index)
                    for index in chain
                ),

                transition_region_indices=(
                    transition_indices
                ),

                stable_region_indices=(
                    stable_indices
                ),

                transition_feature_indices=tuple(
                    int(
                        feature.region_index
                    )
                    for feature
                    in transition_features
                ),

                links=links,

                start_index=start_index,
                end_index=end_index,

                start_time_s=start_time_s,
                end_time_s=end_time_s,
                duration_ms=duration_ms,

                n_regions=len(
                    chain
                ),
                n_transitions=len(
                    transition_indices
                ),
                n_stable_targets=len(
                    stable_indices
                ),

                source_target_st=(
                    source_target
                ),
                destination_target_st=(
                    destination_target
                ),

                has_source_target=(
                    has_source
                ),
                has_destination_target=(
                    has_destination
                ),

                boundary_interval_st=(
                    boundary_interval
                ),

                interior_targets_st=interior_targets_st,

                interior_target_count=len(
                    interior_target_values
                ),

                interior_target_min_st=(
                    interior_min
                ),
                interior_target_max_st=(
                    interior_max
                ),
                interior_target_span_st=(
                    interior_span
                ),

                interior_target_total_ms=(
                    interior_total_ms
                ),
                interior_target_max_duration_ms=(
                    interior_max_duration_ms
                ),

                transition_total_duration_ms=(
                    transition_total_duration_ms
                ),

                shape_path_total_st=(
                    shape_path_total
                ),
                center_path_total_st=(
                    center_path_total
                ),

                residual_rms_weighted_st=(
                    residual_rms_weighted
                ),
                residual_abs_max_st=(
                    residual_abs_max
                ),

                source_return_max=(
                    source_return_max
                ),

                residual_zero_crossings_total=(
                    residual_zero_crossings_total
                ),
                residual_peak_count_total=(
                    residual_peak_count_total
                ),
                residual_trough_count_total=(
                    residual_trough_count_total
                ),

                autocorr_support_max=(
                    autocorr_support_max
                ),
                autocorr_cycles_max=(
                    autocorr_cycles_max
                ),

                begins_with_transition=bool(
                    first_region.kind
                    is RegionKind.TRANSITION
                ),

                ends_with_transition=bool(
                    last_region.kind
                    is RegionKind.TRANSITION
                ),

                contains_short_interior_target=bool(
                    len(
                        stable_indices
                    )
                    > 0
                ),

                returns_near_source_st=(
                    _return_near_source(
                        source_target,
                        destination_target,
                    )
                ),
            )
        )

    represented_tuple = tuple(
        int(index)
        for index in represented_transitions
    )

    # ------------------------------------------------------------
    # HARD INVARIANT:
    # every frozen transition must appear exactly once.
    # ------------------------------------------------------------

    if (
        tuple(
            sorted(
                represented_tuple
            )
        )
        != tuple(
            sorted(
                frozen_transition_indices
            )
        )
    ):
        raise RuntimeError(
            "gesture-object construction changed transition "
            "population"
        )

    if (
        len(
            represented_tuple
        )
        != len(
            set(
                represented_tuple
            )
        )
    ):
        raise RuntimeError(
            "a frozen transition was represented by more "
            "than one maximal gesture object"
        )

    return GestureObjectResult(
        objects=tuple(
            objects
        ),

        analysis_hz=analysis_hz,

        frozen_region_count=len(
            regions
        ),

        frozen_transition_count=len(
            frozen_transition_indices
        ),

        represented_transition_region_indices=(
            represented_tuple
        ),
    )