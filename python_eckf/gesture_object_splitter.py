"""
Observational structural-boundary analysis for maximal gesture objects.

This module consumes the frozen maximal GestureObject representation and its
per-link GestureObjectLink geometry.

IMPORTANT
---------
This layer does NOT assign musical gesture labels.

It does NOT modify:
    - pitch validity
    - corrected F0
    - pitch regions
    - transition regions
    - gesture features
    - GestureObject construction
    - GestureObjectLink construction

Its purpose is narrower:

For every T-S-T link inside a multi-transition maximal object, collect
structural evidence relevant to the future question:

    "Does this short stable target belong inside one local gesture object,
     or is it more plausibly a boundary between neighboring objects?"

V1 is deliberately observational.  It computes evidence but does not yet
split objects automatically.  Automatic splitting should only be enabled
after the evidence has been inspected on regression material.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .gesture_objects import (
    GestureObject,
    GestureObjectLink,
    GestureObjectResult,
)

from .gesture_features import (
    GestureFeatures,
    GestureFeatureResult,
)

class StructuralRelation(Enum):
    """
    Purely geometric relation at one T-S-T connector.

    These are NOT musical labels.
    """

    INCOMPLETE = auto()

    SAME_DIRECTION = auto()
    REVERSAL = auto()

    NEAR_FLAT_ENTRY = auto()
    NEAR_FLAT_EXIT = auto()
    NEAR_FLAT_BOTH = auto()


@dataclass(frozen=True)
class GestureObjectSplitterConfig:
    """
    Configuration for observational link analysis.

    near_flat_st is not a musical threshold.  It only defines when one side
    of a T-S-T connector is so small that its direction should not dominate
    the structural description.

    The existing GestureObjectLink direction fields were created with a
    0.05-semitone directional epsilon, so V1 uses the same scale here.
    """

    near_flat_st: float = 0.05


@dataclass(frozen=True)
class LinkSplitEvidence:
    """
    Observational evidence for one possible internal object boundary.
    """

    object_index: int
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

    relation: StructuralRelation

    outer_targets_available: bool

    same_direction: bool
    direction_reversal: bool

    entry_near_flat: bool
    exit_near_flat: bool

    returns_toward_previous: bool
    returns_toward_next: bool

    stable_to_previous_distance_st: float
    stable_to_next_distance_st: float

    # How much farther the path travels through the stable target than the
    # direct previous-target -> next-target displacement.
    #
    # Example:
    #
    #     66.67 -> 64.02 -> 66.97
    #
    # local path ~= 5.61 st
    # outer displacement ~= 0.30 st
    #
    # therefore this is strongly excursion-like geometrically.
    local_target_path_st: float
    excess_over_outer_st: float

    # Ratio:
    #
    #     outer displacement / local target path
    #
    # Near 1:
    #     movement through the stable target is close to direct.
    #
    # Near 0:
    #     substantial leave-and-return geometry.
    #
    # NaN if both outer targets are unavailable or local path is zero.
    outer_to_path_ratio: float

    # Distance of the final target from the previous target compared with
    # the distance reached at the stable target.
    #
    # Positive:
    #     final target is closer to previous than stable target was.
    #
    # Negative:
    #     final target moved farther away.
    #
    # NaN when outer targets are unavailable.
    return_gain_to_previous_st: float

    # Symmetric quantity looking backward from the next target.
    return_gain_to_next_st: float
    # ------------------------------------------------------------------
    # V2: frozen transition-feature evidence
    # ------------------------------------------------------------------

    left_feature: GestureFeatures
    right_feature: GestureFeatures

    # Direct duration comparison.
    left_transition_duration_ms: float
    right_transition_duration_ms: float
    transition_duration_ratio: float

    # Shape-path comparison.
    left_shape_path_st: float
    right_shape_path_st: float
    shape_path_ratio: float

    # Center-path comparison.
    left_center_path_st: float
    right_center_path_st: float
    center_path_ratio: float

    # How center-dominant each transition is.
    left_center_to_shape_ratio: float
    right_center_to_shape_ratio: float
    center_to_shape_difference: float

    # Residual / expressive activity.
    left_residual_rms_st: float
    right_residual_rms_st: float
    residual_rms_ratio: float

    left_residual_peak_to_peak_st: float
    right_residual_peak_to_peak_st: float
    residual_peak_to_peak_ratio: float

    # Topological oscillation evidence.
    left_residual_full_cycles: float
    right_residual_full_cycles: float

    left_residual_periodicity: float
    right_residual_periodicity: float

    # Autocorrelation is retained as a witness only.
    left_autocorr_support: float
    right_autocorr_support: float

    left_autocorr_cycles_supported: float
    right_autocorr_cycles_supported: float

    # Existing source-return geometry from each frozen transition.
    left_source_return_fraction: float
    right_source_return_fraction: float

    # Simple pair-comparison witnesses.
    duration_similarity: float
    shape_path_similarity: float
    center_path_similarity: float
    residual_rms_similarity: float

@dataclass(frozen=True)
class ObjectSplitEvidence:
    """
    Observational structural evidence for one maximal GestureObject.
    """

    object_index: int

    region_indices: tuple[int, ...]
    transition_region_indices: tuple[int, ...]
    stable_region_indices: tuple[int, ...]

    n_transitions: int
    n_links: int

    links: tuple[LinkSplitEvidence, ...]

    same_direction_links: int
    reversal_links: int
    incomplete_links: int
    near_flat_links: int

    complete_outer_links: int

    return_to_previous_links: int
    return_to_next_links: int
    return_both_links: int

    local_target_path_total_st: float
    excess_over_outer_total_st: float

    # These summarize the link population but do not constitute a split
    # decision.
    max_excess_over_outer_st: float
    min_outer_to_path_ratio: float


@dataclass(frozen=True)
class GestureObjectSplitAnalysis:
    """
    Result of observational structural-boundary analysis.

    No object has been split in V1.
    """

    objects: tuple[ObjectSplitEvidence, ...]

    total_objects: int
    total_links: int

    objects_with_links: int
    objects_without_links: int

    same_direction_links: int
    reversal_links: int
    incomplete_links: int
    near_flat_links: int


def _finite(
    value: float,
) -> bool:

    return bool(
        np.isfinite(
            value
        )
    )


def _sum_finite(
    values,
) -> float:

    finite_values = [
        float(value)
        for value in values
        if _finite(
            value
        )
    ]

    if not finite_values:
        return 0.0

    return float(
        np.sum(
            finite_values
        )
    )


def _max_finite_or_nan(
    values,
) -> float:

    finite_values = [
        float(value)
        for value in values
        if _finite(
            value
        )
    ]

    if not finite_values:
        return float("nan")

    return float(
        np.max(
            finite_values
        )
    )


def _min_finite_or_nan(
    values,
) -> float:

    finite_values = [
        float(value)
        for value in values
        if _finite(
            value
        )
    ]

    if not finite_values:
        return float("nan")

    return float(
        np.min(
            finite_values
        )
    )


def _structural_relation(
    link: GestureObjectLink,
    config: GestureObjectSplitterConfig,
) -> tuple[
    StructuralRelation,
    bool,
    bool,
]:

    entry_available = _finite(
        link.entry_interval_st
    )

    exit_available = _finite(
        link.exit_interval_st
    )

    if not (
        entry_available
        and exit_available
    ):
        return (
            StructuralRelation.INCOMPLETE,
            False,
            False,
        )

    entry_near_flat = (
        abs(
            float(
                link.entry_interval_st
            )
        )
        <= config.near_flat_st
    )

    exit_near_flat = (
        abs(
            float(
                link.exit_interval_st
            )
        )
        <= config.near_flat_st
    )

    if (
        entry_near_flat
        and exit_near_flat
    ):
        relation = (
            StructuralRelation.NEAR_FLAT_BOTH
        )

    elif entry_near_flat:
        relation = (
            StructuralRelation.NEAR_FLAT_ENTRY
        )

    elif exit_near_flat:
        relation = (
            StructuralRelation.NEAR_FLAT_EXIT
        )

    elif link.direction_reversal:
        relation = (
            StructuralRelation.REVERSAL
        )

    elif link.same_direction:
        relation = (
            StructuralRelation.SAME_DIRECTION
        )

    else:
        # With both intervals available and neither side near-flat,
        # this should normally be unreachable if GestureObjectLink's
        # direction fields are internally consistent.
        raise RuntimeError(
            "Unexpected complete T-S-T link geometry: "
            f"object link {link.link_index} has "
            "neither same-direction nor reversal geometry."
        )

    return (
        relation,
        entry_near_flat,
        exit_near_flat,
    )


def _local_geometry(
    link: GestureObjectLink,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:

    if not (
        _finite(
            link.entry_abs_interval_st
        )
        and _finite(
            link.exit_abs_interval_st
        )
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    local_target_path_st = float(
        link.entry_abs_interval_st
        + link.exit_abs_interval_st
    )

    if (
        _finite(
            link.outer_abs_interval_st
        )
    ):
        outer_abs = float(
            link.outer_abs_interval_st
        )

        excess_over_outer_st = float(
            local_target_path_st
            - outer_abs
        )

        if local_target_path_st > 0.0:
            outer_to_path_ratio = float(
                outer_abs
                / local_target_path_st
            )
        else:
            outer_to_path_ratio = float(
                "nan"
            )

    else:
        excess_over_outer_st = float(
            "nan"
        )

        outer_to_path_ratio = float(
            "nan"
        )

    if (
        _finite(
            link.previous_target_st
        )
        and _finite(
            link.stable_target_st
        )
        and _finite(
            link.next_target_st
        )
    ):
        stable_from_previous = abs(
            float(
                link.stable_target_st
                - link.previous_target_st
            )
        )

        next_from_previous = abs(
            float(
                link.next_target_st
                - link.previous_target_st
            )
        )

        return_gain_to_previous_st = float(
            stable_from_previous
            - next_from_previous
        )

        stable_from_next = abs(
            float(
                link.stable_target_st
                - link.next_target_st
            )
        )

        previous_from_next = abs(
            float(
                link.previous_target_st
                - link.next_target_st
            )
        )

        return_gain_to_next_st = float(
            stable_from_next
            - previous_from_next
        )

    else:
        return_gain_to_previous_st = float(
            "nan"
        )

        return_gain_to_next_st = float(
            "nan"
        )

    return (
        local_target_path_st,
        excess_over_outer_st,
        outer_to_path_ratio,
        return_gain_to_previous_st,
        return_gain_to_next_st,
    )

def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if not (
        _finite(numerator)
        and _finite(denominator)
    ):
        return float("nan")

    denominator = float(
        denominator
    )

    if denominator <= 0.0:
        return float("nan")

    return float(
        float(numerator)
        / denominator
    )


def _symmetric_ratio(
    a: float,
    b: float,
) -> float:
    """
    Symmetric magnitude ratio in [0, 1].

    1.0:
        equal magnitudes.

    Near 0:
        strongly different magnitudes.

    This is observational only.
    """

    if not (
        _finite(a)
        and _finite(b)
    ):
        return float("nan")

    a = abs(
        float(a)
    )

    b = abs(
        float(b)
    )

    largest = max(
        a,
        b,
    )

    if largest == 0.0:
        return 1.0

    return float(
        min(
            a,
            b,
        )
        / largest
    )

def _global_feature_map(
    objects: GestureObjectResult,
    features: GestureFeatureResult,
) -> dict[int, GestureFeatures]:
    """
    Map frozen PitchRegion transition indices to their GestureFeatures.

    IMPORTANT:
    GestureFeatures.region_index is not the PitchRegion index.
    GestureObject already preserves the explicit correspondence:

        transition_region_indices
            <->
        transition_feature_indices

    Therefore that frozen mapping is authoritative here.
    """

    mapping: dict[
        int,
        GestureFeatures,
    ] = {}

    gestures = features.gestures

    for obj in objects.objects:

        transition_regions = (
            obj.transition_region_indices
        )

        feature_indices = (
            obj.transition_feature_indices
        )

        if (
            len(transition_regions)
            != len(feature_indices)
        ):
            raise RuntimeError(
                "Gesture-object transition/feature "
                "mapping invariant failed: "
                f"object {obj.object_index} has "
                f"{len(transition_regions)} transition "
                "regions but "
                f"{len(feature_indices)} feature indices."
            )

        for (
            region_index,
            feature_index,
        ) in zip(
            transition_regions,
            feature_indices,
        ):

            region_index = int(
                region_index
            )

            feature_index = int(
                feature_index
            )

            # GestureObject.transition_feature_indices stores the
            # 1-based GestureFeatures.region_index ordinal.
            #
            # Convert that ordinal to the 0-based tuple position.
            feature_position = (
                feature_index - 1
            )

            if (
                feature_position < 0
                or feature_position >= len(gestures)
            ):
                raise RuntimeError(
                    "Gesture-object feature ordinal "
                    "out of range: "
                    f"object {obj.object_index}, "
                    f"transition region {region_index}, "
                    f"feature ordinal {feature_index}, "
                    f"feature count {len(gestures)}."
                )

            feature = gestures[
                feature_position
            ]

            # Protect the 1-based-ordinal assumption explicitly.
            if (
                int(feature.region_index)
                != feature_index
            ):
                raise RuntimeError(
                    "Gesture-object feature ordinal mismatch: "
                    f"object {obj.object_index}, "
                    f"transition region {region_index}, "
                    f"stored ordinal {feature_index}, "
                    f"resolved GestureFeatures.region_index "
                    f"{feature.region_index}."
                )

            if region_index in mapping:

                existing = mapping[
                    region_index
                ]

                if existing is not feature:
                    raise RuntimeError(
                        "Conflicting GestureFeatures "
                        "mapping for transition region "
                        f"{region_index}."
                    )

                continue

            mapping[
                region_index
            ] = feature

    return mapping

def _analyse_link(
    obj: GestureObject,
    link: GestureObjectLink,
    feature_map: dict[int, GestureFeatures],
    config: GestureObjectSplitterConfig,
) -> LinkSplitEvidence:

    (
        relation,
        entry_near_flat,
        exit_near_flat,
    ) = _structural_relation(
        link=link,
        config=config,
    )

    (
        local_target_path_st,
        excess_over_outer_st,
        outer_to_path_ratio,
        return_gain_to_previous_st,
        return_gain_to_next_st,
    ) = _local_geometry(
        link
    )

    # ------------------------------------------------------------------
    # V2: retrieve the frozen GestureFeatures records belonging to the
    # transitions immediately before and after this stable target.
    # ------------------------------------------------------------------

    left_region_index = int(
        link.left_transition_region_index
    )

    right_region_index = int(
        link.right_transition_region_index
    )

    if left_region_index not in feature_map:
        raise RuntimeError(
            "Missing frozen GestureFeatures for "
            f"left transition region "
            f"{left_region_index} in "
            f"object {obj.object_index}."
        )

    if right_region_index not in feature_map:
        raise RuntimeError(
            "Missing frozen GestureFeatures for "
            f"right transition region "
            f"{right_region_index} in "
            f"object {obj.object_index}."
        )

    left_feature = feature_map[
        left_region_index
    ]

    right_feature = feature_map[
        right_region_index
    ]

    # ------------------------------------------------------------------
    # V2: direct left/right transition measurements.
    # ------------------------------------------------------------------

    left_transition_duration_ms = float(
        left_feature.duration_ms
    )

    right_transition_duration_ms = float(
        right_feature.duration_ms
    )

    transition_duration_ratio = _safe_ratio(
        right_transition_duration_ms,
        left_transition_duration_ms,
    )

    left_shape_path_st = float(
        left_feature.shape_path_st
    )

    right_shape_path_st = float(
        right_feature.shape_path_st
    )

    shape_path_ratio = _safe_ratio(
        right_shape_path_st,
        left_shape_path_st,
    )

    left_center_path_st = float(
        left_feature.center_path_st
    )

    right_center_path_st = float(
        right_feature.center_path_st
    )

    center_path_ratio = _safe_ratio(
        right_center_path_st,
        left_center_path_st,
    )

    left_center_to_shape_ratio = float(
        left_feature.center_to_shape_path_ratio
    )

    right_center_to_shape_ratio = float(
        right_feature.center_to_shape_path_ratio
    )

    if (
        _finite(
            left_center_to_shape_ratio
        )
        and _finite(
            right_center_to_shape_ratio
        )
    ):
        center_to_shape_difference = float(
            right_center_to_shape_ratio
            - left_center_to_shape_ratio
        )

    else:
        center_to_shape_difference = float(
            "nan"
        )

    left_residual_rms_st = float(
        left_feature.residual_rms_st
    )

    right_residual_rms_st = float(
        right_feature.residual_rms_st
    )

    residual_rms_ratio = _safe_ratio(
        right_residual_rms_st,
        left_residual_rms_st,
    )

    left_residual_peak_to_peak_st = float(
        left_feature.residual_peak_to_peak_st
    )

    right_residual_peak_to_peak_st = float(
        right_feature.residual_peak_to_peak_st
    )

    residual_peak_to_peak_ratio = _safe_ratio(
        right_residual_peak_to_peak_st,
        left_residual_peak_to_peak_st,
    )

    left_residual_full_cycles = float(
        left_feature.residual_full_cycles_est
    )

    right_residual_full_cycles = float(
        right_feature.residual_full_cycles_est
    )

    left_residual_periodicity = float(
        left_feature.residual_periodicity
    )

    right_residual_periodicity = float(
        right_feature.residual_periodicity
    )

    left_autocorr_support = float(
        left_feature.residual_autocorr_support
    )

    right_autocorr_support = float(
        right_feature.residual_autocorr_support
    )

    left_autocorr_cycles_supported = float(
        left_feature.residual_autocorr_cycles_supported
    )

    right_autocorr_cycles_supported = float(
        right_feature.residual_autocorr_cycles_supported
    )

    left_source_return_fraction = float(
        left_feature.source_return_fraction
    )

    right_source_return_fraction = float(
        right_feature.source_return_fraction
    )

    # ------------------------------------------------------------------
    # V2: symmetric pair-comparison witnesses.
    #
    # These are descriptive similarities, NOT split probabilities.
    # ------------------------------------------------------------------

    duration_similarity = _symmetric_ratio(
        left_transition_duration_ms,
        right_transition_duration_ms,
    )

    shape_path_similarity = _symmetric_ratio(
        left_shape_path_st,
        right_shape_path_st,
    )

    center_path_similarity = _symmetric_ratio(
        left_center_path_st,
        right_center_path_st,
    )

    residual_rms_similarity = _symmetric_ratio(
        left_residual_rms_st,
        right_residual_rms_st,
    )

    return LinkSplitEvidence(
        object_index=int(
            obj.object_index
        ),

        link_index=int(
            link.link_index
        ),

        left_transition_region_index=int(
            link.left_transition_region_index
        ),

        stable_region_index=int(
            link.stable_region_index
        ),

        right_transition_region_index=int(
            link.right_transition_region_index
        ),

        stable_target_st=float(
            link.stable_target_st
        ),

        stable_duration_ms=float(
            link.stable_duration_ms
        ),

        previous_target_st=float(
            link.previous_target_st
        ),

        next_target_st=float(
            link.next_target_st
        ),

        entry_interval_st=float(
            link.entry_interval_st
        ),

        exit_interval_st=float(
            link.exit_interval_st
        ),

        outer_interval_st=float(
            link.outer_interval_st
        ),

        entry_abs_interval_st=float(
            link.entry_abs_interval_st
        ),

        exit_abs_interval_st=float(
            link.exit_abs_interval_st
        ),

        outer_abs_interval_st=float(
            link.outer_abs_interval_st
        ),

        relation=relation,

        outer_targets_available=bool(
            link.outer_targets_available
        ),

        same_direction=bool(
            link.same_direction
        ),

        direction_reversal=bool(
            link.direction_reversal
        ),

        entry_near_flat=bool(
            entry_near_flat
        ),

        exit_near_flat=bool(
            exit_near_flat
        ),

        returns_toward_previous=bool(
            link.returns_toward_previous
        ),

        returns_toward_next=bool(
            link.returns_toward_next
        ),

        stable_to_previous_distance_st=float(
            link.stable_to_previous_distance_st
        ),

        stable_to_next_distance_st=float(
            link.stable_to_next_distance_st
        ),

        local_target_path_st=float(
            local_target_path_st
        ),

        excess_over_outer_st=float(
            excess_over_outer_st
        ),

        outer_to_path_ratio=float(
            outer_to_path_ratio
        ),

        return_gain_to_previous_st=float(
            return_gain_to_previous_st
        ),

        return_gain_to_next_st=float(
            return_gain_to_next_st
        ),

        # V2 ------------------------------------------------------------

        left_feature=left_feature,
        right_feature=right_feature,

        left_transition_duration_ms=(
            left_transition_duration_ms
        ),

        right_transition_duration_ms=(
            right_transition_duration_ms
        ),

        transition_duration_ratio=float(
            transition_duration_ratio
        ),

        left_shape_path_st=(
            left_shape_path_st
        ),

        right_shape_path_st=(
            right_shape_path_st
        ),

        shape_path_ratio=float(
            shape_path_ratio
        ),

        left_center_path_st=(
            left_center_path_st
        ),

        right_center_path_st=(
            right_center_path_st
        ),

        center_path_ratio=float(
            center_path_ratio
        ),

        left_center_to_shape_ratio=(
            left_center_to_shape_ratio
        ),

        right_center_to_shape_ratio=(
            right_center_to_shape_ratio
        ),

        center_to_shape_difference=float(
            center_to_shape_difference
        ),

        left_residual_rms_st=(
            left_residual_rms_st
        ),

        right_residual_rms_st=(
            right_residual_rms_st
        ),

        residual_rms_ratio=float(
            residual_rms_ratio
        ),

        left_residual_peak_to_peak_st=(
            left_residual_peak_to_peak_st
        ),

        right_residual_peak_to_peak_st=(
            right_residual_peak_to_peak_st
        ),

        residual_peak_to_peak_ratio=float(
            residual_peak_to_peak_ratio
        ),

        left_residual_full_cycles=(
            left_residual_full_cycles
        ),

        right_residual_full_cycles=(
            right_residual_full_cycles
        ),

        left_residual_periodicity=(
            left_residual_periodicity
        ),

        right_residual_periodicity=(
            right_residual_periodicity
        ),

        left_autocorr_support=(
            left_autocorr_support
        ),

        right_autocorr_support=(
            right_autocorr_support
        ),

        left_autocorr_cycles_supported=(
            left_autocorr_cycles_supported
        ),

        right_autocorr_cycles_supported=(
            right_autocorr_cycles_supported
        ),

        left_source_return_fraction=(
            left_source_return_fraction
        ),

        right_source_return_fraction=(
            right_source_return_fraction
        ),

        duration_similarity=float(
            duration_similarity
        ),

        shape_path_similarity=float(
            shape_path_similarity
        ),

        center_path_similarity=float(
            center_path_similarity
        ),

        residual_rms_similarity=float(
            residual_rms_similarity
        ),
    )

def _analyse_object(
    obj: GestureObject,
    feature_map: dict[int, GestureFeatures],
    config: GestureObjectSplitterConfig,
) -> ObjectSplitEvidence:

    links = tuple(
        _analyse_link(
            obj=obj,
            link=link,
            feature_map=feature_map,
            config=config,
        )
        for link
        in obj.links
    )

    expected_links = max(
        0,
        int(
            obj.n_transitions
        )
        - 1,
    )

    if len(
        links
    ) != expected_links:
        raise RuntimeError(
            "Gesture-object splitter invariant failed: "
            f"object {obj.object_index} contains "
            f"{obj.n_transitions} transitions but "
            f"{len(links)} links; expected "
            f"{expected_links}."
        )

    same_direction_links = sum(
        1
        for link
        in links
        if (
            link.relation
            is StructuralRelation.SAME_DIRECTION
        )
    )

    reversal_links = sum(
        1
        for link
        in links
        if (
            link.relation
            is StructuralRelation.REVERSAL
        )
    )

    incomplete_links = sum(
        1
        for link
        in links
        if (
            link.relation
            is StructuralRelation.INCOMPLETE
        )
    )

    near_flat_links = sum(
        1
        for link
        in links
        if (
            link.relation
            in (
                StructuralRelation.NEAR_FLAT_ENTRY,
                StructuralRelation.NEAR_FLAT_EXIT,
                StructuralRelation.NEAR_FLAT_BOTH,
            )
        )
    )

    complete_outer_links = sum(
        1
        for link
        in links
        if link.outer_targets_available
    )

    return_to_previous_links = sum(
        1
        for link
        in links
        if link.returns_toward_previous
    )

    return_to_next_links = sum(
        1
        for link
        in links
        if link.returns_toward_next
    )

    return_both_links = sum(
        1
        for link
        in links
        if (
            link.returns_toward_previous
            and link.returns_toward_next
        )
    )

    local_target_path_total_st = (
        _sum_finite(
            link.local_target_path_st
            for link
            in links
        )
    )

    excess_over_outer_total_st = (
        _sum_finite(
            link.excess_over_outer_st
            for link
            in links
        )
    )

    max_excess_over_outer_st = (
        _max_finite_or_nan(
            link.excess_over_outer_st
            for link
            in links
        )
    )

    min_outer_to_path_ratio = (
        _min_finite_or_nan(
            link.outer_to_path_ratio
            for link
            in links
        )
    )

    return ObjectSplitEvidence(
        object_index=int(
            obj.object_index
        ),

        region_indices=tuple(
            int(index)
            for index
            in obj.region_indices
        ),

        transition_region_indices=tuple(
            int(index)
            for index
            in obj.transition_region_indices
        ),

        stable_region_indices=tuple(
            int(index)
            for index
            in obj.stable_region_indices
        ),

        n_transitions=int(
            obj.n_transitions
        ),

        n_links=len(
            links
        ),

        links=links,

        same_direction_links=int(
            same_direction_links
        ),

        reversal_links=int(
            reversal_links
        ),

        incomplete_links=int(
            incomplete_links
        ),

        near_flat_links=int(
            near_flat_links
        ),

        complete_outer_links=int(
            complete_outer_links
        ),

        return_to_previous_links=int(
            return_to_previous_links
        ),

        return_to_next_links=int(
            return_to_next_links
        ),

        return_both_links=int(
            return_both_links
        ),

        local_target_path_total_st=float(
            local_target_path_total_st
        ),

        excess_over_outer_total_st=float(
            excess_over_outer_total_st
        ),

        max_excess_over_outer_st=float(
            max_excess_over_outer_st
        ),

        min_outer_to_path_ratio=float(
            min_outer_to_path_ratio
        ),
    )


def analyse_gesture_object_splits(
    objects: GestureObjectResult,
    features: GestureFeatureResult,
    config: GestureObjectSplitterConfig | None = None,
) -> GestureObjectSplitAnalysis:
    """
    Build observational structural-boundary evidence for every maximal
    GestureObject.

    V1 deliberately returns evidence only.  It does NOT split objects.
    """

    if config is None:
        config = (
            GestureObjectSplitterConfig()
        )

    feature_map = _global_feature_map(
        objects=objects,
        features=features,
    )

    expected_link_feature_regions = {
        int(region_index)
        for obj in objects.objects
        for link in obj.links
        for region_index in (
            link.left_transition_region_index,
            link.right_transition_region_index,
        )
    }

    missing_link_feature_regions = (
        expected_link_feature_regions
        - set(feature_map)
    )


    if missing_link_feature_regions:
        raise RuntimeError(
            "Gesture-object splitter V2 link-feature "
            "coverage invariant failed. Missing frozen "
            "GestureFeatures for link transition regions: "
            f"{sorted(missing_link_feature_regions)}"
        )

    analysed = tuple(
        _analyse_object(
            obj=obj,
            feature_map=feature_map,
            config=config,
        )
        for obj
        in objects.objects
    )

    if len(
        analysed
    ) != len(
        objects.objects
    ):
        raise RuntimeError(
            "Gesture-object splitter invariant failed: "
            "object count changed during analysis."
        )

    total_links = sum(
        obj.n_links
        for obj in analysed
    )

    expected_total_links = sum(
        max(
            0,
            int(
                obj.n_transitions
            )
            - 1,
        )
        for obj
        in objects.objects
    )

    if (
        total_links
        != expected_total_links
    ):
        raise RuntimeError(
            "Gesture-object splitter invariant failed: "
            f"observed {total_links} links but expected "
            f"{expected_total_links}."
        )

    objects_with_links = sum(
        1
        for obj in analysed
        if obj.n_links > 0
    )

    objects_without_links = (
        len(
            analysed
        )
        - objects_with_links
    )

    same_direction_links = sum(
        obj.same_direction_links
        for obj in analysed
    )

    reversal_links = sum(
        obj.reversal_links
        for obj in analysed
    )

    incomplete_links = sum(
        obj.incomplete_links
        for obj in analysed
    )

    near_flat_links = sum(
        obj.near_flat_links
        for obj in analysed
    )

    return GestureObjectSplitAnalysis(
        objects=analysed,

        total_objects=len(
            analysed
        ),

        total_links=int(
            total_links
        ),

        objects_with_links=int(
            objects_with_links
        ),

        objects_without_links=int(
            objects_without_links
        ),

        same_direction_links=int(
            same_direction_links
        ),

        reversal_links=int(
            reversal_links
        ),

        incomplete_links=int(
            incomplete_links
        ),

        near_flat_links=int(
            near_flat_links
        ),
    )