from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .gesture_object_splitter import (
    GestureObjectSplitAnalysis,
    LinkSplitEvidence,
    StructuralRelation,
)


class StructuralDecision(Enum):
    """
    Structural decision only.

    These are NOT musical gesture labels.
    """

    JOIN = auto()
    SPLIT = auto()
    UNRESOLVED = auto()


@dataclass(frozen=True)
class StructuralSplitterConfig:
    """
    Conservative V3 structural decision rules.

    IMPORTANT:
        These thresholds define strong observational evidence.
        They are NOT musical gesture thresholds.

    No weighted split score is constructed.

    A link is split only when multiple independent boundary
    evidence families agree and no strong connector protection
    is present.
    """

    # ------------------------------------------------------------
    # Strong leave-and-return connector geometry
    # ------------------------------------------------------------

    connector_outer_to_path_max: float = 0.35
    connector_min_excess_st: float = 0.50

    # ------------------------------------------------------------
    # Strong direct-through connector geometry
    # ------------------------------------------------------------

    direct_outer_to_path_min: float = 0.90
    direct_max_excess_st: float = 0.15

    # ------------------------------------------------------------
    # Transition-regime discontinuity
    # ------------------------------------------------------------

    duration_similarity_low: float = 0.35
    shape_path_similarity_low: float = 0.30
    center_path_similarity_low: float = 0.30
    residual_rms_similarity_low: float = 0.30

    # Center/shape behavior may change even when absolute paths
    # happen to be similar.
    center_to_shape_difference_high: float = 0.40

    # ------------------------------------------------------------
    # Oscillation/topology discontinuity
    # ------------------------------------------------------------

    topology_cycle_difference_high: float = 2.0
    topology_periodicity_difference_high: float = 0.40
    topology_autocorr_support_difference_high: float = 0.40

    # ------------------------------------------------------------
    # Decision requirements
    # ------------------------------------------------------------

    min_boundary_families_for_split: int = 2


@dataclass(frozen=True)
class StructuralEvidenceFamilies:
    """
    Independent evidence families for one internal T-S-T link.

    These are booleans/witnesses, not probabilities.
    """

    # Evidence that the two sides belong together.
    excursion_connector: bool
    direct_through_connector: bool
    transition_similarity_connector: bool

    # Evidence that the two sides belong to different regimes.
    duration_discontinuity: bool
    path_discontinuity: bool
    center_shape_discontinuity: bool
    residual_discontinuity: bool
    topology_discontinuity: bool

    connector_family_count: int
    boundary_family_count: int


@dataclass(frozen=True)
class StructuralLinkDecision:
    object_index: int
    link_index: int

    left_transition_region_index: int
    stable_region_index: int
    right_transition_region_index: int

    decision: StructuralDecision

    evidence: StructuralEvidenceFamilies

    # Preserve the complete V2 record.
    v2: LinkSplitEvidence


@dataclass(frozen=True)
class StructuralSegment:
    """
    One structural segment produced from a maximal gesture object.

    A segment is cut only at links receiving an explicit SPLIT
    decision.

    UNRESOLVED links remain inside a segment but are preserved in
    unresolved_link_indices so later interpretation knows that the
    internal relationship was not established.
    """

    object_index: int
    segment_index: int

    transition_region_indices: tuple[int, ...]
    stable_region_indices: tuple[int, ...]

    internal_link_indices: tuple[int, ...]
    unresolved_link_indices: tuple[int, ...]

    n_transitions: int
    n_internal_links: int


@dataclass(frozen=True)
class StructuralObjectResult:
    object_index: int

    link_decisions: tuple[StructuralLinkDecision, ...]
    segments: tuple[StructuralSegment, ...]

    n_splits: int
    n_joins: int
    n_unresolved: int


@dataclass(frozen=True)
class StructuralSplitResult:
    objects: tuple[StructuralObjectResult, ...]

    total_objects: int
    total_links: int

    split_links: int
    join_links: int
    unresolved_links: int

    total_segments: int


def _finite(
    value: float,
) -> bool:

    return bool(
        np.isfinite(
            value
        )
    )


def _abs_difference(
    a: float,
    b: float,
) -> float:

    if not (
        _finite(a)
        and _finite(b)
    ):
        return float("nan")

    return float(
        abs(
            float(a)
            - float(b)
        )
    )


def _transition_similarity_connector(
    link: LinkSplitEvidence,
) -> bool:
    """
    Conservative witness that the transitions on the two sides
    have broadly similar shape.

    This deliberately does NOT require every feature to match.
    """

    similarities = (
        link.duration_similarity,
        link.shape_path_similarity,
        link.center_path_similarity,
        link.residual_rms_similarity,
    )

    finite = [
        float(value)
        for value in similarities
        if _finite(value)
    ]

    if len(finite) < 3:
        return False

    # Strong agreement across most available measurements.
    strong = sum(
        value >= 0.65
        for value in finite
    )

    return bool(
        strong >= 3
    )


def _evidence_families(
    link: LinkSplitEvidence,
    config: StructuralSplitterConfig,
) -> StructuralEvidenceFamilies:

    # ============================================================
    # CONNECTOR EVIDENCE
    # ============================================================

    excursion_connector = False

    if (
        link.outer_targets_available
        and link.direction_reversal
        and _finite(
            link.outer_to_path_ratio
        )
        and _finite(
            link.excess_over_outer_st
        )
    ):

        excursion_connector = bool(
            link.outer_to_path_ratio
            <= config.connector_outer_to_path_max
            and link.excess_over_outer_st
            >= config.connector_min_excess_st
            and (
                link.returns_toward_previous
                or link.returns_toward_next
            )
        )

    direct_through_connector = False

    if (
        link.outer_targets_available
        and _finite(
            link.outer_to_path_ratio
        )
        and _finite(
            link.excess_over_outer_st
        )
    ):

        direct_through_connector = bool(
            link.outer_to_path_ratio
            >= config.direct_outer_to_path_min
            and link.excess_over_outer_st
            <= config.direct_max_excess_st
            and (
                link.relation
                in (
                    StructuralRelation.SAME_DIRECTION,
                    StructuralRelation.NEAR_FLAT_ENTRY,
                    StructuralRelation.NEAR_FLAT_EXIT,
                    StructuralRelation.NEAR_FLAT_BOTH,
                )
            )
        )

    transition_similarity_connector = (
        _transition_similarity_connector(
            link
        )
    )

    # ============================================================
    # BOUNDARY EVIDENCE
    # ============================================================

    duration_discontinuity = bool(
        _finite(
            link.duration_similarity
        )
        and link.duration_similarity
        <= config.duration_similarity_low
    )

    path_discontinuity = bool(
        (
            _finite(
                link.shape_path_similarity
            )
            and link.shape_path_similarity
            <= config.shape_path_similarity_low
        )
        or (
            _finite(
                link.center_path_similarity
            )
            and link.center_path_similarity
            <= config.center_path_similarity_low
        )
    )

    center_shape_discontinuity = bool(
        _finite(
            link.center_to_shape_difference
        )
        and abs(
            link.center_to_shape_difference
        )
        >= config.center_to_shape_difference_high
    )

    residual_discontinuity = bool(
        _finite(
            link.residual_rms_similarity
        )
        and link.residual_rms_similarity
        <= config.residual_rms_similarity_low
    )

    cycle_difference = _abs_difference(
        link.left_residual_full_cycles,
        link.right_residual_full_cycles,
    )

    periodicity_difference = _abs_difference(
        link.left_residual_periodicity,
        link.right_residual_periodicity,
    )

    autocorr_difference = _abs_difference(
        link.left_autocorr_support,
        link.right_autocorr_support,
    )

    topology_discontinuity = bool(
        (
            _finite(
                cycle_difference
            )
            and cycle_difference
            >= config.topology_cycle_difference_high
        )
        or (
            _finite(
                periodicity_difference
            )
            and periodicity_difference
            >= config.topology_periodicity_difference_high
        )
        or (
            _finite(
                autocorr_difference
            )
            and autocorr_difference
            >= config.topology_autocorr_support_difference_high
        )
    )

    connector_family_count = int(
        excursion_connector
    ) + int(
        direct_through_connector
    ) + int(
        transition_similarity_connector
    )

    boundary_family_count = int(
        duration_discontinuity
    ) + int(
        path_discontinuity
    ) + int(
        center_shape_discontinuity
    ) + int(
        residual_discontinuity
    ) + int(
        topology_discontinuity
    )

    return StructuralEvidenceFamilies(
        excursion_connector=(
            excursion_connector
        ),
        direct_through_connector=(
            direct_through_connector
        ),
        transition_similarity_connector=(
            transition_similarity_connector
        ),

        duration_discontinuity=(
            duration_discontinuity
        ),
        path_discontinuity=(
            path_discontinuity
        ),
        center_shape_discontinuity=(
            center_shape_discontinuity
        ),
        residual_discontinuity=(
            residual_discontinuity
        ),
        topology_discontinuity=(
            topology_discontinuity
        ),

        connector_family_count=(
            connector_family_count
        ),
        boundary_family_count=(
            boundary_family_count
        ),
    )


def _decide_link(
    link: LinkSplitEvidence,
    config: StructuralSplitterConfig,
) -> StructuralLinkDecision:

    evidence = _evidence_families(
        link=link,
        config=config,
    )

    # ============================================================
    # STRONG CONNECTOR PROTECTION
    # ============================================================
    #
    # EXCURSION is the strongest connector witness because it
    # describes the geometry of the complete leave-and-return
    # construction.
    #
    # DIRECT is weaker: a nearly direct sequence of target centers
    # can still sit at a genuine transition-regime boundary.
    # ============================================================

    if evidence.excursion_connector:

        decision = (
            StructuralDecision.JOIN
        )

    # Direct-through geometry protects the link only when there is
    # not overwhelming independent boundary evidence.
    elif (
        evidence.direct_through_connector
        and evidence.boundary_family_count < 3
    ):

        decision = (
            StructuralDecision.JOIN
        )

    # ============================================================
    # STRONG BOUNDARY OVERRIDING WEAK CONNECTOR EVIDENCE
    # ============================================================
    #
    # Three or more independent boundary families may override
    # DIRECT or SIMILAR evidence.
    #
    # EXCURSION cannot reach this branch because it has already
    # received precedence above.
    # ============================================================

    elif (
        evidence.boundary_family_count >= 3
        and not evidence.excursion_connector
    ):

        decision = (
            StructuralDecision.SPLIT
        )

    # ============================================================
    # ORDINARY STRONG BOUNDARY
    # ============================================================
    #
    # With no connector evidence at all, agreement between two
    # independent boundary families is sufficient.
    # ============================================================

    elif (
        evidence.boundary_family_count
        >= config.min_boundary_families_for_split
        and evidence.connector_family_count == 0
    ):

        decision = (
            StructuralDecision.SPLIT
        )

    # ============================================================
    # POSITIVE JOIN
    # ============================================================

    elif (
        evidence.transition_similarity_connector
        and evidence.boundary_family_count == 0
    ):

        decision = (
            StructuralDecision.JOIN
        )

    # ============================================================
    # CONFLICT / INSUFFICIENT EVIDENCE
    # ============================================================

    else:

        decision = (
            StructuralDecision.UNRESOLVED
        )

    return StructuralLinkDecision(
        object_index=int(
            link.object_index
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

        decision=decision,
        evidence=evidence,
        v2=link,
    )


def _make_segments(
    object_index: int,
    transition_region_indices: tuple[int, ...],
    stable_region_indices: tuple[int, ...],
    decisions: tuple[
        StructuralLinkDecision,
        ...
    ],
) -> tuple[StructuralSegment, ...]:
    """
    Cut the transition chain only at explicit SPLIT links.

    For N transitions there must be N-1 link decisions.
    """

    n_transitions = len(
        transition_region_indices
    )

    if n_transitions == 0:
        return tuple()

    expected_links = max(
        0,
        n_transitions - 1,
    )

    if len(decisions) != expected_links:
        raise RuntimeError(
            "Structural segmentation invariant failed: "
            f"object {object_index} has "
            f"{n_transitions} transitions but "
            f"{len(decisions)} link decisions; "
            f"expected {expected_links}."
        )

    # Each split position is a link after transition position i.
    split_after = {
        i
        for i, decision
        in enumerate(decisions)
        if (
            decision.decision
            is StructuralDecision.SPLIT
        )
    }

    segments = []

    transition_start = 0
    segment_index = 1

    for transition_end in range(
        n_transitions
    ):

        is_last = (
            transition_end
            == n_transitions - 1
        )

        should_cut = (
            transition_end
            in split_after
        )

        if not (
            should_cut
            or is_last
        ):
            continue

        transition_slice = (
            transition_region_indices[
                transition_start:
                transition_end + 1
            ]
        )

        # Stable targets internal to this transition slice.
        #
        # For transitions [a ... b], internal links are
        # [a ... b-1], and each such link corresponds to one
        # stable target.
        stable_slice = (
            stable_region_indices[
                transition_start:
                transition_end
            ]
        )

        internal_decisions = (
            decisions[
                transition_start:
                transition_end
            ]
        )

        internal_link_indices = tuple(
            int(
                decision.link_index
            )
            for decision
            in internal_decisions
        )

        unresolved_link_indices = tuple(
            int(
                decision.link_index
            )
            for decision
            in internal_decisions
            if (
                decision.decision
                is StructuralDecision.UNRESOLVED
            )
        )

        segments.append(
            StructuralSegment(
                object_index=int(
                    object_index
                ),

                segment_index=int(
                    segment_index
                ),

                transition_region_indices=tuple(
                    int(value)
                    for value
                    in transition_slice
                ),

                stable_region_indices=tuple(
                    int(value)
                    for value
                    in stable_slice
                ),

                internal_link_indices=(
                    internal_link_indices
                ),

                unresolved_link_indices=(
                    unresolved_link_indices
                ),

                n_transitions=len(
                    transition_slice
                ),

                n_internal_links=len(
                    internal_link_indices
                ),
            )
        )

        segment_index += 1
        transition_start = (
            transition_end + 1
        )

    # Every transition must appear exactly once.
    represented = tuple(
        transition
        for segment
        in segments
        for transition
        in segment.transition_region_indices
    )

    if represented != tuple(
        transition_region_indices
    ):
        raise RuntimeError(
            "Structural segmentation coverage "
            "invariant failed for object "
            f"{object_index}."
        )

    return tuple(
        segments
    )


def apply_structural_splitting(
    split_evidence: GestureObjectSplitAnalysis,
    config: StructuralSplitterConfig = (
        StructuralSplitterConfig()
    ),
) -> StructuralSplitResult:
    """
    Convert V2 observational split evidence into conservative
    structural segments.

    IMPORTANT:
        - no musical gesture labels;
        - no pitch modification;
        - no transition modification;
        - no target modification;
        - V2 evidence remains immutable;
        - only explicit SPLIT decisions create boundaries.
    """

    object_results = []

    total_links = 0
    split_links = 0
    join_links = 0
    unresolved_links = 0
    total_segments = 0

    for obj in split_evidence.objects:

        decisions = tuple(
            _decide_link(
                link=link,
                config=config,
            )
            for link
            in obj.links
        )

        expected_links = max(
            0,
            int(
                obj.n_transitions
            ) - 1,
        )

        if len(decisions) != expected_links:
            raise RuntimeError(
                "Structural decision invariant failed: "
                f"object {obj.object_index} has "
                f"{obj.n_transitions} transitions but "
                f"{len(decisions)} decisions."
            )

        n_splits = sum(
            decision.decision
            is StructuralDecision.SPLIT
            for decision
            in decisions
        )

        n_joins = sum(
            decision.decision
            is StructuralDecision.JOIN
            for decision
            in decisions
        )

        n_unresolved = sum(
            decision.decision
            is StructuralDecision.UNRESOLVED
            for decision
            in decisions
        )

        if (
            n_splits
            + n_joins
            + n_unresolved
            != len(decisions)
        ):
            raise RuntimeError(
                "Structural decision population "
                "invariant failed."
            )

        segments = _make_segments(
            object_index=int(
                obj.object_index
            ),

            transition_region_indices=tuple(
                int(value)
                for value
                in obj.transition_region_indices
            ),

            stable_region_indices=tuple(
                int(value)
                for value
                in obj.stable_region_indices
            ),

            decisions=decisions,
        )

        # N explicit cuts must create N+1 segments for any object
        # containing at least one transition.
        if obj.n_transitions > 0:

            expected_segments = (
                n_splits + 1
            )

            if len(segments) != expected_segments:
                raise RuntimeError(
                    "Structural segment-count "
                    "invariant failed: "
                    f"object {obj.object_index}, "
                    f"{n_splits} splits produced "
                    f"{len(segments)} segments; "
                    f"expected {expected_segments}."
                )

        object_results.append(
            StructuralObjectResult(
                object_index=int(
                    obj.object_index
                ),

                link_decisions=decisions,
                segments=segments,

                n_splits=int(
                    n_splits
                ),

                n_joins=int(
                    n_joins
                ),

                n_unresolved=int(
                    n_unresolved
                ),
            )
        )

        total_links += len(
            decisions
        )

        split_links += int(
            n_splits
        )

        join_links += int(
            n_joins
        )

        unresolved_links += int(
            n_unresolved
        )

        total_segments += len(
            segments
        )

    if total_links != split_evidence.total_links:
        raise RuntimeError(
            "Structural splitter global link "
            "coverage invariant failed: "
            f"V2 has {split_evidence.total_links} links, "
            f"V3 represented {total_links}."
        )

    if (
        split_links
        + join_links
        + unresolved_links
        != total_links
    ):
        raise RuntimeError(
            "Structural splitter global decision "
            "population invariant failed."
        )

    return StructuralSplitResult(
        objects=tuple(
            object_results
        ),

        total_objects=len(
            object_results
        ),

        total_links=int(
            total_links
        ),

        split_links=int(
            split_links
        ),

        join_links=int(
            join_links
        ),

        unresolved_links=int(
            unresolved_links
        ),

        total_segments=int(
            total_segments
        ),
    )