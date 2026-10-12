from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from python_eckf.trajectory_interpreter import (
    RegionKind,
    interpret_pitch_trajectory,
)

from python_eckf.expressive_smoothing import (
    ExpressiveSmoothingConfig,
    smooth_expressive_pitch,
)

from python_eckf.gesture_features import (
    GestureFeatureConfig,
    extract_gesture_features,
)

from python_eckf.gesture_objects import (
    GestureObjectConfig,
    GestureObjectKind,
    construct_gesture_objects,
)


def load_offline_csv(
    path: Path,
):

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        required = {
            "time_s",
            "clean_f0_hz",
            "corrected_valid",
        }

        missing = (
            required
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:

            raise RuntimeError(
                "CSV is missing columns: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        rows.extend(
            reader
        )

    time_s = np.asarray(
        [
            float(
                row[
                    "time_s"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    clean_f0 = np.asarray(
        [
            (
                float(
                    row[
                        "clean_f0_hz"
                    ]
                )
                if row[
                    "clean_f0_hz"
                ].strip()
                else np.nan
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    valid = np.asarray(
        [
            bool(
                int(
                    row[
                        "corrected_valid"
                    ]
                )
            )
            for row in rows
        ],
        dtype=bool,
    )

    if len(
        time_s
    ) >= 2:

        dt = float(
            np.median(
                np.diff(
                    time_s
                )
            )
        )

        analysis_hz = float(
            1.0 / dt
        )

    else:

        analysis_hz = 100.0

    return (
        time_s,
        clean_f0,
        valid,
        analysis_hz,
    )


def fmt(
    value: float,
    digits: int = 2,
) -> str:

    if not np.isfinite(
        value
    ):
        return "--"

    return (
        f"{value:.{digits}f}"
    )

def fmt_targets(
    values,
) -> str:

    if not values:
        return "[]"

    return "[" + ",".join(
        f"{value:.2f}"
        for value in values
    ) + "]"

def short_kind(
    kind: GestureObjectKind,
) -> str:

    if (
        kind
        is GestureObjectKind.SINGLE_TRANSITION
    ):
        return "T"

    if (
        kind
        is GestureObjectKind.TRANSITION_TARGET_TRANSITION
    ):
        return "T-S-T"

    if (
        kind
        is GestureObjectKind.TRANSITION_CHAIN
    ):
        return "CHAIN"

    return kind.name


def region_pattern(
    obj,
    interpretation,
) -> str:

    symbols = []

    for index in obj.region_indices:

        kind = (
            interpretation.regions[
                index
            ].kind
        )

        if (
            kind
            is RegionKind.TRANSITION
        ):
            symbols.append(
                "T"
            )

        elif (
            kind
            is RegionKind.STABLE_TARGET
        ):
            symbols.append(
                "S"
            )

        elif (
            kind
            is RegionKind.UNRESOLVED
        ):
            symbols.append(
                "U"
            )

        else:
            symbols.append(
                "?"
            )

    return "-".join(
        symbols
    )

def direction_symbol(
    value: int,
) -> str:

    if value > 0:
        return "UP"

    if value < 0:
        return "DN"

    return "--"


def print_links(
    obj,
):

    if not obj.links:
        return

    for link in obj.links:

        print(
            "      "
            f"L{link.link_index + 1}: "
            f"T{link.left_transition_region_index}"
            f"-S{link.stable_region_index}"
            f"-T{link.right_transition_region_index} "
            f"prev={fmt(link.previous_target_st):>6} "
            f"stable={fmt(link.stable_target_st):>6} "
            f"next={fmt(link.next_target_st):>6} "
            f"dur={fmt(link.stable_duration_ms, 0):>4}ms "
            f"in={fmt(link.entry_interval_st):>6} "
            f"out={fmt(link.exit_interval_st):>6} "
            f"outer={fmt(link.outer_interval_st):>6} "
            f"dir={direction_symbol(link.direction_in)}"
            f"->{direction_symbol(link.direction_out)} "
            f"same={int(link.same_direction)} "
            f"reverse={int(link.direction_reversal)} "
            f"retPrev={int(link.returns_toward_previous)} "
            f"retNext={int(link.returns_toward_next)}"
        )

def print_object(
    obj,
    interpretation,
):

    print(
        f"{obj.object_index:3d} "
        f"{obj.start_time_s:7.3f}-"
        f"{obj.end_time_s:7.3f} "
        f"{obj.duration_ms:5.0f}ms "
        f"{short_kind(obj.kind):>5} "
        f"pattern={region_pattern(obj, interpretation):<15} "
        f"regions={str(obj.region_indices):<22} "
        f"T={obj.n_transitions:2d} "
        f"S={obj.n_stable_targets:2d} "
        f"src={fmt(obj.source_target_st):>6} "
        f"dst={fmt(obj.destination_target_st):>6} "
        f"d={fmt(obj.boundary_interval_st):>6} "

        f"inner={obj.interior_target_count:2d} "
        f"targets={fmt_targets(obj.interior_targets_st):<28} "
        f"innerMaxMs={fmt(obj.interior_target_max_duration_ms, 0):>4} "
        f"innerMin={fmt(obj.interior_target_min_st):>6} "
        f"innerMax={fmt(obj.interior_target_max_st):>6} "
        f"innerSpan={fmt(obj.interior_target_span_st):>5} "
              
        f"shapePath={fmt(obj.shape_path_total_st):>6} "
        f"centerPath={fmt(obj.center_path_total_st):>6} "
        f"resRMS={fmt(obj.residual_rms_weighted_st):>5} "
        f"resMax={fmt(obj.residual_abs_max_st):>5} "
        f"zc={obj.residual_zero_crossings_total:2d} "
        f"pk/tr="
        f"{obj.residual_peak_count_total:2d}/"
        f"{obj.residual_trough_count_total:2d} "
        f"acSup={fmt(obj.autocorr_support_max):>4} "
        f"acCyc={fmt(obj.autocorr_cycles_max, 1):>4} "
        f"returnD={fmt(obj.returns_near_source_st):>5}"
    )


def main():

    if len(
        sys.argv
    ) != 2:

        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_gesture_objects.py "
            "<offline.eckf.csv>"
        )

    path = Path(
        sys.argv[
            1
        ]
    )

    (
        time_s,
        clean_f0,
        valid,
        analysis_hz,
    ) = load_offline_csv(
        path
    )

    interpretation = (
        interpret_pitch_trajectory(
            clean_f0_hz=clean_f0,
            valid=valid,
            analysis_hz=analysis_hz,
        )
    )

    smoothing = (
        smooth_expressive_pitch(
            pitch_st=(
                interpretation.pitch_st
            ),
            valid=valid,
            analysis_hz=analysis_hz,
            config=(
                ExpressiveSmoothingConfig(
                    analysis_hz=analysis_hz,
                    window_ms=50.0,
                    polyorder=2,
                    center_window_ms=250.0,
                )
            ),
        )
    )

    features = (
        extract_gesture_features(
            interpretation=interpretation,
            smoothing=smoothing,
            config=(
                GestureFeatureConfig()
            ),
        )
    )

    object_config = (
        GestureObjectConfig(
            short_target_max_ms=250.0,
            max_chain_transitions=8,
        )
    )

    result = (
        construct_gesture_objects(
            interpretation=interpretation,
            features=features,
            config=object_config,
        )
    )

    objects = list(
        result.objects
    )

    frozen_transition_count = sum(
        1
        for region
        in interpretation.regions
        if (
            region.kind
            is RegionKind.TRANSITION
        )
    )

    represented_transition_count = sum(
        obj.n_transitions
        for obj in objects
    )

    represented_unique = set()

    for obj in objects:

        represented_unique.update(
            obj.transition_region_indices
        )

    print(
        "=" * 220
    )

    print(
        "GESTURE OBJECTS V1 — "
        "OBSERVATIONAL GROUPING ONLY"
    )

    print(
        "=" * 220
    )

    print(
        f"Rows:                         "
        f"{len(clean_f0)}"
    )

    print(
        f"Analysis rate:                "
        f"{analysis_hz:.3f} Hz"
    )

    print(
        f"Valid rows:                   "
        f"{int(np.sum(valid))}"
    )

    print(
        f"Frozen pitch regions:         "
        f"{len(interpretation.regions)}"
    )

    print(
        f"Frozen transitions:           "
        f"{frozen_transition_count}"
    )

    print(
        f"Gesture feature objects:      "
        f"{len(features.gestures)}"
    )

    print(
        f"Candidate grouped objects:    "
        f"{len(objects)}"
    )

    print(
        f"Transitions represented:      "
        f"{represented_transition_count}"
    )

    print(
        f"Unique transitions represented:"
        f" {len(represented_unique)}"
    )

    print(
        f"Short-target threshold:       "
        f"{object_config.short_target_max_ms:.0f} ms"
    )

    print()

    print(
        "Expected frozen invariants:"
    )

    print(
        "  pitch regions remain 229"
    )

    print(
        "  transitions remain 97"
    )

    print(
        "  gesture features remain 97"
    )

    print(
        "  represented transitions remain 97 exactly once"
    )

    print()

    print(
        "No musical labels are assigned."
    )

    # ------------------------------------------------------------
    # Object population
    # ------------------------------------------------------------

    counts = Counter(
        short_kind(
            obj.kind
        )
        for obj in objects
    )

    print()
    print(
        "=" * 220
    )

    print(
        "OBJECT POPULATION"
    )

    print(
        "=" * 220
    )

    for name in (
        "T",
        "T-S-T",
        "CHAIN",
    ):

        print(
            f"{name:10s}: "
            f"{counts.get(name, 0)}"
        )

    multi = [
        obj
        for obj in objects
        if (
            obj.n_transitions
            > 1
        )
    ]

    print(
        f"\nMulti-transition objects: "
        f"{len(multi)}"
    )

    print(
        "Transition compression: "
        f"{frozen_transition_count} transitions "
        f"-> {len(objects)} candidate objects"
    )

    # ------------------------------------------------------------
    # All objects
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "ALL CANDIDATE OBJECTS"
    )

    print(
        "=" * 220
    )

    for obj in objects:
        print_object(
            obj,
            interpretation,
        )

    # ------------------------------------------------------------
    # Multi-transition candidates
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "MULTI-TRANSITION CANDIDATES"
    )

    print(
        "=" * 220
    )

    if not multi:

        print(
            "NONE"
        )

    else:

        for obj in multi:
            print_object(
                obj,
                interpretation,
            )

            print_links(
                obj,
            )

    # ------------------------------------------------------------
    # Longest chains
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "LARGEST CANDIDATE CHAINS"
    )

    print(
        "=" * 220
    )

    ranked = sorted(
        objects,
        key=lambda obj: (
            obj.n_transitions,
            obj.n_regions,
            obj.duration_ms,
        ),
        reverse=True,
    )

    for obj in ranked[
        :30
    ]:
        print_object(
            obj,
            interpretation,
        )

    # ------------------------------------------------------------
    # Closest complete return to source
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "MULTI-TRANSITION OBJECTS WITH "
        "CLOSEST SOURCE/DESTINATION TARGETS"
    )

    print(
        "=" * 220
    )

    return_candidates = [
        obj
        for obj in multi
        if np.isfinite(
            obj.returns_near_source_st
        )
    ]

    return_candidates = sorted(
        return_candidates,
        key=lambda obj: (
            obj.returns_near_source_st,
            -obj.shape_path_total_st,
        ),
    )

    for obj in return_candidates[
        :30
    ]:
        print_object(
            obj,
            interpretation,
        )

    # ------------------------------------------------------------
    # Largest interior-target excursions
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "MULTI-TRANSITION OBJECTS WITH "
        "LARGEST INTERIOR TARGET SPAN"
    )

    print(
        "=" * 220
    )

    interior_ranked = [
        obj
        for obj in multi
        if np.isfinite(
            obj.interior_target_span_st
        )
    ]

    interior_ranked = sorted(
        interior_ranked,
        key=lambda obj: (
            obj.interior_target_span_st,
            obj.shape_path_total_st,
        ),
        reverse=True,
    )

    for obj in interior_ranked[
        :30
    ]:
        print_object(
            obj,
            interpretation,
        )

    # ------------------------------------------------------------
    # Regression neighborhoods
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "REGRESSION NEIGHBORHOODS"
    )

    print(
        "=" * 220
    )

    regression_times = [
        14.70,
        23.78,
        27.15,
        27.45,
        27.67,
        47.66,
        71.07,
        71.80,
        72.39,
        75.40,
    ]

    for wanted in regression_times:

        matches = [
            obj
            for obj in objects
            if (
                obj.start_time_s
                <= wanted
                < obj.end_time_s
            )
        ]

        if not matches:

            print(
                f"{wanted:7.3f}s: "
                "NO CANDIDATE OBJECT"
            )

            continue

        for number, obj in enumerate(
            matches
        ):

            if number == 0:

                print(
                    f"{wanted:7.3f}s -> ",
                    end="",
                )

            else:

                print(
                    "           -> ",
                    end="",
                )

            print_object(
                obj,
                interpretation,
            )


if __name__ == "__main__":
    main()