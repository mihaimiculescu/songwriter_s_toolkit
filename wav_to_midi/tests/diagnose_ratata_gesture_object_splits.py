import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    REPO_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            REPO_ROOT
        ),
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
    construct_gesture_objects,
)
from python_eckf.gesture_object_splitter import (
    GestureObjectSplitterConfig,
    StructuralRelation,
    analyse_gesture_object_splits,
)
from python_eckf.trajectory_interpreter import (
    interpret_pitch_trajectory,
)


def load_offline_csv(
    path: Path,
):

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )

    time_s = np.asarray(
        data["time_s"],
        dtype=float,
    )

    clean_f0 = np.asarray(
        data["clean_f0_hz"],
        dtype=float,
    )

    valid = np.asarray(
        data["corrected_valid"],
        dtype=bool,
    )

    if len(
        time_s
    ) < 2:
        raise RuntimeError(
            "CSV contains too few rows."
        )

    dt = float(
        np.median(
            np.diff(
                time_s
            )
        )
    )

    analysis_hz = float(
        1.0
        / dt
    )

    return (
        time_s,
        clean_f0,
        valid,
        analysis_hz,
    )


def fmt(
    value,
    digits=2,
):

    if not np.isfinite(
        value
    ):
        return "--"

    return (
        f"{float(value):.{digits}f}"
    )


def relation_name(
    relation: StructuralRelation,
) -> str:

    names = {
        StructuralRelation.INCOMPLETE:
            "INCOMPLETE",

        StructuralRelation.SAME_DIRECTION:
            "SAME_DIR",

        StructuralRelation.REVERSAL:
            "REVERSAL",

        StructuralRelation.NEAR_FLAT_ENTRY:
            "FLAT_IN",

        StructuralRelation.NEAR_FLAT_EXIT:
            "FLAT_OUT",

        StructuralRelation.NEAR_FLAT_BOTH:
            "FLAT_BOTH",
    }

    return names[
        relation
    ]


def print_object(
    obj,
):

    print(
        f"\nOBJECT {obj.object_index:3d} "
        f"T={obj.n_transitions} "
        f"links={obj.n_links} "
        f"same={obj.same_direction_links} "
        f"rev={obj.reversal_links} "
        f"flat={obj.near_flat_links} "
        f"incomplete={obj.incomplete_links} "
        f"returnPrev={obj.return_to_previous_links} "
        f"returnNext={obj.return_to_next_links} "
        f"returnBoth={obj.return_both_links} "
        f"path={fmt(obj.local_target_path_total_st)} "
        f"excess={fmt(obj.excess_over_outer_total_st)} "
        f"maxExcess={fmt(obj.max_excess_over_outer_st)} "
        f"minRatio={fmt(obj.min_outer_to_path_ratio)}"
    )

    for link in obj.links:

        print(
            "    "
            f"L{link.link_index + 1} "
            f"T{link.left_transition_region_index}"
            f"-S{link.stable_region_index}"
            f"-T{link.right_transition_region_index} "
            f"{relation_name(link.relation):>10} "
            f"prev={fmt(link.previous_target_st):>6} "
            f"S={fmt(link.stable_target_st):>6} "
            f"next={fmt(link.next_target_st):>6} "
            f"dur={fmt(link.stable_duration_ms, 0):>4}ms "
            f"in={fmt(link.entry_interval_st):>6} "
            f"out={fmt(link.exit_interval_st):>6} "
            f"outer={fmt(link.outer_interval_st):>6} "
            f"path={fmt(link.local_target_path_st):>5} "
            f"excess={fmt(link.excess_over_outer_st):>5} "
            f"ratio={fmt(link.outer_to_path_ratio):>5} "
            f"gainPrev="
            f"{fmt(link.return_gain_to_previous_st):>5} "
            f"gainNext="
            f"{fmt(link.return_gain_to_next_st):>5} "
            f"ret="
            f"{int(link.returns_toward_previous)}/"
            f"{int(link.returns_toward_next)}"
        )
        print(
            "        "
            f"FEATURES "
            f"dur="
            f"{fmt(link.left_transition_duration_ms, 0)}"
            f"/"
            f"{fmt(link.right_transition_duration_ms, 0)}ms "
            f"durSim={fmt(link.duration_similarity)} "
            f"shape="
            f"{fmt(link.left_shape_path_st)}"
            f"/"
            f"{fmt(link.right_shape_path_st)} "
            f"shapeSim={fmt(link.shape_path_similarity)} "
            f"center="
            f"{fmt(link.left_center_path_st)}"
            f"/"
            f"{fmt(link.right_center_path_st)} "
            f"centerSim={fmt(link.center_path_similarity)} "
            f"C/S="
            f"{fmt(link.left_center_to_shape_ratio)}"
            f"/"
            f"{fmt(link.right_center_to_shape_ratio)} "
            f"resRMS="
            f"{fmt(link.left_residual_rms_st)}"
            f"/"
            f"{fmt(link.right_residual_rms_st)} "
            f"resSim={fmt(link.residual_rms_similarity)} "
            f"resPP="
            f"{fmt(link.left_residual_peak_to_peak_st)}"
            f"/"
            f"{fmt(link.right_residual_peak_to_peak_st)}"
        )

        print(
            "        "
            f"TOPOLOGY "
            f"cycles="
            f"{fmt(link.left_residual_full_cycles, 1)}"
            f"/"
            f"{fmt(link.right_residual_full_cycles, 1)} "
            f"period="
            f"{fmt(link.left_residual_periodicity)}"
            f"/"
            f"{fmt(link.right_residual_periodicity)} "
            f"acSup="
            f"{fmt(link.left_autocorr_support)}"
            f"/"
            f"{fmt(link.right_autocorr_support)} "
            f"acCyc="
            f"{fmt(link.left_autocorr_cycles_supported, 1)}"
            f"/"
            f"{fmt(link.right_autocorr_cycles_supported, 1)} "
            f"srcRet="
            f"{fmt(link.left_source_return_fraction)}"
            f"/"
            f"{fmt(link.right_source_return_fraction)}"
        )

def main():

    if len(
        sys.argv
    ) != 2:

        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_gesture_object_splits.py "
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

    objects = (
        construct_gesture_objects(
            interpretation=interpretation,
            features=features,
            config=(
                GestureObjectConfig(
                    short_target_max_ms=250.0,
                    max_chain_transitions=8,
                )
            ),
        )
    )

    split_analysis = (
        analyse_gesture_object_splits(
            objects=objects,
            features=features,
            config=(
                GestureObjectSplitterConfig(
                    near_flat_st=0.05,
                )
            ),
        )
    )

    print(
        "=" * 120
    )

    print(
        "GESTURE OBJECT STRUCTURAL SPLIT EVIDENCE V2"
    )

    print(
        "=" * 120
    )

    print(
        f"Rows:                 "
        f"{len(clean_f0)}"
    )

    print(
        f"Analysis rate:        "
        f"{analysis_hz:.3f} Hz"
    )

    print(
        f"Valid rows:           "
        f"{int(np.sum(valid))}"
    )

    print(
        f"Maximal objects:      "
        f"{split_analysis.total_objects}"
    )

    print(
        f"Observed links:       "
        f"{split_analysis.total_links}"
    )

    print(
        f"Objects with links:   "
        f"{split_analysis.objects_with_links}"
    )

    print(
        f"Objects without links:"
        f" {split_analysis.objects_without_links}"
    )

    print(
        f"Same-direction links: "
        f"{split_analysis.same_direction_links}"
    )

    print(
        f"Reversal links:       "
        f"{split_analysis.reversal_links}"
    )

    print(
        f"Near-flat links:      "
        f"{split_analysis.near_flat_links}"
    )

    print(
        f"Incomplete links:     "
        f"{split_analysis.incomplete_links}"
    )

    print()

    print(
        "Expected invariants:"
    )

    print(
        "  maximal objects remain 55"
    )

    print(
        "  observed links remain 42"
    )

    print(
        "  original 97 transitions remain untouched"
    )

    print(
        "  NO objects are actually split in V2"
    )

    print(
        "  NO musical labels are assigned"
    )

    print()
    print(
        "=" * 120
    )
    print(
        "ALL MULTI-TRANSITION OBJECTS"
    )
    print(
        "=" * 120
    )

    for obj in split_analysis.objects:

        if obj.n_links == 0:
            continue

        print_object(
            obj
        )

    print()
    print(
        "=" * 120
    )
    print(
        "REGRESSION OBJECTS"
    )
    print(
        "=" * 120
    )

    wanted_objects = {
        8,
        19,
        24,
        42,
        44,
        46,
        47,
    }

    for obj in split_analysis.objects:

        if (
            obj.object_index
            not in wanted_objects
        ):
            continue

        print_object(
            obj
        )


if __name__ == "__main__":
    main()