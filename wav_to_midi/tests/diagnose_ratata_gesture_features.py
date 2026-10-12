from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from python_eckf.trajectory_interpreter import (
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

    if len(time_s) >= 2:

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


def boundary_type(
    gesture,
) -> str:

    if (
        gesture.has_source_target
        and gesture.has_destination_target
    ):
        return "A->B"

    if gesture.has_source_target:
        return "A->-"

    if gesture.has_destination_target:
        return "-->B"

    return "----"

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

def print_gesture(
    gesture,
):

    print(
        f"{gesture.region_index:3d} "
        f"{gesture.start_time_s:7.3f}-"
        f"{gesture.end_time_s:7.3f} "
        f"{gesture.duration_ms:5.0f} "
        f"{boundary_type(gesture):>5} "
        f"int={fmt(gesture.target_interval_st):>6} "
        f"shapeNet={fmt(gesture.shape_net_st):>6} "
        f"shapePath={fmt(gesture.shape_path_st):>6} "
        f"centerNet={fmt(gesture.center_net_st):>6} "
        f"centerPath={fmt(gesture.center_path_st):>6} "
        f"ratio={fmt(gesture.center_to_shape_path_ratio):>5} "
        f"resRMS={fmt(gesture.residual_rms_st):>5} "
        f"resPP={fmt(gesture.residual_peak_to_peak_st):>5} "
        f"zc={gesture.residual_zero_crossings:2d} "
        f"pk/tr="
        f"{gesture.residual_peak_count:2d}/"
        f"{gesture.residual_trough_count:2d} "
        f"cyc={fmt(gesture.residual_full_cycles_est, 1):>4} "
        f"zcRate={fmt(gesture.residual_cycle_rate_hz):>5} "
        f"per={fmt(gesture.residual_periodicity):>4} "
        f"acPeak={fmt(gesture.residual_autocorr_peak):>5} "
        f"acLag={fmt(gesture.residual_autocorr_lag_ms, 0):>4}ms "
        f"acRate={fmt(gesture.residual_autocorr_rate_hz):>5} "
        f"acCyc={fmt(gesture.residual_autocorr_cycles_supported, 1):>4} "
        f"acSup={fmt(gesture.residual_autocorr_support):>4} "
        f"ret={fmt(gesture.source_return_fraction):>4}"        
    )


def main():

    if len(
        sys.argv
    ) != 2:

        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_gesture_features.py "
            "<offline.eckf.csv>"
        )

    path = Path(
        sys.argv[1]
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

    result = (
        extract_gesture_features(
            interpretation=interpretation,
            smoothing=smoothing,
            config=(
                GestureFeatureConfig()
            ),
        )
    )

    gestures = list(
        result.gestures
    )

    print(
        "=" * 220
    )

    print(
        "GESTURE FEATURES V1 — "
        "OBSERVATIONAL ONLY"
    )

    print(
        "=" * 220
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
        f"Pitch regions:        "
        f"{len(interpretation.regions)}"
    )

    print(
        f"Transitions in:       "
        f"{sum(1 for r in interpretation.regions if r.kind.name == 'TRANSITION')}"
    )

    print(
        f"Gesture objects out:  "
        f"{len(gestures)}"
    )

    print()

    print(
        "No musical labels are assigned."
    )

    print(
        "cycle/rate/periodicity describe residual geometry only."
    )

    print()

    print(
        "=" * 220
    )

    print(
        "ALL TRANSITIONS"
    )

    print(
        "=" * 220
    )

    for gesture in gestures:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Strongest residual excursions
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "STRONGEST RESIDUAL EXCURSIONS"
    )

    print(
        "=" * 220
    )

    ranked = sorted(
        gestures,
        key=lambda g: (
            g.residual_rms_st
        ),
        reverse=True,
    )

    for gesture in ranked[
        :30
    ]:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Most periodic residuals
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "MOST PERIODIC RESIDUALS"
    )

    print(
        "=" * 220
    )

    ranked = sorted(
        gestures,
        key=lambda g: (
            g.residual_periodicity
        ),
        reverse=True,
    )

    for gesture in ranked[
        :30
    ]:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Strongest autocorrelation-supported repetition
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "STRONGEST AUTOCORRELATION-SUPPORTED REPETITION"
    )

    print(
        "=" * 220
    )

    ranked = sorted(
        gestures,
        key=lambda g: (
            g.residual_autocorr_support
        ),
        reverse=True,
    )

    for gesture in ranked[
        :30
    ]:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Strongest source-return geometry
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "STRONGEST SOURCE-RETURN GEOMETRY"
    )

    print(
        "=" * 220
    )

    source_return = [
        gesture
        for gesture in gestures
        if np.isfinite(
            gesture.source_return_fraction
        )
    ]

    source_return = sorted(
        source_return,
        key=lambda g: (
            g.source_return_fraction,
            g.leaves_source_st,
        ),
        reverse=True,
    )

    for gesture in source_return[
        :30
    ]:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Center-dominant
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "MOST CENTER-DOMINANT MOVEMENT"
    )

    print(
        "=" * 220
    )

    ranked = sorted(
        gestures,
        key=lambda g: (
            g.center_to_shape_path_ratio
        ),
        reverse=True,
    )

    for gesture in ranked[
        :30
    ]:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Residual-dominant
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "MOST RESIDUAL-DOMINANT MOVEMENT"
    )

    print(
        "=" * 220
    )

    ranked = sorted(
        gestures,
        key=lambda g: (
            g.center_to_shape_path_ratio
        ),
    )

    for gesture in ranked[
        :30
    ]:
        print_gesture(
            gesture
        )

    # ------------------------------------------------------------
    # Regression windows
    # ------------------------------------------------------------

    print()
    print(
        "=" * 220
    )

    print(
        "REGRESSION WINDOWS"
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
    ]

    for wanted in regression_times:

        matches = [
            gesture
            for gesture in gestures
            if (
                gesture.start_time_s
                <= wanted
                < gesture.end_time_s
            )
        ]

        if not matches:

            print(
                f"{wanted:7.3f}s: "
                "NO TRANSITION"
            )

            continue

        print(
            f"{wanted:7.3f}s -> ",
            end="",
        )

        print_gesture(
            matches[0]
        )


if __name__ == "__main__":
    main()