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
    RegionKind,
    interpret_pitch_trajectory,
)

from python_eckf.expressive_smoothing import (
    ExpressiveSmoothingConfig,
    smooth_expressive_pitch,
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

        reader = csv.DictReader(f)

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
                    sorted(missing)
                )
            )

        rows.extend(
            reader
        )

    time_s = np.asarray(
        [
            float(
                row["time_s"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    clean_f0 = np.asarray(
        [
            (
                float(
                    row["clean_f0_hz"]
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
                np.diff(time_s)
            )
        )

        analysis_hz = (
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


def path_length(
    pitch: np.ndarray,
) -> float:

    pitch = np.asarray(
        pitch,
        dtype=np.float64,
    )

    if len(pitch) <= 1:
        return 0.0

    return float(
        np.sum(
            np.abs(
                np.diff(pitch)
            )
        )
    )


def span(
    pitch: np.ndarray,
) -> float:

    if len(pitch) == 0:
        return np.nan

    return float(
        np.max(pitch)
        - np.min(pitch)
    )


def rms(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                values * values
            )
        )
    )


def percentile_abs(
    values: np.ndarray,
    percentile: float,
) -> float:

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            np.abs(values),
            percentile,
        )
    )


def movement_ratio(
    center_path: float,
    shape_path: float,
) -> float:

    if shape_path <= 1e-12:
        return 0.0

    return float(
        center_path
        / shape_path
    )


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_pitch_decomposition.py "
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

    smoothing_config = (
        ExpressiveSmoothingConfig(
            analysis_hz=analysis_hz,

            # Frozen light expressive smoother.
            window_ms=50.0,
            polyorder=2,

            # Diagnostic centerline candidate.
            center_window_ms=250.0,
        )
    )

    decomposition = (
        smooth_expressive_pitch(
            pitch_st=(
                interpretation.pitch_st
            ),
            valid=valid,
            analysis_hz=analysis_hz,
            config=smoothing_config,
        )
    )

    print(
        "=" * 160
    )

    print(
        "PITCH DECOMPOSITION V1 — "
        "SHAPE / CENTER / RESIDUAL"
    )

    print(
        "=" * 160
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
        f"Shape smoothing:      "
        f"{smoothing_config.window_ms:.1f} ms "
        f"SG poly="
        f"{smoothing_config.polyorder}"
    )

    print(
        f"Center window:        "
        f"{smoothing_config.center_window_ms:.1f} ms "
        f"symmetric median"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "  movement ratio = center path / shape path"
    )

    print(
        "  low ratio  -> movement is mainly around the center"
    )

    print(
        "  high ratio -> the center itself is moving"
    )

    print(
        "  This is geometry only; no vibrato/glide labels are assigned."
    )

    print()
    print(
        "=" * 160
    )

    print(
        "TRANSITION DECOMPOSITION"
    )

    print(
        "=" * 160
    )

    print(
        " #   range             dur type "
        "shapeSpan centerSpan "
        "shapePath centerPath ratio "
        "resRMS resP90 resMax "
        "centerNet"
    )

    print(
        "--- ----------------- ----- ----- "
        "--------- ---------- "
        "--------- ---------- ----- "
        "------ ------ ------ "
        "---------"
    )

    transition_rows = []

    transition_number = 0

    for region in interpretation.regions:

        if (
            region.kind
            is not RegionKind.TRANSITION
        ):
            continue

        transition_number += 1

        start = int(
            region.start_index
        )

        end = int(
            region.end_index
        )

        shape_pitch = (
            decomposition.shape_pitch_st[
                start:end
            ]
        )

        center_pitch = (
            decomposition.center_pitch_st[
                start:end
            ]
        )

        residual = (
            decomposition.residual_pitch_st[
                start:end
            ]
        )

        if (
            np.isfinite(
                region.previous_target_st
            )
            and np.isfinite(
                region.next_target_st
            )
        ):
            boundary = "A->B"

        elif np.isfinite(
            region.previous_target_st
        ):
            boundary = "A->-"

        elif np.isfinite(
            region.next_target_st
        ):
            boundary = "-->B"

        else:
            boundary = "----"

        shape_path = path_length(
            shape_pitch
        )

        center_path = path_length(
            center_pitch
        )

        ratio = movement_ratio(
            center_path,
            shape_path,
        )

        shape_span = span(
            shape_pitch
        )

        center_span = span(
            center_pitch
        )

        residual_rms = rms(
            residual
        )

        residual_p90 = percentile_abs(
            residual,
            90.0,
        )

        residual_max = float(
            np.max(
                np.abs(residual)
            )
        )

        if len(center_pitch) >= 2:
            center_net = float(
                center_pitch[-1]
                - center_pitch[0]
            )
        else:
            center_net = 0.0

        transition_rows.append(
            (
                transition_number,
                region,
                boundary,
                shape_span,
                center_span,
                shape_path,
                center_path,
                ratio,
                residual_rms,
                residual_p90,
                residual_max,
                center_net,
            )
        )

        print(
            f"{transition_number:3d} "
            f"{region.start_time_s:7.3f}-"
            f"{region.end_time_s:7.3f} "
            f"{region.duration_ms:5.0f} "
            f"{boundary:>5} "
            f"{shape_span:9.2f} "
            f"{center_span:10.2f} "
            f"{shape_path:9.2f} "
            f"{center_path:10.2f} "
            f"{ratio:5.2f} "
            f"{residual_rms:6.2f} "
            f"{residual_p90:6.2f} "
            f"{residual_max:6.2f} "
            f"{center_net:9.2f}"
        )

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------

    ratios = np.asarray(
        [
            row[7]
            for row in transition_rows
        ],
        dtype=np.float64,
    )

    residual_rms_values = np.asarray(
        [
            row[8]
            for row in transition_rows
        ],
        dtype=np.float64,
    )

    print()
    print(
        "=" * 160
    )

    print(
        "AGGREGATE"
    )

    print(
        "=" * 160
    )

    print(
        f"Transitions:              "
        f"{len(transition_rows)}"
    )

    print(
        f"Median movement ratio:    "
        f"{np.median(ratios):.3f}"
    )

    print(
        f"Median residual RMS:      "
        f"{np.median(residual_rms_values):.3f} st"
    )

    print(
        f"90th percentile res RMS:  "
        f"{np.percentile(residual_rms_values, 90):.3f} st"
    )

    # ------------------------------------------------------------
    # Lowest center/shape ratios
    # ------------------------------------------------------------

    print()
    print(
        "=" * 160
    )

    print(
        "LOWEST CENTER-MOVEMENT RATIOS"
    )

    print(
        "=" * 160
    )

    print(
        "These are trajectories whose expressive path is large relative "
        "to movement of the slower center."
    )

    print()

    ranked_low = sorted(
        transition_rows,
        key=lambda row: row[7],
    )

    print(
        " #   range             dur type "
        "shapePath centerPath ratio "
        "resRMS resMax centerNet"
    )

    print(
        "--- ----------------- ----- ----- "
        "--------- ---------- ----- "
        "------ ------ ---------"
    )

    for (
        number,
        region,
        boundary,
        shape_span,
        center_span,
        shape_path,
        center_path,
        ratio,
        residual_rms,
        residual_p90,
        residual_max,
        center_net,
    ) in ranked_low[:30]:

        print(
            f"{number:3d} "
            f"{region.start_time_s:7.3f}-"
            f"{region.end_time_s:7.3f} "
            f"{region.duration_ms:5.0f} "
            f"{boundary:>5} "
            f"{shape_path:9.2f} "
            f"{center_path:10.2f} "
            f"{ratio:5.2f} "
            f"{residual_rms:6.2f} "
            f"{residual_max:6.2f} "
            f"{center_net:9.2f}"
        )

    # ------------------------------------------------------------
    # Highest center/shape ratios
    # ------------------------------------------------------------

    print()
    print(
        "=" * 160
    )

    print(
        "HIGHEST CENTER-MOVEMENT RATIOS"
    )

    print(
        "=" * 160
    )

    print(
        "These are trajectories where the slower center itself carries "
        "a large fraction of the expressive movement."
    )

    print()

    ranked_high = sorted(
        transition_rows,
        key=lambda row: row[7],
        reverse=True,
    )

    print(
        " #   range             dur type "
        "shapePath centerPath ratio "
        "resRMS resMax centerNet"
    )

    print(
        "--- ----------------- ----- ----- "
        "--------- ---------- ----- "
        "------ ------ ---------"
    )

    for (
        number,
        region,
        boundary,
        shape_span,
        center_span,
        shape_path,
        center_path,
        ratio,
        residual_rms,
        residual_p90,
        residual_max,
        center_net,
    ) in ranked_high[:30]:

        print(
            f"{number:3d} "
            f"{region.start_time_s:7.3f}-"
            f"{region.end_time_s:7.3f} "
            f"{region.duration_ms:5.0f} "
            f"{boundary:>5} "
            f"{shape_path:9.2f} "
            f"{center_path:10.2f} "
            f"{ratio:5.2f} "
            f"{residual_rms:6.2f} "
            f"{residual_max:6.2f} "
            f"{center_net:9.2f}"
        )

    # ------------------------------------------------------------
    # Specific regression windows
    # ------------------------------------------------------------

    print()
    print(
        "=" * 160
    )

    print(
        "REGRESSION WINDOWS"
    )

    print(
        "=" * 160
    )

    regression_times = [
        14.70,   # clean ~2 st target change
        23.78,   # same-target leave/return
        27.15,   # tiny fast sequence
        27.45,
        27.67,
        47.66,   # long leave/return
        71.07,   # long same-target-ish movement
        71.80,
        72.39,   # direct ~1 st change
    ]

    for wanted in regression_times:

        matches = [
            row
            for row in transition_rows
            if (
                row[1].start_time_s
                <= wanted
                < row[1].end_time_s
            )
        ]

        if not matches:
            print(
                f"{wanted:7.3f}s: "
                f"NO TRANSITION"
            )
            continue

        row = matches[0]

        (
            number,
            region,
            boundary,
            shape_span,
            center_span,
            shape_path,
            center_path,
            ratio,
            residual_rms,
            residual_p90,
            residual_max,
            center_net,
        ) = row

        print(
            f"{wanted:7.3f}s -> "
            f"#{number:03d} "
            f"{region.start_time_s:.3f}-"
            f"{region.end_time_s:.3f}s "
            f"{boundary} | "
            f"shapePath={shape_path:.2f} "
            f"centerPath={center_path:.2f} "
            f"ratio={ratio:.2f} "
            f"centerNet={center_net:+.2f} "
            f"resRMS={residual_rms:.2f} "
            f"resMax={residual_max:.2f}"
        )


if __name__ == "__main__":
    main()