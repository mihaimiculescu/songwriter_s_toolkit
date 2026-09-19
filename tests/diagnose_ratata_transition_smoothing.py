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

from python_eckf.transition_geometry import (
    compare_transition_geometry,
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


def ratio(
    after: float,
    before: float,
) -> float:

    if (
        not np.isfinite(after)
        or not np.isfinite(before)
        or abs(before) <= 1e-12
    ):
        return np.nan

    return float(
        after / before
    )


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_transition_smoothing.py "
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
            window_ms=50.0,
            polyorder=2,
        )
    )

    smoothing = (
        smooth_expressive_pitch(
            pitch_st=(
                interpretation.pitch_st
            ),
            valid=valid,
            analysis_hz=analysis_hz,
            config=smoothing_config,
        )
    )

    result = (
        compare_transition_geometry(
            interpretation=(
                interpretation
            ),
            smoothing=smoothing,
        )
    )

    print(
        "=" * 170
    )

    print(
        "TRANSITION GEOMETRY V2 — "
        "RAW vs BIDIRECTIONAL SHAPE"
    )

    print(
        "=" * 170
    )

    print(
        f"Rows:             "
        f"{len(clean_f0)}"
    )

    print(
        f"Analysis rate:    "
        f"{analysis_hz:.3f} Hz"
    )

    print(
        f"Smoothing:        "
        f"{smoothing_config.window_ms:.1f} ms "
        f"Savitzky-Golay, "
        f"polyorder="
        f"{smoothing_config.polyorder}"
    )

    print(
        f"Transitions:      "
        f"{len(result.transitions)}"
    )

    print()

    print(
        " #   range             dur "
        "type  "
        "rawPath shpPath ratio  "
        "rawRev shpRev  "
        "rawMono shpMono  "
        "rawStep shpStep  "
        "rawSpan shpSpan  "
        "rawRet shpRet"
    )

    print(
        "--- ----------------- ----- "
        "----- "
        "------- ------- -----  "
        "------ ------  "
        "------- -------  "
        "------- -------  "
        "------- -------  "
        "------ ------"
    )

    for i, pair in enumerate(
        result.transitions,
        start=1,
    ):

        raw = pair.raw
        shape = pair.shape

        if (
            raw.has_source
            and raw.has_destination
        ):
            kind = "A->B"

        elif raw.has_source:
            kind = "A->-"

        elif raw.has_destination:
            kind = "-->B"

        else:
            kind = "----"

        path_ratio = ratio(
            shape.path_length_st,
            raw.path_length_st,
        )

        print(
            f"{i:3d} "
            f"{raw.start_time_s:7.3f}-"
            f"{raw.end_time_s:7.3f} "
            f"{raw.duration_ms:5.0f} "
            f"{kind:>5} "
            f"{raw.path_length_st:7.2f} "
            f"{shape.path_length_st:7.2f} "
            f"{path_ratio:5.2f}  "
            f"{raw.direction_reversals:6d} "
            f"{shape.direction_reversals:6d}  "
            f"{raw.monotonic_fraction:7.2f} "
            f"{shape.monotonic_fraction:7.2f}  "
            f"{raw.max_adjacent_step_st:7.2f} "
            f"{shape.max_adjacent_step_st:7.2f}  "
            f"{raw.span_st:7.2f} "
            f"{shape.span_st:7.2f}  "
            f"{raw.return_evidence:6.2f} "
            f"{shape.return_evidence:6.2f}"
        )

    # ------------------------------------------------------------
    # Aggregate effects
    # ------------------------------------------------------------

    raw_paths = np.asarray(
        [
            p.raw.path_length_st
            for p in result.transitions
        ],
        dtype=np.float64,
    )

    shape_paths = np.asarray(
        [
            p.shape.path_length_st
            for p in result.transitions
        ],
        dtype=np.float64,
    )

    raw_reversals = np.asarray(
        [
            p.raw.direction_reversals
            for p in result.transitions
        ],
        dtype=np.float64,
    )

    shape_reversals = np.asarray(
        [
            p.shape.direction_reversals
            for p in result.transitions
        ],
        dtype=np.float64,
    )

    raw_steps = np.asarray(
        [
            p.raw.max_adjacent_step_st
            for p in result.transitions
        ],
        dtype=np.float64,
    )

    shape_steps = np.asarray(
        [
            p.shape.max_adjacent_step_st
            for p in result.transitions
        ],
        dtype=np.float64,
    )

    print()
    print(
        "=" * 170
    )

    print(
        "AGGREGATE EFFECT"
    )

    print(
        "=" * 170
    )

    print(
        "Median path length:"
    )

    print(
        f"  raw:    "
        f"{np.median(raw_paths):.3f} st"
    )

    print(
        f"  shape:  "
        f"{np.median(shape_paths):.3f} st"
    )

    print(
        f"  ratio:  "
        f"{np.median(shape_paths / np.maximum(raw_paths, 1e-12)):.3f}"
    )

    print()

    print(
        "Median direction reversals:"
    )

    print(
        f"  raw:    "
        f"{np.median(raw_reversals):.1f}"
    )

    print(
        f"  shape:  "
        f"{np.median(shape_reversals):.1f}"
    )

    print()

    print(
        "Median maximum adjacent step:"
    )

    print(
        f"  raw:    "
        f"{np.median(raw_steps):.3f} st"
    )

    print(
        f"  shape:  "
        f"{np.median(shape_steps):.3f} st"
    )

    # ------------------------------------------------------------
    # Complete A->B transitions
    # ------------------------------------------------------------

    print()
    print(
        "=" * 170
    )

    print(
        "A->B ONLY — RAW vs SHAPE"
    )

    print(
        "=" * 170
    )

    print(
        " #   range             dur "
        "src->dst       int   "
        "rawPath shpPath "
        "rawEff shpEff "
        "rawRev shpRev "
        "rawRet shpRet"
    )

    print(
        "--- ----------------- ----- "
        "------------- ------ "
        "------- ------- "
        "------ ------ "
        "------ ------ "
        "------ ------"
    )

    ab_index = 0

    for pair in result.transitions:

        raw = pair.raw
        shape = pair.shape

        if not (
            raw.has_source
            and raw.has_destination
        ):
            continue

        ab_index += 1

        print(
            f"{ab_index:3d} "
            f"{raw.start_time_s:7.3f}-"
            f"{raw.end_time_s:7.3f} "
            f"{raw.duration_ms:5.0f} "
            f"{raw.source_target_st:5.2f}"
            f"->{raw.destination_target_st:5.2f} "
            f"{raw.target_interval_st:6.2f} "
            f"{raw.path_length_st:7.2f} "
            f"{shape.path_length_st:7.2f} "
            f"{raw.path_efficiency:6.2f} "
            f"{shape.path_efficiency:6.2f} "
            f"{raw.direction_reversals:6d} "
            f"{shape.direction_reversals:6d} "
            f"{raw.return_evidence:6.2f} "
            f"{shape.return_evidence:6.2f}"
        )


if __name__ == "__main__":
    main()