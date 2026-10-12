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

from python_eckf.transition_geometry import (
    analyse_transition_geometry,
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

        for row in reader:
            rows.append(row)

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


def fmt(
    value: float,
    width: int = 6,
    decimals: int = 2,
) -> str:

    if not np.isfinite(value):
        return "-".rjust(width)

    return (
        f"{value:{width}.{decimals}f}"
    )


def boundary_type(
    has_source: bool,
    has_destination: bool,
) -> str:

    if (
        has_source
        and has_destination
    ):
        return "A->B"

    if has_source:
        return "A->-"

    if has_destination:
        return "-->B"

    return "----"


def direction_text(
    direction: int,
) -> str:

    if direction > 0:
        return "UP"

    if direction < 0:
        return "DOWN"

    return "MIX"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_transition_geometry.py "
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

    result = (
        analyse_transition_geometry(
            interpretation
        )
    )

    transitions = (
        result.transitions
    )

    print(
        "=" * 170
    )

    print(
        "CORRECTED F0 — "
        "TRANSITION GEOMETRY V1"
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
        f"Valid rows:       "
        f"{int(np.sum(valid))}"
    )

    print(
        f"Pitch regions:    "
        f"{len(interpretation.regions)}"
    )

    print(
        f"Transitions:      "
        f"{len(transitions)}"
    )

    both = sum(
        t.has_source
        and t.has_destination
        for t in transitions
    )

    source_only = sum(
        t.has_source
        and not t.has_destination
        for t in transitions
    )

    destination_only = sum(
        not t.has_source
        and t.has_destination
        for t in transitions
    )

    neither = sum(
        not t.has_source
        and not t.has_destination
        for t in transitions
    )

    print()
    print(
        "BOUNDARY TYPES"
    )

    print(
        f"  A->B:  {both}"
    )

    print(
        f"  A->-:  {source_only}"
    )

    print(
        f"  -->B:  {destination_only}"
    )

    print(
        f"  ----:  {neither}"
    )

    print()
    print(
        "=" * 170
    )

    print(
        "TRANSITIONS"
    )

    print(
        "=" * 170
    )

    print(
        " #   range             dur  type "
        "   src    dst    int   "
        "start    end    span    net   path  "
        "eff  mono rev maxst "
        "over under  "
        "dep   arr  nearS nearD  leave return retEv"
    )

    print(
        "--- ----------------- ----- ----- "
        "------ ------ ------ "
        "------ ------ ------ ------ ------ "
        "---- ---- --- ----- "
        "----- ----- "
        "----- ----- ----- ----- "
        "------ ------ -----"
    )

    for i, t in enumerate(
        transitions,
        start=1,
    ):

        print(
            f"{i:3d} "
            f"{t.start_time_s:7.3f}-"
            f"{t.end_time_s:7.3f} "
            f"{t.duration_ms:5.0f} "
            f"{boundary_type(t.has_source, t.has_destination):>5} "
            f"{fmt(t.source_target_st)} "
            f"{fmt(t.destination_target_st)} "
            f"{fmt(t.target_interval_st)} "
            f"{fmt(t.start_pitch_st)} "
            f"{fmt(t.end_pitch_st)} "
            f"{fmt(t.span_st)} "
            f"{fmt(t.net_movement_st)} "
            f"{fmt(t.path_length_st)} "
            f"{t.path_efficiency:4.2f} "
            f"{t.monotonic_fraction:4.2f} "
            f"{t.direction_reversals:3d} "
            f"{t.max_adjacent_step_st:5.2f} "
            f"{fmt(t.overshoot_above_st, 5)} "
            f"{fmt(t.undershoot_below_st, 5)} "
            f"{fmt(t.departure_from_source_st, 5)} "
            f"{fmt(t.arrival_error_destination_st, 5)} "
            f"{fmt(t.fraction_near_source, 5)} "
            f"{fmt(t.fraction_near_destination, 5)} "
            f"{fmt(t.leaves_source_st, 6)} "
            f"{fmt(t.returns_toward_source_st, 6)} "
            f"{fmt(t.return_evidence, 5)}"
        )

    # ------------------------------------------------------------
    # A->B transitions separately
    # ------------------------------------------------------------

    print()
    print(
        "=" * 170
    )

    print(
        "A->B TRANSITIONS — "
        "TARGET-TO-TARGET GEOMETRY"
    )

    print(
        "=" * 170
    )

    ab = [
        t
        for t in transitions
        if (
            t.has_source
            and t.has_destination
        )
    ]

    print(
        " #   range             dur "
        "src->dst       int   dir   "
        "net   path   eff  mono rev "
        "over under  arr"
    )

    print(
        "--- ----------------- ----- "
        "------------- ------ ----- "
        "------ ------ ---- ---- --- "
        "----- ----- -----"
    )

    for i, t in enumerate(
        ab,
        start=1,
    ):

        print(
            f"{i:3d} "
            f"{t.start_time_s:7.3f}-"
            f"{t.end_time_s:7.3f} "
            f"{t.duration_ms:5.0f} "
            f"{t.source_target_st:5.2f}"
            f"->{t.destination_target_st:5.2f} "
            f"{t.target_interval_st:6.2f} "
            f"{direction_text(t.dominant_direction):>5} "
            f"{t.net_movement_st:6.2f} "
            f"{t.path_length_st:6.2f} "
            f"{t.path_efficiency:4.2f} "
            f"{t.monotonic_fraction:4.2f} "
            f"{t.direction_reversals:3d} "
            f"{t.overshoot_above_st:5.2f} "
            f"{t.undershoot_below_st:5.2f} "
            f"{t.arrival_error_destination_st:5.2f}"
        )

    # ------------------------------------------------------------
    # Strongest leave-and-return shapes
    # ------------------------------------------------------------

    print()
    print(
        "=" * 170
    )

    print(
        "STRONGEST SOURCE-RETURN GEOMETRY "
        "(OBSERVATIONAL ONLY)"
    )

    print(
        "=" * 170
    )

    return_candidates = [
        t
        for t in transitions
        if (
            t.has_source
            and np.isfinite(
                t.return_evidence
            )
        )
    ]

    return_candidates.sort(
        key=lambda t: (
            t.return_evidence,
            t.leaves_source_st,
        ),
        reverse=True,
    )

    print(
        " #   range             dur "
        "source  leave  recovered retEv "
        "end-src  rev  path"
    )

    print(
        "--- ----------------- ----- "
        "------ ------ --------- ----- "
        "------- ---- ------"
    )

    for i, t in enumerate(
        return_candidates[:30],
        start=1,
    ):

        end_source = (
            t.end_pitch_st
            - t.source_target_st
        )

        print(
            f"{i:3d} "
            f"{t.start_time_s:7.3f}-"
            f"{t.end_time_s:7.3f} "
            f"{t.duration_ms:5.0f} "
            f"{t.source_target_st:6.2f} "
            f"{t.leaves_source_st:6.2f} "
            f"{t.returns_toward_source_st:9.2f} "
            f"{t.return_evidence:5.2f} "
            f"{end_source:7.2f} "
            f"{t.direction_reversals:4d} "
            f"{t.path_length_st:6.2f}"
        )


if __name__ == "__main__":
    main()