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
    interpret_pitch_trajectory,
)


def load_offline_csv(path: Path):
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
            - set(reader.fieldnames or [])
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
            float(row["time_s"])
            for row in rows
        ],
        dtype=np.float64,
    )

    clean_f0 = np.asarray(
        [
            (
                float(row["clean_f0_hz"])
                if row["clean_f0_hz"].strip()
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

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python "
            "tests/"
            "diagnose_ratata_trajectory_interpreter.py "
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
    ) = load_offline_csv(path)

    result = interpret_pitch_trajectory(
        clean_f0_hz=clean_f0,
        valid=valid,
        analysis_hz=analysis_hz,
    )

    print("=" * 132)
    print(
        "CORRECTED F0 — "
        "PITCH-TARGET SEGMENTATION V2"
    )
    print("=" * 132)

    print(
        f"Rows:           {len(clean_f0)}"
    )
    print(
        f"Analysis rate:  "
        f"{analysis_hz:.3f} Hz"
    )
    print(
        f"Valid rows:     "
        f"{int(np.sum(valid))}"
    )
    print(
        f"Stable rows:    "
        f"{int(np.sum(result.stable_mask))}"
    )
    print(
        f"Regions:        "
        f"{len(result.regions)}"
    )

    counts = Counter(
        region.kind.name
        for region in result.regions
    )

    print()
    print("REGION COUNTS")

    for name in sorted(counts):
        print(
            f"  {name:<16} "
            f"{counts[name]:5d}"
        )

    print()
    print("=" * 132)
    print("REGIONS")
    print("=" * 132)

    print(
        " #    range             dur  "
        "kind             med    span   "
        "rawmed   previous   next"
    )

    print(
        "---  ----------------  ----- "
        "----------------  ------ ----- "
        "------  --------  --------"
    )

    for region in result.regions:

        prev_text = (
            f"{region.previous_target_st:8.2f}"
            if np.isfinite(
                region.previous_target_st
            )
            else "       -"
        )

        next_text = (
            f"{region.next_target_st:8.2f}"
            if np.isfinite(
                region.next_target_st
            )
            else "       -"
        )

        print(
            f"{region.index + 1:3d}  "
            f"{region.start_time_s:7.3f}-"
            f"{region.end_time_s:7.3f}  "
            f"{region.duration_ms:5.0f} "
            f"{region.kind.name:<16} "
            f"{region.median_pitch_st:6.2f} "
            f"{region.span_st:5.2f} "
            f"{region.raw_median_pitch_st:6.2f} "
            f"{prev_text} "
            f"{next_text}"
        )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()