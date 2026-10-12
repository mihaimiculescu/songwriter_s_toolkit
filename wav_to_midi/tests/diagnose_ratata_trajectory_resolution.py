#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from python_eckf.trajectory_resolver import (
    ResolverConfig,
    resolve_valid_runs,
)

from python_eckf.validity import (
    PitchValidityConfig,
    analyse_pitch_validity,
)


def fmt(value, digits=2):
    if not np.isfinite(value):
        return "---"

    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="100 Hz ECKF CSV",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    data = np.genfromtxt(
        args.csv,
        delimiter=",",
        names=True,
    )

    t = np.asarray(
        data["time_s"],
        dtype=np.float64,
    )

    f0 = np.asarray(
        data["f0_hz"],
        dtype=np.float64,
    )

    analysis_hz = (
        1.0
        / float(np.median(np.diff(t)))
    )

    validity_config = PitchValidityConfig(
        analysis_hz=analysis_hz,
        min_vocal_hz=120.0,
    )

    validity = analyse_pitch_validity(
        f0_hz=f0,
        sample_rate=analysis_hz,
        config=validity_config,
    )

    valid = np.asarray(
        validity.valid,
        dtype=bool,
    )

    resolver_config = ResolverConfig(
        analysis_hz=analysis_hz,
    )

    resolutions = resolve_valid_runs(
        f0_hz=f0,
        valid=valid,
        config=resolver_config,
    )

    counts = Counter(
        r.classification.value
        for r in resolutions
    )

    print()
    print("=" * 132)
    print(
        "OFFLINE TRAJECTORY RESOLUTION — OBSERVATIONAL PASS"
    )
    print("=" * 132)

    print(
        f"Rows:             {len(t)}"
    )

    print(
        f"Analysis rate:    {analysis_hz:.3f} Hz"
    )

    print(
        f"Valid rows:       {np.sum(valid)} / {len(valid)}"
    )

    print(
        f"Valid islands:    {len(resolutions)}"
    )

    print()
    print("CLASSIFICATION COUNTS")

    for name, count in sorted(
        counts.items()
    ):
        print(
            f"  {name:35s} {count}"
        )

    # Harmonic/chaotic first, then ambiguity, then everything else.
    priority = {
        "PROBABLE_2X_HARMONIC_LOCK": 5,
        "PROBABLE_3X_HARMONIC_LOCK": 5,
        "PROBABLE_1_2_SUBHARMONIC_LOCK": 5,
        "PROBABLE_1_3_SUBHARMONIC_LOCK": 5,
        "CHAOTIC_TRACKING_FAILURE": 4,
        "AMBIGUOUS": 3,
        "SMOOTH_GLIDE": 2,
        "PLAUSIBLE_NEW_TRAJECTORY": 1,
        "CONTINUATION": 0,
    }

    ranked = sorted(
        resolutions,
        key=lambda r: (
            priority.get(
                r.classification.value,
                0,
            ),
            r.sandwich_score,
            r.harmonic_pair_evidence,
            r.context_evidence,
            max(
                r.entry_vocal_penalty,
                r.exit_vocal_penalty,
            ),
            r.confidence,
        ),
        reverse=True,
    )

    print()
    print("=" * 132)
    print("RANKED ISLANDS")
    print("=" * 132)
    
    print(
        " #   time range        dur   class"
        "                               conf  "
        "L->I     I->R     L->R   "
        "harm   herr   hL    hR    hPair  ctx   sand   "
        "span   maxstep  entryP exitP"
    )
    print(
        "---  ----------------  -----  "
        "----------------------------------  "
        "----  -------  -------  -------  "
        "-----  -----  ----  ----  -----  ----  -----  "
        "-----  -------  ------ -----"
    )

    for rank, r in enumerate(
        ranked[:args.top],
        start=1,
    ):
        start_time = t[r.start]

        end_time = t[
            max(r.start, r.end - 1)
        ]

        relation = (
            r.harmonic_relation
            if r.harmonic_relation is not None
            else "-"
        )

        print(
            f"{rank:3d}  "
            f"{start_time:7.3f}-{end_time:7.3f}  "
            f"{r.duration_ms:5.0f}  "
            f"{r.classification.value:34s}  "
            f"{r.confidence:4.2f}  "
            f"{fmt(r.entry_interval_st):>7s}  "
            f"{fmt(r.exit_interval_st):>7s}  "
            f"{fmt(r.left_right_interval_st):>7s}  "
            f"{relation:>5s}  "
            f"{fmt(r.harmonic_error_st):>5s}  "
            f"{r.left_harmonic_evidence:4.2f}  "
            f"{r.right_harmonic_evidence:4.2f}  "
            f"{r.harmonic_pair_evidence:5.2f}  "
            f"{r.context_evidence:4.2f}  "
            f"{r.sandwich_score:5.2f}  "
            f"{r.span_st:5.2f}  "
            f"{r.max_step_st:7.2f}  "
            f"{r.entry_vocal_penalty:6.2f} "
            f"{r.exit_vocal_penalty:5.2f}"
        )

    print()
    print("=" * 132)
    print(
        "IMPORTANT: OBSERVATIONAL ONLY — NO F0 OR VALIDITY "
        "SAMPLES WERE MODIFIED."
    )
    print("=" * 132)


if __name__ == "__main__":
    main()