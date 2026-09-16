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
    hz_to_midi,
    true_runs,
    vocal_transition_penalty,
    harmonic_evidence,
    same_pitch_context_evidence,
)

from python_eckf.validity import (
    PitchValidityConfig,
    analyse_pitch_validity,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def fmt(value, digits=2):
    if not np.isfinite(value):
        return "---"

    return f"{value:.{digits}f}"


def median_finite(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    return float(np.median(x))


def hz_to_midi_scalar(f):
    if not np.isfinite(f) or f <= 0:
        return np.nan

    return float(
        69.0 + 12.0 * np.log2(f / 440.0)
    )


def contiguous_finite_runs(mask):
    return true_runs(
        np.asarray(mask, dtype=bool)
    )


def accepted_context_pitch(
    midi,
    valid,
    boundary,
    side,
    analysis_hz,
    context_ms=200.0,
    guard_ms=20.0,
):
    n = len(midi)

    context_steps = max(
        1,
        int(
            round(
                context_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    guard_steps = max(
        0,
        int(
            round(
                guard_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    if side == "left":
        hi = max(
            0,
            boundary - guard_steps,
        )

        lo = max(
            0,
            hi - context_steps,
        )

    elif side == "right":
        lo = min(
            n,
            boundary + guard_steps,
        )

        hi = min(
            n,
            lo + context_steps,
        )

    else:
        raise ValueError(side)

    x = midi[lo:hi]
    m = (
        valid[lo:hi]
        & np.isfinite(x)
    )

    return median_finite(
        x[m]
    )


# ---------------------------------------------------------------------
# Rejected-run trajectory metrics
# ---------------------------------------------------------------------


def trajectory_metrics(pitch):
    p = np.asarray(
        pitch,
        dtype=np.float64,
    )

    p = p[np.isfinite(p)]

    if p.size == 0:
        return None

    if p.size == 1:
        return {
            "count": 1,
            "span": 0.0,
            "endpoint": 0.0,
            "path": 0.0,
            "efficiency": 1.0,
            "max_step": 0.0,
            "median_step": 0.0,
            "monotonic_fraction": 1.0,
            "direction_changes": 0,
        }

    d = np.diff(p)
    ad = np.abs(d)

    span = float(
        np.max(p) - np.min(p)
    )

    endpoint = float(
        p[-1] - p[0]
    )

    path = float(
        np.sum(ad)
    )

    max_step = float(
        np.max(ad)
    )

    median_step = float(
        np.median(ad)
    )

    direct = abs(endpoint)

    efficiency = (
        direct / path
        if path > 1e-12
        else 1.0
    )

    # Ignore microscopic movement when estimating direction.
    moving = d[
        np.abs(d) >= 0.05
    ]

    if moving.size == 0:
        monotonic_fraction = 1.0
        direction_changes = 0

    else:
        if endpoint > 0:
            preferred = moving > 0

        elif endpoint < 0:
            preferred = moving < 0

        else:
            positive = np.sum(
                moving > 0
            )

            negative = np.sum(
                moving < 0
            )

            if positive >= negative:
                preferred = moving > 0
            else:
                preferred = moving < 0

        monotonic_fraction = float(
            np.mean(preferred)
        )

        signs = np.sign(moving)

        direction_changes = int(
            np.sum(
                signs[1:]
                != signs[:-1]
            )
        )

    return {
        "count": int(p.size),
        "span": span,
        "endpoint": endpoint,
        "path": path,
        "efficiency": float(efficiency),
        "max_step": max_step,
        "median_step": median_step,
        "monotonic_fraction": monotonic_fraction,
        "direction_changes": direction_changes,
    }


# ---------------------------------------------------------------------
# Shape evidence
# ---------------------------------------------------------------------


def smoothness_score(metrics):
    """
    High for locally coherent trajectories.

    This does NOT require small total span.
    """

    if metrics is None:
        return 0.0

    max_step = metrics["max_step"]

    # 0-2 st / 10 ms: strongly coherent
    # ~4 st: still usable
    # >=8 st: essentially chaotic
    step_score = float(
        np.exp(
            -0.5 * (max_step / 3.0) ** 2
        )
    )

    path_score = float(
        np.clip(
            metrics["efficiency"],
            0.0,
            1.0,
        )
    )

    monotonic = float(
        np.clip(
            metrics["monotonic_fraction"],
            0.0,
            1.0,
        )
    )

    return float(
        0.45 * step_score
        + 0.30 * monotonic
        + 0.25 * path_score
    )


def chaos_score(metrics):
    """
    High for tracker explosions / impossible local jumps.
    """

    if metrics is None:
        return 1.0

    max_step = metrics["max_step"]

    step_chaos = float(
        np.clip(
            (max_step - 4.0) / 8.0,
            0.0,
            1.0,
        )
    )

    reversal_chaos = float(
        np.clip(
            metrics["direction_changes"] / 8.0,
            0.0,
            1.0,
        )
    )

    return float(
        0.75 * step_chaos
        + 0.25 * reversal_chaos
    )


def glide_score(metrics):
    """
    Evidence for a coherent continuous glide.
    """

    if metrics is None:
        return 0.0

    if metrics["span"] < 2.0:
        return 0.0

    return float(
        np.clip(
            0.45
            * metrics["monotonic_fraction"]
            + 0.35
            * metrics["efficiency"]
            + 0.20
            * np.exp(
                -0.5
                * (metrics["max_step"] / 2.5) ** 2
            ),
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------
# Boundary relationship
# ---------------------------------------------------------------------


def boundary_metrics(
    island_median,
    left_pitch,
    right_pitch,
    duration_ms,
    resolver_config,
):
    entry = (
        island_median - left_pitch
        if np.isfinite(left_pitch)
        else np.nan
    )

    exit_interval = (
        right_pitch - island_median
        if np.isfinite(right_pitch)
        else np.nan
    )

    left_right = (
        right_pitch - left_pitch
        if (
            np.isfinite(left_pitch)
            and np.isfinite(right_pitch)
        )
        else np.nan
    )

    entry_penalty = vocal_transition_penalty(
        entry,
        duration_ms,
    )

    exit_penalty = vocal_transition_penalty(
        exit_interval,
        duration_ms,
    )

    (
        left_relation,
        left_error,
        left_harmonic_evidence,
    ) = harmonic_evidence(
        entry,
        resolver_config,
    )

    island_vs_right = (
        island_median - right_pitch
        if np.isfinite(right_pitch)
        else np.nan
    )

    (
        right_relation,
        right_error,
        right_harmonic_evidence,
    ) = harmonic_evidence(
        island_vs_right,
        resolver_config,
    )

    same_family = (
        left_relation is not None
        and right_relation is not None
        and left_relation == right_relation
    )

    if same_family:
        harmonic_pair = float(
            np.sqrt(
                left_harmonic_evidence
                * right_harmonic_evidence
            )
        )

        harmonic_relation = (
            left_relation
        )

    else:
        harmonic_pair = 0.0
        harmonic_relation = None

    context_evidence = (
        same_pitch_context_evidence(
            left_right,
            resolver_config,
        )
    )

    return {
        "entry": float(entry),
        "exit": float(exit_interval),
        "left_right": float(left_right),

        "entry_penalty": float(
            entry_penalty
        ),

        "exit_penalty": float(
            exit_penalty
        ),

        "left_relation": left_relation,
        "right_relation": right_relation,

        "left_harmonic_evidence": float(
            left_harmonic_evidence
        ),

        "right_harmonic_evidence": float(
            right_harmonic_evidence
        ),

        "harmonic_pair": float(
            harmonic_pair
        ),

        "harmonic_relation": (
            harmonic_relation
        ),

        "context_evidence": float(
            context_evidence
        ),

        "left_error": float(
            left_error
        ),

        "right_error": float(
            right_error
        ),
    }


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------


def classify_rejected_run(
    duration_ms,
    metrics,
    boundary,
):
    smooth = smoothness_score(
        metrics
    )

    chaos = chaos_score(
        metrics
    )

    glide = glide_score(
        metrics
    )

    span = (
        metrics["span"]
        if metrics is not None
        else np.nan
    )

    max_step = (
        metrics["max_step"]
        if metrics is not None
        else np.inf
    )

    endpoint = (
        metrics["endpoint"]
        if metrics is not None
        else np.nan
    )

    monotonic = (
        metrics["monotonic_fraction"]
        if metrics is not None
        else 0.0
    )

    # -------------------------------------------------------------
    # 1. Explicit chaotic tracker failure
    # -------------------------------------------------------------

    if (
        max_step >= 8.0
        or chaos >= 0.70
    ):
        return (
            "CHAOTIC_TRACKING_FAILURE",
            min(
                1.0,
                0.70 + 0.30 * chaos,
            ),
            smooth,
            glide,
            chaos,
        )

    # -------------------------------------------------------------
    # 2. Harmonic-lock-shaped rejected boundary
    #
    # Do not call it a correction yet. Just identify the shape.
    # -------------------------------------------------------------

    if (
        boundary["harmonic_relation"]
        is not None
        and boundary["harmonic_pair"] >= 0.45
        and boundary["context_evidence"] >= 0.35
    ):
        return (
            "HARMONIC_LOCK_BOUNDARY",
            float(
                np.clip(
                    0.50
                    + 0.25
                    * boundary["harmonic_pair"]
                    + 0.20
                    * boundary["context_evidence"],
                    0.0,
                    1.0,
                )
            ),
            smooth,
            glide,
            chaos,
        )

    # -------------------------------------------------------------
    # 3. Smooth continuous glide
    #
    # Total span is allowed to be large.
    # -------------------------------------------------------------

    if (
        span >= 3.0
        and glide >= 0.70
        and max_step <= 4.0
    ):
        return (
            "PLAUSIBLE_SMOOTH_GLIDE",
            float(
                np.clip(
                    0.55
                    + 0.40 * glide,
                    0.0,
                    1.0,
                )
            ),
            smooth,
            glide,
            chaos,
        )

    # -------------------------------------------------------------
    # 4. Short coherent attack scoop / grace-like movement
    #
    # This is deliberately conservative.
    # -------------------------------------------------------------

    if (
        duration_ms <= 180.0
        and smooth >= 0.65
        and max_step <= 4.0
        and span <= 5.0
    ):
        return (
            "PLAUSIBLE_ATTACK_SCOOP",
            float(
                np.clip(
                    0.55
                    + 0.35 * smooth,
                    0.0,
                    1.0,
                )
            ),
            smooth,
            glide,
            chaos,
        )

    # -------------------------------------------------------------
    # 5. Short coherent larger discrete gesture
    #
    # Could later become grace note / mordent / short target note.
    # -------------------------------------------------------------

    if (
        duration_ms <= 250.0
        and smooth >= 0.60
        and max_step <= 4.0
    ):
        return (
            "PLAUSIBLE_DISCRETE_GESTURE",
            float(
                np.clip(
                    0.50
                    + 0.35 * smooth,
                    0.0,
                    1.0,
                )
            ),
            smooth,
            glide,
            chaos,
        )

    # -------------------------------------------------------------
    # 6. Coherent but not yet semantically understood
    # -------------------------------------------------------------

    if (
        smooth >= 0.65
        and max_step <= 4.0
    ):
        return (
            "COHERENT_REJECTED_TRAJECTORY",
            float(
                np.clip(
                    0.45
                    + 0.30 * smooth,
                    0.0,
                    1.0,
                )
            ),
            smooth,
            glide,
            chaos,
        )

    return (
        "UNRESOLVED",
        0.40,
        smooth,
        glide,
        chaos,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="100 Hz ECKF CSV",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=60,
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
        / float(
            np.median(
                np.diff(t)
            )
        )
    )

    validity_config = (
        PitchValidityConfig(
            analysis_hz=analysis_hz,
            min_vocal_hz=120.0,
        )
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

    midi = hz_to_midi(
        f0
    )

    resolver_config = ResolverConfig(
        analysis_hz=analysis_hz,
    )

    rejected = (
        ~valid
        & np.isfinite(midi)
    )

    runs = contiguous_finite_runs(
        rejected
    )

    results = []

    for start, end in runs:
        raw_pitch = midi[
            start:end
        ]

        metrics = trajectory_metrics(
            raw_pitch
        )

        if metrics is None:
            continue

        duration_ms = (
            (end - start)
            / analysis_hz
            * 1000.0
        )

        island_median = (
            median_finite(
                raw_pitch
            )
        )

        left_pitch = (
            accepted_context_pitch(
                midi=midi,
                valid=valid,
                boundary=start,
                side="left",
                analysis_hz=analysis_hz,
            )
        )

        right_pitch = (
            accepted_context_pitch(
                midi=midi,
                valid=valid,
                boundary=end,
                side="right",
                analysis_hz=analysis_hz,
            )
        )

        boundary = boundary_metrics(
            island_median=island_median,
            left_pitch=left_pitch,
            right_pitch=right_pitch,
            duration_ms=duration_ms,
            resolver_config=resolver_config,
        )

        (
            classification,
            confidence,
            smooth,
            glide,
            chaos,
        ) = classify_rejected_run(
            duration_ms=duration_ms,
            metrics=metrics,
            boundary=boundary,
        )

        results.append(
            {
                "start": start,
                "end": end,

                "duration_ms": float(
                    duration_ms
                ),

                "median_pitch": float(
                    island_median
                ),

                "left_pitch": float(
                    left_pitch
                ),

                "right_pitch": float(
                    right_pitch
                ),

                "classification": (
                    classification
                ),

                "confidence": float(
                    confidence
                ),

                "smooth": float(
                    smooth
                ),

                "glide": float(
                    glide
                ),

                "chaos": float(
                    chaos
                ),

                "metrics": metrics,
                "boundary": boundary,
            }
        )

    counts = Counter(
        r["classification"]
        for r in results
    )

    print()
    print("=" * 166)
    print(
        "REJECTED TRAJECTORY DIAGNOSTIC — OBSERVATIONAL PASS"
    )
    print("=" * 166)

    print(
        f"Rows:                {len(t)}"
    )

    print(
        f"Analysis rate:       {analysis_hz:.3f} Hz"
    )

    print(
        f"Accepted rows:       {np.sum(valid)} / {len(valid)}"
    )

    print(
        f"Rejected F0 runs:    {len(results)}"
    )

    print()
    print(
        "CLASSIFICATION COUNTS"
    )

    for name, count in sorted(
        counts.items()
    ):
        print(
            f"  {name:35s} {count}"
        )

    # -------------------------------------------------------------
    # Rank interesting cases first.
    # -------------------------------------------------------------

    priority = {
        "HARMONIC_LOCK_BOUNDARY": 7,
        "CHAOTIC_TRACKING_FAILURE": 6,
        "PLAUSIBLE_SMOOTH_GLIDE": 5,
        "PLAUSIBLE_ATTACK_SCOOP": 4,
        "PLAUSIBLE_DISCRETE_GESTURE": 3,
        "COHERENT_REJECTED_TRAJECTORY": 2,
        "UNRESOLVED": 1,
    }

    ranked = sorted(
        results,
        key=lambda r: (
            priority.get(
                r["classification"],
                0,
            ),
            r["confidence"],
            r["glide"],
            r["smooth"],
            -r["chaos"],
        ),
        reverse=True,
    )

    print()
    print("=" * 166)
    print("RANKED REJECTED RUNS")
    print("=" * 166)

    print(
        " #   time range        dur   class"
        "                               conf  "
        "span   endpt  maxstep mono  smooth glide chaos  "
        "L->I     I->R     L->R   "
        "harm hPair ctx   entryP exitP"
    )

    print(
        "---  ----------------  -----  "
        "----------------------------------  "
        "----  -----  -----  ------- ----  "
        "------ ----- -----  "
        "-------  -------  -------  "
        "---- ----- ----  ------ -----"
    )

    for rank, r in enumerate(
        ranked[:args.top],
        start=1,
    ):
        start = r["start"]
        end = r["end"]

        start_time = t[start]

        end_time = t[
            max(
                start,
                end - 1,
            )
        ]

        m = r["metrics"]
        b = r["boundary"]

        relation = (
            b["harmonic_relation"]
            if b["harmonic_relation"]
            is not None
            else "-"
        )

        print(
            f"{rank:3d}  "
            f"{start_time:7.3f}-{end_time:7.3f}  "
            f"{r['duration_ms']:5.0f}  "
            f"{r['classification']:34s}  "
            f"{r['confidence']:4.2f}  "
            f"{m['span']:5.2f}  "
            f"{m['endpoint']:5.2f}  "
            f"{m['max_step']:7.2f} "
            f"{m['monotonic_fraction']:4.2f}  "
            f"{r['smooth']:6.2f} "
            f"{r['glide']:5.2f} "
            f"{r['chaos']:5.2f}  "
            f"{fmt(b['entry']):>7s}  "
            f"{fmt(b['exit']):>7s}  "
            f"{fmt(b['left_right']):>7s}  "
            f"{relation:>4s} "
            f"{b['harmonic_pair']:5.2f} "
            f"{b['context_evidence']:4.2f}  "
            f"{b['entry_penalty']:6.2f} "
            f"{b['exit_penalty']:5.2f}"
        )

    # -------------------------------------------------------------
    # Explicit regression windows
    # -------------------------------------------------------------

    regression_windows = [
        (
            "coherent large descent",
            6.85,
            7.15,
        ),
        (
            "short clean descent",
            9.00,
            9.20,
        ),
        (
            "short plausible gesture",
            22.85,
            23.10,
        ),
        (
            "djwww / noise collapse",
            40.85,
            41.25,
        ),
        (
            "extreme tracker explosion",
            73.80,
            74.10,
        ),
    ]

    print()
    print("=" * 166)
    print("REGRESSION WINDOWS")
    print("=" * 166)

    for label, lo, hi in regression_windows:
        print()
        print(
            f"{label}: {lo:.3f}-{hi:.3f}s"
        )

        matches = []

        for r in results:
            rs = t[
                r["start"]
            ]

            re = t[
                max(
                    r["start"],
                    r["end"] - 1,
                )
            ]

            if (
                re >= lo
                and rs <= hi
            ):
                matches.append(
                    r
                )

        if not matches:
            print(
                "  no rejected run overlaps window"
            )
            continue

        for r in matches:
            rs = t[
                r["start"]
            ]

            re = t[
                max(
                    r["start"],
                    r["end"] - 1,
                )
            ]

            m = r["metrics"]
            b = r["boundary"]

            print(
                f"  {rs:7.3f}-{re:7.3f}s  "
                f"{r['classification']:32s} "
                f"conf={r['confidence']:.2f} "
                f"span={m['span']:.2f} "
                f"endpoint={m['endpoint']:.2f} "
                f"maxstep={m['max_step']:.2f} "
                f"smooth={r['smooth']:.2f} "
                f"glide={r['glide']:.2f} "
                f"chaos={r['chaos']:.2f} "
                f"L->I={fmt(b['entry'])} "
                f"I->R={fmt(b['exit'])}"
            )

    print()
    print("=" * 166)
    print(
        "IMPORTANT: OBSERVATIONAL ONLY — "
        "NO F0 OR VALIDITY SAMPLES WERE MODIFIED."
    )
    print("=" * 166)


if __name__ == "__main__":
    main()