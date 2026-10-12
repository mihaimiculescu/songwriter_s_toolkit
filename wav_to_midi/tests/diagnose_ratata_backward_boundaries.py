#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from python_eckf.trajectory_resolver import (
    hz_to_midi,
    vocal_transition_penalty,
)

from python_eckf.validity import (
    PitchValidityConfig,
    analyse_pitch_validity,
    apply_offline_validity_correction,
)


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class BoundaryDiagnosticConfig:
    analysis_hz: float

    min_vocal_hz: float = 120.0
    max_vocal_hz: float = 1500.0

    # Local continuity only.
    #
    # No restriction on TOTAL pitch span.
    max_adjacent_step_st: float = 4.0

    # Robust accepted-context window.
    context_ms: float = 120.0

    # Minimum duration before a coherent fragment is interpreted
    # musically.
    min_subrun_ms: float = 30.0

    # Short excursion-and-return candidate.
    ornament_max_ms: float = 250.0
    ornament_min_span_st: float = 1.5

    movement_epsilon_st: float = 0.05

    # Descriptive flag for catastrophic enclosing episodes.
    catastrophic_step_st: float = 8.0

    # Current observational classification thresholds.
    rescue_edge_support_min: float = 0.60
    ornament_score_min: float = 0.65


# =====================================================================
# Generic helpers
# =====================================================================


def fmt(value, digits=2):
    if not np.isfinite(value):
        return "---"

    return f"{value:.{digits}f}"


def true_runs(mask):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.size == 0:
        return []

    padded = np.concatenate(
        (
            np.array([False]),
            mask,
            np.array([False]),
        )
    )

    changes = np.diff(
        padded.astype(np.int8)
    )

    starts = np.flatnonzero(
        changes == 1
    )

    ends = np.flatnonzero(
        changes == -1
    )

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def duration_ms(
    start,
    end,
    analysis_hz,
):
    return float(
        (end - start)
        / analysis_hz
        * 1000.0
    )


def median_finite(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.median(values)
    )


def hard_pitch_ok(
    f0,
    config,
):
    return (
        np.isfinite(f0)
        & (f0 >= config.min_vocal_hz)
        & (f0 <= config.max_vocal_hz)
    )


def range_text(
    start,
    end,
    t,
):
    if end <= start:
        return "---"

    return (
        f"{t[start]:.3f}-{t[end - 1]:.3f}"
    )


# =====================================================================
# Vocal-transition evidence
# =====================================================================


def vocal_support(
    interval_st,
    elapsed_ms,
):
    """
    Convert the existing soft vocal interval-vs-duration prior into
    continuous support.

        penalty = 0      -> support = 1
        larger penalty  -> weaker support

    IMPORTANT:
    elapsed_ms must be the ACTUAL elapsed time for the transition being
    evaluated.
    """

    if (
        not np.isfinite(interval_st)
        or not np.isfinite(elapsed_ms)
        or elapsed_ms < 0.0
    ):
        return 0.0

    penalty = vocal_transition_penalty(
        interval_st,
        elapsed_ms,
    )

    return float(
        np.exp(
            -float(penalty)
        )
    )


# =====================================================================
# Nearest accepted edge samples
# =====================================================================


def nearest_accepted_left(
    midi,
    valid,
    index,
):
    """
    Nearest accepted finite pitch strictly before index.
    """

    i = index - 1

    while i >= 0:
        if (
            valid[i]
            and np.isfinite(midi[i])
        ):
            return i

        i -= 1

    return None


def nearest_accepted_right(
    midi,
    valid,
    index,
):
    """
    Nearest accepted finite pitch at or after index.
    """

    n = len(midi)

    i = index

    while i < n:
        if (
            valid[i]
            and np.isfinite(midi[i])
        ):
            return i

        i += 1

    return None


def edge_transition_left(
    midi,
    valid,
    subrun_start,
    config,
):
    """
    Actual transition from nearest trusted accepted past sample to the
    first sample of this coherent sub-run.
    """

    accepted_index = nearest_accepted_left(
        midi,
        valid,
        subrun_start,
    )

    if accepted_index is None:
        return {
            "accepted_index": None,
            "accepted_pitch": np.nan,
            "interval_st": np.nan,
            "elapsed_ms": np.nan,
            "support": 0.0,
        }

    first_pitch = float(
        midi[subrun_start]
    )

    accepted_pitch = float(
        midi[accepted_index]
    )

    interval = float(
        first_pitch
        - accepted_pitch
    )

    elapsed = float(
        (
            subrun_start
            - accepted_index
        )
        / config.analysis_hz
        * 1000.0
    )

    return {
        "accepted_index": int(
            accepted_index
        ),

        "accepted_pitch": float(
            accepted_pitch
        ),

        "interval_st": interval,

        "elapsed_ms": elapsed,

        "support": vocal_support(
            interval,
            elapsed,
        ),
    }


def edge_transition_right(
    midi,
    valid,
    subrun_end,
    config,
):
    """
    Actual transition from the last sample of this coherent sub-run to
    the nearest trusted accepted future sample.

    subrun_end is exclusive.
    """

    last_index = (
        subrun_end - 1
    )

    accepted_index = nearest_accepted_right(
        midi,
        valid,
        subrun_end,
    )

    if accepted_index is None:
        return {
            "accepted_index": None,
            "accepted_pitch": np.nan,
            "interval_st": np.nan,
            "elapsed_ms": np.nan,
            "support": 0.0,
        }

    last_pitch = float(
        midi[last_index]
    )

    accepted_pitch = float(
        midi[accepted_index]
    )

    interval = float(
        accepted_pitch
        - last_pitch
    )

    elapsed = float(
        (
            accepted_index
            - last_index
        )
        / config.analysis_hz
        * 1000.0
    )

    return {
        "accepted_index": int(
            accepted_index
        ),

        "accepted_pitch": float(
            accepted_pitch
        ),

        "interval_st": interval,

        "elapsed_ms": elapsed,

        "support": vocal_support(
            interval,
            elapsed,
        ),
    }


# =====================================================================
# Robust median accepted context
# =====================================================================


def accepted_context(
    midi,
    valid,
    boundary,
    side,
    config,
):
    """
    Median accepted pitch within context_ms of a rejected episode.

    This is LONGER-RANGE CONTEXT.

    It is deliberately NOT the same thing as the immediate boundary
    transition.
    """

    n = len(midi)

    steps = max(
        1,
        int(
            round(
                config.context_ms
                * config.analysis_hz
                / 1000.0
            )
        ),
    )

    if side == "left":
        lo = max(
            0,
            boundary - steps,
        )

        hi = boundary

    elif side == "right":
        lo = boundary

        hi = min(
            n,
            boundary + steps,
        )

    else:
        raise ValueError(side)

    x = midi[
        lo:hi
    ]

    m = (
        valid[lo:hi]
        & np.isfinite(x)
    )

    return median_finite(
        x[m]
    )


def context_pitch_support(
    interval_st,
    config,
):
    """
    Soft support relative to a robust 120-ms median context.

    This is explicitly NOT boundary timing.

    We ask only whether the pitch difference would be plausible across
    roughly the context timescale.
    """

    return vocal_support(
        interval_st,
        config.context_ms,
    )


def context_return_support(
    left_context,
    right_context,
):
    """
    Evidence that accepted pitch before and after an episode occupies
    roughly the same pitch neighbourhood.

    Useful for excursion-and-return interpretation.
    """

    if (
        not np.isfinite(left_context)
        or not np.isfinite(right_context)
    ):
        return 0.0

    difference = abs(
        right_context
        - left_context
    )

    sigma_st = 2.0

    return float(
        np.exp(
            -0.5
            * (difference / sigma_st) ** 2
        )
    )


# =====================================================================
# Trajectory geometry
# =====================================================================


def trajectory_metrics(
    pitch,
    config,
):
    pitch = np.asarray(
        pitch,
        dtype=np.float64,
    )

    pitch = pitch[
        np.isfinite(pitch)
    ]

    if pitch.size == 0:
        return None

    if pitch.size == 1:
        return {
            "count": 1,
            "span": 0.0,
            "endpoint": 0.0,
            "path": 0.0,
            "max_step": 0.0,
            "median_step": 0.0,
            "efficiency": 1.0,
            "monotonic_fraction": 1.0,
            "direction_changes": 0,
            "return_ratio": 1.0,
        }

    d = np.diff(
        pitch
    )

    ad = np.abs(
        d
    )

    span = float(
        np.max(pitch)
        - np.min(pitch)
    )

    endpoint = float(
        pitch[-1]
        - pitch[0]
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

    efficiency = (
        abs(endpoint) / path
        if path > 1e-12
        else 1.0
    )

    return_ratio = (
        abs(endpoint) / span
        if span > 1e-12
        else 1.0
    )

    moving = d[
        np.abs(d)
        >= config.movement_epsilon_st
    ]

    if moving.size == 0:
        monotonic_fraction = 1.0
        direction_changes = 0

    else:
        if endpoint > 0:
            preferred = (
                moving > 0
            )

        elif endpoint < 0:
            preferred = (
                moving < 0
            )

        else:
            positives = int(
                np.sum(
                    moving > 0
                )
            )

            negatives = int(
                np.sum(
                    moving < 0
                )
            )

            preferred = (
                moving > 0
                if positives >= negatives
                else moving < 0
            )

        monotonic_fraction = float(
            np.mean(preferred)
        )

        signs = np.sign(
            moving
        )

        direction_changes = int(
            np.sum(
                signs[1:]
                != signs[:-1]
            )
        )

    return {
        "count": int(
            pitch.size
        ),

        "span": span,
        "endpoint": endpoint,
        "path": path,

        "max_step": max_step,
        "median_step": median_step,

        "efficiency": float(
            efficiency
        ),

        "monotonic_fraction": float(
            monotonic_fraction
        ),

        "direction_changes": int(
            direction_changes
        ),

        "return_ratio": float(
            return_ratio
        ),
    }


# =====================================================================
# Rejected episode segmentation
# =====================================================================


def coherent_subruns(
    f0,
    midi,
    start,
    end,
    config,
):
    """
    Split a rejected episode into locally coherent raw-F0 trajectories.

    Break on:

      * non-finite F0
      * broad vocal-range failure
      * adjacent movement > max_adjacent_step_st

    TOTAL pitch span remains unrestricted.
    """

    hard_ok = hard_pitch_ok(
        f0,
        config,
    )

    runs = []

    current_start = None
    previous_index = None

    for i in range(
        start,
        end,
    ):
        if (
            not hard_ok[i]
            or not np.isfinite(midi[i])
        ):
            if current_start is not None:
                runs.append(
                    (
                        current_start,
                        i,
                    )
                )

            current_start = None
            previous_index = None
            continue

        if current_start is None:
            current_start = i
            previous_index = i
            continue

        step = abs(
            midi[i]
            - midi[previous_index]
        )

        if (
            not np.isfinite(step)
            or step
            > config.max_adjacent_step_st
        ):
            runs.append(
                (
                    current_start,
                    i,
                )
            )

            current_start = i

        previous_index = i

    if current_start is not None:
        runs.append(
            (
                current_start,
                end,
            )
        )

    return runs


# =====================================================================
# Sub-run analysis
# =====================================================================


def analyse_subrun(
    midi,
    valid,
    episode_start,
    episode_end,
    subrun_start,
    subrun_end,
    episode_metrics,
    config,
):
    pitch = midi[
        subrun_start:subrun_end
    ]

    metrics = trajectory_metrics(
        pitch,
        config,
    )

    d_ms = duration_ms(
        subrun_start,
        subrun_end,
        config.analysis_hz,
    )

    first_pitch = float(
        pitch[0]
    )

    last_pitch = float(
        pitch[-1]
    )

    # -------------------------------------------------------------
    # Actual nearest accepted EDGE evidence.
    # -------------------------------------------------------------

    left_edge = edge_transition_left(
        midi=midi,
        valid=valid,
        subrun_start=subrun_start,
        config=config,
    )

    right_edge = edge_transition_right(
        midi=midi,
        valid=valid,
        subrun_end=subrun_end,
        config=config,
    )

    # -------------------------------------------------------------
    # Wider robust CONTEXT evidence.
    # -------------------------------------------------------------

    left_context = accepted_context(
        midi=midi,
        valid=valid,
        boundary=episode_start,
        side="left",
        config=config,
    )

    right_context = accepted_context(
        midi=midi,
        valid=valid,
        boundary=episode_end,
        side="right",
        config=config,
    )

    left_context_interval = (
        first_pitch
        - left_context
        if np.isfinite(left_context)
        else np.nan
    )

    right_context_interval = (
        right_context
        - last_pitch
        if np.isfinite(right_context)
        else np.nan
    )

    left_context_support = (
        context_pitch_support(
            left_context_interval,
            config,
        )
        if np.isfinite(
            left_context_interval
        )
        else 0.0
    )

    right_context_support = (
        context_pitch_support(
            right_context_interval,
            config,
        )
        if np.isfinite(
            right_context_interval
        )
        else 0.0
    )

    touches_left = (
        subrun_start
        == episode_start
    )

    touches_right = (
        subrun_end
        == episode_end
    )

    min_subrun_steps = max(
        1,
        int(
            np.ceil(
                config.min_subrun_ms
                * config.analysis_hz
                / 1000.0
                - 1e-9
            )
        ),
    )

    substantial = (
        (subrun_end - subrun_start)
        >= min_subrun_steps
    )

    episode_catastrophic = bool(
        episode_metrics is not None
        and episode_metrics[
            "max_step"
        ]
        >= config.catastrophic_step_st
    )

    return {
        "start": subrun_start,
        "end": subrun_end,

        "duration_ms": float(
            d_ms
        ),

        "metrics": metrics,

        "first_pitch": first_pitch,
        "last_pitch": last_pitch,

        "left_edge": left_edge,
        "right_edge": right_edge,

        "left_context": float(
            left_context
        ),

        "right_context": float(
            right_context
        ),

        "left_context_interval": float(
            left_context_interval
        ),

        "right_context_interval": float(
            right_context_interval
        ),

        "left_context_support": float(
            left_context_support
        ),

        "right_context_support": float(
            right_context_support
        ),

        "context_return_support": float(
            context_return_support(
                left_context,
                right_context,
            )
        ),

        "touches_left": bool(
            touches_left
        ),

        "touches_right": bool(
            touches_right
        ),

        "touches_trusted_edge": bool(
            touches_left
            or touches_right
        ),

        "substantial": bool(
            substantial
        ),

        "episode_catastrophic": bool(
            episode_catastrophic
        ),
    }


# =====================================================================
# Ornament evidence
# =====================================================================


def ornament_score(
    subrun,
    config,
):
    """
    Short excursion-and-return evidence.

    V3 REQUIREMENT:
    the coherent fragment must actually touch at least one edge of the
    rejected episode. A floating smooth fragment buried in catastrophic
    garbage is NOT promoted to an ornament.
    """

    metrics = subrun[
        "metrics"
    ]

    if metrics is None:
        return 0.0

    if not subrun[
        "touches_trusted_edge"
    ]:
        return 0.0

    if (
        subrun["duration_ms"]
        > config.ornament_max_ms
    ):
        return 0.0

    if (
        metrics["span"]
        < config.ornament_min_span_st
    ):
        return 0.0

    if (
        metrics["max_step"]
        > config.max_adjacent_step_st
    ):
        return 0.0

    # Strong evidence when the fragment makes a meaningful excursion
    # but ends near where it started.
    return_score = float(
        np.exp(
            -2.0
            * metrics["return_ratio"]
        )
    )

    context_return = subrun[
        "context_return_support"
    ]

    edge_support = max(
        subrun[
            "left_edge"
        ][
            "support"
        ],
        subrun[
            "right_edge"
        ][
            "support"
        ],
    )

    context_connection = max(
        subrun[
            "left_context_support"
        ],
        subrun[
            "right_context_support"
        ],
    )

    return float(
        0.55 * return_score
        + 0.20 * context_return
        + 0.15 * edge_support
        + 0.10 * context_connection
    )


# =====================================================================
# Classification
# =====================================================================


def classify_subrun(
    subrun,
    config,
):
    metrics = subrun[
        "metrics"
    ]

    if metrics is None:
        return (
            "NO_VALID_TRAJECTORY",
            0.0,
            0.0,
        )

    if not subrun[
        "substantial"
    ]:
        return (
            "TOO_SHORT_TO_INTERPRET",
            0.0,
            0.0,
        )

    orn = ornament_score(
        subrun,
        config,
    )

    if (
        orn
        >= config.ornament_score_min
    ):
        return (
            "POSSIBLE_ORNAMENT",
            orn,
            orn,
        )

    left_edge_support = subrun[
        "left_edge"
    ][
        "support"
    ]

    right_edge_support = subrun[
        "right_edge"
    ][
        "support"
    ]

    # -------------------------------------------------------------
    # BACKWARDS repair:
    #
    # sub-run reaches the rejected episode's right edge and connects
    # plausibly to the ACTUAL nearest accepted future sample.
    # -------------------------------------------------------------

    if (
        subrun["touches_right"]
        and right_edge_support
        >= config.rescue_edge_support_min
    ):
        confidence = float(
            0.65
            * right_edge_support
            + 0.20
            * subrun[
                "right_context_support"
            ]
            + 0.15
            * min(
                1.0,
                metrics[
                    "monotonic_fraction"
                ],
            )
        )

        return (
            "BACKWARD_RESCUE_CANDIDATE",
            confidence,
            orn,
        )

    # -------------------------------------------------------------
    # FORWARDS repair.
    # -------------------------------------------------------------

    if (
        subrun["touches_left"]
        and left_edge_support
        >= config.rescue_edge_support_min
    ):
        confidence = float(
            0.65
            * left_edge_support
            + 0.20
            * subrun[
                "left_context_support"
            ]
            + 0.15
            * min(
                1.0,
                metrics[
                    "monotonic_fraction"
                ],
            )
        )

        return (
            "FORWARD_RESCUE_CANDIDATE",
            confidence,
            orn,
        )

    return (
        "COHERENT_INTERNAL_ONLY",
        max(
            left_edge_support,
            right_edge_support,
        ),
        orn,
    )


# =====================================================================
# Episode analysis
# =====================================================================


def analyse_episode(
    f0,
    midi,
    valid,
    start,
    end,
    config,
):
    raw_pitch = midi[
        start:end
    ]

    finite_pitch = raw_pitch[
        np.isfinite(raw_pitch)
    ]

    episode_metrics = trajectory_metrics(
        finite_pitch,
        config,
    )

    ranges = coherent_subruns(
        f0=f0,
        midi=midi,
        start=start,
        end=end,
        config=config,
    )

    subruns = []

    for sub_start, sub_end in ranges:
        result = analyse_subrun(
            midi=midi,
            valid=valid,
            episode_start=start,
            episode_end=end,
            subrun_start=sub_start,
            subrun_end=sub_end,
            episode_metrics=episode_metrics,
            config=config,
        )

        (
            classification,
            confidence,
            orn,
        ) = classify_subrun(
            result,
            config,
        )

        result[
            "classification"
        ] = classification

        result[
            "confidence"
        ] = float(
            confidence
        )

        result[
            "ornament_score"
        ] = float(
            orn
        )

        subruns.append(
            result
        )

    return {
        "start": start,
        "end": end,

        "duration_ms": duration_ms(
            start,
            end,
            config.analysis_hz,
        ),

        "metrics": episode_metrics,

        "subruns": subruns,

        "catastrophic": bool(
            episode_metrics is not None
            and episode_metrics[
                "max_step"
            ]
            >= config.catastrophic_step_st
        ),
    }


# =====================================================================
# Pretty printing
# =====================================================================


def edge_text(edge):
    if edge[
        "accepted_index"
    ] is None:
        return "---"

    return (
        f"{edge['interval_st']:+.2f}st/"
        f"{edge['elapsed_ms']:.0f}ms/"
        f"{edge['support']:.2f}"
    )


def print_subrun(
    index,
    subrun,
    t,
):
    m = subrun[
        "metrics"
    ]

    if m is None:
        return

    edges = (
        (
            "L"
            if subrun["touches_left"]
            else "-"
        )
        +
        (
            "R"
            if subrun["touches_right"]
            else "-"
        )
    )

    print(
        f"{index:3d}  "
        f"{range_text(subrun['start'], subrun['end'], t):18s}  "
        f"{subrun['duration_ms']:5.0f}  "
        f"{subrun['classification']:28s}  "
        f"{subrun['confidence']:4.2f}  "
        f"{edges:>2s}  "
        f"{m['span']:5.2f}  "
        f"{m['endpoint']:6.2f}  "
        f"{m['max_step']:7.2f}  "
        f"{m['return_ratio']:6.2f}  "
        f"{edge_text(subrun['left_edge']):>18s}  "
        f"{edge_text(subrun['right_edge']):>18s}  "
        f"{subrun['left_context_support']:5.2f}  "
        f"{subrun['right_context_support']:5.2f}  "
        f"{subrun['context_return_support']:5.2f}  "
        f"{subrun['ornament_score']:5.2f}"
    )


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="100 Hz ECKF CSV",
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

    analysis_hz = float(
        1.0
        / np.median(
            np.diff(t)
        )
    )

    config = BoundaryDiagnosticConfig(
        analysis_hz=analysis_hz,
    )

    validity_config = PitchValidityConfig(
        analysis_hz=analysis_hz,
        min_vocal_hz=config.min_vocal_hz,
    )

    validity = analyse_pitch_validity(
        f0_hz=f0,
        sample_rate=analysis_hz,
        config=validity_config,
    )

    offline_validity = apply_offline_validity_correction(
        f0_hz=f0,
        first_pass=validity,
        sample_rate=analysis_hz,
    )

    valid = np.asarray(
        validity.valid,
        dtype=bool,
    )

    midi = hz_to_midi(
        f0
    )

    rejected_episodes = true_runs(
        ~valid
    )

    episodes = [
        analyse_episode(
            f0=f0,
            midi=midi,
            valid=valid,
            start=start,
            end=end,
            config=config,
        )
        for start, end
        in rejected_episodes
    ]

    all_subruns = [
        subrun
        for episode in episodes
        for subrun in episode[
            "subruns"
        ]
    ]

    counts = Counter(
        subrun[
            "classification"
        ]
        for subrun
        in all_subruns
    )

    # =============================================================
    # Summary
    # =============================================================

    print()
    print("=" * 220)
    print(
        "SUB-RUN / BACKWARDS-BOUNDARY DIAGNOSTIC V3 — "
        "ACTUAL EDGE TIMING + ROBUST CONTEXT"
    )
    print("=" * 220)

    print(
        f"Rows:                    {len(t)}"
    )

    print(
        f"Analysis rate:           {analysis_hz:.3f} Hz"
    )

    print(
        f"Accepted rows:           {np.sum(valid)} / {len(valid)}"
    )

    print(
        f"Rejected episodes:       {len(episodes)}"
    )

    print(
        f"Coherent sub-runs:       {len(all_subruns)}"
    )

    print()

    print(
        "SUB-RUN CLASSIFICATION COUNTS"
    )

    for name, count in sorted(
        counts.items()
    ):
        print(
            f"  {name:30s} {count}"
        )

    # =============================================================
    # Ranked coherent sub-runs
    # =============================================================

    priority = {
        "BACKWARD_RESCUE_CANDIDATE": 5,
        "POSSIBLE_ORNAMENT": 4,
        "FORWARD_RESCUE_CANDIDATE": 3,
        "COHERENT_INTERNAL_ONLY": 2,
        "TOO_SHORT_TO_INTERPRET": 1,
        "NO_VALID_TRAJECTORY": 0,
    }

    ranked = sorted(
        all_subruns,
        key=lambda r: (
            priority.get(
                r["classification"],
                0,
            ),
            r["confidence"],
            r["duration_ms"],
        ),
        reverse=True,
    )

    print()
    print("=" * 220)
    print(
        "RANKED COHERENT SUB-RUNS"
    )
    print("=" * 220)

    print(
        " #   sub-run             dur   class                         "
        "conf  ED  span   endpt  maxstep  return  "
        "LEFT EDGE            RIGHT EDGE           "
        "Lctx  Rctx  ctxR   orn"
    )

    print(
        "---  ------------------  -----  ---------------------------- "
        "----  --  -----  ------  -------  ------  "
        "------------------  ------------------  "
        "----- ----- ----- -----"
    )

    for index, subrun in enumerate(
        ranked,
        start=1,
    ):
        print_subrun(
            index,
            subrun,
            t,
        )

    # =============================================================
    # Episode / sub-run detail
    # =============================================================

    print()
    print("=" * 220)
    print(
        "EPISODE / SUB-RUN DETAIL"
    )
    print("=" * 220)

    for episode in episodes:
        if not episode[
            "subruns"
        ]:
            continue

        start = episode[
            "start"
        ]

        end = episode[
            "end"
        ]

        em = episode[
            "metrics"
        ]

        print()
        print(
            f"EPISODE "
            f"{range_text(start, end, t)}  "
            f"{episode['duration_ms']:.0f} ms"
        )

        if em is None:
            print(
                "  raw episode: no finite pitch"
            )

        else:
            print(
                f"  raw episode: "
                f"span={em['span']:.2f} "
                f"endpoint={em['endpoint']:.2f} "
                f"maxstep={em['max_step']:.2f} "
                f"catastrophic="
                f"{episode['catastrophic']}"
            )

        for subrun in episode[
            "subruns"
        ]:
            sm = subrun[
                "metrics"
            ]

            print(
                f"  SUBRUN "
                f"{range_text(subrun['start'], subrun['end'], t)} "
                f"[{subrun['duration_ms']:.0f} ms]"
            )

            print(
                f"    class:         "
                f"{subrun['classification']} "
                f"conf={subrun['confidence']:.2f}"
            )

            print(
                f"    shape:         "
                f"span={sm['span']:.2f} "
                f"endpoint={sm['endpoint']:.2f} "
                f"maxstep={sm['max_step']:.2f} "
                f"return={sm['return_ratio']:.2f} "
                f"mono={sm['monotonic_fraction']:.2f}"
            )

            print(
                f"    left edge:     "
                f"{edge_text(subrun['left_edge'])}"
            )

            print(
                f"    right edge:    "
                f"{edge_text(subrun['right_edge'])}"
            )

            print(
                f"    left context:  "
                f"pitch={fmt(subrun['left_context'])} "
                f"interval={fmt(subrun['left_context_interval'])} "
                f"support={subrun['left_context_support']:.2f}"
            )

            print(
                f"    right context: "
                f"pitch={fmt(subrun['right_context'])} "
                f"interval={fmt(subrun['right_context_interval'])} "
                f"support={subrun['right_context_support']:.2f}"
            )

            print(
                f"    ctx return:    "
                f"{subrun['context_return_support']:.2f}"
            )

            print(
                f"    ornament:      "
                f"{subrun['ornament_score']:.2f}"
            )

            print(
                f"    edges:         "
                f"touch-left="
                f"{subrun['touches_left']} "
                f"touch-right="
                f"{subrun['touches_right']} "
                f"touch-any="
                f"{subrun['touches_trusted_edge']}"
            )

    # =============================================================
    # Regression windows
    # =============================================================

    regressions = [
        (
            "hidden coherent suffix / backwards rescue",
            8.50,
            9.20,
        ),
        (
            "short leave-and-return / ornament",
            22.85,
            23.10,
        ),
        (
            "suspicious rescue candidate around 25.14",
            24.95,
            25.30,
        ),
        (
            "suspicious rescue candidate around 26.9",
            26.75,
            27.05,
        ),
        (
            "catastrophic episode with fake ornaments",
            36.70,
            37.95,
        ),
        (
            "djwww / noise collapse",
            40.85,
            41.95,
        ),
        (
            "extreme tracker explosion",
            73.80,
            74.20,
        ),
    ]

    print()
    print("=" * 220)
    print(
        "REGRESSION WINDOWS"
    )
    print("=" * 220)

    for label, lo, hi in regressions:
        print()
        print(
            f"{label}: "
            f"{lo:.3f}-{hi:.3f}s"
        )

        found = False

        for episode in episodes:
            episode_start_time = float(
                t[
                    episode["start"]
                ]
            )

            episode_end_time = float(
                t[
                    episode["end"] - 1
                ]
            )

            if (
                episode_end_time < lo
                or episode_start_time > hi
            ):
                continue

            found = True

            print(
                f"  EPISODE "
                f"{episode_start_time:.3f}-"
                f"{episode_end_time:.3f}s "
                f"catastrophic="
                f"{episode['catastrophic']}"
            )

            for subrun in episode[
                "subruns"
            ]:
                sm = subrun[
                    "metrics"
                ]

                le = subrun[
                    "left_edge"
                ]

                re = subrun[
                    "right_edge"
                ]

                print(
                    f"    "
                    f"{range_text(subrun['start'], subrun['end'], t)}  "
                    f"{subrun['classification']:28s} "
                    f"dur={subrun['duration_ms']:.0f} "
                    f"span={sm['span']:.2f} "
                    f"endpoint={sm['endpoint']:.2f} "
                    f"maxstep={sm['max_step']:.2f} "
                    f"Ledge="
                    f"{fmt(le['interval_st'])}/"
                    f"{fmt(le['elapsed_ms'], 0)}/"
                    f"{le['support']:.2f} "
                    f"Redge="
                    f"{fmt(re['interval_st'])}/"
                    f"{fmt(re['elapsed_ms'], 0)}/"
                    f"{re['support']:.2f} "
                    f"Lctx={subrun['left_context_support']:.2f} "
                    f"Rctx={subrun['right_context_support']:.2f} "
                    f"orn={subrun['ornament_score']:.2f} "
                    f"edges="
                    f"{'L' if subrun['touches_left'] else '-'}"
                    f"{'R' if subrun['touches_right'] else '-'}"
                )

        if not found:
            print(
                "  no rejected episode overlaps window"
            )

    print()
    print("=" * 220)
    print(
        "IMPORTANT: OBSERVATIONAL ONLY — "
        "NO F0 OR VALIDITY SAMPLES WERE MODIFIED."
    )
    print("=" * 220)
    print()
    print("=" * 120)
    print("OFFLINE CORRECTION PASS — PROMOTED REGIONS")
    print("=" * 120)

    interesting = {
        "BACKWARD_RESCUED",
        "FORWARD_RESCUED",
        "ORNAMENT_RESCUED",
    }

    reasons = np.asarray(
        offline_validity.correction_reason,
        dtype=object,
    )

    for label in (
        "BACKWARD_RESCUED",
        "FORWARD_RESCUED",
        "ORNAMENT_RESCUED",
    ):
        mask = reasons == label

        runs = true_runs(mask)

        print()
        print(
            f"{label}: "
            f"{int(np.sum(mask))} rows, "
            f"{len(runs)} runs"
        )

        for start, end in runs:
            print(
                f"  {t[start]:.3f}-"
                f"{t[end - 1]:.3f}s "
                f"[{end - start} rows]"
            )
    lost_original_valid = (
        validity.valid
        & ~offline_validity.valid
    )

    if np.any(lost_original_valid):
        raise RuntimeError(
            "Offline correction invalidated samples "
            "that were valid in the first pass."
        )
    newly_valid = (
        offline_validity.valid
        & ~validity.valid
    )

    print()
    print(
        f"Newly valid rows: "
        f"{int(np.sum(newly_valid))}"
    )

if __name__ == "__main__":
    main()