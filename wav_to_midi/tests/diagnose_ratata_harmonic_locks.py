#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from python_eckf.validity import (
    PitchValidityConfig,
    analyse_pitch_validity,
)


NOTE_NAMES = [
    "C", "C#", "D", "D#",
    "E", "F", "F#", "G",
    "G#", "A", "A#", "B",
]


# =====================================================================
# Pitch helpers
# =====================================================================


def hz_to_midi(f0_hz):
    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    out = np.full_like(
        f0_hz,
        np.nan,
        dtype=np.float64,
    )

    good = (
        np.isfinite(f0_hz)
        & (f0_hz > 0.0)
    )

    out[good] = (
        69.0
        + 12.0
        * np.log2(
            f0_hz[good] / 440.0
        )
    )

    return out


def midi_name(value):
    if not np.isfinite(value):
        return "---"

    note = int(round(value))

    octave = (
        note // 12
        - 1
    )

    return (
        f"{NOTE_NAMES[note % 12]}"
        f"{octave}"
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


# =====================================================================
# Runs
# =====================================================================


def true_runs(mask):
    """
    Return contiguous True runs as (start, end), end exclusive.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    runs = []

    i = 0

    while i < len(mask):

        if not mask[i]:
            i += 1
            continue

        start = i

        while (
            i < len(mask)
            and mask[i]
        ):
            i += 1

        runs.append(
            (start, i)
        )

    return runs


# =====================================================================
# Vocal interval / duration prior
# =====================================================================


def rapid_interval_penalty(
    interval_semitones: float,
    duration_ms: float,
) -> float:
    """
    Soft HUMAN-VOCAL plausibility penalty.

    Important musical rule:

        THIRDS AND BELOW (<= 4 semitones)
        receive essentially NO penalty as ordinary rapid
        melodic movement.

    Above a third, increasing interval size requires increasing time.

    This is NOT a validity decision.
    It is a diagnostic score.

    Rough interpretation:

        0.0   ordinary / unsurprising
        0.0-1 mildly demanding
        1-2   suspicious
        2-4   strongly suspicious
        >4    extremely implausible as an instantaneous target change

    Continuous glides are treated separately through trajectory
    coherence, so a large total excursion is not automatically bad.
    """

    interval = abs(
        float(interval_semitones)
    )

    duration_ms = max(
        float(duration_ms),
        1.0,
    )

    # -------------------------------------------------------------
    # Thirds and below:
    # essentially no penalty, even rapidly.
    # -------------------------------------------------------------

    if interval <= 4.0:
        return 0.0

    # -------------------------------------------------------------
    # Above a third:
    #
    # We construct a smoothly increasing "minimum comfortable time".
    #
    #  5 st  ~ fourth
    #  7 st  ~ fifth
    #  9 st  ~ sixth
    # 12 st  ~ octave
    #
    # These are not hard human limits.
    # They are priors for suspicious ECKF jumps.
    # -------------------------------------------------------------

    excess = interval - 4.0

    required_ms = (
        18.0
        * (excess ** 1.35)
    )

    if duration_ms >= required_ms:
        return 0.0

    ratio = (
        required_ms
        / duration_ms
    )

    # Start gently around fourth/fifth territory;
    # escalate much faster for sixths/octaves at tiny durations.
    size_weight = (
        0.35
        + 0.12 * excess
    )

    penalty = (
        max(0.0, ratio - 1.0)
        * size_weight
    )

    return float(
        penalty
    )


def interval_prior_label(penalty):
    if penalty < 0.25:
        return "PLAUSIBLE"

    if penalty < 1.0:
        return "MILD"

    if penalty < 2.0:
        return "SUSPICIOUS"

    if penalty < 4.0:
        return "STRONG"

    return "EXTREME"


# =====================================================================
# Harmonic relationship
# =====================================================================


@dataclass
class HarmonicMatch:
    name: str
    target_semitones: float
    error_semitones: float


HARMONIC_RELATIONS = [
    ("-2 OCT", -24.0),
    ("-3:1", -19.01955),
    ("-1 OCT", -12.0),

    ("+1 OCT", +12.0),
    ("+3:1", +19.01955),
    ("+2 OCT", +24.0),
]


def nearest_harmonic_relation(
    interval_semitones: float,
) -> HarmonicMatch:

    interval = float(
        interval_semitones
    )

    best_name = "NONE"
    best_target = np.nan
    best_error = np.inf

    for name, target in HARMONIC_RELATIONS:

        error = abs(
            interval - target
        )

        if error < best_error:
            best_error = error
            best_name = name
            best_target = target

    return HarmonicMatch(
        name=best_name,
        target_semitones=float(best_target),
        error_semitones=float(best_error),
    )


# =====================================================================
# Trajectory metrics
# =====================================================================


@dataclass
class TrajectoryMetrics:
    span_semitones: float
    endpoint_change: float

    max_adjacent_step: float
    max_adjacent_step_index: int

    path_length: float
    direct_distance: float

    efficiency: float
    monotonic_fraction: float

    max_velocity_st_per_s: float
    median_velocity_st_per_s: float

    max_acceleration_st_per_s2: float


def trajectory_metrics(
    pitches,
    start,
    end,
    analysis_hz,
):
    """
    Metrics for [start, end), assuming finite pitch rows.
    """

    p = np.asarray(
        pitches[start:end],
        dtype=np.float64,
    )

    p = p[
        np.isfinite(p)
    ]

    if p.size < 2:
        return None

    diffs = np.diff(p)

    abs_diffs = np.abs(
        diffs
    )

    span = float(
        np.max(p)
        - np.min(p)
    )

    endpoint = float(
        p[-1]
        - p[0]
    )

    max_step_pos = int(
        np.argmax(abs_diffs)
    )

    max_step = float(
        abs_diffs[max_step_pos]
    )

    path_length = float(
        np.sum(abs_diffs)
    )

    direct = abs(
        endpoint
    )

    if path_length > 0:
        efficiency = (
            direct
            / path_length
        )
    else:
        efficiency = 1.0

    # -------------------------------------------------------------
    # Monotonicity:
    # fraction of non-zero motion agreeing with net direction.
    # -------------------------------------------------------------

    nonzero = diffs[
        np.abs(diffs) > 1e-9
    ]

    if nonzero.size == 0:
        monotonic_fraction = 1.0

    elif endpoint > 0:
        monotonic_fraction = float(
            np.mean(
                nonzero > 0
            )
        )

    elif endpoint < 0:
        monotonic_fraction = float(
            np.mean(
                nonzero < 0
            )
        )

    else:
        monotonic_fraction = 0.5

    velocities = (
        diffs
        * analysis_hz
    )

    abs_velocities = np.abs(
        velocities
    )

    max_velocity = float(
        np.max(abs_velocities)
    )

    median_velocity = float(
        np.median(abs_velocities)
    )

    if velocities.size >= 2:

        accelerations = (
            np.diff(velocities)
            * analysis_hz
        )

        max_acceleration = float(
            np.max(
                np.abs(accelerations)
            )
        )

    else:
        max_acceleration = 0.0

    return TrajectoryMetrics(
        span_semitones=span,
        endpoint_change=endpoint,

        max_adjacent_step=max_step,
        max_adjacent_step_index=(
            start
            + max_step_pos
        ),

        path_length=path_length,
        direct_distance=direct,

        efficiency=float(
            efficiency
        ),

        monotonic_fraction=float(
            monotonic_fraction
        ),

        max_velocity_st_per_s=max_velocity,
        median_velocity_st_per_s=median_velocity,

        max_acceleration_st_per_s2=max_acceleration,
    )


# =====================================================================
# Context
# =====================================================================


def context_pitch(
    midi,
    valid,
    start,
    end,
    side,
    context_steps,
):
    """
    Median trusted pitch immediately before or after a run.
    """

    n = len(midi)

    if side == "before":

        lo = max(
            0,
            start - context_steps,
        )

        hi = start

    elif side == "after":

        lo = end

        hi = min(
            n,
            end + context_steps,
        )

    else:
        raise ValueError(
            side
        )

    mask = (
        valid[lo:hi]
        & np.isfinite(
            midi[lo:hi]
        )
    )

    values = midi[
        lo:hi
    ][mask]

    return median_finite(
        values
    )


# =====================================================================
# Candidate run
# =====================================================================


@dataclass
class Candidate:
    kind: str

    start: int
    end: int

    duration_ms: float

    median_pitch: float

    before_pitch: float
    after_pitch: float

    before_interval: float
    after_interval: float

    before_harmonic: HarmonicMatch
    after_harmonic: HarmonicMatch

    before_penalty: float
    after_penalty: float

    metrics: TrajectoryMetrics

    suspicion_score: float


def analyse_run(
    kind,
    start,
    end,
    midi,
    valid,
    analysis_hz,
    context_steps,
):
    values = midi[
        start:end
    ]

    median_pitch = median_finite(
        values
    )

    before = context_pitch(
        midi=midi,
        valid=valid,
        start=start,
        end=end,
        side="before",
        context_steps=context_steps,
    )

    after = context_pitch(
        midi=midi,
        valid=valid,
        start=start,
        end=end,
        side="after",
        context_steps=context_steps,
    )

    duration_ms = (
        (end - start)
        / analysis_hz
        * 1000.0
    )

    metrics = trajectory_metrics(
        pitches=midi,
        start=start,
        end=end,
        analysis_hz=analysis_hz,
    )

    if metrics is None:
        return None

    if np.isfinite(before):

        before_interval = (
            median_pitch
            - before
        )

        before_harmonic = (
            nearest_harmonic_relation(
                before_interval
            )
        )

        before_penalty = (
            rapid_interval_penalty(
                before_interval,
                duration_ms,
            )
        )

    else:

        before_interval = np.nan

        before_harmonic = HarmonicMatch(
            "NONE",
            np.nan,
            np.inf,
        )

        before_penalty = 0.0

    if np.isfinite(after):

        after_interval = (
            median_pitch
            - after
        )

        after_harmonic = (
            nearest_harmonic_relation(
                after_interval
            )
        )

        after_penalty = (
            rapid_interval_penalty(
                after_interval,
                duration_ms,
            )
        )

    else:

        after_interval = np.nan

        after_harmonic = HarmonicMatch(
            "NONE",
            np.nan,
            np.inf,
        )

        after_penalty = 0.0

    # -------------------------------------------------------------
    # Suspicion score
    #
    # Diagnostic ranking only.
    #
    # Components:
    #
    # 1. Fast implausible context interval.
    # 2. Extremely abrupt local step.
    # 3. Near-octave / harmonic relationship to surrounding
    #    trusted pitch.
    #
    # Smooth long glides should naturally score much lower.
    # -------------------------------------------------------------

    suspicion = (
        max(
            before_penalty,
            after_penalty,
        )
    )

    # Local jump contribution.
    #
    # Thirds and below: no penalty.
    # Evaluate one 10-ms step at its actual duration.

    step_ms = (
        1000.0
        / analysis_hz
    )

    local_step_penalty = (
        rapid_interval_penalty(
            metrics.max_adjacent_step,
            step_ms,
        )
    )

    suspicion += (
        local_step_penalty
    )

    # Harmonic-lock evidence.
    #
    # Only count relationships close to the canonical harmonic
    # offsets. Do not hard-reject on this alone.

    harmonic_bonus = 0.0

    for match in (
        before_harmonic,
        after_harmonic,
    ):

        if (
            np.isfinite(
                match.error_semitones
            )
            and match.error_semitones
            <= 0.75
        ):
            harmonic_bonus += 1.5

        elif (
            np.isfinite(
                match.error_semitones
            )
            and match.error_semitones
            <= 1.5
        ):
            harmonic_bonus += 0.5

    suspicion += harmonic_bonus

    return Candidate(
        kind=kind,

        start=start,
        end=end,

        duration_ms=float(
            duration_ms
        ),

        median_pitch=float(
            median_pitch
        ),

        before_pitch=float(
            before
        ),

        after_pitch=float(
            after
        ),

        before_interval=float(
            before_interval
        ),

        after_interval=float(
            after_interval
        ),

        before_harmonic=before_harmonic,
        after_harmonic=after_harmonic,

        before_penalty=float(
            before_penalty
        ),

        after_penalty=float(
            after_penalty
        ),

        metrics=metrics,

        suspicion_score=float(
            suspicion
        ),
    )


# =====================================================================
# Printing
# =====================================================================


def fmt_pitch(value):
    if not np.isfinite(value):
        return "---"

    return (
        f"{value:6.2f} "
        f"{midi_name(value)}"
    )


def fmt_interval(value):
    if not np.isfinite(value):
        return "---"

    return f"{value:+6.2f} st"


def print_context_rows(
    t,
    f0,
    midi,
    valid,
    reason,
    start,
    end,
    context_steps,
):

    lo = max(
        0,
        start - context_steps,
    )

    hi = min(
        len(t),
        end + context_steps,
    )

    print()
    print(
        "    time_s      f0_hz    midi     note   "
        "valid   reason"
    )

    print(
        "    --------   --------   -------   ----   "
        "-----   ----------------------------"
    )

    for i in range(
        lo,
        hi,
    ):

        if start <= i < end:
            marker = ">>"
        else:
            marker = "  "

        print(
            f"{marker}  "
            f"{t[i]:8.3f}   "
            f"{f0[i]:8.2f}   "
            f"{midi[i]:7.2f}   "
            f"{midi_name(midi[i]):4s}   "
            f"{str(bool(valid[i])):5s}   "
            f"{reason[i]}"
        )


def print_candidate(
    rank,
    candidate,
    t,
    f0,
    midi,
    valid,
    reason,
    context_steps,
):

    c = candidate
    m = c.metrics

    print()
    print("-" * 112)

    print(
        f"{c.kind} #{rank:02d}"
    )

    print(
        f"Run:                    "
        f"{t[c.start]:.3f} .. "
        f"{t[c.end - 1]:.3f} s"
    )

    print(
        f"Duration:               "
        f"{c.duration_ms:.1f} ms"
    )

    print(
        f"Median run pitch:       "
        f"{fmt_pitch(c.median_pitch)}"
    )

    print(
        f"Trusted pitch BEFORE:   "
        f"{fmt_pitch(c.before_pitch)}"
    )

    print(
        f"Trusted pitch AFTER:    "
        f"{fmt_pitch(c.after_pitch)}"
    )

    print()
    print(
        f"Interval from BEFORE:   "
        f"{fmt_interval(c.before_interval)}"
    )

    print(
        f"Interval from AFTER:    "
        f"{fmt_interval(c.after_interval)}"
    )

    print(
        f"Vocal prior BEFORE:     "
        f"{c.before_penalty:.3f}  "
        f"{interval_prior_label(c.before_penalty)}"
    )

    print(
        f"Vocal prior AFTER:      "
        f"{c.after_penalty:.3f}  "
        f"{interval_prior_label(c.after_penalty)}"
    )

    print()
    print(
        f"Nearest harmonic BEFORE:"
        f" {c.before_harmonic.name:8s} "
        f"error={c.before_harmonic.error_semitones:.2f} st"
    )

    print(
        f"Nearest harmonic AFTER: "
        f" {c.after_harmonic.name:8s} "
        f"error={c.after_harmonic.error_semitones:.2f} st"
    )

    print()
    print(
        f"Run span:               "
        f"{m.span_semitones:.2f} st"
    )

    print(
        f"Endpoint change:        "
        f"{m.endpoint_change:+.2f} st"
    )

    print(
        f"Path length:            "
        f"{m.path_length:.2f} st"
    )

    print(
        f"Direct/path efficiency: "
        f"{m.efficiency:.3f}"
    )

    print(
        f"Monotonic fraction:     "
        f"{m.monotonic_fraction:.3f}"
    )

    print(
        f"Maximum adjacent step:  "
        f"{m.max_adjacent_step:.2f} st / "
        f"{1000.0 / (1.0 / np.median(np.diff(t))):.1f} ms"
    )

    print(
        f"Max velocity:           "
        f"{m.max_velocity_st_per_s:.1f} st/s"
    )

    print(
        f"Median velocity:        "
        f"{m.median_velocity_st_per_s:.1f} st/s"
    )

    print(
        f"Max acceleration:       "
        f"{m.max_acceleration_st_per_s2:.1f} st/s²"
    )

    print()
    print(
        f"DIAGNOSTIC SUSPICION:   "
        f"{c.suspicion_score:.3f}"
    )

    print_context_rows(
        t=t,
        f0=f0,
        midi=midi,
        valid=valid,
        reason=reason,
        start=c.start,
        end=c.end,
        context_steps=context_steps,
    )


# =====================================================================
# Main
# =====================================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect possible ECKF harmonic/subharmonic locks and "
            "human-vocal interval plausibility."
        )
    )

    parser.add_argument(
        "csv",
        help="100 Hz ECKF CSV",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--context-ms",
        type=float,
        default=200.0,
        help=(
            "trusted context searched before and after each run"
        ),
    )

    parser.add_argument(
        "--print-context-ms",
        type=float,
        default=80.0,
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

    dt = float(
        np.median(
            np.diff(t)
        )
    )

    analysis_hz = (
        1.0 / dt
    )

    midi = hz_to_midi(
        f0
    )

    config = PitchValidityConfig(
        analysis_hz=analysis_hz,
        min_vocal_hz=120.0,
    )

    result = analyse_pitch_validity(
        f0_hz=f0,
        sample_rate=analysis_hz,
        config=config,
    )

    valid = np.asarray(
        result.valid,
        dtype=bool,
    )

    reason = np.asarray(
        result.reason,
        dtype=object,
    )

    context_steps = max(
        1,
        int(
            round(
                args.context_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    print_context_steps = max(
        1,
        int(
            round(
                args.print_context_ms
                * analysis_hz
                / 1000.0
            )
        ),
    )

    # -------------------------------------------------------------
    # Accepted runs
    # -------------------------------------------------------------

    accepted_mask = (
        valid
        & np.isfinite(midi)
    )

    accepted_runs = true_runs(
        accepted_mask
    )

    accepted = []

    for start, end in accepted_runs:

        candidate = analyse_run(
            kind="ACCEPTED",
            start=start,
            end=end,
            midi=midi,
            valid=valid,
            analysis_hz=analysis_hz,
            context_steps=context_steps,
        )

        if candidate is not None:
            accepted.append(
                candidate
            )

    # -------------------------------------------------------------
    # Rejected but still nominally in vocal range.
    # -------------------------------------------------------------

    rejected_mask = (
        (~valid)
        & np.isfinite(f0)
        & (f0 >= config.min_vocal_hz)
        & (f0 <= config.max_vocal_hz)
    )

    rejected_runs = true_runs(
        rejected_mask
    )

    rejected = []

    for start, end in rejected_runs:

        candidate = analyse_run(
            kind="REJECTED",
            start=start,
            end=end,
            midi=midi,
            valid=valid,
            analysis_hz=analysis_hz,
            context_steps=context_steps,
        )

        if candidate is not None:
            rejected.append(
                candidate
            )

    # -------------------------------------------------------------
    # Rank by suspicion, not merely interval size.
    # -------------------------------------------------------------

    accepted.sort(
        key=lambda x: x.suspicion_score,
        reverse=True,
    )

    rejected.sort(
        key=lambda x: x.suspicion_score,
        reverse=True,
    )

    accepted = accepted[
        :args.top
    ]

    rejected = rejected[
        :args.top
    ]

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print()
    print("=" * 112)
    print(
        "RATATA HARMONIC-LOCK / HUMAN-VOCAL PLAUSIBILITY DIAGNOSTIC"
    )
    print("=" * 112)

    print()
    print(
        f"Rows:                    {len(t)}"
    )

    print(
        f"Analysis rate:           {analysis_hz:.3f} Hz"
    )

    print(
        f"Accepted runs:           {len(accepted_runs)}"
    )

    print(
        f"Rejected vocal runs:     {len(rejected_runs)}"
    )

    print(
        f"Trusted context:         ±{args.context_ms:.1f} ms"
    )

    print()

    print(
        "VOCAL PRIOR:"
    )

    print(
        "  thirds and below (<=4 st): essentially no rapid-change penalty"
    )

    print(
        "  fourth/fifth: increasingly demanding at very short durations"
    )

    print(
        "  sixth/seventh: strongly suspicious when nearly instantaneous"
    )

    print(
        "  octave+: extremely suspicious as an instantaneous target jump"
    )

    print(
        "  large SMOOTH trajectories are not rejected merely for total span"
    )

    # -------------------------------------------------------------
    # Accepted suspicious runs
    # -------------------------------------------------------------

    print()
    print("=" * 112)
    print(
        "A. MOST SUSPICIOUS CURRENTLY ACCEPTED RUNS"
    )
    print("=" * 112)

    for rank, candidate in enumerate(
        accepted,
        start=1,
    ):

        print_candidate(
            rank=rank,
            candidate=candidate,
            t=t,
            f0=f0,
            midi=midi,
            valid=valid,
            reason=reason,
            context_steps=print_context_steps,
        )

    # -------------------------------------------------------------
    # Rejected interesting runs
    # -------------------------------------------------------------

    print()
    print("=" * 112)
    print(
        "B. MOST SUSPICIOUS / INTERESTING CURRENTLY REJECTED RUNS"
    )
    print("=" * 112)

    for rank, candidate in enumerate(
        rejected,
        start=1,
    ):

        print_candidate(
            rank=rank,
            candidate=candidate,
            t=t,
            f0=f0,
            midi=midi,
            valid=valid,
            reason=reason,
            context_steps=print_context_steps,
        )

    print()
    print("=" * 112)


if __name__ == "__main__":
    main()
    