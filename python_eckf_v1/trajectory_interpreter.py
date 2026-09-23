from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class RegionKind(Enum):
    STABLE_TARGET = auto()
    TRANSITION = auto()
    UNRESOLVED = auto()


@dataclass(frozen=True)
class TrajectoryInterpreterConfig:
    analysis_hz: float = 100.0

    # Structural smoothing only.
    # Raw corrected F0 is never replaced by this.
    smoothing_ms: float = 50.0

    # A stable target is detected from a local window.
    stability_window_ms: float = 120.0

    # Maximum robust pitch spread inside that window.
    stable_range_st: float = 0.65

    # Minimum duration of a stable target region.
    min_stable_ms: float = 80.0

    # Adjacent stable regions whose centers are essentially the same
    # target may be merged across a very short interruption.
    same_target_tolerance_st: float = 0.45
    merge_gap_ms: float = 60.0


@dataclass(frozen=True)
class PitchRegion:
    index: int

    start_index: int
    end_index: int

    start_time_s: float
    end_time_s: float
    duration_ms: float

    kind: RegionKind

    median_pitch_st: float
    min_pitch_st: float
    max_pitch_st: float
    span_st: float

    raw_median_pitch_st: float

    previous_target_st: float
    next_target_st: float


@dataclass(frozen=True)
class TrajectoryInterpretation:
    regions: tuple[PitchRegion, ...]

    pitch_st: np.ndarray
    structural_pitch_st: np.ndarray

    region_id: np.ndarray
    stable_mask: np.ndarray

    analysis_hz: float


def hz_to_semitones(
    f0_hz: np.ndarray,
) -> np.ndarray:

    f0 = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    out = np.full(
        f0.shape,
        np.nan,
        dtype=np.float64,
    )

    good = (
        np.isfinite(f0)
        & (f0 > 0.0)
    )

    out[good] = (
        69.0
        + 12.0
        * np.log2(
            f0[good] / 440.0
        )
    )

    return out


def _true_runs(
    mask: np.ndarray,
) -> list[tuple[int, int]]:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if len(mask) == 0:
        return []

    padded = np.concatenate(
        (
            [False],
            mask,
            [False],
        )
    )

    d = np.diff(
        padded.astype(np.int8)
    )

    starts = np.flatnonzero(
        d == 1
    )

    ends = np.flatnonzero(
        d == -1
    )

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def _odd_steps(
    milliseconds: float,
    analysis_hz: float,
    minimum: int = 1,
) -> int:

    n = max(
        minimum,
        int(
            round(
                milliseconds
                * analysis_hz
                / 1000.0
            )
        ),
    )

    if n % 2 == 0:
        n += 1

    return n


def _median_filter_valid_run(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Median smoothing inside ONE already-valid run.

    No values are borrowed across invalid boundaries.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n == 0:
        return values.copy()

    radius = window // 2

    out = np.empty_like(values)

    for i in range(n):
        lo = max(
            0,
            i - radius,
        )

        hi = min(
            n,
            i + radius + 1,
        )

        out[i] = np.median(
            values[lo:hi]
        )

    return out


def _make_structural_pitch(
    pitch_st: np.ndarray,
    valid: np.ndarray,
    config: TrajectoryInterpreterConfig,
) -> np.ndarray:
    """
    Create a lightly smoothed representation used ONLY for structural
    segmentation.

    The original corrected trajectory remains untouched.
    """

    out = np.full_like(
        pitch_st,
        np.nan,
    )

    window = _odd_steps(
        config.smoothing_ms,
        config.analysis_hz,
        minimum=3,
    )

    for start, end in _true_runs(valid):

        out[start:end] = (
            _median_filter_valid_run(
                pitch_st[start:end],
                window,
            )
        )

    return out


def _local_stability_mask(
    structural_pitch: np.ndarray,
    valid: np.ndarray,
    config: TrajectoryInterpreterConfig,
) -> np.ndarray:
    """
    Mark samples whose surrounding structural trajectory occupies a
    sufficiently narrow pitch band.

    This detects target-like portions of the melody without quantizing
    them to MIDI notes.
    """

    n = len(
        structural_pitch
    )

    stable = np.zeros(
        n,
        dtype=bool,
    )

    window = _odd_steps(
        config.stability_window_ms,
        config.analysis_hz,
        minimum=3,
    )

    radius = window // 2

    for run_start, run_end in _true_runs(
        valid
    ):

        for i in range(
            run_start,
            run_end,
        ):

            lo = max(
                run_start,
                i - radius,
            )

            hi = min(
                run_end,
                i + radius + 1,
            )

            local = (
                structural_pitch[
                    lo:hi
                ]
            )

            if len(local) < 3:
                continue

            q10, q90 = np.percentile(
                local,
                [10.0, 90.0],
            )

            robust_range = float(
                q90 - q10
            )

            if (
                robust_range
                <= config.stable_range_st
            ):
                stable[i] = True

    return stable


def _remove_short_stable_runs(
    stable: np.ndarray,
    config: TrajectoryInterpreterConfig,
) -> np.ndarray:

    stable = np.asarray(
        stable,
        dtype=bool,
    ).copy()

    min_steps = max(
        1,
        int(
            np.ceil(
                config.min_stable_ms
                * config.analysis_hz
                / 1000.0
                - 1e-9
            )
        ),
    )

    for start, end in _true_runs(
        stable
    ):
        if (
            end - start
            < min_steps
        ):
            stable[start:end] = False

    return stable


def _merge_same_target_regions(
    stable: np.ndarray,
    structural_pitch: np.ndarray,
    valid: np.ndarray,
    config: TrajectoryInterpreterConfig,
) -> np.ndarray:
    """
    Merge two stable regions when:

      * they belong to the same continuous valid island,
      * the interruption is very short,
      * their median pitch centers are essentially identical.

    This prevents a tiny vibrato excursion or structural-filter wobble
    from unnecessarily splitting one pitch target.
    """

    stable = np.asarray(
        stable,
        dtype=bool,
    ).copy()

    max_gap_steps = max(
        0,
        int(
            round(
                config.merge_gap_ms
                * config.analysis_hz
                / 1000.0
            )
        ),
    )

    changed = True

    while changed:
        changed = False

        runs = _true_runs(
            stable
        )

        if len(runs) < 2:
            break

        for (
            left_start,
            left_end,
        ), (
            right_start,
            right_end,
        ) in zip(
            runs[:-1],
            runs[1:],
        ):

            gap = (
                right_start
                - left_end
            )

            if gap <= 0:
                continue

            if gap > max_gap_steps:
                continue

            # Never bridge a validity hole.
            if not np.all(
                valid[
                    left_end:right_start
                ]
            ):
                continue

            left_pitch = float(
                np.median(
                    structural_pitch[
                        left_start:left_end
                    ]
                )
            )

            right_pitch = float(
                np.median(
                    structural_pitch[
                        right_start:right_end
                    ]
                )
            )

            if (
                abs(
                    right_pitch
                    - left_pitch
                )
                <= config.same_target_tolerance_st
            ):
                stable[
                    left_end:right_start
                ] = True

                changed = True
                break

    return stable


def _nearest_target_left(
    index: int,
    stable_runs: list[
        tuple[int, int]
    ],
    structural_pitch: np.ndarray,
    island_start: int,
) -> float:

    for start, end in reversed(
        stable_runs
    ):
        if start < island_start:
            break

        if (
            end <= index
            and start >= island_start
        ):
            return float(
                np.median(
                    structural_pitch[
                        start:end
                    ]
                )
            )

    return np.nan

def _nearest_target_right(
    index: int,
    stable_runs: list[
        tuple[int, int]
    ],
    structural_pitch: np.ndarray,
    island_end: int,
) -> float:

    for start, end in stable_runs:

        if start >= island_end:
            break

        if (
            start >= index
            and end <= island_end
        ):
            return float(
                np.median(
                    structural_pitch[
                        start:end
                    ]
                )
            )

    return np.nan

def interpret_pitch_trajectory(
    clean_f0_hz: np.ndarray,
    valid: np.ndarray,
    analysis_hz: float,
    config: TrajectoryInterpreterConfig | None = None,
) -> TrajectoryInterpretation:

    f0 = np.asarray(
        clean_f0_hz,
        dtype=np.float64,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    if f0.ndim != 1:
        raise ValueError(
            "clean_f0_hz must be 1-D"
        )

    if valid.ndim != 1:
        raise ValueError(
            "valid must be 1-D"
        )

    if len(f0) != len(valid):
        raise ValueError(
            "clean_f0_hz and valid must "
            "have identical length"
        )

    if analysis_hz <= 0.0:
        raise ValueError(
            "analysis_hz must be > 0"
        )

    if config is None:
        config = (
            TrajectoryInterpreterConfig(
                analysis_hz=float(
                    analysis_hz
                )
            )
        )

    elif not np.isclose(
        config.analysis_hz,
        analysis_hz,
    ):
        raise ValueError(
            "config.analysis_hz does not "
            "match analysis_hz"
        )

    raw_pitch = hz_to_semitones(
        f0
    )

    trusted = (
        valid
        & np.isfinite(
            raw_pitch
        )
    )

    raw_pitch = np.where(
        trusted,
        raw_pitch,
        np.nan,
    )

    structural_pitch = (
        _make_structural_pitch(
            raw_pitch,
            trusted,
            config,
        )
    )

    stable = (
        _local_stability_mask(
            structural_pitch,
            trusted,
            config,
        )
    )

    stable = (
        _remove_short_stable_runs(
            stable,
            config,
        )
    )

    stable = (
        _merge_same_target_regions(
            stable,
            structural_pitch,
            trusted,
            config,
        )
    )

    stable_runs = _true_runs(
        stable
    )

    regions: list[
        PitchRegion
    ] = []

    region_id = np.full(
        len(f0),
        -1,
        dtype=np.int32,
    )

    # ------------------------------------------------------------
    # Partition each trusted island according to stable/non-stable
    # structure.
    # ------------------------------------------------------------

    for island_start, island_end in (
        _true_runs(trusted)
    ):

        boundaries = {
            island_start,
            island_end,
        }

        for start, end in stable_runs:

            if (
                start >= island_start
                and end <= island_end
            ):
                boundaries.add(
                    start
                )
                boundaries.add(
                    end
                )

        boundaries = sorted(
            boundaries
        )

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        ):

            if end <= start:
                continue

            segment_pitch = (
                structural_pitch[
                    start:end
                ]
            )

            raw_segment = (
                raw_pitch[
                    start:end
                ]
            )

            is_stable = bool(
                np.all(
                    stable[
                        start:end
                    ]
                )
            )

            if is_stable:
                kind = (
                    RegionKind.STABLE_TARGET
                )
            else:
                left_target = (
                    _nearest_target_left(
                        start,
                        stable_runs,
                        structural_pitch,
                        island_start,
                    )
                )

                right_target = (
                    _nearest_target_right(
                        end,
                        stable_runs,
                        structural_pitch,
                        island_end,
                    )
                )

                if (
                    np.isfinite(
                        left_target
                    )
                    or np.isfinite(
                        right_target
                    )
                ):
                    kind = (
                        RegionKind.TRANSITION
                    )
                else:
                    kind = (
                        RegionKind.UNRESOLVED
                    )

            previous_target = (
                _nearest_target_left(
                    start,
                    stable_runs,
                    structural_pitch,
                    island_start,
                )
            )

            next_target = (
                _nearest_target_right(
                    end,
                    stable_runs,
                    structural_pitch,
                    island_end,
                )
            )

            region_index = len(
                regions
            )

            region_id[
                start:end
            ] = region_index

            regions.append(
                PitchRegion(
                    index=region_index,

                    start_index=start,
                    end_index=end,

                    start_time_s=float(
                        start
                        / analysis_hz
                    ),

                    end_time_s=float(
                        end
                        / analysis_hz
                    ),

                    duration_ms=float(
                        1000.0
                        * (end - start)
                        / analysis_hz
                    ),

                    kind=kind,

                    median_pitch_st=float(
                        np.median(
                            segment_pitch
                        )
                    ),

                    min_pitch_st=float(
                        np.min(
                            segment_pitch
                        )
                    ),

                    max_pitch_st=float(
                        np.max(
                            segment_pitch
                        )
                    ),

                    span_st=float(
                        np.max(
                            segment_pitch
                        )
                        - np.min(
                            segment_pitch
                        )
                    ),

                    raw_median_pitch_st=float(
                        np.median(
                            raw_segment
                        )
                    ),

                    previous_target_st=float(
                        previous_target
                    ),

                    next_target_st=float(
                        next_target
                    ),
                )
            )

    return TrajectoryInterpretation(
        regions=tuple(
            regions
        ),
        pitch_st=raw_pitch,
        structural_pitch_st=(
            structural_pitch
        ),
        region_id=region_id,
        stable_mask=stable,
        analysis_hz=float(
            analysis_hz
        ),
    )