from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .frame_evidence import FrameAcousticState, validate_frame_acoustic_states


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

    # Current fixed-aperture production rule.  Kept unchanged while the
    # historical variable-aperture ladder is reintroduced in shadow mode.
    stability_window_ms: float = 120.0

    # V2 variable-aperture shadow ladder.  These probes DO NOT change
    # target formation yet; they expose what 24/40/64/96 ms would see.
    variable_apertures_ms: tuple[float, ...] = (24.0, 40.0, 64.0, 96.0)

    # Maximum robust pitch spread inside a window.
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

    # Explicit downstream contract.  `valid` is the final validity mask
    # actually used by the interpreter.  `frame_acoustic_state` preserves
    # the V2 frame passport when supplied by the caller.
    valid: np.ndarray
    frame_acoustic_state: np.ndarray | None

    region_id: np.ndarray
    stable_mask: np.ndarray

    # V2 target-formation audit.  These arrays explain, sample by sample,
    # why corrected-valid F0 was or was not promoted into a stable target.
    local_stability_point_count: np.ndarray
    local_robust_range_st: np.ndarray
    local_stability_pass: np.ndarray

    # Read-only variable-aperture evidence.  Shape is
    # (trajectory_points, number_of_apertures).
    variable_apertures_ms: tuple[float, ...]
    variable_aperture_point_count: np.ndarray
    variable_aperture_robust_range_st: np.ndarray
    variable_aperture_pass: np.ndarray
    variable_first_passing_aperture_ms: np.ndarray
    variable_largest_passing_aperture_ms: np.ndarray

    stable_run_duration_ms: np.ndarray
    min_duration_pass: np.ndarray
    target_formation_reason: np.ndarray

    stable_range_threshold_st: float
    min_stable_threshold_ms: float
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


def _local_stability_evidence(
    structural_pitch: np.ndarray,
    valid: np.ndarray,
    config: TrajectoryInterpreterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure the local evidence used to decide whether each trusted sample
    looks target-like.

    Returns:
      pass_mask
          True when the local robust pitch spread is narrow enough.
      point_count
          Number of trusted structural-pitch samples in the local window.
      robust_range_st
          q90-q10 pitch spread in semitones, NaN when fewer than three
          points are available.

    This is the same stability rule as V2 used previously; it merely
    exposes the evidence instead of hiding it inside a boolean mask.
    """

    n = len(structural_pitch)

    stable = np.zeros(n, dtype=bool)
    point_count = np.zeros(n, dtype=np.int16)
    robust_range_st = np.full(n, np.nan, dtype=np.float64)

    window = _odd_steps(
        config.stability_window_ms,
        config.analysis_hz,
        minimum=3,
    )

    radius = window // 2

    for run_start, run_end in _true_runs(valid):
        for i in range(run_start, run_end):
            lo = max(run_start, i - radius)
            hi = min(run_end, i + radius + 1)
            local = structural_pitch[lo:hi]
            point_count[i] = len(local)

            if len(local) < 3:
                continue

            q10, q90 = np.percentile(local, [10.0, 90.0])
            robust_range = float(q90 - q10)
            robust_range_st[i] = robust_range

            if robust_range <= config.stable_range_st:
                stable[i] = True

    return stable, point_count, robust_range_st



def _validated_variable_apertures(
    config: TrajectoryInterpreterConfig,
) -> tuple[float, ...]:
    apertures = tuple(float(x) for x in config.variable_apertures_ms)
    if not apertures:
        raise ValueError("variable_apertures_ms must not be empty")
    if any((not np.isfinite(x)) or x <= 0.0 for x in apertures):
        raise ValueError("variable apertures must be finite and > 0")
    if any(b <= a for a, b in zip(apertures[:-1], apertures[1:])):
        raise ValueError("variable apertures must be strictly increasing")
    return apertures


def _variable_aperture_shadow_evidence(
    structural_pitch: np.ndarray,
    valid: np.ndarray,
    config: TrajectoryInterpreterConfig,
) -> tuple[tuple[float, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read-only V2 variable-aperture probe.

    Evaluate the historical 24/40/64/96 ms ladder independently at every
    trusted trajectory point.  This function deliberately does NOT choose
    the production stable mask; the current 120 ms fixed-aperture brain is
    left untouched for the first four-song validation.

    Because the active offline trajectory is sampled at ~100 Hz, nominal
    millisecond apertures map to the nearest centered odd number of trajectory
    samples (minimum three).  The nominal aperture is retained in output so
    the mapping is explicit rather than silently pretending 100 Hz can express
    every millisecond width exactly.
    """
    apertures = _validated_variable_apertures(config)
    n = len(structural_pitch)
    m = len(apertures)
    counts = np.zeros((n, m), dtype=np.int16)
    ranges = np.full((n, m), np.nan, dtype=np.float64)
    passes = np.zeros((n, m), dtype=bool)
    first_pass = np.full(n, np.nan, dtype=np.float64)
    largest_pass = np.full(n, np.nan, dtype=np.float64)

    windows = tuple(
        _odd_steps(ms, config.analysis_hz, minimum=3)
        for ms in apertures
    )

    for run_start, run_end in _true_runs(valid):
        for i in range(run_start, run_end):
            for k, (aperture_ms, window) in enumerate(zip(apertures, windows)):
                radius = window // 2
                lo = max(run_start, i - radius)
                hi = min(run_end, i + radius + 1)
                local = structural_pitch[lo:hi]
                counts[i, k] = len(local)
                if len(local) < 3:
                    continue
                q10, q90 = np.percentile(local, [10.0, 90.0])
                spread = float(q90 - q10)
                ranges[i, k] = spread
                if spread <= config.stable_range_st:
                    passes[i, k] = True
                    if not np.isfinite(first_pass[i]):
                        first_pass[i] = aperture_ms
                    largest_pass[i] = aperture_ms

    return apertures, counts, ranges, passes, first_pass, largest_pass

def _stable_run_duration_ms(
    stable: np.ndarray,
    analysis_hz: float,
) -> np.ndarray:
    """Duration of the pre-duration-filter stable run containing each point."""
    out = np.zeros(len(stable), dtype=np.float64)
    for start, end in _true_runs(stable):
        duration_ms = 1000.0 * (end - start) / analysis_hz
        out[start:end] = duration_ms
    return out


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
    frame_acoustic_state: np.ndarray | None = None,
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

    acoustic_state = None

    if frame_acoustic_state is not None:
        acoustic_state = np.asarray(
            frame_acoustic_state,
            dtype=np.int16,
        )

        if acoustic_state.ndim != 1:
            raise ValueError(
                "frame_acoustic_state must be 1-D"
            )

        if len(acoustic_state) != len(valid):
            raise ValueError(
                "frame_acoustic_state and valid must "
                "have identical length"
            )

        validate_frame_acoustic_states(
            acoustic_state
        )

        tracked = (
            acoustic_state
            == int(FrameAcousticState.VOICED_TRACKED)
        )

        # The validity layer is now required to respect the V2 frame
        # passport.  Fail loudly instead of allowing trajectory code to
        # reinterpret silence, unvoiced audio, or unresolved voiced audio.
        impossible = valid & ~tracked
        if np.any(impossible):
            first = int(np.flatnonzero(impossible)[0])
            raise ValueError(
                "valid=True on a non-VOICED_TRACKED frame "
                f"at trajectory index {first}"
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

    if acoustic_state is not None:
        trusted &= (
            acoustic_state
            == int(FrameAcousticState.VOICED_TRACKED)
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

    local_stable, local_point_count, local_robust_range_st = (
        _local_stability_evidence(
            structural_pitch,
            trusted,
            config,
        )
    )

    (
        variable_apertures_ms,
        variable_aperture_point_count,
        variable_aperture_robust_range_st,
        variable_aperture_pass,
        variable_first_passing_aperture_ms,
        variable_largest_passing_aperture_ms,
    ) = _variable_aperture_shadow_evidence(
        structural_pitch,
        trusted,
        config,
    )

    stable_run_duration_ms = _stable_run_duration_ms(
        local_stable,
        config.analysis_hz,
    )

    stable_after_duration = (
        _remove_short_stable_runs(
            local_stable,
            config,
        )
    )

    min_duration_pass = stable_after_duration.copy()

    stable = (
        _merge_same_target_regions(
            stable_after_duration,
            structural_pitch,
            trusted,
            config,
        )
    )

    target_formation_reason = np.full(
        len(f0),
        "NOT_TRUSTED",
        dtype=object,
    )

    for i in np.flatnonzero(trusted):
        if local_point_count[i] < 3:
            reason = "TOO_FEW_LOCAL_POINTS"
        elif not local_stable[i]:
            reason = "LOCAL_SPREAD_TOO_WIDE"
        elif not stable_after_duration[i]:
            reason = "STABLE_RUN_TOO_SHORT"
        elif stable[i]:
            reason = "STABLE_TARGET"
        else:
            # Defensive: should not occur with the current merge-only postpass.
            reason = "NOT_PROMOTED"
        target_formation_reason[i] = reason

    # Samples that became stable only because two same-target regions were
    # merged across a short trusted interruption get an explicit provenance.
    merged_only = stable & ~stable_after_duration & trusted
    target_formation_reason[merged_only] = "MERGED_SAME_TARGET_GAP"

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
        valid=trusted.copy(),
        frame_acoustic_state=(
            acoustic_state.copy()
            if acoustic_state is not None
            else None
        ),
        region_id=region_id,
        stable_mask=stable,
        local_stability_point_count=local_point_count,
        local_robust_range_st=local_robust_range_st,
        local_stability_pass=local_stable,
        variable_apertures_ms=variable_apertures_ms,
        variable_aperture_point_count=variable_aperture_point_count,
        variable_aperture_robust_range_st=variable_aperture_robust_range_st,
        variable_aperture_pass=variable_aperture_pass,
        variable_first_passing_aperture_ms=variable_first_passing_aperture_ms,
        variable_largest_passing_aperture_ms=variable_largest_passing_aperture_ms,
        stable_run_duration_ms=stable_run_duration_ms,
        min_duration_pass=min_duration_pass,
        target_formation_reason=target_formation_reason,
        stable_range_threshold_st=float(config.stable_range_st),
        min_stable_threshold_ms=float(config.min_stable_ms),
        analysis_hz=float(
            analysis_hz
        ),
    )