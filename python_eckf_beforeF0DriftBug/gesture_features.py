from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trajectory_interpreter import (
    RegionKind,
    TrajectoryInterpretation,
)

from .expressive_smoothing import (
    ExpressiveSmoothingResult,
)


@dataclass(frozen=True)
class GestureFeatureConfig:
    """
    Purely observational feature extraction.

    Nothing in this configuration assigns musical meaning.

    movement_epsilon_st:
        Ignore microscopic pitch movements when counting direction
        changes.

    residual_zero_epsilon_st:
        Residual values inside this band are treated as zero when
        examining excursion topology.

    residual_peak_min_st:
        Minimum absolute residual excursion worth counting as a
        meaningful local extremum.

    min_half_cycle_ms:
        Reject implausibly tiny residual half-cycles when estimating
        oscillation structure.
    """

    movement_epsilon_st: float = 0.05
    residual_zero_epsilon_st: float = 0.05
    residual_peak_min_st: float = 0.15
    min_half_cycle_ms: float = 25.0
    # ------------------------------------------------------------
    # Residual autocorrelation
    # ------------------------------------------------------------
    #
    # Broad diagnostic band only.
    #
    # This is NOT yet a definition of "vibrato".
    # It merely limits the lags searched for repeated oscillatory
    # structure.
    #
    autocorr_min_rate_hz: float = 3.0
    autocorr_max_rate_hz: float = 12.0

    # At least this many periods must fit inside the observed object
    # before we consider its autocorrelation peak well-supported.
    autocorr_min_cycles: float = 2.0

@dataclass(frozen=True)
class GestureFeatures:
    """
    Geometry of one already-existing transition region.

    IMPORTANT:
        This object does NOT classify the transition.
    """

    region_index: int

    start_index: int
    end_index: int

    start_time_s: float
    end_time_s: float
    duration_ms: float
    n_samples: int

    # ------------------------------------------------------------
    # Frozen target context
    # ------------------------------------------------------------

    source_target_st: float
    destination_target_st: float
    target_interval_st: float

    has_source_target: bool
    has_destination_target: bool

    # ------------------------------------------------------------
    # Shape trajectory
    # ------------------------------------------------------------

    shape_start_st: float
    shape_end_st: float
    shape_net_st: float
    shape_span_st: float
    shape_path_st: float
    shape_efficiency: float

    shape_up_path_st: float
    shape_down_path_st: float

    shape_up_fraction: float
    shape_down_fraction: float

    shape_reversals: int
    shape_max_step_st: float

    # ------------------------------------------------------------
    # Slow centerline
    # ------------------------------------------------------------

    center_start_st: float
    center_end_st: float
    center_net_st: float
    center_span_st: float
    center_path_st: float
    center_efficiency: float

    center_up_path_st: float
    center_down_path_st: float

    center_up_fraction: float
    center_down_fraction: float

    center_reversals: int
    center_max_step_st: float

    center_to_shape_path_ratio: float

    # ------------------------------------------------------------
    # Residual amplitude
    # ------------------------------------------------------------

    residual_mean_st: float
    residual_rms_st: float

    residual_min_st: float
    residual_max_st: float
    residual_peak_to_peak_st: float

    residual_abs_p50_st: float
    residual_abs_p90_st: float
    residual_abs_max_st: float

    # ------------------------------------------------------------
    # Residual signed topology
    # ------------------------------------------------------------

    residual_positive_fraction: float
    residual_negative_fraction: float
    residual_near_zero_fraction: float

    residual_zero_crossings: int

    residual_positive_excursions: int
    residual_negative_excursions: int

    residual_peak_count: int
    residual_trough_count: int

    residual_direction_reversals: int

    # ------------------------------------------------------------
    # Oscillation / repetition evidence
    # ------------------------------------------------------------

    residual_half_cycles: int
    residual_full_cycles_est: float

    residual_median_half_cycle_ms: float
    residual_median_cycle_ms: float

    residual_cycle_rate_hz: float

    residual_cycle_interval_cv: float
    residual_peak_amplitude_cv: float

    residual_periodicity: float

    # ------------------------------------------------------------
    # Residual autocorrelation
    # ------------------------------------------------------------

    residual_autocorr_peak: float
    residual_autocorr_lag_samples: int
    residual_autocorr_lag_ms: float
    residual_autocorr_rate_hz: float

    residual_autocorr_cycles_supported: float
    residual_autocorr_support: float

    # ------------------------------------------------------------
    # Relationship to targets
    # ------------------------------------------------------------

    source_distance_start_st: float
    source_distance_end_st: float

    destination_distance_start_st: float
    destination_distance_end_st: float

    shape_max_distance_from_source_st: float

    leaves_source_st: float
    returns_toward_source_st: float
    source_return_fraction: float


@dataclass(frozen=True)
class GestureFeatureResult:
    gestures: tuple[GestureFeatures, ...]
    analysis_hz: float


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if not np.isfinite(
        numerator
    ):
        return np.nan

    if (
        not np.isfinite(
            denominator
        )
        or abs(denominator) <= 1e-12
    ):
        return 0.0

    return float(
        numerator / denominator
    )


def _span(
    values: np.ndarray,
) -> float:

    if len(values) == 0:
        return 0.0

    return float(
        np.max(values)
        - np.min(values)
    )


def _path_geometry(
    values: np.ndarray,
    movement_epsilon_st: float,
) -> dict:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return {
            "start": np.nan,
            "end": np.nan,
            "net": 0.0,
            "span": 0.0,
            "path": 0.0,
            "efficiency": 0.0,
            "up_path": 0.0,
            "down_path": 0.0,
            "up_fraction": 0.0,
            "down_fraction": 0.0,
            "reversals": 0,
            "max_step": 0.0,
        }

    start = float(
        values[0]
    )

    end = float(
        values[-1]
    )

    net = float(
        end - start
    )

    span = _span(
        values
    )

    if len(values) == 1:

        return {
            "start": start,
            "end": end,
            "net": net,
            "span": span,
            "path": 0.0,
            "efficiency": 0.0,
            "up_path": 0.0,
            "down_path": 0.0,
            "up_fraction": 0.0,
            "down_fraction": 0.0,
            "reversals": 0,
            "max_step": 0.0,
        }

    delta = np.diff(
        values
    )

    abs_delta = np.abs(
        delta
    )

    path = float(
        np.sum(
            abs_delta
        )
    )

    up_path = float(
        np.sum(
            delta[
                delta > 0.0
            ]
        )
    )

    down_path = float(
        np.sum(
            -delta[
                delta < 0.0
            ]
        )
    )

    efficiency = _safe_ratio(
        abs(net),
        path,
    )

    up_fraction = _safe_ratio(
        up_path,
        path,
    )

    down_fraction = _safe_ratio(
        down_path,
        path,
    )

    signs = np.zeros(
        len(delta),
        dtype=np.int8,
    )

    signs[
        delta > movement_epsilon_st
    ] = 1

    signs[
        delta < -movement_epsilon_st
    ] = -1

    moving_signs = signs[
        signs != 0
    ]

    if len(moving_signs) <= 1:
        reversals = 0
    else:
        reversals = int(
            np.sum(
                moving_signs[1:]
                != moving_signs[:-1]
            )
        )

    max_step = float(
        np.max(
            abs_delta
        )
    )

    return {
        "start": start,
        "end": end,
        "net": net,
        "span": span,
        "path": path,
        "efficiency": efficiency,
        "up_path": up_path,
        "down_path": down_path,
        "up_fraction": up_fraction,
        "down_fraction": down_fraction,
        "reversals": reversals,
        "max_step": max_step,
    }


def _signed_states(
    residual: np.ndarray,
    epsilon_st: float,
) -> np.ndarray:

    states = np.zeros(
        len(residual),
        dtype=np.int8,
    )

    states[
        residual > epsilon_st
    ] = 1

    states[
        residual < -epsilon_st
    ] = -1

    return states


def _fill_zero_states(
    states: np.ndarray,
) -> np.ndarray:
    """
    Fill zero-valued gaps between signed residual states.

    This is ONLY for topology measurement.

    It does not modify the residual trajectory itself.

    Example:

        + + 0 0 + +  -> + + + + + +
        + + 0 0 - -  -> + + + - - -

    For a zero gap between opposite signs, the split occurs near the
    middle of the gap.
    """

    states = np.asarray(
        states,
        dtype=np.int8,
    ).copy()

    n = len(
        states
    )

    if n == 0:
        return states

    nonzero = np.flatnonzero(
        states != 0
    )

    if len(nonzero) == 0:
        return states

    first = int(
        nonzero[0]
    )

    last = int(
        nonzero[-1]
    )

    states[
        :first
    ] = states[first]

    states[
        last + 1:
    ] = states[last]

    i = first

    while i <= last:

        if states[i] != 0:
            i += 1
            continue

        start = i

        while (
            i <= last
            and states[i] == 0
        ):
            i += 1

        end = i

        left_sign = (
            states[start - 1]
            if start > 0
            else 0
        )

        right_sign = (
            states[end]
            if end < n
            else 0
        )

        if left_sign == right_sign:
            states[
                start:end
            ] = left_sign

        elif (
            left_sign != 0
            and right_sign != 0
        ):

            midpoint = (
                start
                + (
                    end - start
                ) // 2
            )

            states[
                start:midpoint
            ] = left_sign

            states[
                midpoint:end
            ] = right_sign

    return states


def _count_signed_runs(
    states: np.ndarray,
    sign: int,
) -> int:

    if len(states) == 0:
        return 0

    target = (
        states == sign
    )

    padded = np.concatenate(
        (
            [False],
            target,
            [False],
        )
    )

    diff = np.diff(
        padded.astype(
            np.int8
        )
    )

    return int(
        np.sum(
            diff == 1
        )
    )


def _local_extrema(
    residual: np.ndarray,
    minimum_amplitude_st: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return local maxima and minima indices.

    Plateaus are intentionally not treated specially yet. This is an
    observational diagnostic, not a final peak detector.
    """

    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    if len(residual) < 3:
        return (
            np.asarray(
                [],
                dtype=np.int64,
            ),
            np.asarray(
                [],
                dtype=np.int64,
            ),
        )

    left = residual[
        1:-1
    ] - residual[
        :-2
    ]

    right = residual[
        2:
    ] - residual[
        1:-1
    ]

    peak_mask = (
        (left > 0.0)
        & (right <= 0.0)
        & (
            residual[
                1:-1
            ]
            >= minimum_amplitude_st
        )
    )

    trough_mask = (
        (left < 0.0)
        & (right >= 0.0)
        & (
            residual[
                1:-1
            ]
            <= -minimum_amplitude_st
        )
    )

    peaks = (
        np.flatnonzero(
            peak_mask
        )
        + 1
    )

    troughs = (
        np.flatnonzero(
            trough_mask
        )
        + 1
    )

    return (
        peaks.astype(
            np.int64
        ),
        troughs.astype(
            np.int64
        ),
    )


def _direction_reversals(
    values: np.ndarray,
    epsilon_st: float,
) -> int:

    if len(values) <= 2:
        return 0

    delta = np.diff(
        values
    )

    signs = np.zeros(
        len(delta),
        dtype=np.int8,
    )

    signs[
        delta > epsilon_st
    ] = 1

    signs[
        delta < -epsilon_st
    ] = -1

    signs = signs[
        signs != 0
    ]

    if len(signs) <= 1:
        return 0

    return int(
        np.sum(
            signs[1:]
            != signs[:-1]
        )
    )


def _residual_oscillation_features(
    residual: np.ndarray,
    analysis_hz: float,
    config: GestureFeatureConfig,
) -> dict:

    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    n = len(
        residual
    )

    if n == 0:

        return {
            "positive_fraction": 0.0,
            "negative_fraction": 0.0,
            "near_zero_fraction": 0.0,
            "zero_crossings": 0,
            "positive_excursions": 0,
            "negative_excursions": 0,
            "peak_count": 0,
            "trough_count": 0,
            "direction_reversals": 0,
            "half_cycles": 0,
            "full_cycles_est": 0.0,
            "median_half_cycle_ms": np.nan,
            "median_cycle_ms": np.nan,
            "cycle_rate_hz": np.nan,
            "cycle_interval_cv": np.nan,
            "peak_amplitude_cv": np.nan,
            "periodicity": 0.0,
            "autocorr_peak": np.nan,
            "autocorr_lag_samples": 0,
            "autocorr_lag_ms": np.nan,
            "autocorr_rate_hz": np.nan,
            "autocorr_cycles_supported": 0.0,
            "autocorr_support": 0.0,
        }


    autocorr = (
        _residual_autocorrelation(
            residual=residual,
            analysis_hz=analysis_hz,
            config=config,
        )
    )

    states = _signed_states(
        residual,
        config.residual_zero_epsilon_st,
    )

    positive_fraction = float(
        np.mean(
            states == 1
        )
    )

    negative_fraction = float(
        np.mean(
            states == -1
        )
    )

    near_zero_fraction = float(
        np.mean(
            states == 0
        )
    )

    topology_states = (
        _fill_zero_states(
            states
        )
    )

    nonzero = topology_states[
        topology_states != 0
    ]

    if len(nonzero) <= 1:
        zero_crossings = 0
    else:
        zero_crossings = int(
            np.sum(
                nonzero[1:]
                != nonzero[:-1]
            )
        )

    positive_excursions = (
        _count_signed_runs(
            states,
            1,
        )
    )

    negative_excursions = (
        _count_signed_runs(
            states,
            -1,
        )
    )

    (
        peaks,
        troughs,
    ) = _local_extrema(
        residual,
        config.residual_peak_min_st,
    )

    direction_reversals = (
        _direction_reversals(
            residual,
            config.movement_epsilon_st,
        )
    )

    # ------------------------------------------------------------
    # Half-cycle timing from alternating signed residual states
    # ------------------------------------------------------------

    sign_change_indices = []

    if len(topology_states) >= 2:

        change = np.flatnonzero(
            topology_states[1:]
            != topology_states[:-1]
        )

        sign_change_indices = (
            change + 1
        ).astype(
            np.int64
        )

    if len(sign_change_indices) >= 2:

        half_cycle_samples = (
            np.diff(
                sign_change_indices
            )
        )

        half_cycle_ms = (
            half_cycle_samples
            * 1000.0
            / analysis_hz
        )

        half_cycle_ms = (
            half_cycle_ms[
                half_cycle_ms
                >= config.min_half_cycle_ms
            ]
        )

    else:

        half_cycle_ms = np.asarray(
            [],
            dtype=np.float64,
        )

    half_cycles = int(
        len(
            half_cycle_ms
        )
    )

    full_cycles_est = float(
        half_cycles / 2.0
    )

    if len(half_cycle_ms) > 0:

        median_half_cycle_ms = float(
            np.median(
                half_cycle_ms
            )
        )

        median_cycle_ms = float(
            2.0
            * median_half_cycle_ms
        )

        if median_cycle_ms > 0.0:

            cycle_rate_hz = float(
                1000.0
                / median_cycle_ms
            )

        else:

            cycle_rate_hz = np.nan

        if (
            len(half_cycle_ms) >= 2
            and np.mean(
                half_cycle_ms
            ) > 0.0
        ):

            cycle_interval_cv = float(
                np.std(
                    half_cycle_ms
                )
                / np.mean(
                    half_cycle_ms
                )
            )

        else:

            cycle_interval_cv = np.nan

    else:

        median_half_cycle_ms = np.nan
        median_cycle_ms = np.nan
        cycle_rate_hz = np.nan
        cycle_interval_cv = np.nan

    extrema_indices = np.sort(
        np.concatenate(
            (
                peaks,
                troughs,
            )
        )
    )

    if len(extrema_indices) >= 2:

        extrema_amplitude = np.abs(
            residual[
                extrema_indices
            ]
        )

        mean_amplitude = float(
            np.mean(
                extrema_amplitude
            )
        )

        if mean_amplitude > 1e-12:

            peak_amplitude_cv = float(
                np.std(
                    extrema_amplitude
                )
                / mean_amplitude
            )

        else:

            peak_amplitude_cv = np.nan

    else:

        peak_amplitude_cv = np.nan

    # ------------------------------------------------------------
    # Periodicity evidence
    # ------------------------------------------------------------
    #
    # This is deliberately a continuous observational score.
    #
    # It is NOT:
    #     periodicity > X => vibrato
    #
    # We combine:
    #   - evidence that multiple half-cycles exist,
    #   - regularity of half-cycle timing,
    #   - regularity of alternating extrema amplitudes.
    # ------------------------------------------------------------

    cycle_support = float(
        1.0
        - np.exp(
            -full_cycles_est
        )
    )

    if np.isfinite(
        cycle_interval_cv
    ):
        interval_regularity = float(
            np.exp(
                -cycle_interval_cv
            )
        )
    else:
        interval_regularity = 0.0

    if np.isfinite(
        peak_amplitude_cv
    ):
        amplitude_regularity = float(
            np.exp(
                -peak_amplitude_cv
            )
        )
    else:
        amplitude_regularity = 0.0

    periodicity = float(
        cycle_support
        * np.sqrt(
            interval_regularity
            * amplitude_regularity
        )
    )

    return {
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "near_zero_fraction": near_zero_fraction,
        "zero_crossings": zero_crossings,
        "positive_excursions": positive_excursions,
        "negative_excursions": negative_excursions,
        "peak_count": int(
            len(peaks)
        ),
        "trough_count": int(
            len(troughs)
        ),
        "direction_reversals": direction_reversals,
        "half_cycles": half_cycles,
        "full_cycles_est": full_cycles_est,
        "median_half_cycle_ms": median_half_cycle_ms,
        "median_cycle_ms": median_cycle_ms,
        "cycle_rate_hz": cycle_rate_hz,
        "cycle_interval_cv": cycle_interval_cv,
        "peak_amplitude_cv": peak_amplitude_cv,
        "periodicity": periodicity,
        "autocorr_peak": (
            autocorr[
                "peak"
            ]
        ),
        "autocorr_lag_samples": (
            autocorr[
                "lag_samples"
            ]
        ),
        "autocorr_lag_ms": (
            autocorr[
                "lag_ms"
            ]
        ),
        "autocorr_rate_hz": (
            autocorr[
                "rate_hz"
            ]
        ),
        "autocorr_cycles_supported": (
            autocorr[
                "cycles_supported"
            ]
        ),
        "autocorr_support": (
            autocorr[
                "support"
            ]
        ),
    }

def _residual_autocorrelation(
    residual: np.ndarray,
    analysis_hz: float,
    config: GestureFeatureConfig,
) -> dict:
    """
    Measure repeated structure in the residual using normalized
    autocorrelation.

    This is observational only.

    Important:
        - residual is NOT resampled;
        - no smoothing is added here;
        - no validity boundaries are crossed;
        - no musical label is assigned;
        - short objects are allowed to return weak/no support rather
          than manufacturing a periodic interpretation.
    """

    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    n = len(
        residual
    )

    empty = {
        "peak": np.nan,
        "lag_samples": 0,
        "lag_ms": np.nan,
        "rate_hz": np.nan,
        "cycles_supported": 0.0,
        "support": 0.0,
    }

    if n < 3:
        return empty

    if (
        config.autocorr_min_rate_hz <= 0.0
        or config.autocorr_max_rate_hz <= 0.0
    ):
        raise ValueError(
            "autocorr rates must be > 0"
        )

    if (
        config.autocorr_min_rate_hz
        >= config.autocorr_max_rate_hz
    ):
        raise ValueError(
            "autocorr_min_rate_hz must be "
            "< autocorr_max_rate_hz"
        )

    if config.autocorr_min_cycles <= 0.0:
        raise ValueError(
            "autocorr_min_cycles must be > 0"
        )

    # ------------------------------------------------------------
    # Remove DC bias.
    #
    # The centerline already removes slow pitch structure, but the
    # residual over one finite transition need not have exactly zero
    # mean.
    # ------------------------------------------------------------

    x = (
        residual
        - np.mean(
            residual
        )
    )

    energy = float(
        np.dot(
            x,
            x,
        )
    )

    if (
        not np.isfinite(
            energy
        )
        or energy <= 1e-12
    ):
        return empty

    # ------------------------------------------------------------
    # Convert search-rate band to integer lag band.
    #
    # Fastest allowed oscillation -> smallest lag.
    # Slowest allowed oscillation -> largest lag.
    # ------------------------------------------------------------

    min_lag = int(
        np.ceil(
            analysis_hz
            / config.autocorr_max_rate_hz
        )
    )

    max_lag = int(
        np.floor(
            analysis_hz
            / config.autocorr_min_rate_hz
        )
    )

    min_lag = max(
        1,
        min_lag,
    )

    max_lag = min(
        n - 2,
        max_lag,
    )

    if max_lag < min_lag:
        return empty

    lags = np.arange(
        min_lag,
        max_lag + 1,
        dtype=np.int64,
    )

    correlations = np.full(
        len(lags),
        np.nan,
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # Overlap-normalized correlation.
    #
    # For each lag we correlate only the overlapping pieces and
    # normalize by their own energies. This avoids automatically
    # depressing longer lags merely because fewer samples overlap.
    # ------------------------------------------------------------

    for j, lag in enumerate(
        lags
    ):

        left = x[
            :-lag
        ]

        right = x[
            lag:
        ]

        left_energy = float(
            np.dot(
                left,
                left,
            )
        )

        right_energy = float(
            np.dot(
                right,
                right,
            )
        )

        denominator = float(
            np.sqrt(
                left_energy
                * right_energy
            )
        )

        if denominator <= 1e-12:
            continue

        correlations[j] = float(
            np.dot(
                left,
                right,
            )
            / denominator
        )

    finite = np.isfinite(
        correlations
    )

    if not np.any(
        finite
    ):
        return empty

    # ------------------------------------------------------------
    # Prefer an actual local positive maximum.
    #
    # If no local maximum exists in the search band, use the global
    # maximum as observational fallback.
    # ------------------------------------------------------------

    candidate_indices = []

    for j in range(
        len(correlations)
    ):

        if not np.isfinite(
            correlations[j]
        ):
            continue

        current = correlations[j]

        left_value = (
            correlations[j - 1]
            if j > 0
            else -np.inf
        )

        right_value = (
            correlations[j + 1]
            if j + 1
            < len(correlations)
            else -np.inf
        )

        if (
            current > 0.0
            and current >= left_value
            and current >= right_value
        ):
            candidate_indices.append(
                j
            )

    if candidate_indices:

        best_index = max(
            candidate_indices,
            key=lambda j: (
                correlations[j]
            ),
        )

    else:

        finite_indices = np.flatnonzero(
            finite
        )

        best_index = int(
            finite_indices[
                np.argmax(
                    correlations[
                        finite
                    ]
                )
            ]
        )

    peak = float(
        correlations[
            best_index
        ]
    )

    lag_samples = int(
        lags[
            best_index
        ]
    )

    lag_ms = float(
        lag_samples
        * 1000.0
        / analysis_hz
    )

    rate_hz = float(
        analysis_hz
        / lag_samples
    )

    # ------------------------------------------------------------
    # How many candidate periods are actually present in this object?
    #
    # Use sample support, not printed duration, so this remains
    # internally consistent with the actual trajectory array.
    # ------------------------------------------------------------

    cycles_supported = float(
        n
        / lag_samples
    )

    # ------------------------------------------------------------
    # Continuous evidence support.
    #
    # 1 period  -> weak
    # 2 periods -> reaches our nominal evidence point
    # >2        -> increasingly supported
    #
    # Still NOT a musical classification.
    # ------------------------------------------------------------

    cycle_support = float(
        np.clip(
            cycles_supported
            / config.autocorr_min_cycles,
            0.0,
            1.0,
        )
    )

    positive_peak = float(
        np.clip(
            peak,
            0.0,
            1.0,
        )
    )

    support = float(
        positive_peak
        * cycle_support
    )

    return {
        "peak": peak,
        "lag_samples": lag_samples,
        "lag_ms": lag_ms,
        "rate_hz": rate_hz,
        "cycles_supported": cycles_supported,
        "support": support,
    }

def _source_return_geometry(
    shape: np.ndarray,
    source_target_st: float,
) -> tuple[float, float, float]:

    if (
        len(shape) == 0
        or not np.isfinite(
            source_target_st
        )
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    distance = np.abs(
        shape
        - source_target_st
    )

    start_distance = float(
        distance[0]
    )

    end_distance = float(
        distance[-1]
    )

    max_distance = float(
        np.max(
            distance
        )
    )

    leaves_source = float(
        max(
            0.0,
            max_distance
            - start_distance,
        )
    )

    returns = float(
        max(
            0.0,
            max_distance
            - end_distance,
        )
    )

    if leaves_source <= 1e-12:
        return (
            max_distance,
            leaves_source,
            0.0,
        )

    return_fraction = float(
        np.clip(
            returns
            / leaves_source,
            0.0,
            1.0,
        )
    )

    return (
        max_distance,
        leaves_source,
        return_fraction,
    )


def extract_gesture_features(
    interpretation: TrajectoryInterpretation,
    smoothing: ExpressiveSmoothingResult,
    config: GestureFeatureConfig | None = None,
) -> GestureFeatureResult:
    """
    Measure every frozen TRANSITION region.

    No segmentation changes.
    No validity changes.
    No musical classification.
    """

    if config is None:
        config = GestureFeatureConfig()

    analysis_hz = float(
        interpretation.analysis_hz
    )

    if not np.isclose(
        analysis_hz,
        smoothing.analysis_hz,
    ):
        raise ValueError(
            "interpretation and smoothing analysis rates differ"
        )

    n = len(
        interpretation.pitch_st
    )

    for name, array in (
        (
            "shape_pitch_st",
            smoothing.shape_pitch_st,
        ),
        (
            "center_pitch_st",
            smoothing.center_pitch_st,
        ),
        (
            "residual_pitch_st",
            smoothing.residual_pitch_st,
        ),
        (
            "valid",
            smoothing.valid,
        ),
    ):

        if len(array) != n:
            raise ValueError(
                f"{name} length does not match interpretation"
            )

    gestures = []

    gesture_index = 0

    for region in interpretation.regions:

        if (
            region.kind
            is not RegionKind.TRANSITION
        ):
            continue

        gesture_index += 1

        start = int(
            region.start_index
        )

        end = int(
            region.end_index
        )

        shape = np.asarray(
            smoothing.shape_pitch_st[
                start:end
            ],
            dtype=np.float64,
        )

        center = np.asarray(
            smoothing.center_pitch_st[
                start:end
            ],
            dtype=np.float64,
        )

        residual = np.asarray(
            smoothing.residual_pitch_st[
                start:end
            ],
            dtype=np.float64,
        )

        if (
            len(shape) == 0
            or not np.all(
                np.isfinite(
                    shape
                )
            )
            or not np.all(
                np.isfinite(
                    center
                )
            )
            or not np.all(
                np.isfinite(
                    residual
                )
            )
        ):
            raise ValueError(
                "transition contains non-finite derived pitch"
            )

        shape_geometry = (
            _path_geometry(
                shape,
                config.movement_epsilon_st,
            )
        )

        center_geometry = (
            _path_geometry(
                center,
                config.movement_epsilon_st,
            )
        )

        residual_features = (
            _residual_oscillation_features(
                residual,
                analysis_hz,
                config,
            )
        )

        source = float(
            region.previous_target_st
        )

        destination = float(
            region.next_target_st
        )

        has_source = bool(
            np.isfinite(
                source
            )
        )

        has_destination = bool(
            np.isfinite(
                destination
            )
        )

        if (
            has_source
            and has_destination
        ):
            target_interval = float(
                destination
                - source
            )
        else:
            target_interval = np.nan

        if has_source:

            source_distance_start = float(
                abs(
                    shape[0]
                    - source
                )
            )

            source_distance_end = float(
                abs(
                    shape[-1]
                    - source
                )
            )

        else:

            source_distance_start = np.nan
            source_distance_end = np.nan

        if has_destination:

            destination_distance_start = float(
                abs(
                    shape[0]
                    - destination
                )
            )

            destination_distance_end = float(
                abs(
                    shape[-1]
                    - destination
                )
            )

        else:

            destination_distance_start = np.nan
            destination_distance_end = np.nan

        (
            max_distance_from_source,
            leaves_source,
            source_return_fraction,
        ) = _source_return_geometry(
            shape,
            source,
        )

        if (
            has_source
            and np.isfinite(
                max_distance_from_source
            )
        ):

            final_distance = float(
                abs(
                    shape[-1]
                    - source
                )
            )

            max_distance = float(
                max_distance_from_source
            )

            returns_toward_source = float(
                max(
                    0.0,
                    max_distance
                    - final_distance,
                )
            )

        else:

            returns_toward_source = np.nan

        residual_mean = float(
            np.mean(
                residual
            )
        )

        residual_rms = float(
            np.sqrt(
                np.mean(
                    residual * residual
                )
            )
        )

        residual_min = float(
            np.min(
                residual
            )
        )

        residual_max = float(
            np.max(
                residual
            )
        )

        residual_abs = np.abs(
            residual
        )

        gestures.append(
            GestureFeatures(
                region_index=gesture_index,

                start_index=start,
                end_index=end,

                start_time_s=float(
                    region.start_time_s
                ),
                end_time_s=float(
                    region.end_time_s
                ),
                duration_ms=float(
                    region.duration_ms
                ),
                n_samples=len(
                    shape
                ),

                source_target_st=source,
                destination_target_st=destination,
                target_interval_st=target_interval,

                has_source_target=has_source,
                has_destination_target=has_destination,

                shape_start_st=(
                    shape_geometry[
                        "start"
                    ]
                ),
                shape_end_st=(
                    shape_geometry[
                        "end"
                    ]
                ),
                shape_net_st=(
                    shape_geometry[
                        "net"
                    ]
                ),
                shape_span_st=(
                    shape_geometry[
                        "span"
                    ]
                ),
                shape_path_st=(
                    shape_geometry[
                        "path"
                    ]
                ),
                shape_efficiency=(
                    shape_geometry[
                        "efficiency"
                    ]
                ),

                shape_up_path_st=(
                    shape_geometry[
                        "up_path"
                    ]
                ),
                shape_down_path_st=(
                    shape_geometry[
                        "down_path"
                    ]
                ),

                shape_up_fraction=(
                    shape_geometry[
                        "up_fraction"
                    ]
                ),
                shape_down_fraction=(
                    shape_geometry[
                        "down_fraction"
                    ]
                ),

                shape_reversals=(
                    shape_geometry[
                        "reversals"
                    ]
                ),
                shape_max_step_st=(
                    shape_geometry[
                        "max_step"
                    ]
                ),

                center_start_st=(
                    center_geometry[
                        "start"
                    ]
                ),
                center_end_st=(
                    center_geometry[
                        "end"
                    ]
                ),
                center_net_st=(
                    center_geometry[
                        "net"
                    ]
                ),
                center_span_st=(
                    center_geometry[
                        "span"
                    ]
                ),
                center_path_st=(
                    center_geometry[
                        "path"
                    ]
                ),
                center_efficiency=(
                    center_geometry[
                        "efficiency"
                    ]
                ),

                center_up_path_st=(
                    center_geometry[
                        "up_path"
                    ]
                ),
                center_down_path_st=(
                    center_geometry[
                        "down_path"
                    ]
                ),

                center_up_fraction=(
                    center_geometry[
                        "up_fraction"
                    ]
                ),
                center_down_fraction=(
                    center_geometry[
                        "down_fraction"
                    ]
                ),

                center_reversals=(
                    center_geometry[
                        "reversals"
                    ]
                ),
                center_max_step_st=(
                    center_geometry[
                        "max_step"
                    ]
                ),

                center_to_shape_path_ratio=(
                    _safe_ratio(
                        center_geometry[
                            "path"
                        ],
                        shape_geometry[
                            "path"
                        ],
                    )
                ),

                residual_mean_st=(
                    residual_mean
                ),
                residual_rms_st=(
                    residual_rms
                ),

                residual_min_st=(
                    residual_min
                ),
                residual_max_st=(
                    residual_max
                ),
                residual_peak_to_peak_st=float(
                    residual_max
                    - residual_min
                ),

                residual_abs_p50_st=float(
                    np.percentile(
                        residual_abs,
                        50.0,
                    )
                ),
                residual_abs_p90_st=float(
                    np.percentile(
                        residual_abs,
                        90.0,
                    )
                ),
                residual_abs_max_st=float(
                    np.max(
                        residual_abs
                    )
                ),

                residual_positive_fraction=(
                    residual_features[
                        "positive_fraction"
                    ]
                ),
                residual_negative_fraction=(
                    residual_features[
                        "negative_fraction"
                    ]
                ),
                residual_near_zero_fraction=(
                    residual_features[
                        "near_zero_fraction"
                    ]
                ),

                residual_zero_crossings=(
                    residual_features[
                        "zero_crossings"
                    ]
                ),

                residual_positive_excursions=(
                    residual_features[
                        "positive_excursions"
                    ]
                ),
                residual_negative_excursions=(
                    residual_features[
                        "negative_excursions"
                    ]
                ),

                residual_peak_count=(
                    residual_features[
                        "peak_count"
                    ]
                ),
                residual_trough_count=(
                    residual_features[
                        "trough_count"
                    ]
                ),

                residual_direction_reversals=(
                    residual_features[
                        "direction_reversals"
                    ]
                ),

                residual_half_cycles=(
                    residual_features[
                        "half_cycles"
                    ]
                ),
                residual_full_cycles_est=(
                    residual_features[
                        "full_cycles_est"
                    ]
                ),

                residual_median_half_cycle_ms=(
                    residual_features[
                        "median_half_cycle_ms"
                    ]
                ),
                residual_median_cycle_ms=(
                    residual_features[
                        "median_cycle_ms"
                    ]
                ),

                residual_cycle_rate_hz=(
                    residual_features[
                        "cycle_rate_hz"
                    ]
                ),

                residual_cycle_interval_cv=(
                    residual_features[
                        "cycle_interval_cv"
                    ]
                ),
                residual_peak_amplitude_cv=(
                    residual_features[
                        "peak_amplitude_cv"
                    ]
                ),

                residual_periodicity=(
                    residual_features[
                        "periodicity"
                    ]
                ),

                residual_autocorr_peak=(
                    residual_features[
                        "autocorr_peak"
                    ]
                ),

                residual_autocorr_lag_samples=(
                    residual_features[
                        "autocorr_lag_samples"
                    ]
                ),

                residual_autocorr_lag_ms=(
                    residual_features[
                        "autocorr_lag_ms"
                    ]
                ),

                residual_autocorr_rate_hz=(
                    residual_features[
                        "autocorr_rate_hz"
                    ]
                ),

                residual_autocorr_cycles_supported=(
                    residual_features[
                        "autocorr_cycles_supported"
                    ]
                ),

                residual_autocorr_support=(
                    residual_features[
                        "autocorr_support"
                    ]
                ),

                source_distance_start_st=(
                    source_distance_start
                ),
                source_distance_end_st=(
                    source_distance_end
                ),

                destination_distance_start_st=(
                    destination_distance_start
                ),
                destination_distance_end_st=(
                    destination_distance_end
                ),

                shape_max_distance_from_source_st=(
                    max_distance_from_source
                ),

                leaves_source_st=(
                    leaves_source
                ),
                returns_toward_source_st=(
                    returns_toward_source
                ),
                source_return_fraction=(
                    source_return_fraction
                ),
            )
        )

    return GestureFeatureResult(
        gestures=tuple(
            gestures
        ),
        analysis_hz=analysis_hz,
    )
