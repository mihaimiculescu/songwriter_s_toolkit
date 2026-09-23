from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter



@dataclass(frozen=True)
class ExpressiveSmoothingConfig:
    """
    Light, symmetric offline smoothing for gesture-shape analysis.

    This is NOT:
      * target detection,
      * note quantization,
      * a slow melodic centerline,
      * MIDI pitch-bend generation.

    Its only purpose is to suppress ECKF sample-to-sample jitter enough
    that derivatives, path length and direction reversals become useful.

    Filtering is performed independently inside each contiguous valid
    island. Validity holes are absolute walls.
    """

    analysis_hz: float = 100.0

    # Nominal symmetric window.
    # At 100 Hz this becomes 5 samples = 50 ms.
    window_ms: float = 50.0

    # Quadratic Savitzky-Golay.
    polyorder: int = 2
    # Slower robust centerline.
    #
    # This is deliberately much slower than the 50 ms expressive SG.
    # At 100 Hz, 250 ms -> 25 samples.
    #
    # FIRST DIAGNOSTIC VALUE ONLY. Do not freeze until inspected.
    center_window_ms: float = 250.0    

@dataclass(frozen=True)
class ExpressiveSmoothingResult:
    """
    Four representations of the same trusted pitch trajectory.

    raw_pitch_st:
        Corrected but otherwise untouched ECKF pitch.

    shape_pitch_st:
        Light 50 ms symmetric SG trajectory. Preserves expressive
        movement while reducing ECKF micro-jitter.

    center_pitch_st:
        Slower robust bidirectional local centerline.

    residual_pitch_st:
        shape_pitch_st - center_pitch_st.

        This contains local movement around the slower centerline.
        It is NOT automatically "vibrato". It may contain vibrato,
        ornaments, attack/release behavior, tracking error, etc.

    valid:
        Trusted samples. Invalid gaps remain NaN in all derived
        representations.
    """

    raw_pitch_st: np.ndarray
    shape_pitch_st: np.ndarray
    center_pitch_st: np.ndarray
    residual_pitch_st: np.ndarray

    valid: np.ndarray
    analysis_hz: float

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

    diff = np.diff(
        padded.astype(np.int8)
    )

    starts = np.flatnonzero(
        diff == 1
    )

    ends = np.flatnonzero(
        diff == -1
    )

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def _window_steps(
    window_ms: float,
    analysis_hz: float,
) -> int:
    """
    Convert requested milliseconds to the nearest useful odd number
    of analysis samples.

    Example:
        50 ms @ 100 Hz
        = 5 samples.
    """

    steps = int(
        round(
            window_ms
            * analysis_hz
            / 1000.0
        )
    )

    steps = max(
        3,
        steps,
    )

    if steps % 2 == 0:
        steps += 1

    return steps


def _largest_valid_window(
    requested_window: int,
    run_length: int,
    polyorder: int,
) -> int | None:
    """
    Find the largest odd Savitzky-Golay window that fits this valid
    island and can support the requested polynomial order.

    Returns None if the island is too short to filter safely.
    """

    window = min(
        requested_window,
        run_length,
    )

    if window % 2 == 0:
        window -= 1

    minimum = (
        polyorder + 1
    )

    # Savitzky-Golay window must be odd.
    if minimum % 2 == 0:
        minimum += 1

    if window < minimum:
        return None

    return int(window)

def _centerline_valid_run(
    run: np.ndarray,
    requested_window: int,
) -> np.ndarray:
    """
    Symmetric running-median centerline for one continuous valid island.

    Why median rather than another SG pass?

    The centerline should be robust against brief pitch excursions.
    A mordent, scoop or short tracking spike should not drag the local
    pitch center as strongly as it would drag a moving average.

    Edge windows shrink symmetrically instead of borrowing samples from
    outside the valid island.

    No padding across validity boundaries.
    """

    run = np.asarray(
        run,
        dtype=np.float64,
    )

    n = len(run)

    if n == 0:
        return run.copy()

    if n == 1:
        return run.copy()

    half = (
        requested_window // 2
    )

    center = np.empty(
        n,
        dtype=np.float64,
    )

    for i in range(n):

        left = max(
            0,
            i - half,
        )

        right = min(
            n,
            i + half + 1,
        )

        center[i] = float(
            np.median(
                run[left:right]
            )
        )

    return center

def _center_window_steps(
    window_ms: float,
    analysis_hz: float,
) -> int:
    """
    Convert centerline duration to an odd number of analysis samples.

    250 ms @ 100 Hz -> 25 samples.
    """

    steps = int(
        round(
            window_ms
            * analysis_hz
            / 1000.0
        )
    )

    steps = max(
        3,
        steps,
    )

    if steps % 2 == 0:
        steps += 1

    return steps

def smooth_expressive_pitch(
    pitch_st: np.ndarray,
    valid: np.ndarray,
    analysis_hz: float,
    config: ExpressiveSmoothingConfig | None = None,
) -> ExpressiveSmoothingResult:
    """
    Apply light bidirectional smoothing independently to every valid
    island.

    scipy.signal.savgol_filter is a centered/offline filter here:
    output at time t uses samples on both sides of t. There is therefore
    no causal phase lag of the kind we observed in the online ECKF.

    IMPORTANT:
      - invalid gaps are never filled;
      - samples are never borrowed across validity gaps;
      - raw_pitch_st is preserved;
      - short valid islands are copied unchanged;
      - this output is for gesture geometry, not for replacing F0.
    """

    raw = np.asarray(
        pitch_st,
        dtype=np.float64,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    if raw.ndim != 1:
        raise ValueError(
            "pitch_st must be 1-D"
        )

    if valid.ndim != 1:
        raise ValueError(
            "valid must be 1-D"
        )

    if len(raw) != len(valid):
        raise ValueError(
            "pitch_st and valid must have identical length"
        )

    if analysis_hz <= 0.0:
        raise ValueError(
            "analysis_hz must be > 0"
        )

    if config is None:
        config = (
            ExpressiveSmoothingConfig(
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
            "config.analysis_hz does not match analysis_hz"
        )

    if config.window_ms <= 0.0:
        raise ValueError(
            "window_ms must be > 0"
        )

    if config.center_window_ms <= 0.0:
        raise ValueError(
            "center_window_ms must be > 0"
        )

    if config.polyorder < 0:
        raise ValueError(
            "polyorder must be >= 0"
        )

    trusted = (
        valid
        & np.isfinite(raw)
    )

    shape = np.full(
        raw.shape,
        np.nan,
        dtype=np.float64,
    )

    center = np.full(
        raw.shape,
        np.nan,
        dtype=np.float64,
    )

    requested_window = (
        _window_steps(
            config.window_ms,
            analysis_hz,
        )
    )

    center_window = (
        _center_window_steps(
            config.center_window_ms,
            analysis_hz,
        )
    )

    for start, end in _true_runs(
        trusted
    ):

        run = raw[
            start:end
        ]

        run_length = len(
            run
        )

        window = (
            _largest_valid_window(
                requested_window=(
                    requested_window
                ),
                run_length=run_length,
                polyorder=(
                    config.polyorder
                ),
            )
        )

        if window is None:
            # Too short to smooth without inventing information.
            shape[
                start:end
            ] = run

            continue

        shape[
            start:end
        ] = savgol_filter(
            run,
            window_length=window,
            polyorder=config.polyorder,
            deriv=0,
            delta=1.0 / analysis_hz,
            mode="interp",
        )
    # ------------------------------------------------------------
    # Slow robust centerline
    # ------------------------------------------------------------

    for start, end in _true_runs(
        trusted
    ):

        run = shape[
            start:end
        ]

        if len(run) == 0:
            continue

        center[
            start:end
        ] = _centerline_valid_run(
            run=run,
            requested_window=(
                center_window
            ),
        )

    # ------------------------------------------------------------
    # Expressive residual
    # ------------------------------------------------------------

    residual = np.full(
        raw.shape,
        np.nan,
        dtype=np.float64,
    )

    residual[
        trusted
    ] = (
        shape[trusted]
        - center[trusted]
    )


    return ExpressiveSmoothingResult(
        raw_pitch_st=raw.copy(),
        shape_pitch_st=shape,
        center_pitch_st=center,
        residual_pitch_st=residual,
        valid=trusted,
        analysis_hz=float(
            analysis_hz
        ),

    )


