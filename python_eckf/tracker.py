from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ECKFConfig
from .harmonic_change import HarmonicChangeDetector
from .matlab_compat import complex_min_matlab_like
from .silence import is_silent
from .eckf_trace import ECKFTrace

@dataclass
class ECKFResult:
    f0_hz: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    fundamental: np.ndarray
    onset_samples: np.ndarray
    onset_seconds: np.ndarray
    spectral_flatness_per_frame: np.ndarray
    energy_db_per_frame: np.ndarray
    sample_rate: float
    original_length: int
    padded_length: int
    config: ECKFConfig


def _as_mono_float64(audio: np.ndarray) -> np.ndarray:
    y = np.asarray(audio)
    if y.ndim != 1:
        raise ValueError(
            "ECKF expects mono input. This project intentionally does not "
            "auto-downmix; provide a 1-D mono WAV/audio array."
        )
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float64)
    else:
        y = y.astype(np.float64, copy=False)
    if not np.all(np.isfinite(y)):
        raise ValueError("audio contains NaN or Inf")
    return y


def track_pitch(
    audio: np.ndarray,
    sample_rate: float,
    config: ECKFConfig | None = None,
) -> ECKFResult:
    """
    Extended Complex Kalman Filter pitch tracker.

    Based on:
      eckf_pitch_final/eckf_pitch_modified.m
      harmonic_change_detector.m
      is_silent.m
      parabolic_interpolation.m

    The implementation keeps a MATLAB-compatibility mode and a vocal-only
    offline mode.  No MIDI interpretation happens here.
    """
    if config is None:
        config = ECKFConfig()
    config.validate()

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    y0 = _as_mono_float64(audio)
    original_length = len(y0)
    block = config.block_size

    nframes = int(np.ceil(original_length / block)) if original_length else 0
    padded_length = nframes * block
    y = np.pad(y0, (0, padded_length - original_length), mode="constant")

    f0 = np.zeros(padded_length, dtype=np.float64)
    amp = np.zeros(padded_length, dtype=np.float64)
    phase = np.zeros(padded_length, dtype=np.float64)
    x_est = np.zeros(padded_length, dtype=np.complex128)
    q_track = np.zeros(padded_length, dtype=np.float64)

    onset_samples = []
    spf = []
    energy_track = []

    if padded_length == 0:
        return ECKFResult(
            f0, amp, phase, x_est,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0),
            np.zeros(0),
            sample_rate,
            original_length,
            padded_length,
            config,
        )

    Ts = 1.0 / float(sample_rate)
    H = np.array([[0.0, 0.5, 0.5]], dtype=np.complex128)

    K = np.zeros((3, 1), dtype=np.complex128)
    flag = -1
    silent_cur = 1
    harm_prev = 0

    detector = HarmonicChangeDetector(config)
    trace = ECKFTrace(sample_rate)

    start = 0
    n = 0

    P_last = None
    x_last = None

    while start < padded_length:
        end_exclusive = start + block

        # MATLAB source uses:
        #   end_pos = start_pos + blockSize - 1
        #   if end_pos >= length(y), break
        # With 1-based indexing this discards the final complete padded frame.
        if config.mode == "matlab":
            matlab_end_pos_1based = end_exclusive
            if matlab_end_pos_1based >= padded_length:
                break
        else:
            if end_exclusive > padded_length:
                break

        y_frame = y[start:end_exclusive]

        silent_prev = silent_cur
        silent_cur, cur_spf, cur_energy = is_silent(
            y_frame,
            flatness_threshold=config.silence_flatness_threshold,
            energy_db_threshold=config.silence_energy_db_threshold,
        )
        spf.append(cur_spf)
        energy_track.append(cur_energy)

        if silent_cur:
            trace.emit(
                "SILENCE",
                start,
                frame_start=start,
                frame_time_s=start / sample_rate,
                silent_prev=silent_prev,
                silent_cur=silent_cur,
                flag=flag,
            )
            flag = 0
            start += block
            n = start
            continue

        y_prev = y[start - block:start] if start >= block else None

        # A harmonic-change comparison against a frame already classified
        # as silent is meaningless.  On a silence -> sound transition the
        # onset is already established by the silence detector itself.
        #
        # We still analyse the current frame because its f0/amplitude/phase
        # may be needed by the normal initialization path, but there is no
        # previous harmonic spectrum to compare against.
        if config.mode == "offline" and silent_prev == 1:
            analysis_cur = detector.analyze(None, y_frame, sample_rate)
        else:
            analysis_cur = detector.analyze(y_prev, y_frame, sample_rate)

        harm_cur = analysis_cur.flag
        trace.emit(
            "FRAME_ANALYSIS",
            start,
            frame_start=start,
            frame_time_s=start / sample_rate,
            harm_prev=harm_prev,
            harm_cur=harm_cur,
            silent_prev=silent_prev,
            silent_cur=silent_cur,
            flag=flag,
        )

        count = 0
        initialization = None
        analysis_start = start

        if (
            (silent_prev == 1 and silent_cur == 0)
            or (harm_prev == 0 and harm_cur == 1)
        ):
            onset_samples.append(start)

            trace.emit(
                "RESET_TRIGGER",
                start,
                frame_start=start,
                frame_time_s=start / sample_rate,
                harm_prev=harm_prev,
                harm_cur=harm_cur,
                silent_prev=silent_prev,
                silent_cur=silent_cur,
                flag=flag,
            )

            while count < config.num_buf_to_wait:
                count += 1
                start += block

            if start + block < padded_length:
                future_frame = y[start:start + block]
                trace.emit(
                    "LOOKAHEAD_FRAME",
                    start,
                    frame_start=analysis_start,
                    frame_time_s=analysis_start / sample_rate,
                    count=count,
                    flag=flag,
                )

                # MATLAB fills the skipped interval with the previous estimate.
                if n > 0:
                    stop = min(start + 1, padded_length)
                    f0[n:stop] = f0[n - 1]
                    amp[n:stop] = amp[n - 1]
                    phase[n:stop] = phase[n - 1]

                flag = 1
                n = start
                y_frame = future_frame
            else:
                break

        if flag == 1:
            if config.mode == "offline":
                initialization = detector.analyze(
                    None,
                    y_frame,
                    sample_rate,
                )
            else:
                y_prev_init = y[start - block:start] if start >= block else None
                initialization = detector.analyze(
                    y_prev_init,
                    y_frame,
                    sample_rate,
                )

            f1 = initialization.f0_hz
            a1 = initialization.amplitude
            phi1 = initialization.phase
            trace.emit(
                "INITIALIZATION_PROPOSED",
                start,
                frame_start=analysis_start,
                frame_time_s=analysis_start / sample_rate,
                count=count,
                init_f0_hz=f1,
                init_amplitude=a1,
                flag=flag,
            )

            # MATLAB n is 1-based.  At this instant n corresponds to the
            # future/stabilized frame start.
            n_matlab = n + 1

            x0 = np.array(
                [
                    np.exp(1j * 2.0 * np.pi * f1 * Ts),
                    a1 * np.exp(
                        1j * 2.0 * np.pi * f1 * n_matlab * Ts + 1j * phi1
                    ),
                    a1 * np.exp(
                        -1j * 2.0 * np.pi * f1 * n_matlab * Ts - 1j * phi1
                    ),
                ],
                dtype=np.complex128,
            ).reshape(3, 1)

            P0 = np.zeros((3, 3), dtype=np.complex128)

            gain_min_abs = abs(complex_min_matlab_like(K))
            reset_accepted = (
                gain_min_abs
                < config.kalman_gain_reset_threshold
            )

            trace.emit(
                "RESET_DECISION",
                start,
                frame_start=analysis_start,
                frame_time_s=analysis_start / sample_rate,
                count=count,
                init_f0_hz=f1,
                init_amplitude=a1,
                gain_min_abs=gain_min_abs,
                reset_threshold=config.kalman_gain_reset_threshold,
                reset_accepted=int(reset_accepted),
                flag=flag,
            )

            if reset_accepted:
            # if abs(complex_min_matlab_like(K)) < config.kalman_gain_reset_threshold:
                P_last = P0

                if config.mode == "matlab":
                    x_last = x0
                    # Literal source behavior: rewind the time/output cursor
                    # but KEEP y_frame pointing at the future stabilized frame.
                    start -= count * block
                    n = start
                else:
                    # Offline correction:
                    # use the future frame only to obtain stable f1/a1/phi1,
                    # then really resume filtering on the backtracked audio.
                    start -= count * block
                    n = start
                    y_frame = y[start:start + block]

                    # Re-anchor the initialized oscillatory state to the actual
                    # backtracked sample position while retaining the stable
                    # future-frame spectral estimates.
                    n_back_matlab = n + 1
                    x_last = np.array(
                        [
                            np.exp(1j * 2.0 * np.pi * f1 * Ts),
                            a1 * np.exp(
                                1j * 2.0 * np.pi * f1 * n_back_matlab * Ts
                                + 1j * phi1
                            ),
                            a1 * np.exp(
                                -1j * 2.0 * np.pi * f1 * n_back_matlab * Ts
                                - 1j * phi1
                            ),
                        ],
                        dtype=np.complex128,
                    ).reshape(3, 1)

                flag = 0
                trace.emit(
                    "RESET_COMPLETED",
                    n,
                    frame_start=analysis_start,
                    frame_time_s=analysis_start / sample_rate,
                    count=count,
                    init_f0_hz=f1,
                    reset_accepted=1,
                    flag=flag,
                )

        if P_last is None or x_last is None:
            # A non-silent frame should normally reach initialization through
            # the onset condition.  Keep failure explicit rather than inventing
            # a hidden state.
            raise RuntimeError(
                "ECKF state was not initialized before filtering. "
                "Inspect silence/onset logic for this input."
            )

        for k in range(block):
            if n >= padded_length:
                break

            # K = (P_last*H')/(H*P_last*H' + 1)
            numerator = P_last @ H.conj().T
            denominator = (H @ P_last @ H.conj().T)[0, 0] + 1.0
            K = numerator / denominator

            P = P_last - K @ H @ P_last

            sample = y_frame[k]
            innovation = sample - (H @ x_last)[0, 0]
            x = x_last + K * innovation

            x1, x2, x3 = x[:, 0]
            x_next = np.array(
                [
                    x1,
                    x1 * x2,
                    x3 / x1,
                ],
                dtype=np.complex128,
            ).reshape(3, 1)

            F = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [x2, x1, 0.0],
                    [-x3 / (x1 ** 2), 0.0, 1.0 / x1],
                ],
                dtype=np.complex128,
            )

            innovation_after_update = abs(sample - (H @ x)[0, 0])
            q = 10.0 ** (-(config.c - innovation_after_update))
            q_track[n] = float(np.real(q))

            P_next = F @ P @ F.conj().T + q * np.eye(
                3, dtype=np.complex128
            )

            f0[n] = abs(np.log(x1) / (1j * Ts * 2.0 * np.pi))
            trace_stride = max(
                1,
                round(sample_rate * 0.010),
            )

            if n % trace_stride == 0:
                trace.emit(
                    "KALMAN_SAMPLE",
                    n,
                    frame_start=start,
                    frame_time_s=start / sample_rate,
                    f0_hz=f0[n],
                    innovation_abs=abs(innovation),
                    innovation_after_abs=innovation_after_update,
                    q=float(np.real(q)),
                    gain_norm=float(np.linalg.norm(K)),
                    covariance_norm=float(np.linalg.norm(P_last)),
                    state_frequency_hz=(
                        float(
                            abs(
                                np.log(x1)
                                / (1j * Ts * 2.0 * np.pi)
                            )
                        )
                    ),
                )            
            amp[n] = abs(x2)

            if amp[n] > 0.0:
                n_matlab = n + 1
                phase_complex = -1j * (
                    np.log(x2 / amp[n])
                    - (2.0 * np.pi * f0[n] * Ts * n_matlab)
                )
                phase[n] = abs(phase_complex)

            x_est[n] = (H @ x)[0, 0]

            P_last = P_next
            x_last = x_next
            n += 1

        start += block
        harm_prev = harm_cur

    onset_samples_arr = np.asarray(onset_samples, dtype=np.int64)
    trace.close()
    return ECKFResult(
        f0_hz=f0,
        amplitude=amp,
        phase=phase,
        fundamental=x_est,
        onset_samples=onset_samples_arr,
        onset_seconds=onset_samples_arr.astype(np.float64) / sample_rate,
        spectral_flatness_per_frame=np.asarray(spf, dtype=np.float64),
        energy_db_per_frame=np.asarray(energy_track, dtype=np.float64),
        sample_rate=float(sample_rate),
        original_length=original_length,
        padded_length=padded_length,
        config=config,
    )
