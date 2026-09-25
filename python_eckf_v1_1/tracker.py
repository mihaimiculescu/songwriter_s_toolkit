from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ECKFConfig
from .harmonic_change import HarmonicChangeDetector
from .matlab_compat import complex_min_matlab_like
from .silence import is_silent, resolve_silence_energy_threshold
from .eckf_trace import ECKFTrace
from .periodicity import assess_periodicity
from .pitch_status import PitchStatus, validate_pitch_status
from .initialization_candidates import choose_initialization
from .octave_reacquisition import reconcile_initialization, inspect_octave_disagreement

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
    pitch_status: np.ndarray | None = None
    # V2: frame-aligned evidence. One entry per ECKF frame, not per 100 Hz sample.
    frame_start_samples: np.ndarray | None = None
    frame_real_sample_count: np.ndarray | None = None
    energy_silence_per_frame: np.ndarray | None = None
    flatness_silence_per_frame: np.ndarray | None = None
    original_silence_per_frame: np.ndarray | None = None
    periodicity_voiced_per_frame: np.ndarray | None = None
    frame_decision: np.ndarray | None = None


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
    silence_energy_threshold = resolve_silence_energy_threshold(config)

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
    pitch_status = np.full(padded_length, PitchStatus.UNVOICED, dtype=np.uint8)

    onset_samples = []
    spf = []
    energy_track = []
    # Offline evidence is indexed by the original frame, never by traversal order.
    # The MATLAB path is intentionally unchanged.
    frame_start_samples = np.arange(nframes, dtype=np.int64) * block
    frame_real_sample_count = np.minimum(
        block, np.maximum(0, original_length - frame_start_samples)
    ).astype(np.int64)
    energy_silence = np.zeros(nframes, dtype=np.bool_)
    flatness_silence = np.zeros(nframes, dtype=np.bool_)
    original_silence = np.zeros(nframes, dtype=np.bool_)
    periodicity_voiced = np.zeros(nframes, dtype=np.bool_)
    frame_decision = np.full(nframes, "NOT_PROCESSED", dtype="U32")
    previous_frame_eligible = False

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
            pitch_status,
            frame_start_samples, frame_real_sample_count,
            energy_silence, flatness_silence, original_silence,
            periodicity_voiced, frame_decision,
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

        if config.mode == "offline":
            # Offline: assess EVERY frame, including frames after initialization.
            # A non-periodic frame has no pitch, even when it is loud.
            # Previous *eligibility* is not the preceding measured silence flag.
            # A periodicity failure or failed initialization may make the previous
            # frame unavailable even when its sound was not silent.
            silent_prev = int(not previous_frame_eligible)
            silent_cur, cur_spf, cur_energy = is_silent(
                y_frame,
                flatness_threshold=config.silence_flatness_threshold,
                energy_db_threshold=silence_energy_threshold,
            )
            spf.append(cur_spf)
            energy_track.append(cur_energy)
            fi = start // block
            original_silence[fi] = bool(silent_cur)
            energy_silence[fi] = bool(cur_energy < silence_energy_threshold)
            flatness_silence[fi] = bool(
                cur_spf >= config.silence_flatness_threshold
            )
            # Absolute low-energy detection has priority over normalized
            # periodicity: even a barely audible sinusoid has ACF near 1.
            # Retain the historical, non-dBFS energy scale and its threshold.
            # Flatness alone is NOT an authoritative veto: a pitched consonant
            # or breath can have a noisy spectrum yet carry genuine F0.
            if energy_silence[fi]:
                frame_decision[fi] = "LOW_ENERGY_SILENCE"
                trace.emit("SILENCE", start, frame_start=start,
                           frame_time_s=start / sample_rate,
                           silent_prev=silent_prev, silent_cur=1, flag=flag)
                P_last = None
                x_last = None
                previous_frame_eligible = False
                flag = 0
                harm_prev = 0
                start += block
                n = start
                continue

            periodicity = assess_periodicity(y_frame, sample_rate)
            periodicity_voiced[fi] = bool(periodicity.voiced)
            trace.emit(
                "PERIODICITY_DECISION", start,
                frame_start=start,
                frame_time_s=start / sample_rate,
                acf_peak=periodicity.acf_peak,
                cmndf_minimum=periodicity.cmndf_minimum,
                acf_frequency_hz=periodicity.acf_frequency_hz,
                cmndf_frequency_hz=periodicity.cmndf_frequency_hz,
                voiced=int(periodicity.voiced),
                reason=periodicity.reason,
            )
            if not periodicity.voiced:
                frame_decision[fi] = "PERIODICITY_REJECTED"
                # No oscillator is allowed to survive an unvoiced frame.
                # Output arrays were initialized to zero: do not fill or interpolate.
                P_last = None
                x_last = None
                flag = 0
                harm_prev = 0
                previous_frame_eligible = False
                start += block
                n = start
                continue

            y_prev = y[start - block:start] if start >= block else None
            if P_last is None or silent_prev:
                analysis_cur = detector.analyze(None, y_frame, sample_rate)
            else:
                analysis_cur = detector.analyze(y_prev, y_frame, sample_rate)
            harm_cur = analysis_cur.flag
            needs_initialization = (
                P_last is None or x_last is None
                or (harm_prev == 0 and harm_cur == 1)
            )
            #DIAGNOSTIC ONLY
            trace.emit(
                "INITIALIZATION_GATE",
                start,
                frame_start=start,
                periodicity_voiced=int(periodicity.voiced),
                periodicity_f0_hz=periodicity.acf_frequency_hz,
                periodicity_acf_peak=periodicity.acf_peak,
                periodicity_cmndf_minimum=periodicity.cmndf_minimum,
                state_available=int(P_last is not None and x_last is not None),
                harm_prev=harm_prev,
                harm_cur=harm_cur,
                needs_initialization=int(needs_initialization),
                existing_f0_hz=(
                    float(f0[start - 1])
                    if start > 0 and pitch_status[start - 1] == PitchStatus.VOICED_VALID
                    else None
                ),
            )
            #END DIAGNOSTIC ONLY
            if needs_initialization:
                # No lookahead and no backtracking: an unverified frame is never
                # assigned a pitch based on a different frame.
                initialization = detector.analyze(None, y_frame, sample_rate)
                previous_hz = None
                elapsed_ms = None
                if start > 0 and pitch_status[start - 1] == PitchStatus.VOICED_VALID:
                    previous_hz = float(f0[start - 1])
                    elapsed_ms = 1000.0 * block / sample_rate
                choice = choose_initialization(
                    detector, y_frame, sample_rate, start,
                    initialization, periodicity, previous_hz, elapsed_ms,
                )
#DIAGNOSE ONLY
                trace.emit(
                    "INITIALIZATION_CHOICE_BEFORE_OCTAVE",
                    start,
                    frame_start=start,
                    spectral_proposal_hz=initialization.f0_hz,
                    periodicity_hz=periodicity.acf_frequency_hz,
                    choice_hz=choice.frequency_hz,
                    choice_source=choice.source,
                    choice_reason=choice.reason,
                    choice_harmonics=choice.spectral_harmonics,
                )
#END DIAGNOSE ONLY
                # Independently challenge an octave-related initialization only
                # when ACF, CMNDF and measured spectral harmonics all agree.
                choice, octave_evidence = reconcile_initialization(
                    detector, y_frame, sample_rate, start, choice,
                    periodicity, previous_hz, elapsed_ms,
                )
#DIAGNOSE ONLY
                trace.emit(
                    "INITIALIZATION_CHOICE_AFTER_OCTAVE",
                    start,
                    frame_start=start,
                    evidence_state=octave_evidence.state,
                    evidence_reason=octave_evidence.reason,
                    evidence_measured_hz=octave_evidence.measured_hz,
                    final_choice_hz=choice.frequency_hz,
                    final_choice_source=choice.source,
                    final_choice_reason=choice.reason,
                )
#END DIAGNOSE ONLY
                trace.emit(
                    "OCTAVE_INITIALIZATION_AUDIT", start,
                    frame_start=start, decision=octave_evidence.state,
                    reason=octave_evidence.reason,
                    measured_f0_hz=octave_evidence.measured_hz,
                )
                f1 = choice.frequency_hz
                a1 = choice.amplitude
                phi1 = choice.phase
                trace.emit(
                    "INITIALIZATION_CANDIDATES", start,
                    frame_start=start,
                    proposed_f0_hz=initialization.f0_hz,
                    acoustic_f0_hz=periodicity.acf_frequency_hz,
                    selected_f0_hz=f1,
                    selected_source=choice.source,
                    harmonic_support=choice.spectral_harmonics,
                    transition_penalty=choice.transition_penalty,
                    reason=choice.reason,
                )
                trace.emit(
                    "INITIALIZATION_PROPOSED", start,
                    frame_start=start,
                    frame_time_s=start / sample_rate,
                    count=0,
                    init_f0_hz=initialization.f0_hz,
                    init_amplitude=initialization.amplitude,
                    flag=1,
                )
                if f1 is None or a1 is None or phi1 is None:
                    trace.emit("INITIALIZATION_REJECTED", start,
                               frame_start=start, reason=choice.reason)
                    pitch_status[start:end_exclusive] = PitchStatus.VOICED_UNRESOLVED
                    frame_decision[fi] = "INITIALIZATION_UNRESOLVED"
                    P_last = None
                    x_last = None
                    flag = 0
                    harm_prev = 0
                    previous_frame_eligible = False
                    start += block
                    n = start
                    continue
                n_matlab = start + 1
                x_last = np.array([
                    np.exp(1j * 2.0 * np.pi * f1 * Ts),
                    a1 * np.exp(1j * (2.0 * np.pi * f1 * n_matlab * Ts + phi1)),
                    a1 * np.exp(-1j * (2.0 * np.pi * f1 * n_matlab * Ts + phi1)),
                ], dtype=np.complex128).reshape(3, 1)
                P_last = np.zeros((3, 3), dtype=np.complex128)
                K = np.zeros((3, 1), dtype=np.complex128)
                onset_samples.append(start)
                trace.emit("RESET_COMPLETED", start,
                           frame_start=start, count=0, init_f0_hz=f1,
                           reset_accepted=1, flag=0)
            frame_decision[fi] = "VOICED_TRACKED"
            flag = 0
            n = start
        else:
            silent_prev = silent_cur
            silent_cur, cur_spf, cur_energy = is_silent(
                y_frame,
                flatness_threshold=config.silence_flatness_threshold,
                energy_db_threshold=silence_energy_threshold,
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
                    peak_1_hz=float(initialization.peak_frequencies_hz[0]),
                    peak_2_hz=float(initialization.peak_frequencies_hz[1]),
                    peak_3_hz=float(initialization.peak_frequencies_hz[2]),
                    spacing_1_hz=float(
                        initialization.peak_frequencies_hz[1]
                        - initialization.peak_frequencies_hz[0]
                    ),
                    spacing_2_hz=float(
                        initialization.peak_frequencies_hz[2]
                        - initialization.peak_frequencies_hz[1]
                    ),
                    rounded_spacing_1_hz=float(
                        np.floor(
                            initialization.peak_frequencies_hz[1]
                            - initialization.peak_frequencies_hz[0]
                            + 0.5
                        )
                    ),
                    rounded_spacing_2_hz=float(
                        np.floor(
                            initialization.peak_frequencies_hz[2]
                            - initialization.peak_frequencies_hz[1]
                            + 0.5
                        )
                    ),
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
                # TODO - delete if no longer necessary 
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
            if config.mode == "offline":
                pitch_status[n] = PitchStatus.VOICED_VALID
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
                    state_frequency_hz=f0[n],
                    # state_frequency_hz=(
                    #     float(
                    #         abs(
                    #             np.log(x1)
                    #             / (1j * Ts * 2.0 * np.pi)
                    #         )
                    #     )
                    # ),
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

        if config.mode == "offline":
            previous_frame_eligible = True
        start += block
        harm_prev = harm_cur

    onset_samples_arr = np.asarray(onset_samples, dtype=np.int64)
    trace.close()
    if config.mode == "offline":
        validate_pitch_status(f0, pitch_status)
    else:
        pitch_status[np.isfinite(f0) & (f0 > 0)] = PitchStatus.VOICED_VALID
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
        pitch_status=pitch_status,
        frame_start_samples=frame_start_samples,
        frame_real_sample_count=frame_real_sample_count,
        energy_silence_per_frame=energy_silence,
        flatness_silence_per_frame=flatness_silence,
        original_silence_per_frame=original_silence,
        periodicity_voiced_per_frame=periodicity_voiced,
        frame_decision=frame_decision,
    )
