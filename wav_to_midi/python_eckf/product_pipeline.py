from __future__ import annotations

"""Lean production analysis pipeline for WAV -> MIDI.

This module preserves the frozen Stage-6 tracker/adjudication/renderer inputs,
but omits development-only gesture/audit layers and writes only the seven small
compatibility lanes consumed by MRV18.3.  Those lanes live in a private
temporary directory created by :mod:`python_eckf.convert` and never appear in
the user's project folder.
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from .analysis_timing import DEFAULT_OFFLINE_ANALYSIS_HZ, resolve_offline_analysis_grid
from .config import ECKFConfig
from .expressive_amplitude import analyse_expressive_amplitude
from .expressive_smoothing import smooth_expressive_pitch
from .frame_evidence import frame_acoustic_state_name
from .harmonic_detective import build_harmonic_detective
from .juror_bench import build_juror_bench
from .juror_evidence import build_juror_evidence, derive_v22_range_calibration
from .pitch_candidate_field import build_pitch_candidate_field
from .primary_reconciliation_v3r9 import apply_primary_reconciliation_v3r9
from .range_reference_evaluator import build_v13_style_range_references
from .reacquisition_adjudication import apply_reacquisition_adjudication_feedback
from .register_readjudication_v18 import apply_register_readjudication_v18
from .runtime_resources import detect_runtime_resources
from .tracker import track_pitch
from .traffic_control import PersistentTrafficController
from .trajectory_interpreter import interpret_pitch_trajectory
from .validity import PitchValidityConfig, analyse_pitch_validity, apply_offline_validity_correction


# Frozen product settings.  These are intentionally not exposed as artist-facing
# command-line switches.
BLOCK_SIZE = 2048
C_VALUE = 7.0
WAIT_FRAMES = 3
NPEAKS = 3
HARMONIC_CHANGE_SEMITONES = 2.0
VOCAL_FLOOR_HZ = 60.0
SILENCE_THRESHOLD = -13.7734


def _build_render_interpretation_stack(
    *,
    offline_validity,
    sampled_frame_state,
    analysis_hz: float,
    audio: np.ndarray,
    sr: int,
    sample_idx: np.ndarray,
    runtime_resources,
    traffic_controller,
):
    """Build only the interpretation layers that MRV18.3 consumes.

    Development-only transition geometry, gesture objects, gesture splitting,
    interpretation-candidate diagnostics, interval-authority audit, and
    structural-collaboration audit are intentionally absent here.
    """
    interpretation = interpret_pitch_trajectory(
        clean_f0_hz=offline_validity.clean_f0_hz,
        valid=offline_validity.valid,
        analysis_hz=analysis_hz,
        frame_acoustic_state=sampled_frame_state,
        runtime_resources=runtime_resources,
        traffic_controller=traffic_controller,
        parallel_mode="auto",
        report_parallel=False,
    )
    expressive = smooth_expressive_pitch(
        pitch_st=interpretation.pitch_st,
        valid=interpretation.valid,
        analysis_hz=analysis_hz,
    )
    amplitude = analyse_expressive_amplitude(
        audio=audio,
        sample_rate=sr,
        sample_idx=sample_idx,
        trusted_pitch=interpretation.valid,
        pitch_residual_st=expressive.residual_pitch_st,
        analysis_hz=analysis_hz,
        runtime_resources=runtime_resources,
        traffic_controller=traffic_controller,
        parallel_mode="auto",
        report_parallel=False,
    )
    return interpretation, expressive, amplitude


def _write_renderer_sidecars(
    *,
    prefix: Path,
    sr: int,
    result,
    sample_idx: np.ndarray,
    times_s: np.ndarray,
    offline_validity,
    interpretation,
    expressive,
    amplitude,
    range_calibration,
) -> None:
    """Write only fields MRV18.3 actually reads.

    These files are a private compatibility boundary while the mature renderer
    still consumes its historical row format.  They are temporary and deleted
    automatically after the MIDI has been written.
    """
    prefix = Path(prefix)

    # Primary periodicity lane used only by bounded orphan-attack rescue.
    with Path(str(prefix) + ".frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "start_time_s",
            "periodicity_acf_peak",
            "periodicity_cmndf_minimum",
            "periodicity_acf_frequency_hz",
            "periodicity_cmndf_frequency_hz",
        ])
        for fi, start in enumerate(result.frame_start_samples):
            w.writerow([
                float(start / sr),
                float(result.periodicity_acf_peak_per_frame[fi]) if np.isfinite(result.periodicity_acf_peak_per_frame[fi]) else "",
                float(result.periodicity_cmndf_minimum_per_frame[fi]) if np.isfinite(result.periodicity_cmndf_minimum_per_frame[fi]) else "",
                float(result.periodicity_acf_frequency_hz_per_frame[fi]) if np.isfinite(result.periodicity_acf_frequency_hz_per_frame[fi]) else "",
                float(result.periodicity_cmndf_frequency_hz_per_frame[fi]) if np.isfinite(result.periodicity_cmndf_frequency_hz_per_frame[fi]) else "",
            ])

    # Structural target lane.
    with Path(str(prefix) + ".target_formation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s", "structural_pitch_st", "final_stable", "local_stability_pass",
            "min_stable_threshold_ms", "stable_range_threshold_st",
        ])
        for j in range(len(sample_idx)):
            w.writerow([
                float(times_s[j]),
                float(interpretation.structural_pitch_st[j]) if np.isfinite(interpretation.structural_pitch_st[j]) else "",
                int(interpretation.stable_mask[j]),
                int(interpretation.local_stability_pass[j]),
                float(interpretation.min_stable_threshold_ms),
                float(interpretation.stable_range_threshold_st),
            ])

    # Corrected expressive pitch lane.
    with Path(str(prefix) + ".expressive.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "shape_pitch_st", "corrected_valid"])
        for j in range(len(sample_idx)):
            w.writerow([
                float(times_s[j]),
                float(expressive.shape_pitch_st[j]) if np.isfinite(expressive.shape_pitch_st[j]) else "",
                int(interpretation.valid[j]),
            ])

    # Stable-target region starts used as renderer boundary proposals.
    with Path(str(prefix) + ".trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "start_time_s", "median_pitch_st"])
        for region in interpretation.regions:
            w.writerow([
                region.kind.name,
                float(region.start_time_s),
                float(region.median_pitch_st) if np.isfinite(region.median_pitch_st) else "",
            ])

    # Tracker reset landmarks.
    with Path(str(prefix) + ".onsets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s"])
        for seconds in result.onset_seconds:
            w.writerow([float(seconds)])

    # Only the zero-confidence boundaries are consumed by PitchEvidence.
    with Path(str(prefix) + ".range_calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zero_low_midi", "zero_high_midi"])
        w.writerow([
            "" if range_calibration.zero_low_midi is None else range_calibration.zero_low_midi,
            "" if range_calibration.zero_high_midi is None else range_calibration.zero_high_midi,
        ])

    # CC11 shape/residual lane.
    with Path(str(prefix) + ".amplitude_expression.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "shape_rms_dbfs", "amplitude_residual_db", "corrected_valid"])
        for j in range(len(sample_idx)):
            w.writerow([
                float(times_s[j]),
                float(amplitude.shape_rms_dbfs[j]) if np.isfinite(amplitude.shape_rms_dbfs[j]) else "",
                float(amplitude.residual_db[j]) if np.isfinite(amplitude.residual_db[j]) else "",
                int(interpretation.valid[j]),
            ])


def prepare_renderer_inputs(wav_path: Path, prefix: Path) -> None:
    """Run frozen Stage-6 analysis and create MRV18.3's private input lanes."""
    wav_path = Path(wav_path)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    resources = detect_runtime_resources()
    controller = PersistentTrafficController(resources)

    try:
        audio, sr = sf.read(wav_path, always_2d=False, dtype="float64")
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 1:
            raise ValueError("Input WAV must be mono")

        cfg = ECKFConfig(
            block_size=BLOCK_SIZE,
            c=C_VALUE,
            num_buf_to_wait=WAIT_FRAMES,
            npeaks=NPEAKS,
            nsemitones=HARMONIC_CHANGE_SEMITONES,
            mode="offline",
            vocal_floor_hz=VOCAL_FLOOR_HZ,
            silence_mode="adaptive",
            silence_energy_db_threshold=SILENCE_THRESHOLD,
            eckf_gain_normalization="peak",
        )

        result = track_pitch(
            audio,
            sr,
            cfg,
            kalman_engine="auto",
            runtime_resources=resources,
            frame_preflight_mode="auto",
            harmonic_preflight_mode="auto",
            variable_aperture_preflight_mode="auto",
            initialization_period_preflight_mode="auto",
            traffic_controller=controller,
        )

        grid = resolve_offline_analysis_grid(sr, DEFAULT_OFFLINE_ANALYSIS_HZ)
        sample_idx = np.arange(0, result.original_length, grid.hop_samples, dtype=np.int64)
        times_s = sample_idx.astype(np.float64) / float(sr)
        frame_index = sample_idx // cfg.block_size
        sampled_frame_state = result.frame_acoustic_state[frame_index]
        export_f0 = result.f0_hz[sample_idx]

        validity_cfg = PitchValidityConfig(analysis_hz=grid.actual_hz)
        first_pass = analyse_pitch_validity(
            f0_hz=export_f0,
            sample_rate=grid.actual_hz,
            config=validity_cfg,
            frame_acoustic_state=sampled_frame_state,
        )

        pitch_candidates = build_pitch_candidate_field(
            audio=audio,
            sample_rate=sr,
            config=cfg,
            tracker_result=result,
            frame_acoustic_state_name_fn=frame_acoustic_state_name,
            runtime_resources=resources,
            parallel_mode="auto",
            report=False,
            traffic_controller=controller,
        )

        preliminary_refs = build_v13_style_range_references(
            pitch_candidates=pitch_candidates,
            audio=audio,
            sample_rate=sr,
            sample_frame_index=frame_index,
            clean_f0_hz=first_pass.cleaned_f0_hz,
            valid=first_pass.valid,
        )
        preliminary_range = derive_v22_range_calibration(
            preliminary_refs.references_hz,
            source="pre_rescue_first_pass_v13_refs",
        )

        offline_validity = apply_offline_validity_correction(
            f0_hz=export_f0,
            first_pass=first_pass,
            sample_rate=grid.actual_hz,
            frame_acoustic_state=sampled_frame_state,
            range_zero_low_midi=(preliminary_range.zero_low_midi if preliminary_range.available else None),
            range_zero_high_midi=(preliminary_range.zero_high_midi if preliminary_range.available else None),
        )

        range_refs = build_v13_style_range_references(
            pitch_candidates=pitch_candidates,
            audio=audio,
            sample_rate=sr,
            sample_frame_index=frame_index,
            clean_f0_hz=offline_validity.clean_f0_hz,
            valid=offline_validity.valid,
        )
        juror_evidence, range_calibration = build_juror_evidence(
            pitch_candidates,
            supported_reference_hz=range_refs.references_hz,
            range_reference_rows=range_refs.rows,
        )
        juror_pairs, juror_bench, _ = build_juror_bench(
            juror_evidence,
            audio=audio,
            sample_rate=sr,
            runtime_resources=resources,
            parallel_mode="auto",
            report=False,
            traffic_controller=controller,
        )
        _, _, final_adjudication = build_harmonic_detective(
            audio=audio,
            sr=sr,
            evidence_rows=juror_evidence,
            bench_rows=juror_bench,
            pair_rows=juror_pairs,
        )

        apply_reacquisition_adjudication_feedback(
            offline_validity=offline_validity,
            sample_times_s=times_s,
            sample_frame_index=frame_index,
            audio=audio,
            sample_rate=sr,
            evidence_rows=juror_evidence,
            bench_rows=juror_bench,
            pair_rows=juror_pairs,
            final_adjudication=final_adjudication,
        )

        apply_primary_reconciliation_v3r9(
            offline_validity=offline_validity,
            sample_times_s=times_s,
            sample_frame_index=frame_index,
            frame_start_times_s=np.asarray(result.frame_start_samples, dtype=np.float64) / float(sr),
            frame_acoustic_state=result.frame_acoustic_state,
            periodicity_acf_peak_per_frame=result.periodicity_acf_peak_per_frame,
            periodicity_cmndf_minimum_per_frame=result.periodicity_cmndf_minimum_per_frame,
            periodicity_acf_frequency_hz_per_frame=result.periodicity_acf_frequency_hz_per_frame,
            periodicity_cmndf_frequency_hz_per_frame=result.periodicity_cmndf_frequency_hz_per_frame,
            range_zero_low_midi=(range_calibration.zero_low_midi if range_calibration.available else None),
            range_zero_high_midi=(range_calibration.zero_high_midi if range_calibration.available else None),
        )

        sampled_frame_state = result.frame_acoustic_state[frame_index]
        interpretation, expressive, amplitude = _build_render_interpretation_stack(
            offline_validity=offline_validity,
            sampled_frame_state=sampled_frame_state,
            analysis_hz=grid.actual_hz,
            audio=audio,
            sr=sr,
            sample_idx=sample_idx,
            runtime_resources=resources,
            traffic_controller=controller,
        )

        register_rows = apply_register_readjudication_v18(
            offline_validity=offline_validity,
            interpretation=interpretation,
            sample_times_s=times_s,
            sample_frame_index=frame_index,
            audio=audio,
            sample_rate=sr,
            evidence_rows=juror_evidence,
            pair_rows=juror_pairs,
            final_adjudication=final_adjudication,
        )
        if register_rows:
            interpretation, expressive, amplitude = _build_render_interpretation_stack(
                offline_validity=offline_validity,
                sampled_frame_state=sampled_frame_state,
                analysis_hz=grid.actual_hz,
                audio=audio,
                sr=sr,
                sample_idx=sample_idx,
                runtime_resources=resources,
                traffic_controller=controller,
            )

        _write_renderer_sidecars(
            prefix=prefix,
            sr=sr,
            result=result,
            sample_idx=sample_idx,
            times_s=times_s,
            offline_validity=offline_validity,
            interpretation=interpretation,
            expressive=expressive,
            amplitude=amplitude,
            range_calibration=range_calibration,
        )
    finally:
        controller.shutdown()
