from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import csv
from dataclasses import fields
from enum import Enum

from .config import ECKFConfig
from .tracker import track_pitch
from .frame_evidence import frame_acoustic_state_name
from .trajectory_interpreter import interpret_pitch_trajectory
from .expressive_smoothing import smooth_expressive_pitch
from .expressive_amplitude import analyse_expressive_amplitude
from .transition_geometry import compare_transition_geometry
from .gesture_features import extract_gesture_features
from .gesture_objects import construct_gesture_objects
from .gesture_object_splitter import analyse_gesture_object_splits
from .gesture_structural_splitter import apply_structural_splitting
from .gesture_hypotheses import build_interpretation_candidates
from .pitch_candidate_field import build_pitch_candidate_field
from .juror_evidence import build_juror_evidence
from .range_reference_evaluator import build_v13_style_range_references
from .juror_bench import build_juror_bench
from .harmonic_detective import build_harmonic_detective
from .validity import (
    PitchValidityConfig,
    analyse_pitch_validity,
    apply_offline_validity_correction,
)

def main():
    p = argparse.ArgumentParser(
        description="ECKF monophonic pitch tracker: MATLAB-compatible / vocal-offline modes"
    )
    p.add_argument("wav", type=Path)
    p.add_argument("--mode", choices=["matlab", "offline"], default="offline")
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--c", type=float, default=7.0)
    p.add_argument("--wait", type=int, default=2,
                   help="MATLAB: fixed skip; offline: maximum adaptive lookahead frames (0 disables).")
    p.add_argument("--npeaks", type=int, default=3)
    p.add_argument("--nsemitones", type=float, default=2.0)
    p.add_argument("--vocal-floor-hz", type=float, default=60.0)
    p.add_argument(
        "--silence-mode", choices=["fixed", "adaptive"], default="fixed",
        help="Silence calibration mode. Only fixed is implemented; adaptive explicitly errors.",
    )
    p.add_argument(
        "--silence-energy-threshold", type=float, default=-50.0,
        help="Fixed threshold on original 20*log10(sum(frame**2)) scale (NOT dBFS).",
    )
    p.add_argument(
        "--eckf-gain-normalization", choices=["none", "peak"], default="peak",
        help="Offline ECKF internal coordinate normalization. peak is gain-invariant; MATLAB mode is always historical/raw.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <wav>.eckf.csv)",
    )
    args = p.parse_args()

    audio, sr = sf.read(args.wav, always_2d=False, dtype="float64")
    if np.asarray(audio).ndim != 1:
        raise SystemExit(
            "Input must be mono. No automatic downmix is performed."
        )

    cfg = ECKFConfig(
        block_size=args.block_size,
        c=args.c,
        num_buf_to_wait=args.wait,
        npeaks=args.npeaks,
        nsemitones=args.nsemitones,
        mode=args.mode,
        vocal_floor_hz=args.vocal_floor_hz,
        silence_mode=args.silence_mode,
        silence_energy_db_threshold=args.silence_energy_threshold,
        eckf_gain_normalization=args.eckf_gain_normalization,
    )

    result = track_pitch(audio, sr, cfg)

    out = args.csv
    if out is None:
        out = args.wav.with_suffix(args.wav.suffix + ".eckf.csv")

    n = result.original_length

    # Diagnostic export only.
    # The ECKF itself remains full sample-resolution.
    EXPORT_HZ = 100.0

    export_step = max(
        1,
        int(round(sr / EXPORT_HZ)),
    )

    sample_idx = np.arange(
        0,
        n,
        export_step,
        dtype=np.int64,
    )

    t = sample_idx.astype(np.float64) / sr

    # -------------------------------------------------------------
    # Offline trajectory validity / bidirectional correction.
    #
    # IMPORTANT:
    # The ECKF itself remains full sample-resolution.
    # Validity operates only on the 100 Hz offline trajectory.
    # MATLAB mode is left completely untouched.
    # -------------------------------------------------------------

    offline_validity = None
    offline_interpretation = None
    offline_expressive = None
    offline_transition_geometry = None
    offline_expressive_amplitude = None
    offline_gesture_features = None
    offline_gesture_objects = None
    offline_gesture_split_evidence = None
    offline_gesture_structure = None
    offline_interpretation_candidates = None
    offline_pitch_candidates = None
    offline_juror_evidence = None
    offline_range_calibration = None
    offline_range_reference_result = None
    offline_juror_pairs = None
    offline_juror_bench = None
    offline_detective_candidates = None
    offline_detective_verdicts = None
    offline_final_adjudication = None
    frame_index = sample_idx // cfg.block_size

    if cfg.mode == "offline":
        export_f0 = result.f0_hz[sample_idx]

        analysis_hz = (
            sr / float(export_step)
        )

        validity_config = PitchValidityConfig(
            analysis_hz=analysis_hz,
        )

        sampled_frame_state = result.frame_acoustic_state[frame_index]

        first_pass_validity = analyse_pitch_validity(
            f0_hz=export_f0,
            sample_rate=analysis_hz,
            config=validity_config,
            frame_acoustic_state=sampled_frame_state,
        )

        offline_validity = apply_offline_validity_correction(
            f0_hz=export_f0,
            first_pass=first_pass_validity,
            sample_rate=analysis_hz,
            frame_acoustic_state=sampled_frame_state,
        )

        # V2 active trajectory stage.  The interpreter consumes the exact
        # final validity mask and the explicit frame-acoustic-state passport;
        # it is no longer merely a library helper that callers may bypass.
        offline_interpretation = interpret_pitch_trajectory(
            clean_f0_hz=offline_validity.clean_f0_hz,
            valid=offline_validity.valid,
            analysis_hz=analysis_hz,
            frame_acoustic_state=sampled_frame_state,
        )

        # ---------------------------------------------------------
        # V2 expressive trajectory lane.
        #
        # This is deliberately observational.  It preserves three views
        # of the same trusted trajectory for later ornament/MIDI work:
        #   raw    = corrected F0, never replaced;
        #   shape  = light symmetric smoothing for motion geometry;
        #   center = slower robust centerline;
        #   residual = shape - center.
        #
        # No note quantization and no ornament classification occurs here.
        # Stable-target formation remains exactly as decided above.
        # ---------------------------------------------------------
        offline_expressive = smooth_expressive_pitch(
            pitch_st=offline_interpretation.pitch_st,
            valid=offline_interpretation.valid,
            analysis_hz=analysis_hz,
        )

        # Parallel amplitude-expression lane.  This preserves raw RMS for
        # provenance plus gain-robust local modulation evidence for future
        # note-velocity / CC-expression rendering.  It does not classify
        # amplitude vibrato or emit MIDI controller decisions.
        offline_expressive_amplitude = analyse_expressive_amplitude(
            audio=audio,
            sample_rate=sr,
            sample_idx=sample_idx,
            trusted_pitch=offline_interpretation.valid,
            pitch_residual_st=offline_expressive.residual_pitch_st,
            analysis_hz=analysis_hz,
        )

        offline_transition_geometry = compare_transition_geometry(
            interpretation=offline_interpretation,
            smoothing=offline_expressive,
        )

        # V2 gesture-structure lane. Structural only: no ornament labels,
        # note quantisation, or MIDI rendering decisions.
        offline_gesture_features = extract_gesture_features(
            interpretation=offline_interpretation,
            smoothing=offline_expressive,
        )

        offline_gesture_objects = construct_gesture_objects(
            interpretation=offline_interpretation,
            features=offline_gesture_features,
        )

        offline_gesture_split_evidence = analyse_gesture_object_splits(
            objects=offline_gesture_objects,
            features=offline_gesture_features,
        )

        offline_gesture_structure = apply_structural_splitting(
            split_evidence=offline_gesture_split_evidence,
        )

        # V2 interpretation-candidate substrate.  This deliberately keeps
        # multiple possible musical/MIDI representations alive; no ornament
        # label or rendering winner is selected here.
        offline_interpretation_candidates = build_interpretation_candidates(
            structure=offline_gesture_structure,
            interpretation=offline_interpretation,
            smoothing=offline_expressive,
            gesture_features=offline_gesture_features,
            amplitude_expression=offline_expressive_amplitude,
            analysis_hz=analysis_hz,
        )

        # Observational pitch-candidate field for the future adjudication bench.
        # No juror is seated yet and no existing decision is changed.
        offline_pitch_candidates = build_pitch_candidate_field(
            audio=audio,
            sample_rate=sr,
            config=cfg,
            tracker_result=result,
            frame_acoustic_state_name_fn=frame_acoustic_state_name,
        )

        # Historical V13-style range-reference evaluator restored.  The
        # range population is no longer "every corrected-valid sample".  A
        # candidate must first be a unique locally supported measured period,
        # using the old 35c selection / 70c competitor / ACF+component-strength
        # contract.  The V22 q02/q98 + plateau/shoulder geometry remains
        # unchanged downstream.
        offline_range_reference_result = build_v13_style_range_references(
            pitch_candidates=offline_pitch_candidates,
            audio=np.asarray(audio, dtype=np.float64),
            sample_rate=sr,
            sample_frame_index=frame_index,
            clean_f0_hz=offline_validity.clean_f0_hz,
            valid=offline_validity.valid,
        )
        offline_juror_evidence, offline_range_calibration = build_juror_evidence(
            offline_pitch_candidates,
            supported_reference_hz=offline_range_reference_result.references_hz,
        )
        # Jurors are now seated: verdict-producing, but still observational.
        offline_juror_pairs, offline_juror_bench = build_juror_bench(offline_juror_evidence)
        # Conditional expert witness: called ONLY on jury abstentions. Existing
        # jury champions are immutable. Hard range/acoustic gate failures are
        # never rescued.
        offline_detective_candidates, offline_detective_verdicts, offline_final_adjudication = build_harmonic_detective(
            audio=np.asarray(audio, dtype=np.float64),
            sr=sr,
            evidence_rows=offline_juror_evidence,
            bench_rows=offline_juror_bench,
        )

        # Per-sample region kind for diagnostic export.  Untrusted samples
        # remain empty rather than being silently classified as trajectory.
        trajectory_region_kind = np.full(
            len(sample_idx),
            "",
            dtype=object,
        )
        for region in offline_interpretation.regions:
            trajectory_region_kind[
                region.start_index:region.end_index
            ] = region.kind.name

    if cfg.mode == "offline":
        with out.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "sample",
                    "time_s",
                    "f0_hz",
                    "amplitude",
                    "phase",
                    "fundamental_real",
                    "fundamental_imag",
                    "first_pass_valid",
                    "corrected_valid",
                    "clean_f0_hz",
                    "first_pass_reason",
                    "validity_reason",
                    "correction_reason",
                    "pitch_status",
                    "frame_start_sample",
                    "frame_real_sample_count",
                    "energy_silence",
                    "flatness_silence",
                    "original_silence",
                    "periodicity_voiced",
                    "frame_acoustic_state",
                    "frame_acoustic_state_name",
                    "periodicity_acf_peak",
                    "periodicity_cmndf_minimum",
                    "periodicity_acf_frequency_hz",
                    "periodicity_cmndf_frequency_hz",
                    "periodicity_reason",
                    "initialization_source",
                    "initialization_frequency_hz",
                    "lookahead_frames",
                    "lookahead_reason",
                    "frame_decision",
                    "trajectory_region_id",
                    "trajectory_region_kind",
                    "trajectory_stable",
                    "pitch_st",
                    "structural_pitch_st",
                ]
            )

            for j, sample in enumerate(
                sample_idx
            ):
                writer.writerow(
                    [
                        int(sample),
                        float(t[j]),
                        float(
                            result.f0_hz[sample]
                        ),
                        float(
                            result.amplitude[sample]
                        ),
                        float(
                            result.phase[sample]
                        ),
                        float(
                            np.real(
                                result.fundamental[
                                    sample
                                ]
                            )
                        ),
                        float(
                            np.imag(
                                result.fundamental[
                                    sample
                                ]
                            )
                        ),
                        int(
                            offline_validity.first_pass_valid[
                                j
                            ]
                        ),
                        int(
                            offline_validity.valid[
                                j
                            ]
                        ),
                        (
                            float(
                                offline_validity.clean_f0_hz[
                                    j
                                ]
                            )
                            if np.isfinite(
                                offline_validity.clean_f0_hz[
                                    j
                                ]
                            )
                            else ""
                        ),
                        str(
                            offline_validity.first_pass_reason[
                                j
                            ]
                        ),
                        str(
                            offline_validity.reason[
                                j
                            ]
                        ),
                        str(
                            offline_validity.correction_reason[
                                j
                            ]
                        ),
                        int(result.pitch_status[sample]),
                        int(result.frame_start_samples[frame_index[j]]),
                        int(result.frame_real_sample_count[frame_index[j]]),
                        int(result.energy_silence_per_frame[frame_index[j]]),
                        int(result.flatness_silence_per_frame[frame_index[j]]),
                        int(result.original_silence_per_frame[frame_index[j]]),
                        int(result.periodicity_voiced_per_frame[frame_index[j]]),
                        int(result.frame_acoustic_state[frame_index[j]]),
                        frame_acoustic_state_name(result.frame_acoustic_state[frame_index[j]]),
                        (float(result.periodicity_acf_peak_per_frame[frame_index[j]])
                         if np.isfinite(result.periodicity_acf_peak_per_frame[frame_index[j]]) else ""),
                        (float(result.periodicity_cmndf_minimum_per_frame[frame_index[j]])
                         if np.isfinite(result.periodicity_cmndf_minimum_per_frame[frame_index[j]]) else ""),
                        (float(result.periodicity_acf_frequency_hz_per_frame[frame_index[j]])
                         if np.isfinite(result.periodicity_acf_frequency_hz_per_frame[frame_index[j]]) else ""),
                        (float(result.periodicity_cmndf_frequency_hz_per_frame[frame_index[j]])
                         if np.isfinite(result.periodicity_cmndf_frequency_hz_per_frame[frame_index[j]]) else ""),
                        str(result.periodicity_reason_per_frame[frame_index[j]]),
                        str(result.initialization_source_per_frame[frame_index[j]]),
                        (float(result.initialization_frequency_hz_per_frame[frame_index[j]])
                         if np.isfinite(result.initialization_frequency_hz_per_frame[frame_index[j]]) else ""),
                        int(result.lookahead_frames_per_frame[frame_index[j]]),
                        str(result.lookahead_reason_per_frame[frame_index[j]]),
                        str(result.frame_decision[frame_index[j]]),
                        int(offline_interpretation.region_id[j]),
                        str(trajectory_region_kind[j]),
                        int(offline_interpretation.stable_mask[j]),
                        (float(offline_interpretation.pitch_st[j])
                         if np.isfinite(offline_interpretation.pitch_st[j]) else ""),
                        (float(offline_interpretation.structural_pitch_st[j])
                         if np.isfinite(offline_interpretation.structural_pitch_st[j]) else ""),
                    ]
                )

        frames_out = out.with_suffix(out.suffix + ".frames.csv")
        with frames_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_index", "start_sample", "start_time_s",
                "real_end_sample_exclusive", "real_sample_count",
                "energy_on_matlab_scale", "spectral_flatness",
                "energy_silence", "flatness_silence", "original_silence",
                "periodicity_voiced", "frame_acoustic_state", "frame_acoustic_state_name",
                "periodicity_acf_peak", "periodicity_cmndf_minimum",
                "periodicity_acf_frequency_hz", "periodicity_cmndf_frequency_hz",
                "periodicity_reason", "initialization_source",
                "initialization_frequency_hz", "lookahead_frames", "lookahead_reason",
                "frame_decision",
            ])
            for fi, start in enumerate(result.frame_start_samples):
                writer.writerow([
                    fi, int(start), float(start / sr),
                    int(start + result.frame_real_sample_count[fi]),
                    int(result.frame_real_sample_count[fi]),
                    float(result.energy_db_per_frame[fi]),
                    float(result.spectral_flatness_per_frame[fi]),
                    int(result.energy_silence_per_frame[fi]),
                    int(result.flatness_silence_per_frame[fi]),
                    int(result.original_silence_per_frame[fi]),
                    int(result.periodicity_voiced_per_frame[fi]),
                    int(result.frame_acoustic_state[fi]),
                    frame_acoustic_state_name(result.frame_acoustic_state[fi]),
                    (float(result.periodicity_acf_peak_per_frame[fi])
                     if np.isfinite(result.periodicity_acf_peak_per_frame[fi]) else ""),
                    (float(result.periodicity_cmndf_minimum_per_frame[fi])
                     if np.isfinite(result.periodicity_cmndf_minimum_per_frame[fi]) else ""),
                    (float(result.periodicity_acf_frequency_hz_per_frame[fi])
                     if np.isfinite(result.periodicity_acf_frequency_hz_per_frame[fi]) else ""),
                    (float(result.periodicity_cmndf_frequency_hz_per_frame[fi])
                     if np.isfinite(result.periodicity_cmndf_frequency_hz_per_frame[fi]) else ""),
                    str(result.periodicity_reason_per_frame[fi]),
                    str(result.initialization_source_per_frame[fi]),
                    (float(result.initialization_frequency_hz_per_frame[fi])
                     if np.isfinite(result.initialization_frequency_hz_per_frame[fi]) else ""),
                    int(result.lookahead_frames_per_frame[fi]),
                    str(result.lookahead_reason_per_frame[fi]),
                    str(result.frame_decision[fi]),
                ])
        print(f"Frame evidence: {frames_out}")

        trajectory_out = out.with_suffix(out.suffix + ".trajectory.csv")
        with trajectory_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "region_index", "kind",
                "start_index", "end_index",
                "start_time_s", "end_time_s", "duration_ms",
                "median_pitch_st", "min_pitch_st", "max_pitch_st",
                "span_st", "raw_median_pitch_st",
                "previous_target_st", "next_target_st",
            ])
            for region in offline_interpretation.regions:
                writer.writerow([
                    int(region.index),
                    region.kind.name,
                    int(region.start_index),
                    int(region.end_index),
                    float(region.start_time_s),
                    float(region.end_time_s),
                    float(region.duration_ms),
                    (float(region.median_pitch_st) if np.isfinite(region.median_pitch_st) else ""),
                    (float(region.min_pitch_st) if np.isfinite(region.min_pitch_st) else ""),
                    (float(region.max_pitch_st) if np.isfinite(region.max_pitch_st) else ""),
                    (float(region.span_st) if np.isfinite(region.span_st) else ""),
                    (float(region.raw_median_pitch_st) if np.isfinite(region.raw_median_pitch_st) else ""),
                    (float(region.previous_target_st) if np.isfinite(region.previous_target_st) else ""),
                    (float(region.next_target_st) if np.isfinite(region.next_target_st) else ""),
                ])
        print(f"Trajectory regions: {trajectory_out}")

        target_out = out.with_suffix(out.suffix + ".target_formation.csv")
        with target_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trajectory_index", "sample", "time_s",
                "corrected_valid", "clean_f0_hz",
                "pitch_st", "structural_pitch_st",
                "local_stability_point_count",
                "local_robust_range_st",
                "stable_range_threshold_st",
                "local_stability_pass",
                "va_first_passing_aperture_ms",
                "va_largest_passing_aperture_ms",
                "va_largest_contiguous_passing_aperture_ms",
                "va_core_candidate",
                "va_core_after_duration",
                "va_edge_grown",
                "va_24_point_count", "va_24_robust_range_st", "va_24_pass",
                "va_40_point_count", "va_40_robust_range_st", "va_40_pass",
                "va_64_point_count", "va_64_robust_range_st", "va_64_pass",
                "va_96_point_count", "va_96_robust_range_st", "va_96_pass",
                "stable_run_duration_ms",
                "min_stable_threshold_ms",
                "min_duration_pass",
                "final_stable",
                "target_formation_reason",
                "trajectory_region_id",
                "trajectory_region_kind",
            ])
            for j, sample in enumerate(sample_idx):
                writer.writerow([
                    int(j),
                    int(sample),
                    float(t[j]),
                    int(offline_validity.valid[j]),
                    (float(offline_validity.clean_f0_hz[j])
                     if np.isfinite(offline_validity.clean_f0_hz[j]) else ""),
                    (float(offline_interpretation.pitch_st[j])
                     if np.isfinite(offline_interpretation.pitch_st[j]) else ""),
                    (float(offline_interpretation.structural_pitch_st[j])
                     if np.isfinite(offline_interpretation.structural_pitch_st[j]) else ""),
                    int(offline_interpretation.local_stability_point_count[j]),
                    (float(offline_interpretation.local_robust_range_st[j])
                     if np.isfinite(offline_interpretation.local_robust_range_st[j]) else ""),
                    float(offline_interpretation.stable_range_threshold_st),
                    int(offline_interpretation.local_stability_pass[j]),
                    (float(offline_interpretation.variable_first_passing_aperture_ms[j])
                     if np.isfinite(offline_interpretation.variable_first_passing_aperture_ms[j]) else ""),
                    (float(offline_interpretation.variable_largest_passing_aperture_ms[j])
                     if np.isfinite(offline_interpretation.variable_largest_passing_aperture_ms[j]) else ""),
                    (float(offline_interpretation.variable_largest_contiguous_passing_aperture_ms[j])
                     if np.isfinite(offline_interpretation.variable_largest_contiguous_passing_aperture_ms[j]) else ""),
                    int(offline_interpretation.variable_core_candidate[j]),
                    int(offline_interpretation.variable_core_after_duration[j]),
                    int(offline_interpretation.variable_edge_grown[j]),
                    int(offline_interpretation.variable_aperture_point_count[j, 0]),
                    (float(offline_interpretation.variable_aperture_robust_range_st[j, 0])
                     if np.isfinite(offline_interpretation.variable_aperture_robust_range_st[j, 0]) else ""),
                    int(offline_interpretation.variable_aperture_pass[j, 0]),
                    int(offline_interpretation.variable_aperture_point_count[j, 1]),
                    (float(offline_interpretation.variable_aperture_robust_range_st[j, 1])
                     if np.isfinite(offline_interpretation.variable_aperture_robust_range_st[j, 1]) else ""),
                    int(offline_interpretation.variable_aperture_pass[j, 1]),
                    int(offline_interpretation.variable_aperture_point_count[j, 2]),
                    (float(offline_interpretation.variable_aperture_robust_range_st[j, 2])
                     if np.isfinite(offline_interpretation.variable_aperture_robust_range_st[j, 2]) else ""),
                    int(offline_interpretation.variable_aperture_pass[j, 2]),
                    int(offline_interpretation.variable_aperture_point_count[j, 3]),
                    (float(offline_interpretation.variable_aperture_robust_range_st[j, 3])
                     if np.isfinite(offline_interpretation.variable_aperture_robust_range_st[j, 3]) else ""),
                    int(offline_interpretation.variable_aperture_pass[j, 3]),
                    float(offline_interpretation.stable_run_duration_ms[j]),
                    float(offline_interpretation.min_stable_threshold_ms),
                    int(offline_interpretation.min_duration_pass[j]),
                    int(offline_interpretation.stable_mask[j]),
                    str(offline_interpretation.target_formation_reason[j]),
                    int(offline_interpretation.region_id[j]),
                    str(trajectory_region_kind[j]),
                ])
        print(f"Target formation audit: {target_out}")

        # -------------------------------------------------------------
        # Expressive lane sample export.
        #
        # This file is intentionally future-MIDI-friendly: the raw
        # corrected trajectory is kept beside a light shape trajectory
        # and a slower centerline.  Later code can decide whether a
        # movement becomes discrete notes, pitch bend, or remains
        # unresolved without reconstructing information discarded here.
        # -------------------------------------------------------------
        expressive_out = out.with_suffix(out.suffix + ".expressive.csv")
        with expressive_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trajectory_index", "sample", "time_s",
                "corrected_valid",
                "trajectory_region_id", "trajectory_region_kind",
                "trajectory_stable",
                "raw_pitch_st",
                "shape_pitch_st",
                "center_pitch_st",
                "residual_pitch_st",
                "raw_rms_linear",
                "raw_rms_dbfs",
                "shape_rms_dbfs",
                "baseline_rms_dbfs",
                "amplitude_residual_db",
                "amplitude_mod_depth_peak_to_peak_db",
                "amplitude_mod_depth_rms_db",
                "amplitude_mod_rate_hz",
                "amplitude_mod_periodicity",
                "pitch_amplitude_residual_correlation",
            ])
            for j, sample in enumerate(sample_idx):
                writer.writerow([
                    int(j),
                    int(sample),
                    float(t[j]),
                    int(offline_interpretation.valid[j]),
                    int(offline_interpretation.region_id[j]),
                    str(trajectory_region_kind[j]),
                    int(offline_interpretation.stable_mask[j]),
                    (float(offline_expressive.raw_pitch_st[j])
                     if np.isfinite(offline_expressive.raw_pitch_st[j]) else ""),
                    (float(offline_expressive.shape_pitch_st[j])
                     if np.isfinite(offline_expressive.shape_pitch_st[j]) else ""),
                    (float(offline_expressive.center_pitch_st[j])
                     if np.isfinite(offline_expressive.center_pitch_st[j]) else ""),
                    (float(offline_expressive.residual_pitch_st[j])
                     if np.isfinite(offline_expressive.residual_pitch_st[j]) else ""),
                    (float(offline_expressive_amplitude.raw_rms_linear[j])
                     if np.isfinite(offline_expressive_amplitude.raw_rms_linear[j]) else ""),
                    (float(offline_expressive_amplitude.raw_rms_dbfs[j])
                     if np.isfinite(offline_expressive_amplitude.raw_rms_dbfs[j]) else ""),
                    (float(offline_expressive_amplitude.shape_rms_dbfs[j])
                     if np.isfinite(offline_expressive_amplitude.shape_rms_dbfs[j]) else ""),
                    (float(offline_expressive_amplitude.baseline_rms_dbfs[j])
                     if np.isfinite(offline_expressive_amplitude.baseline_rms_dbfs[j]) else ""),
                    (float(offline_expressive_amplitude.residual_db[j])
                     if np.isfinite(offline_expressive_amplitude.residual_db[j]) else ""),
                    (float(offline_expressive_amplitude.local_depth_peak_to_peak_db[j])
                     if np.isfinite(offline_expressive_amplitude.local_depth_peak_to_peak_db[j]) else ""),
                    (float(offline_expressive_amplitude.local_depth_rms_db[j])
                     if np.isfinite(offline_expressive_amplitude.local_depth_rms_db[j]) else ""),
                    (float(offline_expressive_amplitude.local_modulation_rate_hz[j])
                     if np.isfinite(offline_expressive_amplitude.local_modulation_rate_hz[j]) else ""),
                    (float(offline_expressive_amplitude.local_modulation_periodicity[j])
                     if np.isfinite(offline_expressive_amplitude.local_modulation_periodicity[j]) else ""),
                    (float(offline_expressive_amplitude.local_pitch_amplitude_correlation[j])
                     if np.isfinite(offline_expressive_amplitude.local_pitch_amplitude_correlation[j]) else ""),
                ])
        print(f"Expressive trajectory: {expressive_out}")

        # -------------------------------------------------------------
        # Dedicated amplitude-expression export.
        #
        # Raw dBFS is kept for provenance.  The residual/depth/rate fields
        # are relative local-dynamics evidence and are the future-facing
        # representation for amplitude vibrato/tremolo and MIDI expression.
        # No MIDI CC or ornament label is assigned here.
        # -------------------------------------------------------------
        amplitude_out = out.with_suffix(out.suffix + ".amplitude_expression.csv")
        with amplitude_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trajectory_index", "sample", "time_s",
                "corrected_valid", "trajectory_region_id", "trajectory_region_kind",
                "trajectory_stable",
                "raw_rms_linear", "raw_rms_dbfs",
                "shape_rms_dbfs", "baseline_rms_dbfs",
                "amplitude_residual_db",
                "mod_depth_peak_to_peak_db", "mod_depth_rms_db",
                "mod_rate_hz", "mod_periodicity",
                "pitch_amplitude_residual_correlation",
            ])
            for j, sample in enumerate(sample_idx):
                writer.writerow([
                    int(j), int(sample), float(t[j]),
                    int(offline_interpretation.valid[j]),
                    int(offline_interpretation.region_id[j]),
                    str(trajectory_region_kind[j]),
                    int(offline_interpretation.stable_mask[j]),
                    (float(offline_expressive_amplitude.raw_rms_linear[j])
                     if np.isfinite(offline_expressive_amplitude.raw_rms_linear[j]) else ""),
                    (float(offline_expressive_amplitude.raw_rms_dbfs[j])
                     if np.isfinite(offline_expressive_amplitude.raw_rms_dbfs[j]) else ""),
                    (float(offline_expressive_amplitude.shape_rms_dbfs[j])
                     if np.isfinite(offline_expressive_amplitude.shape_rms_dbfs[j]) else ""),
                    (float(offline_expressive_amplitude.baseline_rms_dbfs[j])
                     if np.isfinite(offline_expressive_amplitude.baseline_rms_dbfs[j]) else ""),
                    (float(offline_expressive_amplitude.residual_db[j])
                     if np.isfinite(offline_expressive_amplitude.residual_db[j]) else ""),
                    (float(offline_expressive_amplitude.local_depth_peak_to_peak_db[j])
                     if np.isfinite(offline_expressive_amplitude.local_depth_peak_to_peak_db[j]) else ""),
                    (float(offline_expressive_amplitude.local_depth_rms_db[j])
                     if np.isfinite(offline_expressive_amplitude.local_depth_rms_db[j]) else ""),
                    (float(offline_expressive_amplitude.local_modulation_rate_hz[j])
                     if np.isfinite(offline_expressive_amplitude.local_modulation_rate_hz[j]) else ""),
                    (float(offline_expressive_amplitude.local_modulation_periodicity[j])
                     if np.isfinite(offline_expressive_amplitude.local_modulation_periodicity[j]) else ""),
                    (float(offline_expressive_amplitude.local_pitch_amplitude_correlation[j])
                     if np.isfinite(offline_expressive_amplitude.local_pitch_amplitude_correlation[j]) else ""),
                ])
        print(f"Amplitude expression: {amplitude_out}")

        # -------------------------------------------------------------
        # Transition geometry export.
        #
        # Each existing TRANSITION region is measured twice:
        #   raw   = untouched corrected trajectory
        #   shape = lightly smoothed trajectory
        # This is geometry only.  No glissando/portamento/mordent/etc.
        # label is assigned.
        # -------------------------------------------------------------
        transition_out = out.with_suffix(out.suffix + ".transitions.csv")
        with transition_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "region_index",
                "start_index", "end_index",
                "start_time_s", "end_time_s", "duration_ms",
                "source_target_st", "destination_target_st",
                "has_source", "has_destination", "target_interval_st",
                "raw_start_pitch_st", "raw_end_pitch_st",
                "raw_median_pitch_st", "raw_span_st",
                "raw_net_movement_st", "raw_path_length_st",
                "raw_path_efficiency",
                "raw_upward_fraction", "raw_downward_fraction",
                "raw_dominant_direction", "raw_monotonic_fraction",
                "raw_direction_reversals",
                "raw_max_adjacent_step_st", "raw_median_adjacent_step_st",
                "raw_overshoot_above_st", "raw_undershoot_below_st",
                "raw_min_distance_source_st",
                "raw_min_distance_destination_st",
                "raw_fraction_near_source",
                "raw_fraction_near_destination",
                "raw_source_return_evidence",
                "shape_start_pitch_st", "shape_end_pitch_st",
                "shape_median_pitch_st", "shape_span_st",
                "shape_net_movement_st", "shape_path_length_st",
                "shape_path_efficiency",
                "shape_upward_fraction", "shape_downward_fraction",
                "shape_dominant_direction", "shape_monotonic_fraction",
                "shape_direction_reversals",
                "shape_max_adjacent_step_st", "shape_median_adjacent_step_st",
                "shape_overshoot_above_st", "shape_undershoot_below_st",
                "shape_min_distance_source_st",
                "shape_min_distance_destination_st",
                "shape_fraction_near_source",
                "shape_fraction_near_destination",
                "shape_source_return_evidence",
            ])

            def _finite_csv(value):
                try:
                    return float(value) if np.isfinite(value) else ""
                except (TypeError, ValueError):
                    return ""

            for comparison in offline_transition_geometry.transitions:
                raw = comparison.raw
                shape = comparison.shape
                writer.writerow([
                    int(raw.region_index),
                    int(raw.start_index), int(raw.end_index),
                    float(raw.start_time_s), float(raw.end_time_s),
                    float(raw.duration_ms),
                    _finite_csv(raw.source_target_st),
                    _finite_csv(raw.destination_target_st),
                    int(raw.has_source), int(raw.has_destination),
                    _finite_csv(raw.target_interval_st),
                    _finite_csv(raw.start_pitch_st),
                    _finite_csv(raw.end_pitch_st),
                    _finite_csv(raw.median_pitch_st),
                    _finite_csv(raw.span_st),
                    _finite_csv(raw.net_movement_st),
                    _finite_csv(raw.path_length_st),
                    _finite_csv(raw.path_efficiency),
                    _finite_csv(raw.upward_fraction),
                    _finite_csv(raw.downward_fraction),
                    int(raw.dominant_direction),
                    _finite_csv(raw.monotonic_fraction),
                    int(raw.direction_reversals),
                    _finite_csv(raw.max_adjacent_step_st),
                    _finite_csv(raw.median_adjacent_step_st),
                    _finite_csv(raw.overshoot_above_st),
                    _finite_csv(raw.undershoot_below_st),
                    _finite_csv(raw.min_distance_source_st),
                    _finite_csv(raw.min_distance_destination_st),
                    _finite_csv(raw.fraction_near_source),
                    _finite_csv(raw.fraction_near_destination),
                    _finite_csv(raw.return_evidence),
                    _finite_csv(shape.start_pitch_st),
                    _finite_csv(shape.end_pitch_st),
                    _finite_csv(shape.median_pitch_st),
                    _finite_csv(shape.span_st),
                    _finite_csv(shape.net_movement_st),
                    _finite_csv(shape.path_length_st),
                    _finite_csv(shape.path_efficiency),
                    _finite_csv(shape.upward_fraction),
                    _finite_csv(shape.downward_fraction),
                    int(shape.dominant_direction),
                    _finite_csv(shape.monotonic_fraction),
                    int(shape.direction_reversals),
                    _finite_csv(shape.max_adjacent_step_st),
                    _finite_csv(shape.median_adjacent_step_st),
                    _finite_csv(shape.overshoot_above_st),
                    _finite_csv(shape.undershoot_below_st),
                    _finite_csv(shape.min_distance_source_st),
                    _finite_csv(shape.min_distance_destination_st),
                    _finite_csv(shape.fraction_near_source),
                    _finite_csv(shape.fraction_near_destination),
                    _finite_csv(shape.return_evidence),
                ])
        print(f"Transition geometry: {transition_out}")

        # ---------------------------------------------------------
        # Gesture-structure diagnostic exports.
        # ---------------------------------------------------------
        # These files define future interpretation units without
        # assigning musical labels.  Pitch and amplitude expression are
        # summarized but never discarded; the complete per-sample lanes
        # remain available in .expressive.csv / .amplitude_expression.csv.

        def _csv_atom(value):
            if isinstance(value, Enum):
                return value.name
            if isinstance(value, (tuple, list)):
                return ";".join(str(_csv_atom(v)) for v in value)
            if isinstance(value, (np.bool_, bool)):
                return int(value)
            if isinstance(value, (np.integer, int)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                return float(value) if np.isfinite(value) else ""
            return value

        def _amp_summary(start, end):
            start = max(0, int(start))
            end = min(len(sample_idx), int(end))
            if end <= start:
                return {
                    "amp_shape_delta_db": "",
                    "amp_residual_rms_db": "",
                    "amp_residual_peak_to_peak_db": "",
                    "amp_modulation_rate_median_hz": "",
                    "amp_modulation_periodicity_median": "",
                    "pitch_amplitude_correlation_median": "",
                }

            shape = np.asarray(offline_expressive_amplitude.shape_rms_dbfs[start:end], dtype=np.float64)
            residual = np.asarray(offline_expressive_amplitude.residual_db[start:end], dtype=np.float64)
            rate = np.asarray(offline_expressive_amplitude.local_modulation_rate_hz[start:end], dtype=np.float64)
            periodicity = np.asarray(offline_expressive_amplitude.local_modulation_periodicity[start:end], dtype=np.float64)
            corr = np.asarray(offline_expressive_amplitude.local_pitch_amplitude_correlation[start:end], dtype=np.float64)

            def finite_values(a):
                return a[np.isfinite(a)]

            shape_f = finite_values(shape)
            residual_f = finite_values(residual)
            rate_f = finite_values(rate)
            periodicity_f = finite_values(periodicity)
            corr_f = finite_values(corr)

            return {
                "amp_shape_delta_db": (
                    float(shape_f[-1] - shape_f[0]) if len(shape_f) >= 2 else ""
                ),
                "amp_residual_rms_db": (
                    float(np.sqrt(np.mean(residual_f ** 2))) if len(residual_f) else ""
                ),
                "amp_residual_peak_to_peak_db": (
                    float(np.max(residual_f) - np.min(residual_f)) if len(residual_f) else ""
                ),
                "amp_modulation_rate_median_hz": (
                    float(np.median(rate_f)) if len(rate_f) else ""
                ),
                "amp_modulation_periodicity_median": (
                    float(np.median(periodicity_f)) if len(periodicity_f) else ""
                ),
                "pitch_amplitude_correlation_median": (
                    float(np.median(corr_f)) if len(corr_f) else ""
                ),
            }

        gesture_features_out = out.with_suffix(out.suffix + ".gesture_features.csv")
        gesture_feature_names = [f.name for f in fields(type(offline_gesture_features.gestures[0]))] if offline_gesture_features.gestures else []
        amp_columns = [
            "amp_shape_delta_db",
            "amp_residual_rms_db",
            "amp_residual_peak_to_peak_db",
            "amp_modulation_rate_median_hz",
            "amp_modulation_periodicity_median",
            "pitch_amplitude_correlation_median",
        ]
        with gesture_features_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(gesture_feature_names + amp_columns)
            for feature in offline_gesture_features.gestures:
                amp = _amp_summary(feature.start_index, feature.end_index)
                writer.writerow(
                    [_csv_atom(getattr(feature, name)) for name in gesture_feature_names]
                    + [amp[name] for name in amp_columns]
                )
        print(f"Gesture features: {gesture_features_out}")

        gesture_objects_out = out.with_suffix(out.suffix + ".gesture_objects.csv")
        object_skip = {"links"}
        object_names = []
        if offline_gesture_objects.objects:
            object_names = [
                f.name for f in fields(type(offline_gesture_objects.objects[0]))
                if f.name not in object_skip
            ]
        with gesture_objects_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(object_names + amp_columns)
            for obj in offline_gesture_objects.objects:
                amp = _amp_summary(obj.start_index, obj.end_index)
                writer.writerow(
                    [_csv_atom(getattr(obj, name)) for name in object_names]
                    + [amp[name] for name in amp_columns]
                )
        print(f"Gesture objects: {gesture_objects_out}")

        gesture_links_out = out.with_suffix(out.suffix + ".gesture_links.csv")
        link_header = [
            "object_index", "link_index",
            "left_transition_region_index", "stable_region_index", "right_transition_region_index",
            "decision",
            "connector_family_count", "boundary_family_count",
            "excursion_connector", "direct_through_connector", "transition_similarity_connector",
            "duration_discontinuity", "path_discontinuity", "center_shape_discontinuity",
            "residual_discontinuity", "topology_discontinuity",
            "stable_target_st", "stable_duration_ms",
            "previous_target_st", "next_target_st",
            "entry_interval_st", "exit_interval_st", "outer_interval_st",
            "relation", "same_direction", "direction_reversal",
            "returns_toward_previous", "returns_toward_next",
            "local_target_path_st", "excess_over_outer_st", "outer_to_path_ratio",
            "duration_similarity", "shape_path_similarity", "center_path_similarity",
            "residual_rms_similarity",
        ]
        with gesture_links_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(link_header)
            for obj_result in offline_gesture_structure.objects:
                for decision in obj_result.link_decisions:
                    ev = decision.evidence
                    link = decision.v2
                    writer.writerow([
                        decision.object_index, decision.link_index,
                        decision.left_transition_region_index, decision.stable_region_index,
                        decision.right_transition_region_index,
                        decision.decision.name,
                        ev.connector_family_count, ev.boundary_family_count,
                        int(ev.excursion_connector), int(ev.direct_through_connector),
                        int(ev.transition_similarity_connector),
                        int(ev.duration_discontinuity), int(ev.path_discontinuity),
                        int(ev.center_shape_discontinuity), int(ev.residual_discontinuity),
                        int(ev.topology_discontinuity),
                        _csv_atom(link.stable_target_st), _csv_atom(link.stable_duration_ms),
                        _csv_atom(link.previous_target_st), _csv_atom(link.next_target_st),
                        _csv_atom(link.entry_interval_st), _csv_atom(link.exit_interval_st),
                        _csv_atom(link.outer_interval_st), link.relation.name,
                        int(link.same_direction), int(link.direction_reversal),
                        int(link.returns_toward_previous), int(link.returns_toward_next),
                        _csv_atom(link.local_target_path_st), _csv_atom(link.excess_over_outer_st),
                        _csv_atom(link.outer_to_path_ratio), _csv_atom(link.duration_similarity),
                        _csv_atom(link.shape_path_similarity), _csv_atom(link.center_path_similarity),
                        _csv_atom(link.residual_rms_similarity),
                    ])
        print(f"Gesture links: {gesture_links_out}")

        gesture_segments_out = out.with_suffix(out.suffix + ".gesture_segments.csv")
        region_map = {i: region for i, region in enumerate(offline_interpretation.regions)}
        segment_header = [
            "object_index", "segment_index", "transition_region_indices", "stable_region_indices",
            "internal_link_indices", "unresolved_link_indices", "n_transitions", "n_internal_links",
            "start_index", "end_index", "start_time_s", "end_time_s", "duration_ms",
            "pitch_shape_span_st", "pitch_center_span_st", "pitch_residual_rms_st",
        ] + amp_columns
        with gesture_segments_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(segment_header)
            for obj_result in offline_gesture_structure.objects:
                for segment in obj_result.segments:
                    region_indices = sorted(set(segment.transition_region_indices + segment.stable_region_indices))
                    if region_indices:
                        start_index = min(region_map[i].start_index for i in region_indices)
                        end_index = max(region_map[i].end_index for i in region_indices)
                    else:
                        start_index = end_index = 0
                    shape = np.asarray(offline_expressive.shape_pitch_st[start_index:end_index], dtype=np.float64)
                    center = np.asarray(offline_expressive.center_pitch_st[start_index:end_index], dtype=np.float64)
                    residual = np.asarray(offline_expressive.residual_pitch_st[start_index:end_index], dtype=np.float64)
                    shape = shape[np.isfinite(shape)]
                    center = center[np.isfinite(center)]
                    residual = residual[np.isfinite(residual)]
                    shape_span = float(np.max(shape) - np.min(shape)) if len(shape) else ""
                    center_span = float(np.max(center) - np.min(center)) if len(center) else ""
                    residual_rms = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else ""
                    amp = _amp_summary(start_index, end_index)
                    writer.writerow([
                        segment.object_index, segment.segment_index,
                        _csv_atom(segment.transition_region_indices),
                        _csv_atom(segment.stable_region_indices),
                        _csv_atom(segment.internal_link_indices),
                        _csv_atom(segment.unresolved_link_indices),
                        segment.n_transitions, segment.n_internal_links,
                        start_index, end_index,
                        float(start_index / analysis_hz), float(end_index / analysis_hz),
                        float((end_index - start_index) * 1000.0 / analysis_hz),
                        shape_span, center_span, residual_rms,
                    ] + [amp[name] for name in amp_columns])
        print(f"Gesture segments: {gesture_segments_out}")

        gesture_candidates_out = out.with_suffix(out.suffix + ".gesture_candidates.csv")
        candidate_names = [f.name for f in fields(type(offline_interpretation_candidates[0]))] if offline_interpretation_candidates else []
        with gesture_candidates_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(candidate_names)
            for candidate in offline_interpretation_candidates:
                writer.writerow([_csv_atom(getattr(candidate, name)) for name in candidate_names])
        print(f"Gesture interpretation candidates: {gesture_candidates_out}")

        print(
            "Gesture structure: "
            f"objects={offline_gesture_structure.total_objects}, "
            f"links={offline_gesture_structure.total_links}, "
            f"join={offline_gesture_structure.join_links}, "
            f"split={offline_gesture_structure.split_links}, "
            f"unresolved={offline_gesture_structure.unresolved_links}, "
            f"segments={offline_gesture_structure.total_segments}"
        )
    else:
        # Keep MATLAB-mode export exactly as before.
        table = np.column_stack(
            [
                sample_idx,
                t,
                result.f0_hz[
                    sample_idx
                ],
                result.amplitude[
                    sample_idx
                ],
                result.phase[
                    sample_idx
                ],
                np.real(
                    result.fundamental[
                        sample_idx
                    ]
                ),
                np.imag(
                    result.fundamental[
                        sample_idx
                    ]
                ),
            ]
        )

        np.savetxt(
            out,
            table,
            delimiter=",",
            header=(
                "sample,time_s,f0_hz,"
                "amplitude,phase,"
                "fundamental_real,"
                "fundamental_imag"
            ),
            comments="",
        )

    print(f"Input:       {args.wav}")
    print(f"Sample rate: {sr} Hz")
    print(f"Mode:        {cfg.mode}")
    print(f"Block size:  {cfg.block_size} samples")
    print(f"Frame time:  {1000.0 * cfg.block_size / sr:.3f} ms")
    print(f"c:           {cfg.c}")
    print(f"wait:        {cfg.num_buf_to_wait}")
    print(f"npeaks:      {cfg.npeaks}")
    print(f"harmonic-change threshold: {cfg.nsemitones} semitones")
    if cfg.mode == "offline":
        print(f"Vocal floor: {cfg.vocal_floor_hz:.3f} Hz")
        first_valid_count = int(
            np.sum(
                offline_validity.first_pass_valid
            )
        )

        corrected_valid_count = int(
            np.sum(
                offline_validity.valid
            )
        )

        newly_valid_count = int(
            np.sum(
                offline_validity.valid
                & ~offline_validity.first_pass_valid
            )
        )

        print(
            f"Valid F0:    "
            f"{first_valid_count} -> "
            f"{corrected_valid_count}"
        )

        print(
            f"Rescued:     "
            f"{newly_valid_count} rows"
        )
    print(f"Onsets:      {len(result.onset_samples)}")
    print(f"CSV:         {out}")

    if cfg.mode == "offline" and offline_pitch_candidates is not None:
        pitch_candidates_path = Path(str(out) + ".pitch_candidates.csv")
        with pitch_candidates_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_index", "frame_start_sample", "time_s",
                "frame_real_sample_count", "acoustic_state_name", "source",
                "candidate_hz", "nearest_midi", "cents_from_nearest_midi",
                "measured_period_hz", "measured_period_lag",
                "measured_period_acf", "measured_period_cmndf",
                "spectral_harmonic_count", "spectral_harmonics",
                "measured_component_amplitude", "transition_penalty",
                "selected_by_initializer", "initializer_selected_hz",
                "note_group_midi",
            ])
            for r in offline_pitch_candidates:
                writer.writerow([
                    r.frame_index, r.frame_start_sample, r.time_s,
                    r.frame_real_sample_count, r.acoustic_state_name, r.source,
                    r.candidate_hz, r.nearest_midi, r.cents_from_nearest_midi,
                    r.measured_period_hz, r.measured_period_lag,
                    r.measured_period_acf, r.measured_period_cmndf,
                    r.spectral_harmonic_count,
                    ";".join(str(x) for x in r.spectral_harmonics),
                    r.measured_component_amplitude,
                    "" if r.transition_penalty is None else r.transition_penalty,
                    int(r.selected_by_initializer),
                    "" if r.initializer_selected_hz is None else r.initializer_selected_hz,
                    r.note_group_midi,
                ])
        print(f"Pitch candidate field: {pitch_candidates_path}")


    if cfg.mode == "offline" and offline_range_reference_result is not None:
        range_refs_path = Path(str(out) + ".range_references.csv")
        with range_refs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_index","frame_start_sample","time_s","observational_hz",
                "selected_measured_hz","selected_acf","selected_component_amplitude",
                "local_candidate_count","competing_candidate_count","acoustic_measurement",
                "relative_energy","energy_slope_db","low_energy_flag","decaying_flag",
                "prior_reference_hz","penalty_elapsed_ms","transition_penalty",
                "interval_energy_joint_challenge","evidence_status",
                "diagnostic_settled","usable_as_next_reference",
            ])
            for rr in offline_range_reference_result.rows:
                writer.writerow([
                    rr.frame_index,rr.frame_start_sample,rr.time_s,
                    "" if rr.observational_hz is None else rr.observational_hz,
                    "" if rr.selected_measured_hz is None else rr.selected_measured_hz,
                    "" if rr.selected_acf is None else rr.selected_acf,
                    "" if rr.selected_component_amplitude is None else rr.selected_component_amplitude,
                    rr.local_candidate_count,rr.competing_candidate_count,rr.acoustic_measurement,
                    "" if rr.relative_energy is None else rr.relative_energy,
                    "" if rr.energy_slope_db is None else rr.energy_slope_db,
                    int(rr.low_energy_flag),int(rr.decaying_flag),
                    "" if rr.prior_reference_hz is None else rr.prior_reference_hz,
                    "" if rr.penalty_elapsed_ms is None else rr.penalty_elapsed_ms,
                    "" if rr.transition_penalty is None else rr.transition_penalty,
                    int(rr.interval_energy_joint_challenge),rr.evidence_status,
                    int(rr.diagnostic_settled),int(rr.usable_as_next_reference),
                ])
        print(f"Historical range references: {range_refs_path}")

    if cfg.mode == "offline" and offline_range_calibration is not None:
        range_path = Path(str(out) + ".range_calibration.csv")
        r = offline_range_calibration
        with range_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "available","anchor_count","source","low_hz","high_hz",
                "low_anchor_midi","high_anchor_midi",
                "plateau_low_midi","plateau_high_midi",
                "zero_low_midi","zero_high_midi",
                "lo_quantile","hi_quantile","plateau_extension_st","shoulder_st",
                "w_range",
            ])
            writer.writerow([
                int(r.available),r.anchor_count,r.source,
                "" if r.low_hz is None else r.low_hz,
                "" if r.high_hz is None else r.high_hz,
                "" if r.low_anchor_midi is None else r.low_anchor_midi,
                "" if r.high_anchor_midi is None else r.high_anchor_midi,
                "" if r.plateau_low_midi is None else r.plateau_low_midi,
                "" if r.plateau_high_midi is None else r.plateau_high_midi,
                "" if r.zero_low_midi is None else r.zero_low_midi,
                "" if r.zero_high_midi is None else r.zero_high_midi,
                r.lo_quantile,r.hi_quantile,r.plateau_extension_st,r.shoulder_st,0.5,
            ])
        print(f"V22 range calibration: {range_path}")

    if cfg.mode == "offline" and offline_juror_evidence is not None:
        juror_path = Path(str(out) + ".juror_evidence.csv")
        with juror_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_index","time_s","group_id","note_group_midi","within_49c","cents_from_note","representative_hz",
                "representative_source","candidate_count_in_group",
                "acf_median","cmndf_median","harmonic_count_median",
                "component_amplitude_median","range_confidence",
                "interval_previous_component","interval_following_component",
                "temporal_prev_same_note","temporal_next_same_note",
                "temporal_persistence_count","selected_by_initializer",
            ])
            for r in offline_juror_evidence:
                writer.writerow([
                    r.frame_index,r.time_s,r.group_id,r.note_group_midi,int(r.within_49c),r.cents_from_note,r.representative_hz,
                    r.representative_source,r.candidate_count_in_group,
                    "" if r.acf_median is None else r.acf_median,
                    "" if r.cmndf_median is None else r.cmndf_median,
                    "" if r.harmonic_count_median is None else r.harmonic_count_median,
                    "" if r.component_amplitude_median is None else r.component_amplitude_median,
                    "" if r.range_confidence is None else r.range_confidence,
                    "" if r.interval_previous_component is None else r.interval_previous_component,
                    "" if r.interval_following_component is None else r.interval_following_component,
                    int(r.temporal_prev_same_note),int(r.temporal_next_same_note),
                    r.temporal_persistence_count,int(r.selected_by_initializer),
                ])
        print(f"Juror evidence: {juror_path}")

    if cfg.mode == "offline" and offline_juror_pairs is not None:
        pair_path = Path(str(out) + ".juror_pairs.csv")
        with pair_path.open("w", newline="", encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["frame_index","time_s","a_group_id","a_midi","a_hz","b_group_id","b_midi","b_hz","a_range_admitted","b_range_admitted","a_admissible","b_admissible","a_score","b_score","a_spectral","b_spectral","a_temporal","b_temporal","a_interval","b_interval","a_range_component","b_range_component","winner_group_id","winner_midi","winner_hz","outcome","diagnostic"])
            for r in offline_juror_pairs:
                writer.writerow([r.frame_index,r.time_s,r.a_group_id,r.a_midi,r.a_hz,r.b_group_id,r.b_midi,r.b_hz,int(r.a_range_admitted),int(r.b_range_admitted),int(r.a_admissible),int(r.b_admissible),"" if r.a_score is None else r.a_score,"" if r.b_score is None else r.b_score,"" if r.a_spectral is None else r.a_spectral,"" if r.b_spectral is None else r.b_spectral,"" if r.a_temporal is None else r.a_temporal,"" if r.b_temporal is None else r.b_temporal,"" if r.a_interval is None else r.a_interval,"" if r.b_interval is None else r.b_interval,"" if r.a_range_component is None else r.a_range_component,"" if r.b_range_component is None else r.b_range_component,"" if r.winner_group_id is None else r.winner_group_id,"" if r.winner_midi is None else r.winner_midi,"" if r.winner_hz is None else r.winner_hz,r.outcome,r.diagnostic])
        print(f"Juror pair verdicts: {pair_path}")

    if cfg.mode == "offline" and offline_juror_bench is not None:
        bench_path = Path(str(out) + ".juror_bench.csv")
        with bench_path.open("w", newline="", encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["frame_index","time_s","candidate_count","status","winner_group_id","winner_midi","winner_hz","decisive_edges","undecided_pairs","abstention_category"])
            for r in offline_juror_bench:
                writer.writerow([r.frame_index,r.time_s,r.candidate_count,r.status,"" if r.winner_group_id is None else r.winner_group_id,"" if r.winner_midi is None else r.winner_midi,"" if r.winner_hz is None else r.winner_hz,r.decisive_edges,r.undecided_pairs,r.abstention_category])
        print(f"Juror bench: {bench_path}")

    if cfg.mode == "offline" and offline_detective_candidates is not None:
        dcp = Path(str(out) + ".harmonic_detective_candidates.csv")
        with dcp.open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["frame_index","time_s","group_id","midi","hz","valid_windows","harmonic_score","harmonic_support","harmonic_strength","window_details_json","failure_reasons"])
            for r in offline_detective_candidates:
                w.writerow([r.frame_index,r.time_s,r.group_id,r.midi,r.hz,r.valid_windows,"" if r.harmonic_score is None else r.harmonic_score,"" if r.harmonic_support is None else r.harmonic_support,"" if r.harmonic_strength is None else r.harmonic_strength,r.window_details_json,r.failure_reasons])
        dvp = Path(str(out) + ".harmonic_detective.csv")
        with dvp.open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["frame_index","time_s","jury_status","detective_called","eligible_representatives","best_group_id","best_midi","best_hz","best_score","best_support","runner_group_id","runner_hz","runner_score","harmonic_margin","detective_winner_group_id","detective_winner_midi","detective_winner_hz","reason"])
            for r in offline_detective_verdicts:
                w.writerow([r.frame_index,r.time_s,r.jury_status,int(r.detective_called),r.eligible_representatives,r.best_group_id or "","" if r.best_midi is None else r.best_midi,"" if r.best_hz is None else r.best_hz,"" if r.best_score is None else r.best_score,"" if r.best_support is None else r.best_support,r.runner_group_id or "","" if r.runner_hz is None else r.runner_hz,"" if r.runner_score is None else r.runner_score,"" if r.harmonic_margin is None else r.harmonic_margin,r.detective_winner_group_id or "","" if r.detective_winner_midi is None else r.detective_winner_midi,"" if r.detective_winner_hz is None else r.detective_winner_hz,r.reason])
        fap = Path(str(out) + ".adjudication_final.csv")
        with fap.open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["frame_index","time_s","provenance","status","winner_group_id","winner_midi","winner_hz","original_jury_status","original_abstention_category"])
            for r in offline_final_adjudication:
                w.writerow([r.frame_index,r.time_s,r.provenance,r.status,r.winner_group_id or "","" if r.winner_midi is None else r.winner_midi,"" if r.winner_hz is None else r.winner_hz,r.original_jury_status,r.original_abstention_category])
        print(f"Harmonic Detective: {dvp}")
        print(f"Final adjudication: {fap}")

    print(
        f"CSV rate:    {EXPORT_HZ:g} Hz "
        f"(every {export_step} samples)"
    )

if __name__ == "__main__":
    main()
