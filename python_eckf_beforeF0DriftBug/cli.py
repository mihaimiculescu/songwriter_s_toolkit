from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import csv

from .config import ECKFConfig
from .tracker import track_pitch
from .frame_evidence import frame_acoustic_state_name
from .trajectory_interpreter import interpret_pitch_trajectory
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
    print(f"threshold:   {cfg.nsemitones} semitones")
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
    print(
        f"CSV rate:    {EXPORT_HZ:g} Hz "
        f"(every {export_step} samples)"
    )

if __name__ == "__main__":
    main()
