from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import csv

from .config import ECKFConfig
from .tracker import track_pitch
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
    p.add_argument("--wait", type=int, default=2)
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
    frame_index = sample_idx // cfg.block_size

    if cfg.mode == "offline":
        export_f0 = result.f0_hz[sample_idx]

        analysis_hz = (
            sr / float(export_step)
        )

        validity_config = PitchValidityConfig(
            analysis_hz=analysis_hz,
        )

        first_pass_validity = analyse_pitch_validity(
            f0_hz=export_f0,
            sample_rate=analysis_hz,
            config=validity_config,
        )

        offline_validity = apply_offline_validity_correction(
            f0_hz=export_f0,
            first_pass=first_pass_validity,
            sample_rate=analysis_hz,
        )
        # A downstream trajectory rescue must never revive an explicitly
        # energy-silent frame, even when its neighbours carry the same note.
        vetoed = result.energy_silence_per_frame[frame_index]
        offline_validity.first_pass_valid[vetoed] = False
        offline_validity.valid[vetoed] = False
        offline_validity.clean_f0_hz[vetoed] = np.nan
        offline_validity.first_pass_reason[vetoed] = "LOW_ENERGY_SILENCE"
        offline_validity.reason[vetoed] = "LOW_ENERGY_SILENCE"
        offline_validity.correction_reason[vetoed] = "LOW_ENERGY_SILENCE"

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
                    "frame_decision",
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
                        str(result.frame_decision[frame_index[j]]),
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
                "periodicity_voiced", "frame_decision",
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
                    str(result.frame_decision[fi]),
                ])
        print(f"Frame evidence: {frames_out}")
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
