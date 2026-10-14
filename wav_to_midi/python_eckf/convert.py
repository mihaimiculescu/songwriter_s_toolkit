from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Iterable

from .analysis_timing import DEFAULT_OFFLINE_ANALYSIS_HZ
from . import cli as tracker_cli
from . import render_midi as renderer_cli


def _parse_time_signature_text(text: str) -> str:
    try:
        num_s, den_s = text.split("/", 1)
        num = int(num_s)
        den = int(den_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError("time signature must look like 4/4") from exc
    if num <= 0 or den <= 0 or den & (den - 1):
        raise argparse.ArgumentTypeError("time signature denominator must be a power of two")
    return f"{num}/{den}"


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "R4 single-process WAV -> frozen V3R9 analysis -> MRV18.3 MIDI. "
            "Musical logic is unchanged; R4 changes execution mechanics only."
        )
    )
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output MIDI path")
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--time-signature", type=_parse_time_signature_text, default="4/4")
    ap.add_argument(
        "--analysis-prefix", type=Path, default=None,
        help="tracker CSV prefix; default: <output-stem>.eckf.csv",
    )

    # Frozen tracker parameters. Defaults mirror python_eckf.cli.
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--c", type=float, default=7.0)
    ap.add_argument("--wait", type=int, default=2)
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--nsemitones", type=float, default=2.0)
    ap.add_argument("--vocal-floor-hz", type=float, default=60.0)
    ap.add_argument("--analysis-hz", type=float, default=DEFAULT_OFFLINE_ANALYSIS_HZ)
    ap.add_argument("--silence-mode", choices=["fixed", "adaptive"], default=None)
    ap.add_argument("--silence-energy-threshold", type=float, default=None)
    ap.add_argument("--eckf-gain-normalization", choices=["none", "peak"], default="peak")

    # R4 execution-only controls.
    ap.add_argument("--kalman-engine", choices=["auto", "python", "numba"], default="auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--memory-budget-mib", type=int, default=None)
    ap.add_argument(
        "--frame-preflight", choices=["auto", "serial", "off"], default="auto",
        help=(
            "R4 Stage 2 traffic police for parallel-safe per-frame acoustic evidence. "
            "Use off to reproduce frozen V3R9 on-demand measurements."
        ),
    )
    ap.add_argument(
        "--harmonic-preflight", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 4 parallel-safe harmonic-detector preflight.",
    )
    ap.add_argument(
        "--aperture-preflight", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 4 parallel-safe variable-aperture acoustic preflight.",
    )
    ap.add_argument(
        "--initialization-preflight", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 6 exact measured-period initialization preflight.",
    )
    ap.add_argument(
        "--candidate-parallel", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 3 parallel-safe observational pitch-candidate measurements.",
    )
    ap.add_argument(
        "--juror-parallel", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 3 parallel-safe juror pair measurements.",
    )
    ap.add_argument(
        "--interpretation-parallel", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 5 parallel-safe variable-aperture trajectory evidence.",
    )
    ap.add_argument(
        "--amplitude-parallel", choices=["auto", "serial", "off"], default="auto",
        help="R4 Stage 5 parallel-safe expressive-amplitude modulation evidence.",
    )

    # Frozen renderer parameters.
    ap.add_argument("--velocity", type=int, default=None)
    ap.add_argument("--channel", type=int, default=0)

    args = ap.parse_args(list(argv) if argv is not None else None)

    out_mid = args.out
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.analysis_prefix or out_mid.with_suffix(".eckf.csv")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    tracker_args = [
        str(args.wav),
        "--mode", "offline",
        "--block-size", str(args.block_size),
        "--c", str(args.c),
        "--wait", str(args.wait),
        "--npeaks", str(args.npeaks),
        "--nsemitones", str(args.nsemitones),
        "--vocal-floor-hz", str(args.vocal_floor_hz),
        "--analysis-hz", str(args.analysis_hz),
        "--eckf-gain-normalization", args.eckf_gain_normalization,
        "--kalman-engine", args.kalman_engine,
        "--frame-preflight", args.frame_preflight,
        "--harmonic-preflight", args.harmonic_preflight,
        "--aperture-preflight", args.aperture_preflight,
        "--initialization-preflight", args.initialization_preflight,
        "--candidate-parallel", args.candidate_parallel,
        "--juror-parallel", args.juror_parallel,
        "--interpretation-parallel", args.interpretation_parallel,
        "--amplitude-parallel", args.amplitude_parallel,
        "--csv", str(prefix),
    ]
    if args.silence_mode is not None:
        tracker_args.extend(["--silence-mode", args.silence_mode])
    if args.silence_energy_threshold is not None:
        tracker_args.extend(["--silence-energy-threshold", str(args.silence_energy_threshold)])
    if args.workers is not None:
        tracker_args.extend(["--workers", str(args.workers)])
    if args.memory_budget_mib is not None:
        tracker_args.extend(["--memory-budget-mib", str(args.memory_budget_mib)])

    started = time.perf_counter()
    tracker_started = started
    tracker_cli.main(tracker_args)
    tracker_elapsed = time.perf_counter() - tracker_started

    render_args = [
        str(prefix),
        "--wav", str(args.wav),
        "--out", str(out_mid),
        "--bpm", str(args.bpm),
        "--time-signature", args.time_signature,
        "--channel", str(args.channel),
    ]
    if args.velocity is not None:
        render_args.extend(["--velocity", str(args.velocity)])

    render_started = time.perf_counter()
    renderer_cli.main(render_args)
    render_elapsed = time.perf_counter() - render_started
    total_elapsed = time.perf_counter() - started

    print(
        f"R4 timing: tracker+analysis={tracker_elapsed:.3f}s; "
        f"renderer={render_elapsed:.3f}s; total={total_elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
