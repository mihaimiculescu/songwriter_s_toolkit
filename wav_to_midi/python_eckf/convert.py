from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .product_pipeline import prepare_renderer_inputs
from .render_midi import render_midi


def _parse_time_signature(text: str) -> tuple[int, int]:
    try:
        a, b = text.split("/", 1)
        num, den = int(a), int(b)
    except Exception as exc:
        raise argparse.ArgumentTypeError("time signature must look like 4/4") from exc
    if num <= 0 or den <= 0 or den & (den - 1):
        raise argparse.ArgumentTypeError("time signature denominator must be a power of two")
    return num, den


def _temporary_root() -> str | None:
    # Linux songwriter workstations normally have /dev/shm (RAM-backed).  Fall
    # back to the platform temp directory everywhere else.
    root = "/dev/shm"
    return root if os.path.isdir(root) and os.access(root, os.W_OK) else None


def convert_wav_to_midi(
    wav_path: str | Path,
    output_path: str | Path | None = None,
    *,
    bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    velocity: int | None = None,
) -> Path:
    """Convert one mono WAV to one MIDI file.

    Only the MIDI file is persistent.  Stage-6 analysis compatibility lanes are
    private temporary files and are removed automatically.
    """
    wav = Path(wav_path).expanduser().resolve()
    if not wav.exists():
        raise FileNotFoundError(wav)
    if wav.suffix.lower() not in {".wav", ".wave"}:
        raise ValueError("input must be a WAV file")
    if bpm <= 0:
        raise ValueError("bpm must be > 0")

    out = Path(output_path).expanduser() if output_path is not None else wav.with_suffix(".mid")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wav2midi-", dir=_temporary_root()) as td:
        prefix = Path(td) / "analysis"
        prepare_renderer_inputs(wav, prefix)
        render_midi(
            prefix,
            wav,
            out,
            bpm=float(bpm),
            time_signature=time_signature,
            velocity=velocity,
            write_audits=False,
        )
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wav2midi",
        description="Convert a mono vocal WAV to MIDI.",
    )
    parser.add_argument("wav", type=Path, help="input WAV")
    parser.add_argument("--out", type=Path, default=None, help="output MIDI (default: same name as WAV)")
    parser.add_argument("--bpm", type=float, default=120.0, help="MIDI tempo (default: 120)")
    parser.add_argument("--time-signature", type=_parse_time_signature, default=(4, 4), help="default: 4/4")
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("Processing...", flush=True)
    out = convert_wav_to_midi(
        args.wav,
        args.out,
        bpm=args.bpm,
        time_signature=args.time_signature,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
