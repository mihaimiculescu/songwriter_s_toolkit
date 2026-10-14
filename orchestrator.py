#!/usr/bin/env python3
"""
Songwriter's Toolkit host-side orchestrator.

Development layout expected:

    <toolkit-root>/
    ├── Projects/
    ├── dev/
    │   ├── orchestrator.py
    │   ├── .runtime/
    │   │   └── orchestrator.sock
    │   └── gui/
    └── main/
        └── wav_to_midi/
            └── python_eckf/

The orchestrator is a host process. It must be started from the eckf-pitch
Conda environment. The Dockerized GUI talks to it through a Unix-domain socket.

One-shot use:
    python dev/orchestrator.py --project PROJECT_ID

GUI service mode:
    python dev/orchestrator.py --serve
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_ECKF_ENV = "eckf-pitch"
MAX_REQUEST_BYTES = 64 * 1024


class OrchestratorError(RuntimeError):
    """Expected pipeline/configuration failure."""


def log(message: str) -> None:
    print(f"[orchestrator] {message}", flush=True)


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> None:
    rendered = " ".join(shlex_quote(arg) for arg in args)
    if cwd is not None:
        log(f"$ (cd {cwd} && {rendered})")
    else:
        log(f"$ {rendered}")

    if dry_run:
        return

    try:
        subprocess.run(args, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise OrchestratorError(f"Command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise OrchestratorError(
            f"Command failed with exit code {exc.returncode}: {rendered}"
        ) from exc


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise OrchestratorError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise OrchestratorError(f"Expected a JSON object in {path}")
    return data


def require_active_eckf_environment() -> None:
    active = os.environ.get("CONDA_DEFAULT_ENV")
    if active != EXPECTED_ECKF_ENV:
        actual = active if active else "none"
        raise OrchestratorError(
            f"Expected active Conda environment '{EXPECTED_ECKF_ENV}', "
            f"but active environment is '{actual}'. "
            f"Run: conda activate {EXPECTED_ECKF_ENV}"
        )


def resolve_layout() -> tuple[Path, Path, Path, Path]:
    branch_root = Path(__file__).resolve().parent
    toolkit_root = branch_root.parent
    projects_root = toolkit_root / "Projects"
    wav_to_midi_root = toolkit_root / "main" / "wav_to_midi"
    default_socket = branch_root / ".runtime" / "orchestrator.sock"

    if not projects_root.is_dir():
        raise OrchestratorError(
            f"Projects directory not found at expected location: {projects_root}"
        )
    if not wav_to_midi_root.is_dir():
        raise OrchestratorError(
            "Production WAV-to-MIDI directory not found at expected location: "
            f"{wav_to_midi_root}"
        )
    if not (wav_to_midi_root / "python_eckf").is_dir():
        raise OrchestratorError(
            "python_eckf package not found under production WAV-to-MIDI directory: "
            f"{wav_to_midi_root / 'python_eckf'}"
        )

    return toolkit_root, projects_root, wav_to_midi_root, default_socket


def resolve_project(projects_root: Path, project_name: str) -> Path:
    if not project_name or Path(project_name).name != project_name:
        raise OrchestratorError("Project must be a project folder name, not a path")

    project_dir = (projects_root / project_name).resolve()
    root = projects_root.resolve()
    if project_dir.parent != root:
        raise OrchestratorError("Project must live directly under Projects/")
    if not project_dir.is_dir():
        raise OrchestratorError(f"Project directory does not exist: {project_dir}")
    return project_dir


def extract_project_inputs(project_dir: Path) -> tuple[Path, Path, float]:
    payload = load_json(project_dir / "output.json")
    source = payload.get("source")
    metadata = payload.get("metadata")

    if not isinstance(source, dict):
        raise OrchestratorError("output.json is missing object: source")
    if not isinstance(metadata, dict):
        raise OrchestratorError("output.json is missing object: metadata")

    audio_name = source.get("audio")
    if not isinstance(audio_name, str) or not audio_name.strip():
        raise OrchestratorError("output.json source.audio must be a non-empty string")
    if Path(audio_name).name != audio_name:
        raise OrchestratorError(
            "output.json source.audio must contain only the audio filename"
        )

    tempo_raw = metadata.get("tempo")
    if isinstance(tempo_raw, bool) or not isinstance(tempo_raw, (int, float)):
        raise OrchestratorError(
            "output.json metadata.tempo must be a numeric BPM value"
        )
    tempo = float(tempo_raw)
    if tempo <= 0:
        raise OrchestratorError("output.json metadata.tempo must be greater than 0")

    input_wav = project_dir / audio_name
    if not input_wav.is_file():
        raise OrchestratorError(f"Input audio file does not exist: {input_wav}")
    if input_wav.suffix.lower() != ".wav":
        raise OrchestratorError(
            f"Expected source.audio to be a WAV file, got: {input_wav.name}"
        )

    return input_wav, input_wav.with_suffix(".mid"), tempo


def normalize_input_wav(input_wav: Path, *, dry_run: bool) -> None:
    temp_wav = input_wav.with_name(f".{input_wav.stem}.mono.tmp.wav")

    if temp_wav.exists():
        if temp_wav.is_file():
            temp_wav.unlink()
        else:
            raise OrchestratorError(
                f"Temporary path exists but is not a file: {temp_wav}"
            )

    try:
        log("1/3 Converting project acapella to mono with ffmpeg")
        run_command(
            ["ffmpeg", "-y", "-i", str(input_wav), "-ac", "1", str(temp_wav)],
            dry_run=dry_run,
        )

        log("2/3 Peak-normalizing mono WAV to -0.1 dB")
        run_command(
            [
                "ffmpeg-normalize",
                str(temp_wav),
                "-nt", "peak",
                "-t", "-0.1",
                "-c:a", "pcm_s16le",
                "-o", str(input_wav),
                "-f",
            ],
            dry_run=dry_run,
        )
    finally:
        if not dry_run and temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError as exc:
                log(f"WARNING: could not remove temporary WAV {temp_wav}: {exc}")


def run_wav_to_midi(
    input_wav: Path,
    midi_path: Path,
    tempo: float,
    wav_to_midi_root: Path,
    *,
    dry_run: bool,
) -> None:
    if midi_path.exists() and midi_path.is_dir():
        raise OrchestratorError(f"MIDI output path is a directory: {midi_path}")
    if not dry_run and midi_path.exists():
        midi_path.unlink()

    bpm_arg = str(int(tempo)) if tempo.is_integer() else str(tempo)
    log("3/3 Running production WAV-to-MIDI converter from main branch")
    run_command(
        [
            sys.executable,
            "-m", "python_eckf",
            str(input_wav),
            "--out", str(midi_path),
            "--bpm", bpm_arg,
        ],
        cwd=wav_to_midi_root,
        dry_run=dry_run,
    )

    if not dry_run and not midi_path.is_file():
        raise OrchestratorError(
            f"WAV-to-MIDI command completed but output MIDI was not created: {midi_path}"
        )


def check_dependencies() -> None:
    missing: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("ffmpeg-normalize") is None:
        missing.append("ffmpeg-normalize")
    if missing:
        raise OrchestratorError(
            "Missing required command(s) in the active environment/PATH: "
            + ", ".join(missing)
        )


def process_project(project_name: str, *, dry_run: bool = False) -> Path:
    require_active_eckf_environment()
    toolkit_root, projects_root, wav_to_midi_root, _ = resolve_layout()
    project_dir = resolve_project(projects_root, project_name)
    input_wav, midi_path, tempo = extract_project_inputs(project_dir)
    check_dependencies()

    log(f"Toolkit root: {toolkit_root}")
    log(f"Project: {project_dir.name}")
    log(f"Input WAV: {input_wav.name}")
    log(f"Tempo: {tempo:g} BPM")
    log(f"MIDI output: {midi_path.name}")

    normalize_input_wav(input_wav, dry_run=dry_run)
    run_wav_to_midi(
        input_wav,
        midi_path,
        tempo,
        wav_to_midi_root,
        dry_run=dry_run,
    )

    if dry_run:
        log("Dry run completed successfully; no files were changed.")
    else:
        log(f"Completed successfully: {midi_path}")
    return midi_path


def send_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(encoded)


def read_request(conn: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_REQUEST_BYTES:
            raise OrchestratorError("Request is too large")

    if not data:
        raise OrchestratorError("Empty request")

    line = bytes(data).split(b"\n", 1)[0]
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrchestratorError("Invalid JSON request") from exc

    if not isinstance(payload, dict):
        raise OrchestratorError("Request must be a JSON object")
    return payload


def prepare_socket_path(socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if not socket_path.exists() and not socket_path.is_symlink():
        return

    try:
        mode = socket_path.lstat().st_mode
    except FileNotFoundError:
        return

    if not stat.S_ISSOCK(mode):
        raise OrchestratorError(
            f"Refusing to remove non-socket path: {socket_path}"
        )
    socket_path.unlink()


def serve(socket_path: Path) -> int:
    require_active_eckf_environment()
    # Fail immediately at startup if the host installation is incomplete.
    resolve_layout()
    check_dependencies()
    prepare_socket_path(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o660)
        server.listen(4)
        log(f"Host service ready: {socket_path}")
        log("Waiting for Generate requests from the GUI...")

        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = read_request(conn)
                    project = request.get("project")
                    if not isinstance(project, str) or not project:
                        raise OrchestratorError(
                            "Request field 'project' must be a non-empty string"
                        )

                    midi_path = process_project(project)
                    send_json(
                        conn,
                        {
                            "ok": True,
                            "project": project,
                            "midi": midi_path.name,
                        },
                    )
                except Exception as exc:
                    # Expected failures are returned to the GUI. Unexpected Python
                    # failures are also contained so the service remains alive.
                    message = str(exc) or exc.__class__.__name__
                    log(f"ERROR: {message}")
                    try:
                        send_json(conn, {"ok": False, "error": message})
                    except OSError:
                        pass
    except KeyboardInterrupt:
        log("Host service stopped.")
        return 0
    finally:
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Songwriter's Toolkit host orchestrator: process one project or "
            "serve Generate requests from the Dockerized GUI."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--project",
        help="Project folder name directly under <toolkit-root>/Projects/",
    )
    mode.add_argument(
        "--serve",
        action="store_true",
        help="Run the host service used by the Dockerized GUI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --project, validate and print commands without executing them.",
    )
    parser.add_argument(
        "--socket",
        type=Path,
        help=(
            "Unix socket path for --serve. Default: "
            "<branch-root>/.runtime/orchestrator.sock"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _, _, _, default_socket = resolve_layout()
        if args.serve:
            if args.dry_run:
                raise OrchestratorError("--dry-run cannot be used with --serve")
            return serve((args.socket or default_socket).resolve())

        assert args.project is not None
        process_project(args.project, dry_run=args.dry_run)
        return 0
    except OrchestratorError as exc:
        print(f"[orchestrator] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("[orchestrator] Interrupted.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
