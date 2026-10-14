#!/usr/bin/env bash
set -euo pipefail

DEV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_ROOT="$DEV_ROOT/gui"

# Make sudo ask once, at startup.
sudo -v

# Make conda activation available inside this non-interactive shell.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate eckf-pitch

echo "[start] Conda environment: $CONDA_DEFAULT_ENV"

cleanup() {
    echo
    echo "[stop] Stopping Songwriter's Toolkit..."

    if [[ -n "${ORCHESTRATOR_PID:-}" ]]; then
        kill "$ORCHESTRATOR_PID" 2>/dev/null || true
        wait "$ORCHESTRATOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "[start] Starting orchestrator..."
python "$DEV_ROOT/orchestrator.py" --serve &
ORCHESTRATOR_PID=$!

# Give the service a moment to create its Unix socket.
for _ in {1..50}; do
    if [[ -S "$DEV_ROOT/.runtime/orchestrator.sock" ]]; then
        break
    fi

    if ! kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
        echo "[start] ERROR: orchestrator exited during startup."
        exit 1
    fi

    sleep 0.1
done

if [[ ! -S "$DEV_ROOT/.runtime/orchestrator.sock" ]]; then
    echo "[start] ERROR: orchestrator socket was not created."
    exit 1
fi

echo "[start] Orchestrator ready."
echo "[start] Starting GUI..."

cd "$GUI_ROOT"
sudo docker compose -f compose.dev.yaml up