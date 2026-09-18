#!/usr/bin/env bash
# Waveform-only, four-recording ECKF robustness run. Run from repository root.
set -euo pipefail
ROOT="$(pwd)"
DIAG="$ROOT/tests/diagnose_synchronized_eckf.py"
SELECTOR="$ROOT/python_eckf/initialization_candidates.py"
OUT="$ROOT/tests/robustness_v3"
[[ -f "$DIAG" && -f "$SELECTOR" ]] || { echo 'Run from songwriter_s_toolkit root; diagnostic or selector missing.' >&2; exit 2; }
mkdir -p "$OUT"
python - "$ROOT" "$OUT" <<'PY'
import sys, pathlib, hashlib, json
import soundfile as sf
root, out = map(pathlib.Path,sys.argv[1:])
items = []
for stem in ('Ochiitai','PREDESTINATI','RATATA','Trandafiri'):
    choices = [p for p in (root/'tests'/f'{stem}.wav',root/'tests'/f'{stem}.wa') if p.is_file()]
    if len(choices) != 1:
        raise SystemExit(f'ERROR: expected exactly one tests/{stem}.wav or .wa; found {len(choices)}. Check filename.')
    p = choices[0]
    info=sf.info(str(p))
    if not info.frames or not info.samplerate:
        raise SystemExit(f'ERROR: unreadable/empty audio: {p}')
    items.append(dict(name=stem,path=str(p),samplerate=info.samplerate,frames=info.frames,duration_s=info.duration,channels=info.channels,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
module=root/'python_eckf'/'initialization_candidates.py'
manifest=dict(selector_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),ground_truth_used=False,files=items)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for item in items:
    print(f"{item['name']}: {item['duration_s']:.3f}s, {item['samplerate']}Hz, {item['channels']}ch, {item['path']}")
print('Selector SHA256:',manifest['selector_sha256'])
print('Ground truth: NOT read or passed to tracker.')
PY
# Save a real test log; tee ensures failures propagate through pipefail.
: > "$OUT/unit_tests.log"
for pattern in test_initialization_candidates.py test_selector_priority_regressions.py test_selector_family_v3.py; do
  python -m unittest discover -s tests -p "$pattern" -v 2>&1 | tee -a "$OUT/unit_tests.log"
done
for stem in Ochiitai PREDESTINATI RATATA Trandafiri; do
  input="$ROOT/tests/$stem.wav"
  [[ -f "$input" ]] || input="$ROOT/tests/$stem.wa"
  duration="$(python - "$input" <<'PY'
import sys,soundfile as sf
info=sf.info(sys.argv[1]);print(f'{info.duration:.9f}')
PY
)"
  echo "=== Waveform-only: $stem (0 to ${duration}s) ==="
  python "$DIAG" --wav "$input" --start 0 --end "$duration" --output "$OUT/${stem}_v3.txt" > "$OUT/${stem}_stdout.log" 2>&1 || {
    echo "FAILED: $stem; see $OUT/${stem}_stdout.log" >&2
    exit 1
  }
  [[ -s "$OUT/${stem}_v3.txt" && -s "$OUT/${stem}_v3_frames.csv" ]] || {
    echo "FAILED: missing report or frame CSV for $stem" >&2; exit 1;
  }
  python - "$input" "$OUT/${stem}_v3.txt" "$OUT/${stem}_v3_frames.csv" <<'PYVERIFY'
import sys, pathlib, soundfile as sf, csv
wav, report, frames = map(pathlib.Path, sys.argv[1:])
text=report.read_text()
info=sf.info(str(wav))
if f"WAV: {wav}" not in text or f"Sample rate: {info.samplerate} Hz" not in text:
    raise SystemExit(f"FAILED: diagnostic used incorrect WAV or sample rate for {wav}")
with frames.open(newline="") as f:
    rows=list(csv.DictReader(f))
expected=(info.frames+2047)//2048
if len(rows)!=expected:
    raise SystemExit(f"FAILED: expected {expected} output frames for {wav.name}, got {len(rows)}")
print(f"Verified {wav.name}: correct WAV, {info.samplerate} Hz, {len(rows)} frames")
PYVERIFY
  echo "Completed $stem: $OUT/${stem}_v3.txt"
done
echo "All 4 audio runs completed. Outputs: $OUT"
echo 'MIDI was not read. Examine waveform evidence first; check GT only retrospectively.'
