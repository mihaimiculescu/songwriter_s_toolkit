# WAV → MIDI

Production monophonic vocal WAV-to-MIDI converter.

```bash
python -m python_eckf vocal.wav
```

This creates `vocal.mid` next to the source WAV.

Optional project integration:

```bash
python -m python_eckf vocal.wav --out melody.mid --bpm 120
```

The converter automatically detects CPU/RAM resources. It produces no persistent
analysis CSV files or audit files; only the requested MIDI is kept.

Input must be mono WAV.
