V11_04 VOICING JUROR — READ ONLY

Copy audit_voicing_juror_v1.py into tests/ and run FROM REPO ROOT:

python tests/audit_voicing_juror_v1.py \
  --wav-root tests \
  --v11-zip tests/shadow_adjudicator_v22_fixed_v11_04.zip \
  --out tests/voicing_juror_v1

Outputs: four <song>_voicing_observations.csv files, cross_tab_by_song.csv,
example_observations.csv, manifest.json.

Package:
7z a -r tests/voicing_juror_v1.zip tests/voicing_juror_v1

Dependencies: numpy, soundfile (pip install numpy soundfile).

RESTRICTIONS: Diagnostic only. It does not edit WAVs, V11 decisions or produce MIDI.
A 'voiced_like' label means strong evidence for periodic acoustic energy, NOT proof
that the signal is a human vocal rather than music/noise. 'unvoiced_like' does NOT
mean a MIDI rest. All thresholds are diagnostic hypotheses, not calibrated truths.

Inputs: V11_04's observation summaries and original WAV files. Ratata's excluded
tail is absent from observation summaries and never added to the population.
The case-relative energy reference uses the 90th percentile of 40ms observation
RMS levels; it is not an independently measured instrumental/noise floor.

Conservative three-way logic:
* voiced_like: >=2 of 3 window autocorrelations >=.62, >=2 period-contrast >=.12,
  and not >28dB below song's p90 short-window energy.
* unvoiced_like: all 3 ACF <=.27, and either very low relative energy or noise-like
  spectrum in >=2 windows (flatness >=.45).
* otherwise uncertain.
Windows centered on each observation: 40ms, 64ms, 96ms. No interpolation across
waveform boundaries. Independent of existing pitch-candidate admissibility gates.
