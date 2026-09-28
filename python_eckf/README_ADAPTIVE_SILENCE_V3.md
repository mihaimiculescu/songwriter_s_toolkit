# ECKF V2 adaptive silence V3

This revision changes only the digital-mute detector used by per-file silence calibration.

- Digital mute is now defined in RMS terms, not by a tiny peak-amplitude epsilon.
- The threshold is -145 dBFS RMS.
- Linear RMS amplitude equivalent: 10^(-145/20) = 5.6234132519e-8 full scale.
- Frames at or below that RMS level are excluded from the acoustic-noise calibration population and break candidate quiet runs.
- The existing safety rule remains: if the calculated per-file historical-statistic threshold would be below -50, use -50.
- The rest of the V2 silence calibrator is unchanged in this revision.

Important: this revision does not solve a different problem: a file such as Ochiitai can contain no true acoustic-silence plateau because every musical pause is filled by reverb tails. In that case the current long-quiet-region method can still mistake reverb decay for a noise-floor sample. That needs a separate decision/fallback rule rather than a digital-mute threshold tweak.
