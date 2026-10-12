from __future__ import annotations

"""High-resolution acoustic attack evidence for MIDI Renderer V10.

This module deliberately does *not* estimate pitch.  It answers only the timing
question: where does the waveform show a renewed vocal/acoustic articulation?
Pitch authority remains with the ECKF interpretation and its primary acoustic
measurements.
"""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class AcousticAttack:
    time_s: float
    novelty_z: float
    prominence_z: float
    rms_dbfs: float
    source: str = "wav_spectral_attack"


@dataclass
class AttackLane:
    times_s: np.ndarray
    novelty_z: np.ndarray
    rms_dbfs: np.ndarray
    peaks: list[AcousticAttack]
    hop_s: float
    active_rms_floor_db: float

    def is_audible(self, attack: AcousticAttack) -> bool:
        return math.isfinite(attack.rms_dbfs) and attack.rms_dbfs >= self.active_rms_floor_db

    def best_peak(self, start_s: float, end_s: float) -> AcousticAttack | None:
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        candidates = [p for p in self.peaks if start_s <= p.time_s <= end_s and self.is_audible(p)]
        if candidates:
            return max(candidates, key=lambda p: (p.novelty_z + 0.5 * p.prominence_z, -abs(p.time_s - end_s)))

        # A structurally important CLI boundary is allowed to ask the raw lane
        # for its strongest local maximum even if that maximum did not pass the
        # generic free-standing peak detector.  This is evidence fusion, not a
        # hidden onset hint.
        mask = (self.times_s >= start_s) & (self.times_s <= end_s)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        j = int(idx[np.argmax(self.novelty_z[idx])])
        if not math.isfinite(float(self.novelty_z[j])):
            return None
        return AcousticAttack(
            float(self.times_s[j]), float(self.novelty_z[j]), 0.0,
            float(self.rms_dbfs[j]), "wav_local_novelty_max",
        )

    def novelty_at(self, time_s: float) -> float:
        if len(self.times_s) == 0:
            return math.nan
        j = int(np.argmin(np.abs(self.times_s - float(time_s))))
        return float(self.novelty_z[j])

    def rms_at(self, time_s: float) -> float:
        if len(self.times_s) == 0:
            return math.nan
        j = int(np.argmin(np.abs(self.times_s - float(time_s))))
        return float(self.rms_dbfs[j])

    def preceding_valley_db(self, time_s: float, lookback_s: float) -> float:
        """Amplitude drop from the local pre-attack high to the local valley.

        Positive values mean the waveform dipped and then restarted.  This is
        supporting evidence for same-pitch rearticulation; it is never used as
        pitch evidence.
        """
        if len(self.times_s) == 0:
            return 0.0
        mask = (self.times_s >= time_s - lookback_s) & (self.times_s <= time_s)
        idx = np.flatnonzero(mask & np.isfinite(self.rms_dbfs))
        if len(idx) < 2:
            return 0.0
        vals = self.rms_dbfs[idx]
        return max(0.0, float(np.max(vals) - vals[-1]))


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    med = float(np.median(x[finite]))
    mad = float(np.median(np.abs(x[finite] - med))) * 1.4826
    scale = max(mad, 1e-9)
    out = (x - med) / scale
    out[~finite] = 0.0
    return out


def analyze_wav_attacks(wav_path: Path) -> AttackLane:
    wav_path = Path(wav_path)
    audio, sr = sf.read(wav_path, always_2d=True, dtype="float32")
    if audio.size == 0:
        return AttackLane(np.zeros(0), np.zeros(0), np.zeros(0), [], 0.005, -65.0)
    y = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)

    # ~23 ms analysis retains consonant/attack timing while still providing a
    # useful spectral envelope.  5 ms hop is intentionally much finer than the
    # 100-Hz interpretation timeline.
    hop = max(1, int(round(sr * 0.005)))
    win = max(256, int(round(sr * 0.023)))
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n_frames = 1 + (len(y) - win) // hop
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    high_band = freqs >= 1800.0

    flux = np.zeros(n_frames, dtype=np.float64)
    high_flux = np.zeros(n_frames, dtype=np.float64)
    rms_db = np.zeros(n_frames, dtype=np.float64)

    previous_mag: np.ndarray | None = None
    chunk = 3000
    for frame0 in range(0, n_frames, chunk):
        count = min(chunk, n_frames - frame0)
        starts = (frame0 + np.arange(count, dtype=np.int64)) * hop
        idx = starts[:, None] + np.arange(win, dtype=np.int64)[None, :]
        frames = y[idx] * window[None, :]

        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-15)
        rms_db[frame0:frame0 + count] = 20.0 * np.log10(rms + 1e-15)

        mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        # Compressed and frame-normalized spectrum: onset novelty should react
        # to redistribution/restart of energy rather than absolute loudness.
        mag = np.log1p(40.0 * mag)
        mag /= np.linalg.norm(mag, axis=1, keepdims=True) + 1e-12

        if previous_mag is None:
            stacked = mag
            diff = np.maximum(0.0, np.diff(stacked, axis=0))
            diff = np.vstack([np.zeros((1, diff.shape[1]), dtype=np.float32), diff])
        else:
            stacked = np.vstack([previous_mag, mag])
            diff = np.maximum(0.0, np.diff(stacked, axis=0))
        previous_mag = mag[-1:, :]

        flux[frame0:frame0 + count] = np.sqrt(np.mean(diff * diff, axis=1))
        if np.any(high_band):
            high_flux[frame0:frame0 + count] = np.sqrt(np.mean(diff[:, high_band] ** 2, axis=1))

    energy_rise = np.maximum(0.0, rms_db - np.r_[rms_db[0], rms_db[:-1]])
    combined = _robust_z(flux) + 0.35 * _robust_z(high_flux) + 0.35 * _robust_z(energy_rise)

    hop_s = float(hop) / float(sr)
    med_n = max(3, int(round(0.35 / hop_s)))
    if med_n % 2 == 0:
        med_n += 1
    local_baseline = median_filter(combined, size=med_n, mode="nearest")
    novelty = _robust_z(combined - local_baseline)

    # Free-standing attack proposals are intentionally permissive.  Musical
    # acceptance happens later with CLI pitch/ownership evidence.  Keeping this
    # lane permissive avoids making the waveform detector itself a hidden note
    # classifier.
    min_distance = max(1, int(round(0.025 / hop_s)))
    peak_idx, props = find_peaks(
        novelty,
        distance=min_distance,
        height=1.75,
        prominence=1.0,
    )
    times = (np.arange(n_frames, dtype=np.float64) * hop + 0.5 * win) / float(sr)
    prominences = props.get("prominences", np.zeros(len(peak_idx), dtype=float))
    peaks = [
        AcousticAttack(
            float(times[i]), float(novelty[i]), float(prom), float(rms_db[i]),
        )
        for i, prom in zip(peak_idx, prominences)
    ]
    # Conservative adaptive audibility floor.  The absolute guard prevents
    # digital-silence spectral edges from becoming attacks in files with long
    # muted sections; the percentile term adapts to noisier recordings.
    active_floor = max(-65.0, float(np.percentile(rms_db[np.isfinite(rms_db)], 10.0)) + 6.0)
    return AttackLane(times, novelty, rms_db, peaks, hop_s, active_floor)
