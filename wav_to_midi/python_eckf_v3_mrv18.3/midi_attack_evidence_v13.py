from __future__ import annotations

"""High-resolution acoustic attack evidence for MIDI Renderer V13.

V13 keeps the clean-sheet rule that the WAV answers timing, not pitch.  It
changes attack ownership from a local-valley test to an object-to-valley-to-object
test over the complete attack-to-pitch corridor.  A dying precursor cannot keep
ownership merely because the final 65 ms before a valley have already decayed.
Once the earlier object, the collapse, and an independent later restart are all
visible, the later object owns the future pitch.  Closure duration never pardons
a proven separation.
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


@dataclass(frozen=True)
class OwnershipContinuity:
    """Acoustic continuity evidence between an attack and a later pitch anchor."""

    connected: bool
    status: str = "continuous"
    barrier_time_s: float = math.nan
    valley_dbfs: float = math.nan
    left_level_dbfs: float = math.nan
    right_level_dbfs: float = math.nan
    depth_left_db: float = 0.0
    depth_right_db: float = 0.0
    low_duration_s: float = 0.0
    restart_novelty_z: float = math.nan
    restart_time_s: float = math.nan
    object_left_peak_dbfs: float = math.nan
    object_right_peak_dbfs: float = math.nan


def _attack_strength_local(p: AcousticAttack) -> float:
    return float(p.novelty_z + 0.5 * p.prominence_z)


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
        if len(self.times_s) == 0:
            return 0.0
        mask = (self.times_s >= time_s - lookback_s) & (self.times_s <= time_s)
        idx = np.flatnonzero(mask & np.isfinite(self.rms_dbfs))
        if len(idx) < 2:
            return 0.0
        vals = self.rms_dbfs[idx]
        return max(0.0, float(np.max(vals) - vals[-1]))

    def ownership_continuity(
        self,
        attack_time_s: float,
        pitch_anchor_s: float,
        *,
        post_anchor_grace_s: float = 0.03,
    ) -> OwnershipContinuity:
        """Decide whether an attack belongs to the acoustic object reaching pitch.

        V13 deliberately evaluates the *whole* corridor from candidate attack to
        trustworthy pitch.  V12 used short left/right windows around a valley; a
        weak precursor that slowly decayed could therefore look locally flat by
        the time the deepest valley was reached and incorrectly retain ownership.

        A V13 barrier needs three independent pieces of morphology:

        1. an earlier audible object (measured from the candidate through its
           decaying lobe, not only the final few milliseconds before the valley),
        2. a deep collapse, either recording-near-inactive or exceptionally deep
           relative to both objects, and
        3. a later spectral restart leading toward the pitch anchor.

        There is intentionally no minimum-duration pardon.  If a short closure
        proves that one object ended and another began, ownership stops there.
        """
        if len(self.times_s) == 0:
            return OwnershipContinuity(True, "no_wav_lane")
        a = float(attack_time_s)
        b = float(pitch_anchor_s) + max(0.0, float(post_anchor_grace_s))
        if not (math.isfinite(a) and math.isfinite(b)) or b <= a + 0.025:
            return OwnershipContinuity(True, "too_short_to_test")

        mask = (self.times_s >= a) & (self.times_s <= b) & np.isfinite(self.rms_dbfs)
        idx = np.flatnonzero(mask)
        if len(idx) < 5:
            return OwnershipContinuity(True, "too_short_to_test")

        tt = self.times_s[idx]
        rr = median_filter(self.rms_dbfs[idx], size=3, mode="nearest")
        nov = self.novelty_z[idx]
        attack_rms = self.rms_at(a)

        near_floor = self.active_rms_floor_db + 3.0
        restart_novelty_z = 2.0
        ordinary_edge_depth_db = 10.0
        relative_deep_edge_db = 18.0

        # Peak proposals are the independent restart witnesses.  Keep all audible
        # ones in the corridor; the later object's level is measured around the
        # restart itself rather than from a tiny window next to the valley.
        corridor_peaks = [
            p for p in self.peaks
            if a + self.hop_s <= p.time_s <= b and self.is_audible(p)
        ]

        barriers: list[OwnershipContinuity] = []

        for k in range(1, len(tt) - 1):
            valley = float(rr[k])

            # A later independent restart must exist.  This is what keeps an
            # ordinary intra-object amplitude dip from becoming a boundary.
            later_peaks = [p for p in corridor_peaks if p.time_s >= tt[k] + self.hop_s]
            if not later_peaks:
                continue
            # Use the first independently defensible restart after the collapse,
            # not the strongest event arbitrarily later in the corridor.  This
            # keeps a remote later note from retroactively turning a harmless
            # intra-object dip into a barrier.
            strong_restarts = [p for p in later_peaks if _attack_strength_local(p) >= 6.0]
            if not strong_restarts:
                continue
            restart = min(strong_restarts, key=lambda p: p.time_s)
            restart_strength = _attack_strength_local(restart)

            post = (tt >= tt[k] + self.hop_s) & (tt <= min(b, restart.time_s + 0.035))
            restart_z = float(np.max(nov[post])) if np.any(post) else -math.inf
            if not (restart_z >= restart_novelty_z or restart_strength >= 6.0):
                continue

            # OBJECT-LEVEL sides.  The left side begins at the candidate attack
            # and therefore remembers the precursor's real peak even after a long
            # decay.  The right side extends to/through the independent restart.
            left = (tt >= a) & (tt <= tt[k] - self.hop_s)
            right = (tt >= tt[k] + self.hop_s) & (tt <= min(b, restart.time_s + 0.035))
            if np.count_nonzero(left) < 2 or np.count_nonzero(right) < 2:
                continue

            left_peak = float(np.max(rr[left]))
            if math.isfinite(attack_rms):
                left_peak = max(left_peak, float(attack_rms))
            right_peak = float(np.max(rr[right]))
            if math.isfinite(restart.rms_dbfs):
                right_peak = max(right_peak, float(restart.rms_dbfs))

            depth_left = left_peak - valley
            depth_right = right_peak - valley

            # Two defensible collapse modes:
            #   - near-inactive relative to this recording, or
            #   - so deep relative to both objects that a noisy floor should not
            #     hide the separation.
            near_inactive = valley <= near_floor
            deep_relative = min(depth_left, depth_right) >= relative_deep_edge_db
            bilateral = min(depth_left, depth_right) >= ordinary_edge_depth_db

            # Dying weak precursor: allow a slightly smaller left depth only when
            # the later object is dramatically stronger.  This is the morphology
            # seen in the ElTelefono regressions: weak object -> decay -> valley
            # -> much stronger true note.
            detached_precursor = (
                depth_left >= 8.0
                and depth_right >= 15.0
                and right_peak >= left_peak + 8.0
            )

            if not ((near_inactive and (bilateral or detached_precursor)) or deep_relative):
                continue

            # Measure the contiguous low region for audit only.  Duration is not
            # an acceptance gate in V13.
            low_threshold = min(
                near_floor,
                min(left_peak, right_peak) - 6.0,
            )
            low = rr <= low_threshold
            i = k
            while i > 0 and bool(low[i - 1]) and tt[i] - tt[i - 1] <= self.hop_s * 1.6:
                i -= 1
            j = k
            while j + 1 < len(tt) and bool(low[j + 1]) and tt[j + 1] - tt[j] <= self.hop_s * 1.6:
                j += 1
            low_duration = float(tt[j] - tt[i] + self.hop_s)

            barriers.append(OwnershipContinuity(
                connected=False,
                status=(
                    "object_detached_precursor_barrier"
                    if detached_precursor and not bilateral
                    else "object_interruption_barrier"
                ),
                barrier_time_s=float(0.5 * (tt[i] + tt[j])),
                valley_dbfs=float(np.min(rr[i:j + 1])),
                left_level_dbfs=left_peak,
                right_level_dbfs=right_peak,
                depth_left_db=depth_left,
                depth_right_db=depth_right,
                low_duration_s=low_duration,
                restart_novelty_z=restart_z,
                restart_time_s=float(restart.time_s),
                object_left_peak_dbfs=left_peak,
                object_right_peak_dbfs=right_peak,
            ))

        if not barriers:
            return OwnershipContinuity(True, "continuous")

        # The *last* proven collapse defines the start of the final acoustic
        # object feeding the pitch anchor.  An earlier candidate is disconnected
        # no matter how strong it was.
        return max(barriers, key=lambda ev: ev.barrier_time_s)


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
    # interpretation timeline (135 Hz for frozen V3R9, but not hard-coded here).
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
