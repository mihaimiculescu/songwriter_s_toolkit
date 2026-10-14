from __future__ import annotations

"""Pitch/ownership evidence adapter for clean-sheet MIDI Renderer V10.

The renderer does not run a second pitch tracker.  Trusted V3R5 interpretation
is the normal authority.  Raw frame periodicity is exposed only as a bounded
rescue lane for an acoustic attack that would otherwise have no pitch owner.
"""

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left, bisect_right
import csv
import math

import numpy as np


@dataclass(frozen=True)
class PitchIsland:
    start_s: float
    end_s: float
    times_s: np.ndarray
    pitch_st: np.ndarray
    structural_st: np.ndarray
    point_count: int
    duration_s: float
    median_st: float
    stable_fraction: float
    min_stable_s: float
    source: str = "v3r5_corrected_valid"


@dataclass(frozen=True)
class PrimaryPitchPoint:
    time_s: float
    pitch_st: float
    f0_hz: float
    acf: float
    cmndf: float
    source: str


@dataclass(frozen=True)
class StructuralBoundary:
    time_s: float
    pitch_st: float
    previous_pitch_st: float
    source: str = "v3r5_persistent_stable_center"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sidecar(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix) + suffix)


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def hz_to_st(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def st_to_hz(st: float) -> float:
    return 440.0 * (2.0 ** ((float(st) - 69.0) / 12.0))


def cents_between_hz(a: float, b: float) -> float:
    if not (a > 0 and b > 0):
        return math.inf
    return abs(1200.0 * math.log2(float(a) / float(b)))


class PitchEvidence:
    def __init__(self, prefix: Path):
        self.prefix = Path(prefix)
        self.target_rows = _read_csv(_sidecar(prefix, ".target_formation.csv"))
        self.expressive_rows = _read_csv(_sidecar(prefix, ".expressive.csv"))
        self.frame_rows = _read_csv(_sidecar(prefix, ".frames.csv"))
        self.trajectory_rows = _read_csv(_sidecar(prefix, ".trajectory.csv"))
        self.reset_rows = _read_csv(_sidecar(prefix, ".onsets.csv")) if _sidecar(prefix, ".onsets.csv").exists() else []
        self.range_rows = _read_csv(_sidecar(prefix, ".range_calibration.csv")) if _sidecar(prefix, ".range_calibration.csv").exists() else []

        # R4 Stage 5 renderer acceleration: primary_support() used to rescan and
        # reparsed every frame CSV row for every acoustic attack query.  Parse
        # those immutable numeric columns once and keep a stable time-sorted
        # index.  Candidate rules themselves remain unchanged.
        frame_parsed = []
        for order, r in enumerate(self.frame_rows):
            tt = _f(r, "start_time_s")
            if not math.isfinite(tt):
                continue
            frame_parsed.append((
                float(tt), int(order),
                _f(r, "periodicity_acf_peak"),
                _f(r, "periodicity_cmndf_minimum"),
                _f(r, "periodicity_acf_frequency_hz"),
                _f(r, "periodicity_cmndf_frequency_hz"),
            ))
        frame_parsed.sort(key=lambda x: (x[0], x[1]))
        self._frame_times_sorted = [x[0] for x in frame_parsed]
        self._frame_acf_sorted = [x[2] for x in frame_parsed]
        self._frame_cm_sorted = [x[3] for x in frame_parsed]
        self._frame_fa_sorted = [x[4] for x in frame_parsed]
        self._frame_fc_sorted = [x[5] for x in frame_parsed]

        self.t = np.asarray([_f(r, "time_s") for r in self.expressive_rows], dtype=float)
        self.shape = np.asarray([_f(r, "shape_pitch_st") for r in self.expressive_rows], dtype=float)
        self.valid = np.asarray([bool(_i(r, "corrected_valid")) for r in self.expressive_rows], dtype=bool)
        self.valid &= np.isfinite(self.t) & np.isfinite(self.shape)
        self.dt = self._median_dt(self.t)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("renderer cannot infer interpretation cadence from fewer than two timed interpretation points")
        self.interpretation_hz = 1.0 / self.dt

        target_by_time = {round(_f(r, "time_s"), 6): r for r in self.target_rows if math.isfinite(_f(r, "time_s"))}
        self.structural = np.full_like(self.shape, np.nan)
        self.stable = np.zeros(len(self.shape), dtype=bool)
        min_stable_values: list[float] = []
        stable_thr_values: list[float] = []
        for i, tt in enumerate(self.t):
            r = target_by_time.get(round(float(tt), 6))
            if r is None:
                continue
            self.structural[i] = _f(r, "structural_pitch_st")
            self.stable[i] = bool(_i(r, "final_stable") or _i(r, "local_stability_pass"))
            ms = _f(r, "min_stable_threshold_ms")
            if math.isfinite(ms) and ms > 0:
                min_stable_values.append(ms / 1000.0)
            th = _f(r, "stable_range_threshold_st")
            if math.isfinite(th) and th > 0:
                stable_thr_values.append(th)
        self.min_stable_s = float(np.median(min_stable_values)) if min_stable_values else 0.08
        self.stable_range_st = float(np.median(stable_thr_values)) if stable_thr_values else 0.65
        self.structural[~np.isfinite(self.structural)] = self.shape[~np.isfinite(self.structural)]

        self.zero_low_st, self.zero_high_st = self._range_zero_bounds()
        self.reset_times = sorted(
            _f(r, "time_s") for r in self.reset_rows if math.isfinite(_f(r, "time_s"))
        )
        self.islands = self._build_islands()
        self.structural_boundaries = self._build_structural_boundaries()

    @staticmethod
    def _median_dt(t: np.ndarray) -> float:
        x = t[np.isfinite(t)]
        if len(x) < 2:
            return math.nan
        d = np.diff(x)
        d = d[d > 0]
        return float(np.median(d)) if len(d) else math.nan

    def _range_zero_bounds(self) -> tuple[float, float]:
        if self.range_rows:
            lo = _f(self.range_rows[0], "zero_low_midi")
            hi = _f(self.range_rows[0], "zero_high_midi")
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                return float(lo), float(hi)
        vals = self.shape[self.valid]
        if len(vals):
            return float(np.quantile(vals, 0.01) - 4.0), float(np.quantile(vals, 0.99) + 4.0)
        return 24.0, 96.0

    def _build_islands(self) -> list[PitchIsland]:
        out: list[PitchIsland] = []
        i = 0
        n = len(self.t)
        while i < n:
            if not self.valid[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and self.valid[j + 1] and self.t[j + 1] - self.t[j] <= 1.6 * self.dt:
                j += 1
            tt = self.t[i:j + 1].copy()
            pp = self.shape[i:j + 1].copy()
            ss = self.structural[i:j + 1].copy()
            end = float(tt[-1] + self.dt)
            duration = max(self.dt, end - float(tt[0]))
            out.append(PitchIsland(
                float(tt[0]), end, tt, pp, ss, len(tt), duration,
                float(np.median(pp)), float(np.mean(self.stable[i:j + 1])),
                self.min_stable_s,
            ))
            i = j + 1
        return out

    def _build_structural_boundaries(self) -> list[StructuralBoundary]:
        """CLI-frozen stable-target landmarks, not MIDI semitone crossings.

        Each STABLE_TARGET region is already the end product of V3R5 target
        formation.  V10 exposes its start as a *proposal*.  The event assembler
        accepts large target changes directly; near-same-pitch target restarts
        still need independent attack/reset support.
        """
        stable_rows = [r for r in self.trajectory_rows if (r.get("kind") or "") == "STABLE_TARGET"]
        out: list[StructuralBoundary] = []
        previous_st: float | None = None
        for r in stable_rows:
            t = _f(r, "start_time_s")
            st = _f(r, "median_pitch_st")
            if not (math.isfinite(t) and math.isfinite(st)):
                continue
            if previous_st is None:
                previous_st = float(st)
                continue
            out.append(StructuralBoundary(float(t), float(st), float(previous_st), "v3r5_stable_target_start"))
            previous_st = float(st)
        return out

    def reset_near(self, time_s: float, radius_s: float) -> float | None:
        xs = [x for x in self.reset_times if abs(x - time_s) <= radius_s]
        return min(xs, key=lambda x: abs(x - time_s)) if xs else None

    def island_containing(self, time_s: float, tolerance_s: float = 0.0) -> PitchIsland | None:
        for island in self.islands:
            if island.start_s - tolerance_s <= time_s < island.end_s + tolerance_s:
                return island
        return None

    def next_island(self, time_s: float, horizon_s: float) -> PitchIsland | None:
        candidates = [x for x in self.islands if time_s - self.dt <= x.start_s <= time_s + horizon_s]
        return min(candidates, key=lambda x: x.start_s) if candidates else None

    def pitch_curve(self, start_s: float, end_s: float, fallback_st: float) -> tuple[np.ndarray, np.ndarray]:
        mask = self.valid & (self.t >= start_s - 1e-9) & (self.t <= end_s + 1e-9)
        tt = self.t[mask].copy()
        pp = self.shape[mask].copy()
        if len(tt) == 0:
            return np.asarray([start_s, end_s], dtype=float), np.asarray([fallback_st, fallback_st], dtype=float)
        if tt[0] > start_s + 1e-9:
            tt = np.insert(tt, 0, start_s)
            pp = np.insert(pp, 0, pp[0])
        if tt[-1] < end_s - 1e-9:
            tt = np.append(tt, end_s)
            pp = np.append(pp, pp[-1])
        return tt, pp

    def _frame_candidate(self, r: dict[str, str], future_st: float | None) -> PrimaryPitchPoint | None:
        t = _f(r, "start_time_s")
        acf = _f(r, "periodicity_acf_peak")
        cm = _f(r, "periodicity_cmndf_minimum")
        fa = _f(r, "periodicity_acf_frequency_hz")
        fc = _f(r, "periodicity_cmndf_frequency_hz")
        if not (math.isfinite(t) and math.isfinite(acf) and math.isfinite(cm)):
            return None
        # Bounded rescue: this lane is intentionally stricter than ordinary
        # tracking.  It exists only to give an otherwise orphaned attack a
        # defensible pitch owner.
        if acf < 0.78 or cm > 0.24:
            return None
        candidates = [x for x in (fa, fc) if math.isfinite(x) and x > 0]
        if not candidates:
            return None

        chosen: float | None = None
        source = "primary_periodicity_agreement"
        if len(candidates) == 1:
            chosen = candidates[0]
        elif cents_between_hz(candidates[0], candidates[1]) <= 90.0:
            chosen = math.sqrt(candidates[0] * candidates[1])
        else:
            # When ACF and CMNDF disagree (often by an integer factor), prefer
            # the candidate supported by an already-trusted future owner.  If
            # no such owner exists, an exceptionally clean ACF/CMNDF *quality*
            # pair may select the in-range ACF period.  This does not alter the
            # CLI trajectory; it only services an orphan MIDI attack.
            if future_st is not None and math.isfinite(future_st):
                chosen = min(candidates, key=lambda hz: abs(hz_to_st(hz) - future_st))
                if abs(hz_to_st(chosen) - future_st) > 3.0:
                    chosen = None
                else:
                    source = "primary_periodicity_future_owner_match"
            elif acf >= 0.93 and cm <= 0.08 and math.isfinite(fa) and fa > 0:
                chosen = fa
                source = "primary_periodicity_strong_acf"
        if chosen is None:
            return None
        st = hz_to_st(chosen)
        if st < self.zero_low_st - 1e-6 or st > self.zero_high_st + 1e-6:
            return None
        return PrimaryPitchPoint(float(t), float(st), float(chosen), float(acf), float(cm), source)

    def _frame_candidate_values(
        self, t: float, acf: float, cm: float, fa: float, fc: float, future_st: float | None
    ) -> PrimaryPitchPoint | None:
        # Exact numeric equivalent of _frame_candidate(), operating on the
        # pre-parsed immutable frame columns above.
        if not (math.isfinite(t) and math.isfinite(acf) and math.isfinite(cm)):
            return None
        if acf < 0.78 or cm > 0.24:
            return None
        candidates = [x for x in (fa, fc) if math.isfinite(x) and x > 0]
        if not candidates:
            return None

        chosen: float | None = None
        source = "primary_periodicity_agreement"
        if len(candidates) == 1:
            chosen = candidates[0]
        elif cents_between_hz(candidates[0], candidates[1]) <= 90.0:
            chosen = math.sqrt(candidates[0] * candidates[1])
        else:
            if future_st is not None and math.isfinite(future_st):
                chosen = min(candidates, key=lambda hz: abs(hz_to_st(hz) - future_st))
                if abs(hz_to_st(chosen) - future_st) > 3.0:
                    chosen = None
                else:
                    source = "primary_periodicity_future_owner_match"
            elif acf >= 0.93 and cm <= 0.08 and math.isfinite(fa) and fa > 0:
                chosen = fa
                source = "primary_periodicity_strong_acf"
        if chosen is None:
            return None
        st = hz_to_st(chosen)
        if st < self.zero_low_st - 1e-6 or st > self.zero_high_st + 1e-6:
            return None
        return PrimaryPitchPoint(float(t), float(st), float(chosen), float(acf), float(cm), source)

    def primary_support(
        self,
        start_s: float,
        end_s: float,
        *,
        future_st: float | None = None,
    ) -> list[PrimaryPitchPoint]:
        raw: list[PrimaryPitchPoint] = []
        times = self._frame_times_sorted
        lo = bisect_left(times, float(start_s) - 1e-9)
        hi = bisect_right(times, float(end_s) + 1e-9)
        for j in range(lo, hi):
            p = self._frame_candidate_values(
                times[j],
                self._frame_acf_sorted[j],
                self._frame_cm_sorted[j],
                self._frame_fa_sorted[j],
                self._frame_fc_sorted[j],
                future_st,
            )
            if p is not None:
                raw.append(p)
        if not raw:
            return []

        # Keep a coherent local trajectory.  Vocal motion can be fast, so this
        # is intentionally a wide motion envelope; it rejects octave-family
        # chaos, not legitimate melisma.
        best: list[PrimaryPitchPoint] = [raw[0]]
        for p in raw[1:]:
            gap = p.time_s - best[-1].time_s
            if gap > 0.12:
                break
            allowed = 7.0 * max(1.0, gap / 0.05)
            if abs(p.pitch_st - best[-1].pitch_st) <= allowed:
                best.append(p)
        return best
