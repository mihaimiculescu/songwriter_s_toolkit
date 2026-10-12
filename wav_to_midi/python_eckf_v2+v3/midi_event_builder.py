from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResidualF0Event:
    start_s: float
    end_s: float
    median_st: float
    source_id: str
    point_count: int
    trajectory_kinds: str
    min_stable_threshold_s: float = 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    v = row.get(key, '')
    if v in ('', None):
        return default
    try:
        return float(v)
    except Exception:
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    v = row.get(key, '')
    if v in ('', None):
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _sidecar(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix) + suffix)


def _subtract_intervals(start: float, end: float, occupied: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for a, b in sorted(occupied):
        if b <= start or a >= end:
            continue
        new: list[tuple[float, float]] = []
        for x, y in pieces:
            if b <= x or a >= y:
                new.append((x, y))
                continue
            if a > x:
                new.append((x, min(a, y)))
            if b < y:
                new.append((max(b, x), y))
        pieces = [(x, y) for x, y in new if y > x + 1e-9]
        if not pieces:
            break
    return pieces


def valid_f0_islands(prefix: Path) -> list[ResidualF0Event]:
    """Return every contiguous corrected-valid F0 island.

    `ResidualF0Event` is reused as a compact island descriptor here.  The
    renderer decides whether an island extends an existing structural note or
    must become a new residual musical event.
    """
    tf_path = _sidecar(prefix, '.target_formation.csv')
    if not tf_path.exists():
        return []
    rows = _read_csv(tf_path)
    if not rows:
        return []

    t = np.asarray([_f(r, 'time_s') for r in rows], dtype=float)
    valid = np.asarray([bool(_i(r, 'corrected_valid', 0)) for r in rows], dtype=bool)
    st = np.asarray([_f(r, 'structural_pitch_st') for r in rows], dtype=float)
    valid &= np.isfinite(t) & np.isfinite(st)
    finite_t = t[np.isfinite(t)]
    dt = float(np.median(np.diff(finite_t))) if len(finite_t) >= 2 else 0.01
    kinds = [r.get('trajectory_region_kind', '') or '' for r in rows]
    min_stable_ms = np.asarray([_f(r, 'min_stable_threshold_ms') for r in rows], dtype=float)

    out: list[ResidualF0Event] = []
    serial = 0
    i = 0
    while i < len(rows):
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and valid[j + 1] and (t[j + 1] - t[j]) <= 1.5 * dt:
            j += 1
        vals = st[i:j + 1]
        ks = sorted({kinds[k] for k in range(i, j + 1) if kinds[k]})
        out.append(ResidualF0Event(
            start_s=float(t[i]),
            end_s=float(t[j] + dt),
            median_st=float(np.median(vals)),
            source_id=f'valid_f0_island_{serial}',
            point_count=int(len(vals)),
            trajectory_kinds='|'.join(ks),
            min_stable_threshold_s=(float(np.nanmedian(min_stable_ms[i:j + 1])) / 1000.0
                                    if np.any(np.isfinite(min_stable_ms[i:j + 1])) else 0.0),
        ))
        serial += 1
        i = j + 1
    return out


def build_residual_valid_f0_events(
    prefix: Path,
    occupied_intervals: list[tuple[float, float]],
) -> list[ResidualF0Event]:
    """Return only *completely ownerless* corrected-valid F0 islands.

    Fringes of an island that already contains a stable/short target are not
    new notes; the renderer extends the structural event to own those fringes.
    A new residual MIDI event is created only when the whole corrected-valid
    island has no structural note owner at all.
    """
    out: list[ResidualF0Event] = []
    for isl in valid_f0_islands(prefix):
        overlaps = [
            (a, b) for a, b in occupied_intervals
            if a < isl.end_s - 1e-12 and b > isl.start_s + 1e-12
        ]
        if not overlaps:
            # Ownership completeness must not turn every pitched consonant
            # fragment into a note.  Reuse the target-former's *existing*
            # minimum stable-duration scale as the promotion gate for a wholly
            # ownerless F0 island.  No MIDI-specific duration threshold is
            # introduced here.
            duration_s = max(0.0, isl.end_s - isl.start_s)
            if (isl.min_stable_threshold_s > 0.0 and
                    duration_s + 1e-12 >= isl.min_stable_threshold_s):
                out.append(isl)
    return out


def coverage_audit(
    prefix: Path,
    final_event_intervals: list[tuple[float, float]],
) -> tuple[list[dict[str, object]], float]:
    """Audit corrected-valid timeline cells not represented by MIDI.

    Tiny completely-ownerless F0 islands that fail the existing target-former
    minimum stable-duration rule are recorded as explicit rejections rather
    than silently counted as uncovered.  Any other uncovered corrected-valid
    cell remains an error condition.
    """
    tf_path = _sidecar(prefix, '.target_formation.csv')
    if not tf_path.exists():
        return [], 0.0
    rows = _read_csv(tf_path)
    if not rows:
        return [], 0.0
    t = np.asarray([_f(r, 'time_s') for r in rows], dtype=float)
    valid = np.asarray([bool(_i(r, 'corrected_valid', 0)) for r in rows], dtype=bool)
    st = np.asarray([_f(r, 'structural_pitch_st') for r in rows], dtype=float)
    valid &= np.isfinite(t) & np.isfinite(st)
    finite_t = t[np.isfinite(t)]
    dt = float(np.median(np.diff(finite_t))) if len(finite_t) >= 2 else 0.01

    # Identify ownerless islands that were deliberately rejected by the same
    # upstream duration rule used by build_residual_valid_f0_events().
    rejected: list[tuple[float, float]] = []
    for isl in valid_f0_islands(prefix):
        overlaps_event = any(
            a < isl.end_s - 1e-12 and b > isl.start_s + 1e-12
            for a, b in final_event_intervals
        )
        duration_s = max(0.0, isl.end_s - isl.start_s)
        if (not overlaps_event and isl.min_stable_threshold_s > 0.0 and
                duration_s + 1e-12 < isl.min_stable_threshold_s):
            rejected.append((isl.start_s, isl.end_s))

    rows_out: list[dict[str, object]] = []
    uncovered = 0.0
    for idx, (tt, vv, ss) in enumerate(zip(t, valid, st)):
        if not vv:
            continue
        cell_a, cell_b = float(tt), float(tt + dt)
        covered = any(a < cell_b - 1e-12 and b > cell_a + 1e-12 for a, b in final_event_intervals)
        if covered:
            continue
        explicitly_rejected = any(a < cell_b - 1e-12 and b > cell_a + 1e-12 for a, b in rejected)
        status = ('EXPLICITLY_REJECTED_SHORT_OWNERLESS_F0'
                  if explicitly_rejected else 'UNOWNED_VALID_F0')
        if not explicitly_rejected:
            uncovered += dt
        rows_out.append({
            'trajectory_index': idx,
            'time_s': cell_a,
            'end_s': cell_b,
            'structural_pitch_st': float(ss),
            'status': status,
        })
    return rows_out, float(uncovered)

