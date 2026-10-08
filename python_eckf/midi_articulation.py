from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArticulationPoint:
    time_s: float
    source: str
    score: float = 1.0


@dataclass(frozen=True)
class NonMusicalGap:
    start_s: float
    end_s: float
    source: str


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


def _load_reset_landmarks(prefix: Path) -> list[float]:
    """Return the existing ECKF reset landmarks.

    IMPORTANT: these are supporting landmarks, not musical Note-On decisions.
    The tracker's current 2-semitone harmonic-change policy remains untouched.
    """
    p = _sidecar(prefix, '.onsets.csv')
    if p.exists():
        return [_f(r, 'time_s') for r in _read_csv(p) if math.isfinite(_f(r, 'time_s'))]

    # Backward-compatible fallback for result packs generated before .onsets.csv.
    p = _sidecar(prefix, '.frames.csv')
    if not p.exists():
        return []
    out: list[float] = []
    for r in _read_csv(p):
        src = (r.get('initialization_source') or '').strip().lower()
        decision = (r.get('frame_decision') or '').strip().upper()
        if not src or src == 'none' or 'UNRESOLVED' in decision:
            continue
        t = _f(r, 'start_time_s')
        if math.isfinite(t):
            out.append(t)
    return out


def _timeline_dt(prefix: Path) -> float:
    p = _sidecar(prefix, '.target_formation.csv')
    if not p.exists():
        return 0.01
    ts = np.asarray([_f(r, 'time_s') for r in _read_csv(p)], dtype=float)
    ts = ts[np.isfinite(ts)]
    if len(ts) < 2:
        return 0.01
    return float(np.median(np.diff(ts)))


def _project_suppression_s(prefix: Path) -> float:
    """Existing project stable-target timescale used only to merge duplicate cues."""
    p = _sidecar(prefix, '.target_formation.csv')
    if not p.exists():
        return 0.08
    vals = [_f(r, 'min_stable_threshold_ms') for r in _read_csv(p)]
    vals = [x for x in vals if math.isfinite(x) and x > 0]
    return float(np.median(vals)) / 1000.0 if vals else 0.08


def nonmusical_gaps(prefix: Path) -> list[NonMusicalGap]:
    """Find genuine pitch-absent interruptions from existing interpretation.

    This is deliberately semantic rather than another pitch juror.  A gap is a
    contiguous corrected-invalid run on the 100-Hz target-formation timeline
    with trusted pitch on both sides.  It is therefore an interruption between
    two musical pitch islands, not simply leading/trailing silence.

    No new duration threshold is introduced: any represented contiguous gap is
    retained.  The renderer later uses its exact boundaries for Note Off / On.
    """
    p = _sidecar(prefix, '.target_formation.csv')
    if not p.exists():
        return []
    rows = _read_csv(p)
    if not rows:
        return []
    t = np.asarray([_f(r, 'time_s') for r in rows], dtype=float)
    valid = np.asarray([bool(_i(r, 'corrected_valid', 0)) for r in rows], dtype=bool)
    dt = _timeline_dt(prefix)

    out: list[NonMusicalGap] = []
    i = 0
    while i < len(rows):
        if valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and not valid[j + 1]:
            j += 1
        has_left = i > 0 and valid[i - 1]
        has_right = j + 1 < len(rows) and valid[j + 1]
        if has_left and has_right and math.isfinite(t[i]) and math.isfinite(t[j]):
            start = float(t[i])
            end = float(t[j] + dt)
            out.append(NonMusicalGap(start, end, 'corrected_valid_interruption'))
        i = j + 1
    return out


def stable_target_articulations(prefix: Path) -> list[ArticulationPoint]:
    """Legato Note-On proposals from *persistent* target-center changes.

    This deliberately replaces the older ``round(structural_pitch_st)`` run
    splitter.  Rounded semitone crossings are too sensitive to vibrato and to
    small excursions near a MIDI boundary.  Instead we reuse two quantities
    already established by target formation:

      * ``stable_range_threshold_st`` -- how far a pitch may wander while still
        belonging to one stable center;
      * ``min_stable_threshold_ms`` -- how long a new center must persist.

    A new legato articulation is proposed only when a contiguous stable run
    establishes a robust center farther than the existing stability range from
    the previous center *and* persists for the existing minimum stable time.
    No MIDI-specific pitch-distance or duration threshold is introduced.
    """
    p = _sidecar(prefix, '.target_formation.csv')
    if not p.exists():
        return []
    rows = _read_csv(p)
    if not rows:
        return []

    dt = _timeline_dt(prefix)
    pts: list[dict[str, float]] = []
    for r in rows:
        t = _f(r, 'time_s')
        st = _f(r, 'structural_pitch_st')
        valid = bool(_i(r, 'corrected_valid', 0))
        stable = bool(_i(r, 'final_stable', 0) or _i(r, 'local_stability_pass', 0))
        thr = _f(r, 'stable_range_threshold_st')
        min_ms = _f(r, 'min_stable_threshold_ms')
        if not (valid and stable and math.isfinite(t) and math.isfinite(st)):
            continue
        pts.append({
            't': float(t), 'st': float(st),
            'thr': float(thr) if math.isfinite(thr) and thr > 0 else math.nan,
            'min_s': float(min_ms) / 1000.0 if math.isfinite(min_ms) and min_ms > 0 else math.nan,
        })
    if len(pts) < 2:
        return []

    # Split only at real temporal gaps first.  Within one continuous stable
    # span, find persistent center changes with the target-former's own scales.
    spans: list[list[dict[str, float]]] = []
    cur: list[dict[str, float]] = []
    for q in pts:
        if cur and q['t'] - cur[-1]['t'] > 1.5 * dt:
            spans.append(cur); cur = []
        cur.append(q)
    if cur:
        spans.append(cur)

    out: list[ArticulationPoint] = []
    for span in spans:
        if len(span) < 2:
            continue
        thresholds = [q['thr'] for q in span if math.isfinite(q['thr']) and q['thr'] > 0]
        mins = [q['min_s'] for q in span if math.isfinite(q['min_s']) and q['min_s'] > 0]
        stable_thr = float(np.median(thresholds)) if thresholds else 0.65
        min_stable_s = float(np.median(mins)) if mins else _project_suppression_s(prefix)
        need_n = max(1, int(math.ceil(min_stable_s / max(dt, 1e-9))))

        seg_start = 0
        center = float(np.median([q['st'] for q in span[:min(len(span), need_n)]]))
        i = max(1, need_n)
        while i < len(span):
            # Candidate landing begins when pitch leaves the current target's
            # own stability envelope.  It becomes a note boundary only if a
            # full stable-duration aperture confirms the new center.
            if abs(span[i]['st'] - center) <= stable_thr:
                # Keep the reference center robustly attached to the current
                # established target rather than following instantaneous F0.
                vals = [q['st'] for q in span[seg_start:i + 1]]
                center = float(np.median(vals))
                i += 1
                continue

            j = min(len(span), i + need_n)
            if j - i < need_n:
                break
            cand_vals = [q['st'] for q in span[i:j]]
            cand_center = float(np.median(cand_vals))
            cand_spread = float(np.max(cand_vals) - np.min(cand_vals)) if cand_vals else math.inf
            if abs(cand_center - center) > stable_thr and cand_spread <= stable_thr:
                out.append(ArticulationPoint(float(span[i]['t']), 'persistent_stable_target_change', 100.0))
                seg_start = i
                center = cand_center
                i = j
            else:
                i += 1

    return out


def renewed_acoustic_attacks(prefix: Path) -> list[ArticulationPoint]:
    """Detect renewed attacks inside continuous voicing from the existing
    amplitude-expression lane.

    This is deliberately relational, not threshold-tuned to any song.  A
    candidate is a local amplitude valley that:

      * remains inside one corrected-valid voiced run;
      * falls below its local amplitude baseline;
      * rebounds through/above that same baseline; and
      * rebounds more steeply than the immediately preceding decay.

    The articulation time is the first post-valley baseline crossing, i.e. the
    earliest point where the renewed voiced attack has re-established its local
    level.  Symmetric amplitude oscillation is therefore less likely to be
    mistaken for an attack than an asymmetric decay/re-attack envelope.
    """
    p = _sidecar(prefix, '.amplitude_expression.csv')
    if not p.exists():
        return []
    rows = _read_csv(p)
    if len(rows) < 5:
        return []

    t = np.asarray([_f(r, 'time_s') for r in rows], dtype=float)
    y = np.asarray([_f(r, 'shape_rms_dbfs') for r in rows], dtype=float)
    residual = np.asarray([_f(r, 'amplitude_residual_db') for r in rows], dtype=float)
    local_rms = np.asarray([_f(r, 'mod_depth_rms_db') for r in rows], dtype=float)
    valid = np.asarray([bool(_i(r, 'corrected_valid', 0)) for r in rows], dtype=bool)
    dt = _timeline_dt(prefix)
    look_s = _project_suppression_s(prefix)
    look_n = max(2, int(round(look_s / max(dt, 1e-9))))

    out: list[ArticulationPoint] = []
    for i in range(1, len(rows) - 1):
        if not (valid[i] and np.isfinite(y[i]) and np.isfinite(residual[i])):
            continue
        if not (y[i] <= y[i - 1] and y[i] < y[i + 1]):
            continue
        if residual[i] >= 0.0:
            continue
        # A renewed attack must be stronger than the ordinary local amplitude
        # modulation already measured by the amplitude-expression lane.  This
        # prevents shallow vibrato/phrasing valleys from becoming Note Ons.
        if math.isfinite(local_rms[i]) and abs(residual[i]) <= local_rms[i]:
            continue

        left0 = max(0, i - look_n)
        right1 = min(len(rows) - 1, i + look_n)

        # Stay inside the same continuous voiced run.
        li = i - 1
        while li > left0 and valid[li - 1]:
            li -= 1
        ri = i + 1
        while ri < right1 and valid[ri + 1]:
            ri += 1
        if li >= i or ri <= i:
            continue

        # Nearest pre-valley local high and first post-valley baseline crossing.
        pre_slice = np.arange(li, i, dtype=int)
        pre_slice = pre_slice[np.isfinite(y[pre_slice])]
        if len(pre_slice) == 0:
            continue
        pre = int(pre_slice[np.argmax(y[pre_slice])])

        crossing = None
        for j in range(i + 1, ri + 1):
            if not (valid[j] and np.isfinite(residual[j])):
                break
            if residual[j] >= 0.0:
                crossing = j
                break
        if crossing is None:
            continue

        fall_dt = float(t[i] - t[pre])
        rise_dt = float(t[crossing] - t[i])
        if fall_dt <= 0.0 or rise_dt <= 0.0:
            continue
        fall = float(y[pre] - y[i])
        rise = float(y[crossing] - y[i])
        if fall <= 0.0 or rise <= 0.0:
            continue
        decay_slope = fall / fall_dt
        attack_slope = rise / rise_dt
        if attack_slope <= decay_slope:
            continue

        out.append(ArticulationPoint(float(t[crossing]), 'renewed_acoustic_attack', 25.0))

    return out

def _amplitude_valleys(prefix: Path) -> list[float]:
    """Return envelope-valley times as supporting evidence only."""
    p = _sidecar(prefix, '.amplitude_expression.csv')
    if not p.exists():
        return []
    rows = _read_csv(p)
    t = np.asarray([_f(r, 'time_s') for r in rows], dtype=float)
    y = np.asarray([_f(r, 'shape_rms_dbfs') for r in rows], dtype=float)
    valid = np.asarray([bool(_i(r, 'corrected_valid', 0)) for r in rows], dtype=bool)
    out: list[float] = []
    for i in range(1, len(y) - 1):
        if not valid[i] or not np.isfinite(y[i]):
            continue
        if y[i] <= y[i - 1] and y[i] < y[i + 1]:
            out.append(float(t[i]))
    return out


def _near_any(t: float, xs: list[float], radius: float) -> bool:
    return any(abs(t - x) <= radius for x in xs)


def articulation_points(prefix: Path) -> list[ArticulationPoint]:
    """Build consolidated musical Note-On proposals from existing evidence.

    Hierarchy:
      1. corrected-valid interruption -> hard boundary;
      2. persistent stable-target-center change -> legato boundary;
      3. same-pitch/rearticulation boundary may be proposed only when *both*
         an ECKF reset and an amplitude valley support the same event;
      4. pitch motion alone, reset alone, or amplitude change alone never split.

    All proposals are clustered on the existing project stable-target timescale
    so multiple lanes describing one attack yield one MIDI Note On.
    """
    merge_s = _project_suppression_s(prefix)
    resets = _load_reset_landmarks(prefix)
    valleys = _amplitude_valleys(prefix)

    proposals: list[ArticulationPoint] = []

    # 1) Hard consonant/silence/nonmusical interruption: restart at right edge.
    for g in nonmusical_gaps(prefix):
        t = g.end_s
        nearby = [x for x in resets if abs(x - t) <= merge_s]
        if nearby:
            t = min(nearby, key=lambda x: abs(x - t))
            src = 'nonmusical_gap+eckf_reset'
        elif _near_any(t, valleys, merge_s):
            src = 'nonmusical_gap+amplitude_attack'
        else:
            src = 'nonmusical_gap'
        proposals.append(ArticulationPoint(t, src, 1000.0))

    # 2) Same-vowel legato: only a persistent new target center may split.
    proposals.extend(stable_target_articulations(prefix))

    # 2b) Same-pitch / near-same-pitch renewed attack inside continuous
    # voicing.  This uses the envelope's own local baseline and slope
    # asymmetry; pitch variation itself is not a splitter.
    proposals.extend(renewed_acoustic_attacks(prefix))

    # 3) Same-pitch / rearticulation support.  Neither lane is sufficient by
    # itself; require reset + amplitude valley to point to the same attack.
    # Use the reset timestamp because it is the sharper temporal landmark.
    for r in resets:
        if _near_any(r, valleys, merge_s):
            proposals.append(ArticulationPoint(float(r), 'eckf_reset+amplitude_rearticulation', 50.0))

    # One musical attack may generate several observations.  Cluster by the
    # project's existing stable-target timescale, keeping the strongest cue.
    chosen: list[ArticulationPoint] = []
    for p in sorted(proposals, key=lambda x: (-x.score, x.time_s)):
        if all(abs(p.time_s - q.time_s) >= merge_s for q in chosen):
            chosen.append(p)
    return sorted(chosen, key=lambda x: x.time_s)

def gaps_for_span(prefix: Path, start_s: float, end_s: float) -> list[NonMusicalGap]:
    return [g for g in nonmusical_gaps(prefix) if g.start_s < end_s and g.end_s > start_s]


def cuts_for_span(prefix: Path, start_s: float, end_s: float) -> list[ArticulationPoint]:
    return [p for p in articulation_points(prefix) if start_s < p.time_s < end_s]
