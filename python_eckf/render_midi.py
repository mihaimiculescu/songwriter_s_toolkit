from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mido
import numpy as np

from .midi_articulation import articulation_points, nonmusical_gaps
from .midi_event_builder import build_residual_valid_f0_events, valid_f0_islands, coverage_audit


# ---------------------------------------------------------------------------
# MIDI output contract
# ---------------------------------------------------------------------------
PPQ = 960
MELODY_CHANNEL = 0              # MIDI channel 1 in DAW/user-facing numbering.
GM_PROGRAM_NUMBER = 65          # General MIDI #65 = Soprano Saxophone.
MIDO_PROGRAM = GM_PROGRAM_NUMBER - 1  # Program Change payloads are zero-based.
DEFAULT_VELOCITY = 100  # fixed project velocity; amplitude dynamics live on CC11.
CC7_VOLUME = 7
CC7_NEUTRAL = 127
CC11_EXPRESSION = 11
CC11_NEUTRAL = 127
CC11_DB_COMPRESSION = 0.3 #otherwise the dynamic excursion is too wide

# Pitch bend is channel-wide in MIDI 1.0.  This renderer is deliberately
# monophonic.  Every bend-bearing note explicitly sets its own RPN 0,0 range
# and resets Pitch Bend to center *after* Note Off.
PB_CENTER = 0
PB_MIN = -8192
PB_MAX = 8191


@dataclass
class NoteEvent:
    start_s: float
    end_s: float
    base_st: float
    base_midi: int
    source_kind: str
    source_id: str
    pitch_times_s: np.ndarray
    pitch_st: np.ndarray
    velocity: int = DEFAULT_VELOCITY
    attack_dbfs: float = math.nan

    bend_range_st: int = 0
    max_abs_bend_st: float = 0.0


@dataclass
class ExpressionBurst:
    start_s: float
    end_s: float
    times_s: np.ndarray
    residual_db: np.ndarray


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(row: dict[str, str], key: str, default: int = -1) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _nearest_midi(st: float) -> int:
    return int(np.clip(int(round(float(st))), 0, 127))


def _seconds_to_tick(seconds: float, bpm: float, ppq: int = PPQ) -> int:
    beats = float(seconds) * float(bpm) / 60.0
    return int(round(beats * ppq))


def _parse_time_signature(text: str) -> tuple[int, int]:
    try:
        num_s, den_s = text.split("/", 1)
        num, den = int(num_s), int(den_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError("time signature must look like 4/4") from exc
    if num <= 0 or den <= 0 or den & (den - 1):
        raise argparse.ArgumentTypeError("time signature denominator must be a power of two")
    return num, den


def _csv_family(prefix: Path, suffix: str) -> Path:
    # CLI output prefix is usually .../Song.csv and sidecars are
    # Song.csv.trajectory.csv, Song.csv.expressive.csv, etc.
    p = Path(str(prefix) + suffix)
    if not p.exists():
        raise FileNotFoundError(f"required sidecar not found: {p}")
    return p


def _load_expressive(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(_csv_family(prefix, ".expressive.csv"))
    t = np.asarray([_f(r, "time_s") for r in rows], dtype=np.float64)
    shape = np.asarray([_f(r, "shape_pitch_st") for r in rows], dtype=np.float64)
    valid = np.asarray([bool(_i(r, "corrected_valid", 0)) for r in rows], dtype=bool)
    valid &= np.isfinite(shape)
    return t, shape, valid


def _curve_slice(
    times: np.ndarray,
    shape_st: np.ndarray,
    valid: np.ndarray,
    start_s: float,
    end_s: float,
    fallback_st: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (times >= start_s - 1e-9) & (times <= end_s + 1e-9) & valid
    tt = times[mask]
    pp = shape_st[mask]

    if len(tt) == 0:
        return (
            np.asarray([start_s, end_s], dtype=np.float64),
            np.asarray([fallback_st, fallback_st], dtype=np.float64),
        )

    # Make sure exact event boundaries exist.  We do not extrapolate from a
    # different validity island; boundary values use the nearest trusted point
    # already inside this musical event.
    if tt[0] > start_s + 1e-9:
        tt = np.insert(tt, 0, start_s)
        pp = np.insert(pp, 0, pp[0])
    if tt[-1] < end_s - 1e-9:
        tt = np.append(tt, end_s)
        pp = np.append(pp, pp[-1])

    return tt, pp


def _load_amplitude_lane(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(_csv_family(prefix, ".amplitude_expression.csv"))
    t = np.asarray([_f(r, "time_s") for r in rows], dtype=np.float64)
    shape = np.asarray([_f(r, "shape_rms_dbfs") for r in rows], dtype=np.float64)
    residual = np.asarray([_f(r, "amplitude_residual_db") for r in rows], dtype=np.float64)
    valid = np.asarray([bool(_i(r, "corrected_valid", 0)) for r in rows], dtype=bool)
    valid &= np.isfinite(shape) & np.isfinite(residual)
    return t, shape, residual, valid


def _project_timescale_s(prefix: Path) -> float:
    p = Path(str(prefix) + ".target_formation.csv")
    if not p.exists():
        return 0.08
    vals = [_f(r, "min_stable_threshold_ms") for r in _read_csv(p)]
    vals = [x for x in vals if math.isfinite(x) and x > 0]
    return float(np.median(vals)) / 1000.0 if vals else 0.08


def _timeline_dt_from(times: np.ndarray) -> float:
    t = times[np.isfinite(times)]
    if len(t) < 2:
        return 0.01
    return float(np.median(np.diff(t)))


def _assign_note_expression_baselines(prefix: Path, notes: list[NoteEvent], velocity_override: int | None) -> dict[int, int]:
    """Assign fixed Note On velocity and compute per-note CC11 baselines.

    Velocity is deliberately fixed (100 by default).  Measured attack loudness is
    encoded on CC11 instead.  The loudest rendered attack maps to CC11=127;
    quieter attacks preserve their linear RMS ratio via 10**(dB/20).
    """
    if not notes:
        return {}

    fixed_velocity = DEFAULT_VELOCITY if velocity_override is None else int(np.clip(velocity_override, 1, 127))
    t, shape, _residual, valid = _load_amplitude_lane(prefix)
    attack_window = _project_timescale_s(prefix)
    levels: list[float] = []

    for n in notes:
        n.velocity = fixed_velocity
        right = min(n.end_s, n.start_s + attack_window)
        mask = (t >= n.start_s - 1e-9) & (t <= right + 1e-9) & valid
        vals = shape[mask]
        if len(vals):
            level = float(np.max(vals))
        else:
            finite = np.flatnonzero(np.isfinite(shape))
            if len(finite):
                idx = int(finite[np.argmin(np.abs(t[finite] - n.start_s))])
                level = float(shape[idx])
            else:
                level = math.nan
        n.attack_dbfs = level
        levels.append(level)

    finite_levels = [x for x in levels if math.isfinite(x)]
    peak_db = max(finite_levels) if finite_levels else math.nan
    baselines: dict[int, int] = {}
    for idx, n in enumerate(notes):
        if not math.isfinite(n.attack_dbfs) or not math.isfinite(peak_db):
            baselines[idx] = CC11_NEUTRAL
            continue
        ratio = 10.0 ** ((CC11_DB_COMPRESSION * (n.attack_dbfs - peak_db)) / 20.0)
        baselines[idx] = int(np.clip(round(CC11_NEUTRAL * ratio), 1, 127))
    return baselines


def _expression_bursts(prefix: Path, notes: list[NoteEvent]) -> list[ExpressionBurst]:
    """Identify amplitude-modulated continuous musical spans.

    Admission is unchanged from v8: a contiguous note group must show repeated
    amplitude-residual baseline crossings (at least four).  The burst stores the
    measured residual in dB; MIDI projection happens later relative to each
    serviced note's CC11 baseline.  No CC11 reset is generated at span end.
    """
    if not notes:
        return []
    t, _shape, residual, valid = _load_amplitude_lane(prefix)
    dt = _timeline_dt_from(t)

    groups: list[list[NoteEvent]] = []
    cur: list[NoteEvent] = [notes[0]]
    for n in notes[1:]:
        if n.start_s - cur[-1].end_s <= 1.5 * dt:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
    groups.append(cur)

    out: list[ExpressionBurst] = []
    for group in groups:
        a, b = group[0].start_s, group[-1].end_s
        idx = np.flatnonzero((t >= a - 1e-9) & (t <= b + 1e-9) & valid)
        if len(idx) < 5:
            continue
        rr = residual[idx]
        signs = np.sign(rr)
        crossings = 0
        last_sign = 0.0
        for sg in signs:
            if sg == 0:
                continue
            if last_sign != 0 and sg != last_sign:
                crossings += 1
            last_sign = sg
        if crossings < 4:
            continue
        finite = np.isfinite(rr)
        if np.count_nonzero(finite) < 2:
            continue
        out.append(ExpressionBurst(float(a), float(b), t[idx][finite].copy(), rr[finite].copy()))
    return out


def _structural_rows(prefix: Path) -> list[dict[str, str]]:
    path = Path(str(prefix) + ".structural_collaboration.csv")
    if not path.exists():
        return []
    return _read_csv(path)


def _split_note_event(
    n: NoteEvent,
    cuts: list[tuple[float, str]],
    exp_t: np.ndarray,
    exp_shape: np.ndarray,
    exp_valid: np.ndarray,
) -> list[NoteEvent]:
    boundaries = [n.start_s] + [t for t, _ in cuts if n.start_s < t < n.end_s] + [n.end_s]
    boundaries = sorted(set(round(float(x), 9) for x in boundaries))
    out: list[NoteEvent] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a + 1e-6:
            continue
        mask = (exp_t >= a - 1e-9) & (exp_t <= b + 1e-9) & exp_valid
        vals = exp_shape[mask]
        st = float(np.median(vals)) if len(vals) else float(n.base_st)
        base_midi = _nearest_midi(st)
        tt, pp = _curve_slice(exp_t, exp_shape, exp_valid, a, b, st)
        out.append(NoteEvent(
            start_s=a, end_s=b, base_st=st, base_midi=base_midi,
            source_kind=n.source_kind + '+articulated',
            source_id=n.source_id, pitch_times_s=tt, pitch_st=pp,
            velocity=n.velocity,
        ))
    return out or [n]


def build_note_events(prefix: Path) -> list[NoteEvent]:
    """Convert finished interpretation into monophonic articulated MIDI notes.

    Structural regions remain the pitch/gesture authority, but they are no
    longer assumed to equal one MIDI Note On.  V3 uses a semantic articulation
    hierarchy: actual nonmusical interruptions are hard splitters; a newly
    established stable target creates a legato Note On; ECKF reset/amplitude
    cues only support those events.  Pitch motion alone never splits a note.
    Continuous TRANSITION regions remain pitch-bend gestures.
    """
    trajectory = _read_csv(_csv_family(prefix, '.trajectory.csv'))
    exp_t, exp_shape, exp_valid = _load_expressive(prefix)
    structural = _structural_rows(prefix)
    all_articulations = articulation_points(prefix)

    notes: list[NoteEvent] = []

    # 1) Primary stable targets.
    for r in trajectory:
        if r.get('kind') != 'STABLE_TARGET':
            continue
        start_s = _f(r, 'start_time_s')
        end_s = _f(r, 'end_time_s')
        st = _f(r, 'median_pitch_st')
        if not (math.isfinite(start_s) and math.isfinite(end_s) and math.isfinite(st)) or end_s <= start_s:
            continue
        base_midi = _nearest_midi(st)
        tt, pp = _curve_slice(exp_t, exp_shape, exp_valid, start_s, end_s, st)
        notes.append(NoteEvent(start_s, end_s, st, base_midi, 'stable_target',
                               r.get('region_index', ''), tt, pp))

    # 2) Explicit short-target candidates not covered by a stable target.
    for r in structural:
        action = (r.get('action') or '').strip()
        if action not in {'short_target', 'short_target_candidate'}:
            continue
        start_s = _f(r, 'trusted_run_start_s', _f(r, 'frame_start_s'))
        end_s = _f(r, 'trusted_run_end_s', _f(r, 'frame_end_s'))
        if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s <= start_s:
            continue
        if any(not (end_s <= n.start_s or start_s >= n.end_s) for n in notes):
            continue
        mask = (exp_t >= start_s - 1e-9) & (exp_t <= end_s + 1e-9) & exp_valid
        vals = exp_shape[mask]
        if len(vals) == 0:
            continue
        st = float(np.median(vals))
        tt, pp = _curve_slice(exp_t, exp_shape, exp_valid, start_s, end_s, st)
        notes.append(NoteEvent(start_s, end_s, st, _nearest_midi(st), 'short_target',
                               r.get('frame_index', ''), tt, pp))

    notes.sort(key=lambda n: (n.start_s, n.end_s))

    # 2b) Ownership-complete recovery.  Structural target formation is
    # deliberately conservative, but MIDI generation must not silently drop
    # corrected-valid F0 just because its trajectory region remained
    # UNRESOLVED.  Recover only the valid-F0 sub-spans not already owned by a
    # stable/short target; later gesture, boundary, articulation and gap logic
    # remains authoritative.
    # First let each already-owned corrected-valid island extend its
    # structural note(s) to the island edges.  These are note fringes, not new
    # articulations.  Internal articulation is still decided later.
    for isl in valid_f0_islands(prefix):
        owners = [n for n in notes if n.start_s < isl.end_s - 1e-12 and n.end_s > isl.start_s + 1e-12]
        if owners:
            owners.sort(key=lambda n: (n.start_s, n.end_s))
            owners[0].start_s = min(owners[0].start_s, isl.start_s)
            owners[-1].end_s = max(owners[-1].end_s, isl.end_s)

    occupied = [(n.start_s, n.end_s) for n in notes]
    for spec in build_residual_valid_f0_events(prefix, occupied):
        tt, pp = _curve_slice(
            exp_t, exp_shape, exp_valid,
            spec.start_s, spec.end_s, spec.median_st,
        )
        notes.append(NoteEvent(
            spec.start_s, spec.end_s, spec.median_st,
            _nearest_midi(spec.median_st),
            'residual_valid_f0', spec.source_id, tt, pp,
        ))

    notes.sort(key=lambda n: (n.start_s, n.end_s))
    if not notes:
        return []

    # 3) Existing V4 boundary ownership.
    for r in structural:
        action = (r.get('action') or '').strip()
        run_start = _f(r, 'trusted_run_start_s')
        run_end = _f(r, 'trusted_run_end_s')
        if action == 'right_attack' and math.isfinite(run_start):
            future = [n for n in notes if n.start_s >= run_start - 0.25]
            if future:
                n = min(future, key=lambda x: abs(x.start_s - run_start))
                n.start_s = min(n.start_s, run_start)
        elif action == 'left_release' and math.isfinite(run_end):
            past = [n for n in notes if n.end_s <= run_end + 0.25]
            if past:
                n = min(past, key=lambda x: abs(x.end_s - run_end))
                n.end_s = max(n.end_s, run_end)

    notes.sort(key=lambda n: (n.start_s, n.end_s))

    # 4) Transition ownership stays continuous and belongs to the left note.
    transition_ranges = [(_f(r, 'start_time_s'), _f(r, 'end_time_s'))
                         for r in trajectory if r.get('kind') == 'TRANSITION']
    for i in range(len(notes) - 1):
        left, right = notes[i], notes[i + 1]
        if right.start_s <= left.end_s:
            left.end_s = min(left.end_s, right.start_s)
            continue
        gap_a, gap_b = left.end_s, right.start_s
        covered = any(a <= gap_a + 0.011 and b >= gap_b - 0.011 for a, b in transition_ranges)
        if covered:
            left.end_s = right.start_s

    # 5) V3 articulation hierarchy.
    # First split at semantic musical attacks.  Pure pitch motion does not cut.
    articulated: list[NoteEvent] = []
    for n in notes:
        cuts: list[tuple[float, str]] = []
        for a in all_articulations:
            if not (n.start_s < a.time_s < n.end_s):
                continue
            # Stable-target articulation inside a frozen transition is allowed
            # only if target formation itself established the new center.  A
            # gap/restart is also allowed.  Other transition motion stays bend.
            cuts.append((a.time_s, a.source))
        articulated.extend(_split_note_event(n, cuts, exp_t, exp_shape, exp_valid))
    notes = sorted(articulated, key=lambda n: (n.start_s, n.end_s))

    # Carve true nonmusical interruptions out of note spans.  This prevents a
    # consonant/breath gap from being rendered as pitched saxophone.
    gaps = nonmusical_gaps(prefix)
    carved: list[NoteEvent] = []
    for n in notes:
        pieces = [(n.start_s, n.end_s)]
        for g in gaps:
            next_pieces = []
            for a, b in pieces:
                if g.end_s <= a or g.start_s >= b:
                    next_pieces.append((a, b))
                    continue
                if g.start_s > a + 1e-6:
                    next_pieces.append((a, min(g.start_s, b)))
                if g.end_s < b - 1e-6:
                    next_pieces.append((max(g.end_s, a), b))
            pieces = next_pieces
        for a, b in pieces:
            if b <= a + 1e-6:
                continue
            mask = (exp_t >= a - 1e-9) & (exp_t <= b + 1e-9) & exp_valid
            vals = exp_shape[mask]
            st = float(np.median(vals)) if len(vals) else float(n.base_st)
            tt, pp = _curve_slice(exp_t, exp_shape, exp_valid, a, b, st)
            carved.append(NoteEvent(
                start_s=a, end_s=b, base_st=st, base_midi=_nearest_midi(st),
                source_kind=n.source_kind + '+gap_carved', source_id=n.source_id,
                pitch_times_s=tt, pitch_st=pp, velocity=n.velocity,
            ))
    notes = sorted(carved, key=lambda n: (n.start_s, n.end_s))

    # 6) Enforce monophony and build bends independently for each articulation.
    cleaned: list[NoteEvent] = []
    for i, n in enumerate(notes):
        if i + 1 < len(notes):
            n.end_s = min(n.end_s, notes[i + 1].start_s)
        if n.end_s <= n.start_s + 1e-6:
            continue
        tt, pp = _curve_slice(exp_t, exp_shape, exp_valid, n.start_s, n.end_s, n.base_st)
        n.pitch_times_s = tt
        n.pitch_st = pp
        deviations = pp - float(n.base_midi)
        n.max_abs_bend_st = float(np.nanmax(np.abs(deviations))) if len(deviations) else 0.0
        n.bend_range_st = max(1, int(math.ceil(n.max_abs_bend_st - 1e-12))) if n.max_abs_bend_st > 1e-4 else 0
        cleaned.append(n)
    return cleaned


def _rpn_pitch_bend_range(channel: int, semitones: int) -> list[mido.Message]:
    """RPN 0,0 Pitch Bend Sensitivity, followed by RPN Null."""
    semitones = int(np.clip(semitones, 0, 127))
    return [
        mido.Message("control_change", channel=channel, control=101, value=0, time=0),
        mido.Message("control_change", channel=channel, control=100, value=0, time=0),
        mido.Message("control_change", channel=channel, control=6, value=semitones, time=0),
        mido.Message("control_change", channel=channel, control=38, value=0, time=0),
        mido.Message("control_change", channel=channel, control=101, value=127, time=0),
        mido.Message("control_change", channel=channel, control=100, value=127, time=0),
    ]


def _bend_value(pitch_st: float, base_midi: int, range_st: int) -> int:
    if range_st <= 0:
        return 0
    frac = (float(pitch_st) - float(base_midi)) / float(range_st)
    value = int(round(frac * 8192.0))
    return int(np.clip(value, PB_MIN, PB_MAX))


def render_midi(
    prefix: Path,
    output_mid: Path,
    *,
    bpm: float,
    time_signature: tuple[int, int] = (4, 4),
    velocity: int | None = None,
    channel: int = MELODY_CHANNEL,
) -> list[NoteEvent]:
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if not 0 <= channel <= 15 or channel == 9:
        raise ValueError("melody channel must be 0..15 and must not be percussion channel 9")

    notes = build_note_events(prefix)
    cc11_baselines = _assign_note_expression_baselines(prefix, notes, velocity)
    expression_bursts = _expression_bursts(prefix, notes)

    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)

    # Track 0: global/meta only.
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0))
    num, den = time_signature
    meta.append(mido.MetaMessage(
        "time_signature", numerator=num, denominator=den,
        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
    ))
    meta.append(mido.MetaMessage("end_of_track", time=0))

    # Track 1: melody only.
    melody = mido.MidiTrack()
    midi.tracks.append(melody)
    melody.append(mido.MetaMessage("track_name", name="ECKF Melody - GM 65 Soprano Sax", time=0))
    melody.append(mido.Message("program_change", channel=channel, program=MIDO_PROGRAM, time=0))

    # Deterministic channel state at file start, irrespective of whether later
    # notes actually use these controllers.
    melody.append(mido.Message("pitchwheel", channel=channel, pitch=PB_CENTER, time=0))
    melody.append(mido.Message("control_change", channel=channel, control=CC7_VOLUME,
                               value=CC7_NEUTRAL, time=0))
    melody.append(mido.Message("control_change", channel=channel, control=CC11_EXPRESSION,
                               value=CC11_NEUTRAL, time=0))

    # Collect absolute-tick events with deterministic same-tick ordering.
    # Priority matters: Note Off -> Bend reset -> RPN/range -> center -> Note On -> bends.
    events: list[tuple[int, int, mido.Message]] = []

    # CC11 is stateful.  It is initialized once at file start, then every
    # Note On establishes that note's measured amplitude baseline.  Expression
    # modulation is differential around that baseline.  There are NO automatic
    # CC11 resets at Note Off or at modulation-span end.
    #
    # Build per-note modulation samples from admitted residual oscillation spans.
    modulation_by_note: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(notes))}
    for burst in expression_bursts:
        for i, n in enumerate(notes):
            if n.end_s <= burst.start_s or n.start_s >= burst.end_s:
                continue
            mask = (burst.times_s >= n.start_s - 1e-9) & (burst.times_s <= n.end_s + 1e-9)
            tt = burst.times_s[mask]
            rr = burst.residual_db[mask]
            if len(tt) < 2:
                continue
            # Differential modulation: the first valid residual in this note is
            # the zero-reference.  Positive/negative residual changes scale the
            # note's CC11 baseline multiplicatively in linear amplitude space.
            ref = float(rr[0])
            modulation_by_note[i].extend((float(t_s), float(r_db - ref)) for t_s, r_db in zip(tt, rr))

    for note_index, n in enumerate(notes):
        start_tick = _seconds_to_tick(n.start_s, bpm)
        end_tick = max(start_tick + 1, _seconds_to_tick(n.end_s, bpm))

        has_bend = n.bend_range_st > 0
        if has_bend:
            pr = 20
            for msg in _rpn_pitch_bend_range(channel, n.bend_range_st):
                events.append((start_tick, pr, msg))
                pr += 1
            events.append((start_tick, 30, mido.Message("pitchwheel", channel=channel, pitch=0, time=0)))

        # At every Note On, establish the new note's absolute expression baseline
        # before sounding the note.  This replaces dynamic velocity; Note On
        # velocity itself stays fixed at 100 (or explicit CLI override).
        baseline_cc11 = int(cc11_baselines.get(note_index, CC11_NEUTRAL))
        events.append((start_tick, 35, mido.Message(
            "control_change", channel=channel, control=CC11_EXPRESSION,
            value=baseline_cc11, time=0,
        )))

        events.append((start_tick, 40, mido.Message(
            "note_on", channel=channel, note=n.base_midi, velocity=n.velocity, time=0,
        )))

        # Differential CC11 modulation around this note's assigned baseline.
        last_cc11 = baseline_cc11
        for t_s, delta_db in modulation_by_note.get(note_index, []):
            tick = int(np.clip(_seconds_to_tick(float(t_s), bpm), start_tick, end_tick - 1))
            ratio = 10.0 ** ((CC11_DB_COMPRESSION * float(delta_db)) / 20.0)
            value = int(np.clip(round(baseline_cc11 * ratio), 1, 127))
            if value == last_cc11:
                continue
            events.append((tick, 55, mido.Message(
                "control_change", channel=channel, control=CC11_EXPRESSION,
                value=value, time=0,
            )))
            last_cc11 = value

        if has_bend:
            last_value = None
            for t_s, p_st in zip(n.pitch_times_s, n.pitch_st):
                tick = int(np.clip(_seconds_to_tick(float(t_s), bpm), start_tick, end_tick - 1))
                value = _bend_value(float(p_st), n.base_midi, n.bend_range_st)
                if last_value is not None and value == last_value:
                    continue
                events.append((tick, 50, mido.Message("pitchwheel", channel=channel, pitch=value, time=0)))
                last_value = value

        events.append((end_tick, 0, mido.Message(
            "note_off", channel=channel, note=n.base_midi, velocity=0, time=0,
        )))

        # HARD INVARIANT requested by the project: every pitch-bend burst ends
        # with PB=0, and that reset occurs AFTER Note Off of the serviced note.
        if has_bend:
            events.append((end_tick, 10, mido.Message("pitchwheel", channel=channel, pitch=0, time=0)))

    events.sort(key=lambda x: (x[0], x[1]))
    prev_tick = 0
    for tick, _, msg in events:
        delta = max(0, tick - prev_tick)
        msg.time = delta
        melody.append(msg)
        prev_tick = tick
    melody.append(mido.MetaMessage("end_of_track", time=0))

    output_mid.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_mid)

    audit = output_mid.with_suffix(output_mid.suffix + ".events.csv")
    with audit.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "event_index", "start_s", "end_s", "duration_s",
            "base_midi", "base_st", "velocity", "attack_dbfs", "cc11_baseline", "source_kind", "source_id",
            "gm_program_number", "midi_program_payload", "channel_zero_based",
            "bend_range_semitones", "max_abs_bend_semitones",
            "pitch_point_count",
        ])
        for idx, n in enumerate(notes):
            w.writerow([
                idx, n.start_s, n.end_s, n.end_s - n.start_s,
                n.base_midi, n.base_st, n.velocity, n.attack_dbfs, cc11_baselines.get(idx, CC11_NEUTRAL), n.source_kind, n.source_id,
                GM_PROGRAM_NUMBER, MIDO_PROGRAM, channel,
                n.bend_range_st, n.max_abs_bend_st, len(n.pitch_times_s),
            ])

    expression_audit = output_mid.with_suffix(output_mid.suffix + ".expression.csv")
    with expression_audit.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["burst_index", "start_s", "end_s", "duration_s",
                    "point_count", "start_residual_db", "end_residual_db",
                    "min_residual_db", "max_residual_db"])
        for idx, b in enumerate(expression_bursts):
            w.writerow([idx, b.start_s, b.end_s, b.end_s - b.start_s,
                        len(b.times_s), float(b.residual_db[0]), float(b.residual_db[-1]),
                        float(np.min(b.residual_db)), float(np.max(b.residual_db))])

    # Ownership audit: corrected-valid F0 should never silently disappear.
    coverage_rows, uncovered_s = coverage_audit(
        prefix, [(n.start_s, n.end_s) for n in notes],
    )
    coverage_path = output_mid.with_suffix(output_mid.suffix + '.coverage.csv')
    with coverage_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['trajectory_index', 'time_s', 'end_s', 'structural_pitch_st', 'status'],
        )
        w.writeheader()
        w.writerows(coverage_rows)

    if uncovered_s > 1e-9:
        print(f'WARNING: UNOWNED_VALID_F0={uncovered_s:.6f} s; see {coverage_path}')

    return notes


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Render finished python_eckf interpretation sidecars to MIDI 1.0 Type 1 melody.",
    )
    ap.add_argument("prefix", type=Path,
                    help="main ECKF CSV prefix, e.g. tests/Song_v2_structural_collab_v4.csv")
    ap.add_argument("--out", type=Path, required=True, help="output .mid")
    ap.add_argument("--bpm", type=float, required=True,
                    help="song BPM; required so MIDI bars/beats remain compatible downstream")
    ap.add_argument("--time-signature", default="4/4", type=_parse_time_signature)
    ap.add_argument("--velocity", type=int, default=None,
                    help="optional fixed velocity override; default is fixed velocity 100; amplitude dynamics are encoded on CC11")
    ap.add_argument("--channel", type=int, default=MELODY_CHANNEL,
                    help="zero-based MIDI channel; default 0 (DAW channel 1); channel 9 forbidden")
    args = ap.parse_args(list(argv) if argv is not None else None)

    notes = render_midi(
        args.prefix,
        args.out,
        bpm=args.bpm,
        time_signature=args.time_signature,
        velocity=args.velocity,
        channel=args.channel,
    )
    print(f"MIDI:  {args.out}")
    print(f"Audit: {args.out.with_suffix(args.out.suffix + '.events.csv')}")
    print(f"Coverage: {args.out.with_suffix(args.out.suffix + '.coverage.csv')}")
    print(f"Expression: {args.out.with_suffix(args.out.suffix + '.expression.csv')}")
    print(f"Notes: {len(notes)}")
    print(f"Format: MIDI 1.0 SMF Type 1, PPQ={PPQ}")
    print(f"Melody: Track 1, channel {args.channel} (DAW channel {args.channel + 1})")
    print(f"GM program: #{GM_PROGRAM_NUMBER} Soprano Saxophone (Program Change payload {MIDO_PROGRAM})")


if __name__ == "__main__":
    main()
