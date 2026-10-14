from __future__ import annotations

"""MIDI Renderer V17.1 -- residual ownership without artificial reattack.

V17.1 is intentionally a very small repair layer on top of V16.1.  V16.1 remains
responsible for articulation, continuous-motion handling, owner-corridor
adjudication, pitch-only handoffs, and bounded V16-created corridor repair.

V17.1 keeps V17's corrected-valid ownership closure, but removes one topology
artifact: a residual run must not create a new Note-On when it simply continues
an immediately preceding same-family owner and no independent physical reattack
is already accepted near the residual start.  No WAV-only pitch inference,
primary-family rescue, subharmonic adjudication, or transient-note cleanup is
introduced here.
"""

from dataclasses import dataclass
import math

import numpy as np

from .midi_event_assembler_v16_1 import (
    AssemblyResult,
    MusicalEvent,
    AttackDecision,
    assemble_events as assemble_events_v16_1,
)


# Reuse the upstream target-former's existing minimum stable-duration scale.
# This is a multiplier, not a new MIDI-specific note-duration threshold.
RESIDUAL_VALID_MIN_STABLE_FACTOR = 1.0
# An already-accepted articulation may own the residual pitch before the first
# trustworthy corrected-valid point, but only when that decision explicitly
# anchors its pitch inside the residual run and does not overlap the preceding
# event.  This preserves the clean-sheet attack/pitch-authority separation.
RESIDUAL_ATTACK_ANCHOR_TOLERANCE_S = 0.010

# V17.1 same-family ownership handoff.  These are deliberately conservative:
# this repair is not a general event-merging pass.
RESIDUAL_JOIN_MAX_GAP_S = 0.060
RESIDUAL_JOIN_MAX_TAIL_GAP_ST = 0.75
RESIDUAL_REATTACK_PROOF_WINDOW_S = 0.020


@dataclass(frozen=True)
class ResidualValidRun:
    first_index: int
    last_index: int
    start_s: float
    end_s: float
    duration_s: float
    median_st: float
    point_count: int


def _nearest_midi(st: float) -> int:
    return int(np.clip(int(round(float(st))), 0, 127))


def _is_point_owned(t: float, events: list[MusicalEvent]) -> bool:
    return any(e.start_s - 1e-9 <= t < e.end_s + 1e-9 for e in events)


def _residual_valid_runs(result: AssemblyResult) -> list[ResidualValidRun]:
    """Find final corrected-valid V3R9 interpretation points with no MIDI owner."""
    pitch = result.pitch
    unowned = np.asarray([
        bool(v) and not _is_point_owned(float(t), result.events)
        for t, v in zip(pitch.t, pitch.valid)
    ], dtype=bool)

    out: list[ResidualValidRun] = []
    i = 0
    n = len(unowned)
    while i < n:
        if not unowned[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and unowned[j + 1]
            and pitch.valid[j + 1]
            and pitch.t[j + 1] - pitch.t[j] <= 1.6 * pitch.dt
        ):
            j += 1
        start = float(pitch.t[i])
        end = float(pitch.t[j] + pitch.dt)
        vals = pitch.shape[i:j + 1]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out.append(ResidualValidRun(
                i, j, start, end, max(0.0, end - start),
                float(np.median(vals)), int(j - i + 1),
            ))
        i = j + 1
    return out


def _previous_event(events: list[MusicalEvent], t: float) -> MusicalEvent | None:
    candidates = [e for e in events if e.end_s <= t + 1e-9]
    return max(candidates, key=lambda e: e.end_s) if candidates else None


def _next_event(events: list[MusicalEvent], t: float) -> MusicalEvent | None:
    candidates = [e for e in events if e.start_s >= t - 1e-9]
    return min(candidates, key=lambda e: e.start_s) if candidates else None


def _accepted_existing_attack_start(
    result: AssemblyResult,
    run: ResidualValidRun,
    previous: MusicalEvent | None,
) -> tuple[float, AttackDecision | None]:
    """Reuse an already-accepted attack only; V17 performs no new attack search."""
    floor = previous.end_s if previous is not None else -math.inf
    candidates: list[AttackDecision] = []
    for d in result.attack_decisions:
        if not d.accepted:
            continue
        if d.time_s < floor - 1e-9 or d.time_s > run.start_s + 1e-9:
            continue
        if not math.isfinite(d.pitch_anchor_s):
            continue
        if not (
            run.start_s - RESIDUAL_ATTACK_ANCHOR_TOLERANCE_S
            <= d.pitch_anchor_s
            <= run.end_s + RESIDUAL_ATTACK_ANCHOR_TOLERANCE_S
        ):
            continue
        candidates.append(d)
    if not candidates:
        return run.start_s, None
    # The earliest already-proven attack after the preceding event is the
    # physical articulation owner; trustworthy F0 may arrive later.
    chosen = min(candidates, key=lambda d: d.time_s)
    return min(run.start_s, float(chosen.time_s)), chosen


def _tail_pitch_st(event: MusicalEvent) -> float:
    vals = np.asarray(event.pitch_st, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        return float(vals[-1])
    return float(event.base_st)


def _accepted_physical_reattack_near(result: AssemblyResult, t: float) -> AttackDecision | None:
    """Return a pre-existing accepted attack close enough to prove rearticulation.

    Synthetic V17/V17.1 coverage bookkeeping is intentionally ignored.
    """
    candidates: list[AttackDecision] = []
    for d in result.attack_decisions:
        if not d.accepted:
            continue
        if abs(float(d.time_s) - float(t)) > RESIDUAL_REATTACK_PROOF_WINDOW_S + 1e-12:
            continue
        if d.source.startswith("residual_valid_f0") or d.source.startswith("v17"):
            continue
        candidates.append(d)
    if not candidates:
        return None
    return min(candidates, key=lambda d: abs(float(d.time_s) - float(t)))


def _same_family_predecessor_for_residual(
    result: AssemblyResult,
    run: ResidualValidRun,
    previous: MusicalEvent | None,
) -> tuple[MusicalEvent | None, AttackDecision | None]:
    if previous is None:
        return None, None
    gap = float(run.start_s - previous.end_s)
    if gap < -1e-9 or gap > RESIDUAL_JOIN_MAX_GAP_S + 1e-12:
        return None, None
    if int(previous.base_midi) != _nearest_midi(run.median_st):
        return None, None
    if abs(_tail_pitch_st(previous) - float(run.median_st)) > RESIDUAL_JOIN_MAX_TAIL_GAP_ST:
        return None, None
    reattack = _accepted_physical_reattack_near(result, run.start_s)
    if reattack is not None:
        return None, reattack
    return previous, None


def _extend_existing_owner_with_residual(
    event: MusicalEvent,
    result: AssemblyResult,
    run: ResidualValidRun,
    end_s: float,
) -> None:
    pitch = result.pitch
    sl = slice(run.first_index, run.last_index + 1)
    tt = np.asarray(pitch.t[sl], dtype=float)
    pp = np.asarray(pitch.shape[sl], dtype=float)
    finite = np.isfinite(tt) & np.isfinite(pp)
    tt = tt[finite]
    pp = pp[finite]
    if not len(tt):
        return

    old_t = np.asarray(event.pitch_times_s, dtype=float)
    old_p = np.asarray(event.pitch_st, dtype=float)
    keep = np.isfinite(old_t) & np.isfinite(old_p) & (old_t <= event.end_s + 1e-9)
    old_t = old_t[keep]
    old_p = old_p[keep]

    # Keep the preceding owner's final bend through any tiny evidence gap, then
    # hand pitch authority to the first corrected-valid residual sample.
    all_t = np.concatenate([old_t, tt])
    all_p = np.concatenate([old_p, pp])
    order = np.argsort(all_t, kind="stable")
    all_t = all_t[order]
    all_p = all_p[order]
    if len(all_t):
        uniq = np.ones(len(all_t), dtype=bool)
        uniq[1:] = np.diff(all_t) > 1e-9
        all_t = all_t[uniq]
        all_p = all_p[uniq]
    if end_s > all_t[-1] + 1e-9:
        all_t = np.append(all_t, float(end_s))
        all_p = np.append(all_p, float(all_p[-1]))

    event.end_s = float(end_s)
    event.pitch_times_s = all_t
    event.pitch_st = all_p
    deviations = event.pitch_st - float(event.base_midi)
    event.max_abs_bend_st = float(np.nanmax(np.abs(deviations))) if len(deviations) else 0.0
    event.bend_range_st = (
        max(1, int(math.ceil(event.max_abs_bend_st - 1e-12)))
        if event.max_abs_bend_st > 1e-4 else 0
    )


def _make_residual_event(
    result: AssemblyResult,
    run: ResidualValidRun,
    start_s: float,
    end_s: float,
    attack: AttackDecision | None,
    serial: int,
) -> MusicalEvent:
    pitch = result.pitch
    sl = slice(run.first_index, run.last_index + 1)
    tt = pitch.t[sl].astype(float, copy=True)
    pp = pitch.shape[sl].astype(float, copy=True)
    finite = np.isfinite(tt) & np.isfinite(pp)
    tt = tt[finite]
    pp = pp[finite]
    if len(tt) == 0:
        raise ValueError("V17 residual run has no finite corrected-valid pitch")

    if start_s < tt[0] - 1e-9:
        tt = np.insert(tt, 0, float(start_s))
        pp = np.insert(pp, 0, float(pp[0]))
    if end_s > tt[-1] + 1e-9:
        tt = np.append(tt, float(end_s))
        pp = np.append(pp, float(pp[-1]))

    base_st = float(np.median(pitch.shape[sl][np.isfinite(pitch.shape[sl])]))
    score = (
        float(attack.acoustic_score)
        if attack is not None and math.isfinite(attack.acoustic_score)
        else 0.0
    )
    kind = (
        "residual_valid_f0_existing_attack_v17"
        if attack is not None
        else "residual_valid_f0_v17"
    )
    return MusicalEvent(
        float(start_s), float(end_s), base_st, _nearest_midi(base_st), tt, pp,
        kind, f"residual_valid_v17_{serial}@{run.start_s:.6f}", score,
        "v3r9_corrected_valid_residual_v17",
        expression_anchor_s=float(run.start_s),
    )


def _reclean_monophonic(events: list[MusicalEvent]) -> list[MusicalEvent]:
    """Reapply the same final monophonic clipping after residual insertion."""
    events = sorted(events, key=lambda e: (e.start_s, e.end_s, -e.attack_score))
    cleaned: list[MusicalEvent] = []
    for e in events:
        if cleaned and e.start_s < cleaned[-1].end_s - 1e-9:
            cleaned[-1].end_s = e.start_s
            if cleaned[-1].end_s <= cleaned[-1].start_s + 1e-4:
                cleaned.pop()
        if e.end_s <= e.start_s + 1e-4:
            continue
        deviations = e.pitch_st - float(e.base_midi)
        e.max_abs_bend_st = float(np.nanmax(np.abs(deviations))) if len(deviations) else 0.0
        e.bend_range_st = (
            max(1, int(math.ceil(e.max_abs_bend_st - 1e-12)))
            if e.max_abs_bend_st > 1e-4 else 0
        )
        cleaned.append(e)
    return cleaned


def assemble_events(prefix, wav_path) -> AssemblyResult:
    """Run V16.1 unchanged, then close stable corrected-valid ownership debt.

    V17.1 differs from V17 only when a residual run is demonstrably the same
    family as the immediately preceding owner and there is no independent
    physical reattack near the residual start.  In that case ownership is
    extended instead of emitting another Note-On.
    """
    result = assemble_events_v16_1(prefix, wav_path)
    residuals = _residual_valid_runs(result)

    promoted = 0
    joined_same_family = 0
    rejected_short = 0
    added: list[MusicalEvent] = []

    for serial, run in enumerate(residuals):
        min_required = RESIDUAL_VALID_MIN_STABLE_FACTOR * float(result.pitch.min_stable_s)
        if run.duration_s + 1e-12 < min_required:
            rejected_short += 1
            result.attack_decisions.append(AttackDecision(
                run.start_s, False, "residual_valid_f0_v17_1",
                "residual_valid_f0_below_existing_min_stable_v17_1",
                pitch_source="v3r9_corrected_valid_residual_v17_1",
                pitch_anchor_s=run.start_s,
                pitch_anchor_st=run.median_st,
            ))
            continue

        previous = _previous_event(result.events, run.start_s)
        following = _next_event(result.events, run.end_s)
        end_s = float(run.end_s)
        if following is not None and following.start_s < end_s:
            end_s = float(following.start_s)
        if end_s <= run.start_s + 1e-6:
            continue

        join_owner, blocking_reattack = _same_family_predecessor_for_residual(
            result, run, previous
        )
        if join_owner is not None:
            _extend_existing_owner_with_residual(join_owner, result, run, end_s)
            joined_same_family += 1
            result.attack_decisions.append(AttackDecision(
                run.start_s, False, "residual_valid_f0_v17_1",
                "same_family_residual_transferred_without_note_on_v17_1",
                pitch_source="v3r9_corrected_valid_residual_v17_1",
                pitch_anchor_s=run.start_s,
                pitch_anchor_st=run.median_st,
                continuity_status="preceding_same_family_owner_extended",
            ))
            continue

        start_s, attack = _accepted_existing_attack_start(result, run, previous)
        e = _make_residual_event(result, run, start_s, end_s, attack, serial)
        # Label only the V17.1-added event; its musical behavior is otherwise V17.
        e.source_kind = e.source_kind.replace("_v17", "_v17_1")
        e.source_id = e.source_id.replace("residual_valid_v17_", "residual_valid_v17_1_")
        e.pitch_source = "v3r9_corrected_valid_residual_v17_1"
        added.append(e)
        promoted += 1
        result.attack_decisions.append(AttackDecision(
            e.start_s, True, "residual_valid_f0_v17_1",
            "stable_corrected_valid_residual_ownership_v17_1",
            e.attack_score, math.nan,
            e.pitch_source, run.start_s, run.median_st,
            continuity_status=("existing_accepted_attack" if attack is not None else "corrected_valid_residual"),
        ))

    if added:
        result.events = _reclean_monophonic(result.events + added)
    else:
        result.events = _reclean_monophonic(result.events)

    result.attack_decisions.append(AttackDecision(
        -1.0, False, "v17_1_summary",
        (f"residual_runs={len(residuals)};promoted={promoted};"
         f"joined_same_family={joined_same_family};rejected_short={rejected_short}"),
        pitch_source="v3r9_corrected_valid_residual_v17_1",
    ))
    return result
