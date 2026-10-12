from __future__ import annotations

"""MIDI Renderer V17 -- corrected-valid residual ownership closure.

V17 is intentionally a very small layer on top of V16.1.  V16.1 remains
responsible for articulation, continuous-motion handling, owner-corridor
adjudication, pitch-only handoffs, and bounded V16-created corridor repair.

V17 does one job only: after all of those decisions are finished, every
remaining sufficiently-stable corrected-valid V3R9 F0 run must receive MIDI
ownership.  No WAV-only pitch inference, primary-family rescue, subharmonic
adjudication, or transient-note cleanup is introduced here.
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
    """Run V16.1 unchanged, then close stable corrected-valid ownership debt."""
    result = assemble_events_v16_1(prefix, wav_path)
    residuals = _residual_valid_runs(result)

    promoted = 0
    rejected_short = 0
    added: list[MusicalEvent] = []

    for serial, run in enumerate(residuals):
        # Reuse the exact upstream musical timescale already carried by V3R9.
        # V17 must not turn one/two-frame remnants into new micro-notes.
        min_required = RESIDUAL_VALID_MIN_STABLE_FACTOR * float(result.pitch.min_stable_s)
        if run.duration_s + 1e-12 < min_required:
            rejected_short += 1
            result.attack_decisions.append(AttackDecision(
                run.start_s, False, "residual_valid_f0_v17",
                "residual_valid_f0_below_existing_min_stable_v17",
                pitch_source="v3r9_corrected_valid_residual_v17",
                pitch_anchor_s=run.start_s,
                pitch_anchor_st=run.median_st,
            ))
            continue

        previous = _previous_event(result.events, run.start_s)
        following = _next_event(result.events, run.end_s)
        start_s, attack = _accepted_existing_attack_start(result, run, previous)
        end_s = float(run.end_s)
        if following is not None and following.start_s < end_s:
            end_s = float(following.start_s)
        if end_s <= run.start_s + 1e-6:
            continue

        e = _make_residual_event(result, run, start_s, end_s, attack, serial)
        added.append(e)
        promoted += 1
        result.attack_decisions.append(AttackDecision(
            e.start_s, True, "residual_valid_f0_v17",
            "stable_corrected_valid_residual_ownership_v17",
            e.attack_score, math.nan,
            e.pitch_source, run.start_s, run.median_st,
            continuity_status=("existing_accepted_attack" if attack is not None else "corrected_valid_residual"),
        ))

    if added:
        result.events = _reclean_monophonic(result.events + added)

    # A V17 release should make its feature activity visible in the attack audit
    # without changing the AssemblyResult schema.
    result.attack_decisions.append(AttackDecision(
        -1.0, False, "v17_summary",
        f"residual_runs={len(residuals)};promoted={promoted};rejected_short={rejected_short}",
        pitch_source="v3r9_corrected_valid_residual_v17",
    ))
    return result
