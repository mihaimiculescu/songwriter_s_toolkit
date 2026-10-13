from __future__ import annotations

"""MIDI Renderer V18.3 -- revealed micro-remnant topology cooperation.

V18.3 is a strict child of V18.2.  V18.1/V18.2 acoustic thresholds, PB
calibration, repetition recovery, subharmonic adjudication, and corrected-valid
coverage guarantees remain unchanged.

The only new behavior resolves the interaction V18.2 exposed when a truthful
unsupported-start retraction would leave <55 ms of real corrected-valid pitch:

  * first apply the already-approved V18.2 unsupported-start certificate;
  * if >=55 ms remains, perform the ordinary V18.2 timing trim unchanged;
  * if <55 ms remains, do not manufacture a standalone micro-note;
  * when the revealed remnant and the immediately following owner are already
    classified as one continuous acoustic object, fold the remnant pitch curve
    into that following owner as Pitch Bend, preserving the real pitch without
    issuing another Note-On;
  * if that continuity certificate is absent, keep V18.2 behavior unchanged.

No new pitch inference or acoustic threshold is introduced.  The 55 ms scale is
exactly V18.2's existing micro-note guard, and the 30 ms neighbor aperture is
V18's existing transient-head adjacency limit.
"""

from pathlib import Path
import math
import numpy as np

from .midi_event_assembler_v18_1 import (
    AssemblyResult, MusicalEvent, AttackDecision,
    assemble_events as assemble_events_v18_1,
)
from .midi_event_assembler_v18 import (
    _WavProbe, _trim_event_start, _merge_curves,
    TRANSIENT_HEAD_MAX_GAP_TO_NEXT_S,
)
from .midi_event_assembler_v17_1 import _reclean_monophonic
from .midi_event_assembler_v18_2 import (
    UNSUPPORTED_BACKDATE_MIN_LAG_S,
    UNSUPPORTED_BACKDATE_START_DBFS,
    UNSUPPORTED_BACKDATE_MAX_MEDIAN_DBFS,
    UNSUPPORTED_BACKDATE_RMS_SAMPLES,
    UNSUPPORTED_BACKDATE_MIN_REMAINING_S,
    UNSUPPORTED_PRIMARY_MIN_ACF,
    UNSUPPORTED_PRIMARY_MAX_CMNDF,
    UNSUPPORTED_PRIMARY_MAX_ACF_CMNDF_CENTS,
    UNSUPPORTED_PRIMARY_MAX_FAMILY_GAP_ST,
    UNSUPPORTED_PRIMARY_ANCHOR_GUARD_S,
    UNSUPPORTED_VALID_MAX_FAMILY_GAP_ST,
    _target_family_primary_exists,
    _first_target_valid_time,
    _borrowed_corridor_median_dbfs,
)

# No new threshold: V18.3 explicitly reuses V18.2's existing 55 ms guard as the
# definition of a newly revealed micro-remnant.
REVEALED_MICRO_MAX_S = UNSUPPORTED_BACKDATE_MIN_REMAINING_S
REVEALED_MICRO_MAX_GAP_NEXT_S = TRANSIENT_HEAD_MAX_GAP_TO_NEXT_S


def _next_event_after(events: list[MusicalEvent], event: MusicalEvent) -> MusicalEvent | None:
    candidates = [
        e for e in events
        if e is not event and float(e.start_s) >= float(event.end_s) - 1e-9
    ]
    return min(candidates, key=lambda e: float(e.start_s)) if candidates else None


def _continuous_with_following_owner(
    result: AssemblyResult,
    remnant_start_s: float,
    next_event: MusicalEvent,
) -> tuple[bool, str]:
    next_anchor = (
        float(next_event.expression_anchor_s)
        if math.isfinite(float(next_event.expression_anchor_s))
        else float(next_event.start_s)
    )
    grace = float(np.clip(1.5 * result.pitch.dt, 0.012, 0.020))
    continuity = result.attack_lane.ownership_continuity(
        float(remnant_start_s), next_anchor, post_anchor_grace_s=grace
    )
    return bool(continuity.connected), str(continuity.status)


def _retract_and_adjudicate_revealed_micro_remnants(
    result: AssemblyResult,
    probe: _WavProbe,
) -> tuple[int, int]:
    """Apply V18.2's certificate, then cooperate with existing topology logic.

    Returns (ordinary_retractions, micro_remnants_folded).
    """
    ordinary = 0
    folded = 0

    # Snapshot order.  We only remove a revealed micro-remnant after merging it
    # into its immediately following owner; all other V18.1 topology stays put.
    candidates = sorted(list(result.events), key=lambda e: (e.start_s, e.end_s))
    removed_ids: set[int] = set()

    for e in candidates:
        if id(e) in removed_ids or e not in result.events:
            continue
        if "trusted_island_owner" not in e.source_kind:
            continue

        anchor = float(e.expression_anchor_s)
        if not math.isfinite(anchor):
            continue
        old_start = float(e.start_s)
        if anchor - old_start < UNSUPPORTED_BACKDATE_MIN_LAG_S:
            continue

        start_level = probe.rms_db(old_start)
        if start_level > UNSUPPORTED_BACKDATE_START_DBFS:
            continue

        prevalid = (
            result.pitch.valid
            & (result.pitch.t >= old_start - 1e-9)
            & (result.pitch.t < anchor - 1e-9)
        )
        if np.any(prevalid):
            continue

        if _target_family_primary_exists(result, old_start, anchor, float(e.base_st)):
            continue

        median_level = _borrowed_corridor_median_dbfs(probe, old_start, anchor)
        if median_level > UNSUPPORTED_BACKDATE_MAX_MEDIAN_DBFS:
            continue

        first_valid = _first_target_valid_time(result, e)
        if first_valid is None:
            continue
        new_start = float(first_valid)
        if new_start <= old_start + 1e-9:
            continue

        remaining = float(e.end_s) - new_start
        if remaining >= REVEALED_MICRO_MAX_S - 1e-12:
            _trim_event_start(e, new_start)
            e.source_kind = f"{e.source_kind}+unsupported_borrowed_start_trim_v18_2"
            result.attack_decisions.append(AttackDecision(
                old_start,
                False,
                "unsupported_borrowed_start_v18_2",
                "weak_preanchor_corridor_has_no_valid_or_primary_target_family_v18_2",
                pitch_source=e.pitch_source,
                pitch_anchor_s=new_start,
                pitch_anchor_st=float(e.base_st),
                continuity_status=(
                    f"old_start_dbfs={start_level:.3f};"
                    f"borrowed_median_dbfs={median_level:.3f};"
                    f"lag_s={anchor-old_start:.6f};"
                    f"trimmed_to={new_start:.9f}"
                ),
            ))
            ordinary += 1
            continue

        # V18.2 used to cancel the truthful trim here to avoid creating a
        # micro-note.  V18.3 instead asks the already-existing topology lane
        # whether that tiny real pitch is continuous with the next owner.
        nxt = _next_event_after(result.events, e)
        if nxt is None:
            continue
        gap = float(nxt.start_s) - float(e.end_s)
        if gap < -1e-9 or gap > REVEALED_MICRO_MAX_GAP_NEXT_S + 1e-12:
            continue

        connected, continuity_status = _continuous_with_following_owner(
            result, new_start, nxt
        )
        if not connected:
            # No new policy for independent objects: retain V18.2 exactly.
            continue

        # Retract first, then preserve the acoustically real remnant pitch as a
        # PB lead-in of the following owner.  This is the same representation
        # used by V18/V18.1 transient/weak-island folding and therefore keeps
        # corrected-valid pitch rendered without creating another Note-On.
        _trim_event_start(e, new_start)
        remnant_base_st = float(e.base_st)
        remnant_anchor = float(e.expression_anchor_s)
        merged = _merge_curves(e, nxt, keep_base="second")
        merged.source_kind = f"{nxt.source_kind}+revealed_micro_pb_v18_3"
        merged.pitch_source = nxt.pitch_source

        result.events = [x for x in result.events if x is not e]
        removed_ids.add(id(e))
        result.attack_decisions.append(AttackDecision(
            old_start,
            False,
            "unsupported_borrowed_start_micro_fold_v18_3",
            "unsupported_start_retracted_then_micro_remnant_folded_into_continuous_next_owner_v18_3",
            pitch_source=merged.pitch_source,
            pitch_anchor_s=new_start,
            pitch_anchor_st=remnant_base_st,
            continuity_status=(
                f"{continuity_status};"
                f"old_start_dbfs={start_level:.3f};"
                f"borrowed_median_dbfs={median_level:.3f};"
                f"lag_s={anchor-old_start:.6f};"
                f"remnant_s={remaining:.6f};"
                f"remnant_anchor_s={remnant_anchor:.9f};"
                f"folded_into_next_start={float(nxt.start_s):.9f}"
            ),
        ))
        folded += 1

    result.events = _reclean_monophonic(result.events)
    return ordinary, folded


def assemble_events(prefix: Path, wav_path: Path) -> AssemblyResult:
    # Start from frozen V18.1 and apply exactly the V18.2 certificate plus the
    # one V18.3 cooperation rule.  Calling V18.2 directly would have already
    # skipped sub-55 ms remnants, so it cannot expose them for adjudication.
    result = assemble_events_v18_1(prefix, wav_path)
    probe = _WavProbe(wav_path)
    ordinary, folded = _retract_and_adjudicate_revealed_micro_remnants(result, probe)
    result.events = _reclean_monophonic(result.events)
    result.attack_decisions.append(AttackDecision(
        -1.0,
        False,
        "v18_3_summary",
        (
            f"unsupported_borrowed_starts_retracted={ordinary};"
            f"revealed_micro_remnants_folded={folded}"
        ),
        pitch_source="renderer_topology_cooperation_v18_3",
    ))
    return result
