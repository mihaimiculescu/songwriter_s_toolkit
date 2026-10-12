from __future__ import annotations

"""Persistent interval-juror provenance for V3R2.

V3R2 refines V3R1's authority rule without changing the interval curve itself.

Key distinction:
  * every existing octave-pair interval verdict is still recorded;
  * a verdict becomes a downstream veto only when the reference context it
    depends on is authoritative enough to support persistence.

A very specific circularity is treated as provisional:
  1. immediately after an interruption/reacquisition, a frame has an
     initializer-selected owner but NO usable interval component for any
     candidate;
  2. the next frame's interval verdict uses that just-created frame as its
     previous reference;
  3. that verdict is therefore advisory only.  It may describe continuity from
     the provisional owner, but it cannot make that owner self-confirming.

This does NOT discard all first-post-gap verdicts.  Only verdicts whose actual
previous reference is one of those no-interval initializer frames lose veto
authority.  All verdicts remain visible in the audit.
"""

from dataclasses import dataclass
import math
from collections import defaultdict


@dataclass(frozen=True)
class IntervalAuthorityVerdict:
    frame_index: int
    time_s: float
    preferred_group_id: str
    preferred_midi: int
    preferred_hz: float
    rejected_group_id: str
    rejected_midi: int
    rejected_hz: float
    preferred_component: float
    rejected_component: float
    authoritative: bool = True
    context_status: str = "authoritative_context"
    reference_time_s: float | None = None
    reference_hz: float | None = None
    source: str = "juror_pair_existing_interval_verdict"


def _finite(v) -> bool:
    return v is not None and math.isfinite(float(v))


def _time_key(t: float) -> int:
    # Evidence times originate from the same frame clock; microsecond rounding
    # avoids fragile binary-float equality while remaining far below frame size.
    return int(round(float(t) * 1_000_000.0))


def _build_provisional_reference_times(evidence_rows):
    """Return times of initializer-created frames with no interval context.

    These are the exact premises that can create the self-confirming sequence:
    NO_VERDICT -> provisional owner -> next-frame verdict based on that owner.
    """
    by_frame = defaultdict(list)
    for r in evidence_rows or ():
        by_frame[int(r.frame_index)].append(r)

    provisional = {}
    for fi, rows in by_frame.items():
        selected = [r for r in rows if bool(getattr(r, "selected_by_initializer", False))]
        if not selected:
            continue
        any_interval = any(
            _finite(getattr(r, "interval_previous_component", None))
            or _finite(getattr(r, "interval_following_component", None))
            for r in rows
        )
        if any_interval:
            continue
        # Exactly one selected initializer is normal; if there are several,
        # retain all corresponding time/hz values conservatively.
        for r in selected:
            t = getattr(r, "time_s", None)
            hz = getattr(r, "representative_hz", None)
            if _finite(t) and _finite(hz) and float(hz) > 0.0:
                provisional[_time_key(float(t))] = (float(t), float(hz), int(fi))
    return provisional


def _pair_reference_context(pair_row, by_evidence, provisional_times):
    """Describe the interval reference that actually drives this pair verdict.

    For octave pairs the candidate rows normally share the same historical
    previous/following reference.  We only suppress authority when the verdict's
    *previous* reference explicitly points to a provisional no-interval frame.
    """
    rows = by_evidence.get(int(pair_row.frame_index), ())
    gids = {str(pair_row.a_group_id), str(pair_row.b_group_id)}
    pair_rows = [r for r in rows if str(getattr(r, "group_id", "")) in gids]

    prev_refs = []
    for r in pair_rows:
        rt = getattr(r, "interval_previous_reference_time_s", None)
        rh = getattr(r, "interval_previous_reference_hz", None)
        comp = getattr(r, "interval_previous_component", None)
        if _finite(comp) and _finite(rt) and _finite(rh):
            prev_refs.append((float(rt), float(rh)))

    if prev_refs:
        # If either octave candidate is being judged against the provisional
        # premise, the pair verdict is advisory.  (Normally both share it.)
        for rt, rh in prev_refs:
            p = provisional_times.get(_time_key(rt))
            if p is None:
                continue
            pt, phz, _fi = p
            # Reference Hz should be the provisional owner itself; tolerate
            # tiny representation differences only.
            cents = abs(1200.0 * math.log2(float(rh) / float(phz))) if rh > 0 and phz > 0 else math.inf
            if cents < 5.0:
                return False, "provisional_post_gap_reference", rt, rh
        rt, rh = prev_refs[0]
        return True, "authoritative_previous_reference", rt, rh

    # No previous reference.  A following-only interval verdict is not the
    # circularity under investigation, so preserve V3R1 authority semantics.
    following = []
    for r in pair_rows:
        rt = getattr(r, "interval_following_reference_time_s", None)
        rh = getattr(r, "interval_following_reference_hz", None)
        comp = getattr(r, "interval_following_component", None)
        if _finite(comp) and _finite(rt) and _finite(rh):
            following.append((float(rt), float(rh)))
    if following:
        rt, rh = following[0]
        return True, "authoritative_following_reference", rt, rh

    return True, "authoritative_interval_context", None, None


def build_interval_authority(pair_rows, evidence_rows=()):
    """Return frame -> tuple[IntervalAuthorityVerdict, ...].

    All unequal octave-pair interval verdicts are recorded.  Only verdicts with
    ``authoritative=True`` are allowed to veto downstream octave re-attribution.
    """
    by_frame = defaultdict(list)
    by_evidence = defaultdict(list)
    for r in evidence_rows or ():
        by_evidence[int(r.frame_index)].append(r)
    provisional_times = _build_provisional_reference_times(evidence_rows)

    for p in pair_rows:
        if abs(int(p.a_midi) - int(p.b_midi)) != 12:
            continue
        if not (_finite(p.a_interval) and _finite(p.b_interval)):
            continue
        ai, bi = float(p.a_interval), float(p.b_interval)
        if ai == bi:
            continue
        if ai > bi:
            pref = (p.a_group_id, int(p.a_midi), float(p.a_hz), ai)
            rej = (p.b_group_id, int(p.b_midi), float(p.b_hz), bi)
        else:
            pref = (p.b_group_id, int(p.b_midi), float(p.b_hz), bi)
            rej = (p.a_group_id, int(p.a_midi), float(p.a_hz), ai)

        authoritative, context_status, ref_t, ref_hz = _pair_reference_context(
            p, by_evidence, provisional_times
        )
        by_frame[int(p.frame_index)].append(IntervalAuthorityVerdict(
            frame_index=int(p.frame_index),
            time_s=float(p.time_s),
            preferred_group_id=str(pref[0]),
            preferred_midi=int(pref[1]),
            preferred_hz=float(pref[2]),
            rejected_group_id=str(rej[0]),
            rejected_midi=int(rej[1]),
            rejected_hz=float(rej[2]),
            preferred_component=float(pref[3]),
            rejected_component=float(rej[3]),
            authoritative=bool(authoritative),
            context_status=str(context_status),
            reference_time_s=ref_t,
            reference_hz=ref_hz,
        ))
    return {fi: tuple(v) for fi, v in by_frame.items()}


def group_rejected(authority_by_frame, frame_index: int, group_id: str | None) -> bool:
    if group_id is None:
        return False
    return any(
        bool(v.authoritative) and v.rejected_group_id == group_id
        for v in authority_by_frame.get(int(frame_index), ())
    )


def preferred_for_rejected(authority_by_frame, frame_index: int, group_id: str | None):
    if group_id is None:
        return None
    for v in authority_by_frame.get(int(frame_index), ()):
        if bool(v.authoritative) and v.rejected_group_id == group_id:
            return v
    return None
