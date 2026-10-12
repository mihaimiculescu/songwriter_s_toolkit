from __future__ import annotations

"""V17 contradictory-register re-adjudication.

This pass is intentionally based on the V14 production baseline.

Lessons retained from the abandoned V15/V16 experiments:
  * register ownership must remain coherent inside one voiced trajectory object;
  * the existing interval juror is useful for detecting abrupt register
    discontinuities.

Crucially, the interval juror is only a TRIGGER here.  It does not propagate an
owner and it does not insert a wall.  Once a contradiction is detected, the
affected connected burst is re-adjudicated from the already-existing,
in-range, acoustically-admissible candidate field.  The chosen path is the one
that best satisfies the existing interval-juror geometry, then existing
adjudication/acoustic evidence.  No octave rule, bend cap, GT target, song
timing, or new pitch threshold is introduced.
"""

from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np

from .juror_evidence import _juror_quadratic_transition_penalty
from .reacquisition_adjudication import _candidate_rows, _score_field


@dataclass(frozen=True)
class RegisterReadjudicationRow:
    region_id: int
    burst_start_s: float
    burst_end_s: float
    frame_index: int
    original_hz: float | None
    reviewed_hz: float | None
    reviewed_midi: int | None
    source: str
    trigger_penalty_max: float
    applied_sample_count: int


def _midi_float(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def _interval_cost(hz: float, t: float, ref_hz: float, ref_t: float) -> float:
    interval = abs(_midi_float(hz) - _midi_float(ref_hz))
    available_ms = max(abs(float(t) - float(ref_t)) * 1000.0, 1.0)
    return float(_juror_quadratic_transition_penalty(interval, available_ms))


def _connected_edge_components(edge_indices: list[int]) -> list[tuple[int, int]]:
    """Return sample-index bursts from contradiction edges.

    Edge i means the transition i-1 -> i is contradictory.  Consecutive edges
    belong to one burst.  The returned bounds are inclusive sample indices and
    include both sides of the contradiction.
    """
    if not edge_indices:
        return []
    edges = sorted(set(int(i) for i in edge_indices))
    out = []
    start = prev = edges[0]
    for e in edges[1:]:
        if e == prev + 1:
            prev = e
            continue
        out.append((max(0, start - 1), prev))
        start = prev = e
    out.append((max(0, start - 1), prev))
    return out


def _owner_change_cost(final_row, hz: float | None) -> int:
    if final_row is None or getattr(final_row, "status", "") != "resolved" or getattr(final_row, "winner_hz", None) is None:
        return 0
    if hz is None:
        return 1
    ref = float(final_row.winner_hz)
    cents = abs(1200.0 * math.log2(float(hz) / ref))
    return 0 if cents < 2.0 else 1


def _dedup_candidates(rows, current_hz, acoustic_scores):
    out = []
    for r in _candidate_rows(rows):
        hz = float(r.representative_hz)
        if any(abs(1200.0 * math.log2(hz / c["hz"])) < 0.5 for c in out):
            continue
        out.append({
            "hz": hz,
            "midi": int(r.note_group_midi),
            "acoustic": acoustic_scores.get(round(hz, 6), -math.inf),
            "source": "existing_candidate",
        })

    if current_hz is not None and math.isfinite(float(current_hz)) and float(current_hz) > 0:
        hz = float(current_hz)
        if not any(abs(1200.0 * math.log2(hz / c["hz"])) < 0.5 for c in out):
            out.append({
                "hz": hz,
                "midi": int(round(_midi_float(hz))),
                "acoustic": acoustic_scores.get(round(hz, 6), -math.inf),
                "source": "current_owner",
            })
    return out


def _solve_burst(
    *,
    frames,
    frame_times,
    current_by_frame,
    by_evidence,
    by_final,
    audio,
    sr,
    left_anchor,
    right_anchor,
):
    """Solve one contradiction burst as a coherent candidate path.

    Objective (lexicographic, no tunable weights):
      1. minimise existing interval-juror penalty, including immediate trusted
         anchors on either side when available;
      2. maximise retained frames;
      3. minimise disagreement with already-resolved jury/HD winners;
      4. maximise existing Harmonic-Detective acoustic score.

    NULL is allowed only so a frame can abstain rather than force an
    interval-impossible owner.
    """
    ordered = [int(fi) for fi in frames if fi in frame_times]
    if not ordered:
        return {}

    fields = {}
    for fi in ordered:
        rows = by_evidence.get(fi, [])
        scored = _score_field(audio, sr, rows) if rows else []
        amap = {round(float(hz), 6): float(score) for score, _support, hz, _midi, _gid in scored}
        cand = _dedup_candidates(rows, current_by_frame.get(fi), amap)
        cand.append({"hz": None, "midi": None, "acoustic": 0.0, "source": "abstain"})
        fields[fi] = cand

    # DP state:
    # (penalty, negative_retained, adjudication_changes, negative_acoustic,
    #  predecessor_candidate_index)
    dp = []
    first = ordered[0]
    first_states = []
    for c in fields[first]:
        retained = 0 if c["hz"] is None else 1
        ac = c["acoustic"] if retained and math.isfinite(c["acoustic"]) else 0.0
        penalty = 0.0
        if retained and left_anchor is not None:
            ahz, at = left_anchor
            penalty = _interval_cost(c["hz"], frame_times[first], ahz, at)
        first_states.append((
            penalty,
            -retained,
            _owner_change_cost(by_final.get(first), c["hz"]),
            -ac,
            None,
        ))
    dp.append(first_states)

    for k in range(1, len(ordered)):
        fi = ordered[k]
        prev_fi = ordered[k - 1]
        states = []
        for ci, c in enumerate(fields[fi]):
            retained = 0 if c["hz"] is None else 1
            ac = c["acoustic"] if retained and math.isfinite(c["acoustic"]) else 0.0
            best = None
            for pj, pc in enumerate(fields[prev_fi]):
                prev = dp[k - 1][pj]
                tr = 0.0
                if c["hz"] is not None and pc["hz"] is not None:
                    tr = _interval_cost(c["hz"], frame_times[fi], pc["hz"], frame_times[prev_fi])
                val = (
                    prev[0] + tr,
                    prev[1] - retained,
                    prev[2] + _owner_change_cost(by_final.get(fi), c["hz"]),
                    prev[3] - ac,
                    pj,
                )
                if best is None or val[:4] < best[:4]:
                    best = val
            states.append(best)
        dp.append(states)

    def end_key(j):
        s = dp[-1][j]
        penalty = s[0]
        c = fields[ordered[-1]][j]
        if c["hz"] is not None and right_anchor is not None:
            ahz, at = right_anchor
            penalty += _interval_cost(c["hz"], frame_times[ordered[-1]], ahz, at)
        return (penalty, s[1], s[2], s[3])

    j = min(range(len(dp[-1])), key=end_key)
    chosen = [None] * len(ordered)
    for k in range(len(ordered) - 1, -1, -1):
        chosen[k] = j
        j = dp[k][j][4] if k > 0 else None

    return {fi: fields[fi][ci] for fi, ci in zip(ordered, chosen)}


def apply_register_readjudication_v17(
    *,
    offline_validity,
    interpretation,
    sample_times_s,
    sample_frame_index,
    audio,
    sample_rate,
    evidence_rows,
    final_adjudication,
):
    """Re-adjudicate only connected abrupt register contradictions.

    Detection is confined to an already-existing V14 trajectory region.
    An adjacent pair is a trigger only when the existing interval juror assigns
    a non-zero penalty to that exact observed transition.

    The trigger itself never chooses the replacement.  Candidate ownership is
    decided by `_solve_burst` from the existing candidate/adjudication evidence.
    """
    valid = offline_validity.valid
    clean = offline_validity.clean_f0_hz
    reason = offline_validity.reason
    corr_reason = np.asarray(offline_validity.correction_reason, dtype=object)
    times = np.asarray(sample_times_s, dtype=float)
    frame_index = np.asarray(sample_frame_index)
    region_id = np.asarray(interpretation.region_id)

    by_evidence = defaultdict(list)
    for r in evidence_rows:
        by_evidence[int(r.frame_index)].append(r)
    by_final = {int(r.frame_index): r for r in final_adjudication}

    audit = []

    region_values = sorted(set(int(x) for x in region_id if int(x) >= 0))
    for rid in region_values:
        idx = np.flatnonzero((region_id == rid) & valid & np.isfinite(clean) & (clean > 0))
        if len(idx) < 2:
            continue

        # Only truly adjacent timeline samples inside this region can trigger.
        edges = []
        edge_penalty = {}
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue
            p = _interval_cost(float(clean[b]), float(times[b]), float(clean[a]), float(times[a]))
            if p > 0.0:
                edges.append(int(b))
                edge_penalty[int(b)] = float(p)

        for burst_a, burst_b in _connected_edge_components(edges):
            # Constrain burst to this exact trajectory region.
            burst_idx = np.arange(burst_a, burst_b + 1, dtype=int)
            burst_idx = burst_idx[(region_id[burst_idx] == rid) & valid[burst_idx]]
            if len(burst_idx) < 2:
                continue

            frames = []
            for fi in frame_index[burst_idx]:
                fi = int(fi)
                if fi not in frames:
                    frames.append(fi)

            # Frame time and current owner are derived only from samples in the
            # contradiction burst/region; no arbitrary wider propagation.
            frame_times = {}
            current_by_frame = {}
            for fi in frames:
                ii = burst_idx[frame_index[burst_idx] == fi]
                if len(ii) == 0:
                    continue
                frame_times[fi] = float(np.median(times[ii]))
                vals = np.asarray(clean[ii], dtype=float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                current_by_frame[fi] = float(np.median(vals)) if len(vals) else None

            # Immediate same-region trusted owners are anchors.  We do not
            # bridge invalid/unvoiced gaps or cross a trajectory-region border.
            left_anchor = None
            left_i = int(np.min(burst_idx)) - 1
            if left_i >= 0 and valid[left_i] and region_id[left_i] == rid and math.isfinite(float(clean[left_i])) and float(clean[left_i]) > 0:
                left_anchor = (float(clean[left_i]), float(times[left_i]))

            right_anchor = None
            right_i = int(np.max(burst_idx)) + 1
            if right_i < len(valid) and valid[right_i] and region_id[right_i] == rid and math.isfinite(float(clean[right_i])) and float(clean[right_i]) > 0:
                right_anchor = (float(clean[right_i]), float(times[right_i]))

            chosen = _solve_burst(
                frames=frames,
                frame_times=frame_times,
                current_by_frame=current_by_frame,
                by_evidence=by_evidence,
                by_final=by_final,
                audio=audio,
                sr=sample_rate,
                left_anchor=left_anchor,
                right_anchor=right_anchor,
            )
            if not chosen:
                continue

            max_trigger = max((edge_penalty.get(e, 0.0) for e in edges if burst_a <= e <= burst_b), default=0.0)
            for fi in frames:
                c = chosen.get(fi)
                if c is None:
                    continue
                # Apply to all samples of this ECKF frame that belong to the
                # same V14 trajectory region.  That preserves frame ownership
                # coherence without propagating into later continuous material.
                ii = np.flatnonzero((frame_index == fi) & (region_id == rid))
                if len(ii) == 0:
                    continue
                vals = np.asarray(clean[ii], dtype=float)
                finite = vals[np.isfinite(vals) & (vals > 0)]
                original_hz = float(np.median(finite)) if len(finite) else None

                if c["hz"] is None:
                    valid[ii] = False
                    clean[ii] = np.nan
                    reason[ii] = "INTERVAL_READJUDICATION_ABSTAIN"
                    corr_reason[ii] = "INTERVAL_READJUDICATION_ABSTAIN"
                    audit.append(RegisterReadjudicationRow(
                        region_id=rid,
                        burst_start_s=float(times[np.min(burst_idx)]),
                        burst_end_s=float(times[np.max(burst_idx)]),
                        frame_index=fi,
                        original_hz=original_hz,
                        reviewed_hz=None,
                        reviewed_midi=None,
                        source="interval_trigger_readjudication_abstain",
                        trigger_penalty_max=float(max_trigger),
                        applied_sample_count=int(len(ii)),
                    ))
                    continue

                reviewed_hz = float(c["hz"])
                differs = (
                    original_hz is None
                    or abs(1200.0 * math.log2(reviewed_hz / original_hz)) >= 2.0
                    or not np.all(valid[ii])
                )
                if not differs:
                    continue

                valid[ii] = True
                clean[ii] = reviewed_hz
                reason[ii] = "INTERVAL_READJUDICATION_REPLACEMENT"
                corr_reason[ii] = "INTERVAL_READJUDICATION_REPLACEMENT"
                audit.append(RegisterReadjudicationRow(
                    region_id=rid,
                    burst_start_s=float(times[np.min(burst_idx)]),
                    burst_end_s=float(times[np.max(burst_idx)]),
                    frame_index=fi,
                    original_hz=original_hz,
                    reviewed_hz=reviewed_hz,
                    reviewed_midi=int(c["midi"]),
                    source="interval_trigger_readjudication",
                    trigger_penalty_max=float(max_trigger),
                    applied_sample_count=int(len(ii)),
                ))

    return tuple(audit)
