from __future__ import annotations

"""Island-level adjudication feedback for reacquisition / recovery only.

V13 keeps V12's narrow scope but fixes its frame-local inconsistency.  A
reacquisition episode is treated as one contiguous recovery island.  Existing
jury / Harmonic-Detective decisions remain the candidate authority; the
interval juror is reused only to reject an octave owner that creates an
implausible jump *inside the same recovery island*.

No song-specific timings, pitches, octave rules, bend caps, or GT-derived
thresholds are introduced.
"""

from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np

from .harmonic_detective import (
    _spectrum,
    _harmonic_score,
    WINDOWS_MS,
    MIN_SCORE,
    MIN_SUPPORT,
    MIN_MARGIN,
    HIGH_CONF_MIN_SCORE,
    HIGH_CONF_MIN_SUPPORT,
    HIGH_CONF_MIN_MARGIN,
    HIGH_CONF_MIN_VALID_WINDOWS,
)
from .interval_authority import build_interval_authority, group_rejected, preferred_for_rejected
from .ownership_evidence import (
    acoustically_admissible,
    interval_transition_penalty,
    range_admitted,
)


@dataclass(frozen=True)
class ReacquisitionFeedbackRow:
    frame_index: int
    time_s: float
    original_hz: float | None
    original_reason: str
    jury_hz: float | None
    reviewed_hz: float | None
    reviewed_midi: int | None
    review_source: str
    detective_best_score: float | None
    detective_runner_score: float | None
    detective_margin: float | None
    applied_sample_count: int


def _runs(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        a = i
        while i < len(mask) and mask[i]:
            i += 1
        yield a, i


def _midi_float(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def _score_field(audio, sr, rows):
    scored=[]
    for r in rows:
        if not acoustically_admissible(r):
            continue
        if not range_admitted(r):
            continue
        vals=[]
        for window_ms in WINDOWS_MS:
            spec, _reason = _spectrum(audio, sr, float(r.time_s), window_ms)
            if spec is None:
                continue
            s=_harmonic_score(spec, float(r.representative_hz))
            if s is not None:
                vals.append(s)
        if len(vals) < 2:
            continue
        score=float(np.median([v['score'] for v in vals]))
        support=float(np.median([v['support'] for v in vals]))
        scored.append((score, support, float(r.representative_hz), int(r.note_group_midi), r.group_id))
    scored.sort(key=lambda x:(-x[0], x[2]))
    return scored


def _review_provisional(audio, sr, evidence_rows, bench_row):
    """V12 expert review, retained unchanged.

    Only provisional champions with no usable interval-juror reference may be
    reconsidered by the existing Harmonic Detective.
    """
    if getattr(bench_row, 'status', '') != 'provisional_tournament_champion':
        return None
    winner_gid=getattr(bench_row, 'winner_group_id', None)
    winner_rows=[r for r in evidence_rows if r.group_id==winner_gid]
    if not winner_rows:
        return None
    wr=winner_rows[0]
    if wr.interval_previous_component is not None or wr.interval_following_component is not None:
        return None

    scored=_score_field(audio, sr, evidence_rows)
    if not scored:
        return None
    best=scored[0]
    runner=scored[1] if len(scored)>1 else None
    best_score,best_support,best_hz,best_midi,_=best
    runner_score=None if runner is None else runner[0]
    margin=None if runner is None else best_score-runner_score

    ordinary=(
        best_score >= MIN_SCORE and best_support >= MIN_SUPPORT and
        (runner is None or (margin is not None and margin >= MIN_MARGIN))
    )
    high_conf=(
        runner is not None and best_score >= HIGH_CONF_MIN_SCORE and
        best_support >= HIGH_CONF_MIN_SUPPORT and margin is not None and
        margin >= HIGH_CONF_MIN_MARGIN and len(WINDOWS_MS) >= HIGH_CONF_MIN_VALID_WINDOWS
    )
    if not (ordinary or high_conf):
        return None
    return best_hz,best_midi,best[4],'harmonic_detective_reacquisition_review',best_score,runner_score,margin


def _frame_proposal(fi, *, by_final, by_bench, by_evidence, audio, sr, is_reacquired, interval_authority):
    """Return V12's best already-existing frame owner proposal."""
    final=by_final.get(fi)
    bench=by_bench.get(fi)
    jury_hz=None
    hz=None
    midi=None
    source=''
    group_id=None
    best_score=runner_score=margin=None

    if final is not None and getattr(final,'status','')=='resolved' and getattr(final,'winner_hz',None) is not None:
        jury_hz=float(final.winner_hz)
        hz=jury_hz
        midi=(None if getattr(final,'winner_midi',None) is None else int(final.winner_midi))
        source='existing_final_adjudication'
        group_id=getattr(final,'winner_group_id',None)

    if bench is not None and is_reacquired:
        rev=_review_provisional(audio, sr, by_evidence.get(fi,[]), bench)
        if rev is not None:
            hz,midi,group_id,source,best_score,runner_score,margin=rev

    if hz is not None and group_rejected(interval_authority, fi, group_id):
        v=preferred_for_rejected(interval_authority, fi, group_id)
        if v is not None:
            hz=float(v.preferred_hz)
            midi=int(v.preferred_midi)
            group_id=v.preferred_group_id
            source='existing_interval_verdict_preserved'

    return {
        'jury_hz': jury_hz, 'hz': hz, 'midi': midi, 'group_id': group_id, 'source': source,
        'best_score': best_score, 'runner_score': runner_score, 'margin': margin,
    }


def _interval_cost(hz: float, t: float, ref_hz: float, ref_t: float) -> float:
    return interval_transition_penalty(hz, t, ref_hz, ref_t)


def _candidate_rows(rows):
    out=[]
    for r in rows:
        if not acoustically_admissible(r):
            continue
        if not range_admitted(r):
            continue
        hz=float(r.representative_hz)
        if math.isfinite(hz) and hz > 0:
            out.append(r)
    return out


def _refine_island_proposals(*, frames, proposals, by_evidence, audio, sr, interval_authority, left_anchor=None, right_anchor=None):
    """Choose one globally coherent candidate path for a recovery island.

    V14 solves the whole recovery island at once.  Candidate states are only
    already-existing, in-range, acoustically admissible contestants.  A NULL
    state is also allowed: if a frame has no candidate compatible with the
    island's coherent register, that frame is left invalid instead of forcing
    a contradictory octave owner or inventing a replacement F0.

    Objective is lexicographic, with no tunable weights:
      1. minimise the EXISTING cubic interval-juror penalty across retained
         adjacent owners;
      2. among equal-penalty paths, retain as many island frames as possible;
      3. among those, minimise changes from existing adjudicated owners;
      4. among those, maximise EXISTING Harmonic-Detective acoustic score.

    A real legato/note change remains available whenever the existing interval
    juror assigns it zero penalty.  No octave bucket, bend cap, song-specific
    pitch, GT target, or new numeric threshold is introduced.
    """
    if len(frames) < 2:
        return proposals

    times={fi: float(by_evidence[fi][0].time_s) for fi in frames if by_evidence.get(fi)}
    ordered=[fi for fi in frames if fi in times]
    if len(ordered) < 2:
        return proposals

    fields={}
    for fi in ordered:
        rows=[r for r in _candidate_rows(by_evidence.get(fi,[])) if not group_rejected(interval_authority, fi, getattr(r, "group_id", None))]
        scored=_score_field(audio,sr,rows) if rows else []
        acoustic={round(float(hz),6): float(score) for score,_support,hz,_midi,_gid in scored}
        cand=[]
        for r in rows:
            hz=float(r.representative_hz)
            if any(abs(1200.0*math.log2(hz/c['hz'])) < 0.5 for c in cand):
                continue
            cand.append({
                'hz': hz,
                'midi': int(r.note_group_midi),
                'acoustic': acoustic.get(round(hz,6), -math.inf),
                'source': 'island_global_candidate',
            })

        cur=proposals.get(fi)
        if cur and cur.get('hz') is not None:
            hz=float(cur['hz'])
            if math.isfinite(hz) and hz>0 and not any(abs(1200.0*math.log2(hz/c['hz'])) < 0.5 for c in cand):
                cand.append({
                    'hz': hz,
                    'midi': int(cur['midi']) if cur.get('midi') is not None else int(round(_midi_float(hz))),
                    'acoustic': acoustic.get(round(hz,6), -math.inf),
                    'source': cur.get('source','existing_owner'),
                })

        # NULL is a real semantic outcome: "none of the existing candidates is
        # compatible with the coherent recovery island".  It never fabricates
        # pitch and it is considered only by the global path solver.
        cand.append({'hz': None, 'midi': None, 'acoustic': 0.0, 'source': 'island_unowned'})
        fields[fi]=cand

    def owner_change_cost(fi, hz):
        cur=proposals.get(fi)
        if hz is None:
            # Dropping an existing owner is a change, but only after interval
            # coherence and retained-frame count have already been optimised.
            return 1 if cur and cur.get('hz') is not None else 0
        if not cur or cur.get('hz') is None:
            return 1
        cents=abs(1200.0*math.log2(float(hz)/float(cur['hz'])))
        return 0 if cents < 2.0 else 1

    # State tuple:
    # (interval_penalty, negative_retained_count, owner_changes,
    #  negative_acoustic_score_sum, predecessor_index, last_retained_index)
    dp=[]
    first=ordered[0]
    first_states=[]
    for ci,c in enumerate(fields[first]):
        retained=0 if c['hz'] is None else 1
        ac=c['acoustic'] if (c['hz'] is not None and math.isfinite(c['acoustic'])) else 0.0
        anchor_cost=0.0
        if retained and left_anchor is not None:
            anchor_hz,anchor_t=left_anchor
            anchor_cost=_interval_cost(c['hz'],times[first],anchor_hz,anchor_t)
        first_states.append((anchor_cost,-retained,owner_change_cost(first,c['hz']),-ac,None,0 if retained else None))
    dp.append(first_states)

    for k in range(1,len(ordered)):
        fi=ordered[k]
        states=[]
        for ci,c in enumerate(fields[fi]):
            retained=0 if c['hz'] is None else 1
            ac=c['acoustic'] if (c['hz'] is not None and math.isfinite(c['acoustic'])) else 0.0
            best=None
            for pj,pc in enumerate(fields[ordered[k-1]]):
                prev=dp[k-1][pj]
                tr=0.0
                last_idx=prev[5]
                new_last=last_idx
                if c['hz'] is not None:
                    if last_idx is not None:
                        prev_fi=ordered[last_idx]
                        prev_c_index=chosen_state_index(dp, fields, ordered, k-1, pj, last_idx)
                        prev_c = fields[prev_fi][prev_c_index] if prev_c_index is not None else None
                        if prev_c is not None and prev_c['hz'] is not None:
                            tr=_interval_cost(c['hz'],times[fi],prev_c['hz'],times[prev_fi])
                    new_last=k
                val=(
                    prev[0]+tr,
                    prev[1]-retained,
                    prev[2]+owner_change_cost(fi,c['hz']),
                    prev[3]-ac,
                    pj,
                    new_last,
                )
                if best is None or val[:4] < best[:4]:
                    best=val
            states.append(best)
        dp.append(states)

    def final_key(j):
        state=dp[-1][j]
        penalty=state[0]
        if right_anchor is not None and state[5] is not None:
            last_k=state[5]
            last_fi=ordered[last_k]
            last_ci=chosen_state_index(dp, fields, ordered, len(ordered)-1, j, last_k)
            last_c = fields[last_fi][last_ci] if last_ci is not None else None
            if last_c is not None and last_c['hz'] is not None:
                anchor_hz,anchor_t=right_anchor
                penalty += _interval_cost(last_c['hz'],times[last_fi],anchor_hz,anchor_t)
        return (penalty,state[1],state[2],state[3])

    last_idx=min(range(len(dp[-1])), key=final_key)
    chosen=[None]*len(ordered)
    j=last_idx
    for k in range(len(ordered)-1,-1,-1):
        chosen[k]=j
        j=dp[k][j][4] if k>0 else None

    for fi,ci in zip(ordered,chosen):
        c=fields[fi][ci]
        p=proposals.get(fi)
        if p is None:
            continue
        if c['hz'] is None:
            p['hz']=None
            p['midi']=None
            p['source']='island_global_interval_reject'
            p['drop_owner']=True
            continue
        current=p.get('hz')
        if current is not None:
            cents=abs(1200.0*math.log2(float(c['hz'])/float(current)))
            if cents < 2.0:
                continue
        p['hz']=float(c['hz'])
        p['midi']=int(c['midi'])
        p['source']='island_global_interval_path'
        p['drop_owner']=False
    return proposals


def chosen_state_index(dp, fields, ordered, k, state_index, target_k):
    """Backtrack from one DP state to the candidate index at target_k.

    DP predecessor indices are local to each frame's candidate list.  Keep them
    frame-local and fail closed if a malformed path is encountered rather than
    indexing a different frame's candidate field.
    """
    if not (0 <= target_k <= k < len(dp)):
        return None
    j=state_index
    kk=k
    while kk>target_k:
        if j is None or not (0 <= j < len(dp[kk])):
            return None
        j=dp[kk][j][4]
        kk-=1
    if j is None or not (0 <= target_k < len(ordered)):
        return None
    fi=ordered[target_k]
    if not (0 <= j < len(fields.get(fi, []))):
        return None
    return j

def apply_reacquisition_adjudication_feedback(
    *,
    offline_validity,
    sample_times_s,
    sample_frame_index,
    audio,
    sample_rate,
    evidence_rows,
    bench_rows,
    pair_rows,
    final_adjudication,
):
    """Project adjudicated ownership over whole recovery islands consistently."""
    by_evidence=defaultdict(list)
    for r in evidence_rows:
        by_evidence[int(r.frame_index)].append(r)
    by_bench={int(r.frame_index):r for r in bench_rows}
    by_final={int(r.frame_index):r for r in final_adjudication}
    interval_authority=build_interval_authority(pair_rows, evidence_rows)

    fp_reason=np.asarray(offline_validity.first_pass_reason, dtype=object)
    corr_reason=np.asarray(offline_validity.correction_reason, dtype=object)
    valid=offline_validity.valid
    clean=offline_validity.clean_f0_hz
    reason=offline_validity.reason
    frame_index=np.asarray(sample_frame_index)
    times=np.asarray(sample_times_s,dtype=float)

    # V13 island substrate: include the waiting/recovery neighbourhood, not only
    # the individual sample that happened to receive REACQUIRED_*.
    fp_island=np.isin(fp_reason, [
        'WAITING_FOR_REACQUISITION',
        'REACQUIRED_FUTURE_CONFIRMED',
    ])
    corr_island=np.isin(corr_reason, [
        'BACKWARD_RESCUED','FORWARD_RESCUED','ORNAMENT_RESCUED',
        'RANGE_ZERO_VETO','ADJUDICATION_REACQUISITION_REPLACEMENT',
    ])
    island_mask=fp_island | corr_island
    audit=[]

    for isl_a,isl_b in _runs(island_mask):
        isl_idx=np.arange(isl_a,isl_b,dtype=int)
        frames=[]
        for fi in frame_index[isl_idx]:
            fi=int(fi)
            if fi not in frames:
                frames.append(fi)

        proposals={}
        for fi in frames:
            idx=np.flatnonzero((frame_index==fi) & island_mask)
            reacq=bool(np.any(fp_reason[idx]=='REACQUIRED_FUTURE_CONFIRMED')) if idx.size else False
            p=_frame_proposal(
                fi, by_final=by_final, by_bench=by_bench, by_evidence=by_evidence,
                audio=audio, sr=sample_rate, is_reacquired=reacq, interval_authority=interval_authority,
            )
            if p['hz'] is not None:
                proposals[fi]=p

        # Immediate trusted neighbors may anchor the recovery island.  This is
        # not a cross-silence prior: an anchor is admitted only when the sample
        # directly adjacent to the island is already corrected-valid and lies
        # outside the recovery mask.  Future-confirmed reacquisition therefore
        # must reconnect plausibly to the very trajectory that confirmed it.
        left_anchor=None
        if isl_a>0 and bool(valid[isl_a-1]) and not bool(island_mask[isl_a-1]) and math.isfinite(float(clean[isl_a-1])):
            left_anchor=(float(clean[isl_a-1]),float(times[isl_a-1]))
        right_anchor=None
        if isl_b<len(valid) and bool(valid[isl_b]) and not bool(island_mask[isl_b]) and math.isfinite(float(clean[isl_b])):
            right_anchor=(float(clean[isl_b]),float(times[isl_b]))

        proposals=_refine_island_proposals(
            frames=frames, proposals=proposals, by_evidence=by_evidence,
            audio=audio, sr=sample_rate, interval_authority=interval_authority,
            left_anchor=left_anchor, right_anchor=right_anchor,
        )

        for fi in frames:
            p=proposals.get(fi)
            if not p:
                continue
            idx=np.flatnonzero((frame_index==fi) & island_mask)
            if idx.size==0:
                continue

            if p.get('drop_owner'):
                original_vals=np.asarray(clean[idx],dtype=float)
                finite=original_vals[np.isfinite(original_vals)]
                original_hz=float(np.median(finite)) if finite.size else None
                original_reason=str(fp_reason[idx[0]]) if fp_reason[idx[0]] else str(corr_reason[idx[0]])
                valid[idx]=False
                clean[idx]=np.nan
                reason[idx]='ADJUDICATION_REACQUISITION_ISLAND_REJECTED'
                corr_reason[idx]='ADJUDICATION_REACQUISITION_ISLAND_REJECTED'
                audit.append(ReacquisitionFeedbackRow(
                    frame_index=fi, time_s=float(np.median(times[idx])),
                    original_hz=original_hz, original_reason=original_reason,
                    jury_hz=p.get('jury_hz'), reviewed_hz=None, reviewed_midi=None,
                    review_source=p.get('source',''), detective_best_score=p.get('best_score'),
                    detective_runner_score=p.get('runner_score'), detective_margin=p.get('margin'),
                    applied_sample_count=int(idx.size),
                ))
                continue
            if p.get('hz') is None:
                continue
            reviewed_hz=float(p['hz'])
            if not math.isfinite(reviewed_hz) or reviewed_hz<=0:
                continue

            original_vals=np.asarray(clean[idx],dtype=float)
            finite=original_vals[np.isfinite(original_vals)]
            original_hz=float(np.median(finite)) if finite.size else None
            original_reason=str(fp_reason[idx[0]]) if fp_reason[idx[0]] else str(corr_reason[idx[0]])
            differs=(original_hz is None or abs(1200.0*math.log2(reviewed_hz/original_hz)) >= 2.0)
            hole=not np.all(valid[idx])
            if not differs and not hole:
                continue

            # Do not manufacture pitch into arbitrary waiting samples unless an
            # adjudicated owner exists for that frame.  At this point it does.
            valid[idx]=True
            clean[idx]=reviewed_hz
            reason[idx]='ADJUDICATION_REACQUISITION_ISLAND_REPLACEMENT'
            corr_reason[idx]='ADJUDICATION_REACQUISITION_ISLAND_REPLACEMENT'
            audit.append(ReacquisitionFeedbackRow(
                frame_index=fi,
                time_s=float(np.median(times[idx])),
                original_hz=original_hz,
                original_reason=original_reason,
                jury_hz=p.get('jury_hz'),
                reviewed_hz=reviewed_hz,
                reviewed_midi=p.get('midi'),
                review_source=p.get('source',''),
                detective_best_score=p.get('best_score'),
                detective_runner_score=p.get('runner_score'),
                detective_margin=p.get('margin'),
                applied_sample_count=int(idx.size),
            ))

    return tuple(audit)
