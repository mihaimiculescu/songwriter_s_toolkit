from __future__ import annotations

"""Bridge from the pitch-candidate field to the V2 juror bench.

V22 range calibration is restored here.  The per-file observed-performance
range is derived from independently trusted V2 corrected-valid clean F0
references, not from the candidate field itself.  The historical geometry is
kept unchanged: nearest-note anchors, +/-2 semitone plateau extension, then
2-semitone half-cosine shoulders to zero.
"""

from dataclasses import dataclass
from collections import defaultdict
from bisect import bisect_left, bisect_right
import math
import numpy as np

from .trajectory_resolver import vocal_transition_penalty, vocal_required_time_ms


def _median(vals):
    vals=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def _nearest_midi(hz: float) -> int:
    return int(math.floor(69.0 + 12.0 * math.log2(hz / 440.0) + 0.5))


RANGE_LO_Q = 0.02
RANGE_HI_Q = 0.98
MIN_RANGE_ANCHORS = 40
RANGE_PLATEAU_EXTENSION_ST = 2.0
RANGE_SHOULDER_ST = 2.0
INTERVAL_MAX_ADJACENT_GAP_S = 0.050


def _juror_quadratic_transition_penalty(interval_st: float, available_ms: float) -> float:
    """Juror-only convex refinement of the historical transition prior.

    Upstream ECKF/validity/initialization logic keeps the historical
    ``vocal_transition_penalty`` unchanged.  Only the interval juror gets the
    steeper severity requested for grossly impossible leap/time combinations.

    Mild violations remain close to the historical penalty.  Once required
    time greatly exceeds available time, severity rises quadratically via
    ``1 + demand_excess**3``.
    """
    base = float(vocal_transition_penalty(interval_st, available_ms))
    if base <= 0.0:
        return 0.0
    required = float(vocal_required_time_ms(interval_st))
    available = max(float(available_ms), 1.0)
    if required <= available or required <= 0.0:
        return base
    demand_excess = max(0.0, required / available - 1.0)
    return float(base * (1.0 + demand_excess * demand_excess * demand_excess))


@dataclass(frozen=True)
class RangeCalibration:
    available: bool
    anchor_count: int
    source: str
    low_hz: float | None
    high_hz: float | None
    low_anchor_midi: int | None
    high_anchor_midi: int | None
    plateau_low_midi: float | None
    plateau_high_midi: float | None
    zero_low_midi: float | None
    zero_high_midi: float | None
    lo_quantile: float
    hi_quantile: float
    plateau_extension_st: float
    shoulder_st: float


def derive_v22_range_calibration(supported_reference_hz, source: str = "v13_style_unique_local_support_v2") -> RangeCalibration:
    vals=np.asarray([float(v) for v in supported_reference_hz
                     if v is not None and math.isfinite(float(v)) and float(v)>0],
                    dtype=np.float64)
    n=int(vals.size)
    if n < MIN_RANGE_ANCHORS:
        return RangeCalibration(False,n,source,None,None,None,None,
                                None,None,None,None,RANGE_LO_Q,RANGE_HI_Q,
                                RANGE_PLATEAU_EXTENSION_ST,RANGE_SHOULDER_ST)
    lo=float(np.quantile(vals,RANGE_LO_Q))
    hi=float(np.quantile(vals,RANGE_HI_Q))
    lo_note=_nearest_midi(lo)
    hi_note=_nearest_midi(hi)
    plateau_lo=float(lo_note)-RANGE_PLATEAU_EXTENSION_ST
    plateau_hi=float(hi_note)+RANGE_PLATEAU_EXTENSION_ST
    return RangeCalibration(True,n,source,lo,hi,lo_note,hi_note,
                            plateau_lo,plateau_hi,
                            plateau_lo-RANGE_SHOULDER_ST,
                            plateau_hi+RANGE_SHOULDER_ST,
                            RANGE_LO_Q,RANGE_HI_Q,
                            RANGE_PLATEAU_EXTENSION_ST,RANGE_SHOULDER_ST)


def _range_confidence(hz: float, calibration: RangeCalibration):
    if not calibration.available or hz <= 0:
        return None
    lo=float(calibration.plateau_low_midi)
    hi=float(calibration.plateau_high_midi)
    shoulder=float(calibration.shoulder_st)
    note=69.0+12.0*math.log2(hz/440.0)
    dist=(lo-note) if note < lo else ((note-hi) if note > hi else 0.0)
    if dist <= 0: return 1.0
    if dist >= shoulder: return 0.0
    return 0.5*(1.0+math.cos(math.pi*dist/shoulder))


@dataclass(frozen=True)
class JurorEvidenceRow:
    frame_index:int
    time_s:float
    group_id:str
    note_group_midi:int
    within_49c:bool
    cents_from_note:float
    representative_hz:float
    representative_source:str
    candidate_count_in_group:int
    acf_median:float|None
    cmndf_median:float|None
    harmonic_count_median:float|None
    component_amplitude_median:float|None
    range_confidence:float|None
    interval_previous_component:float|None
    interval_following_component:float|None
    interval_previous_reference_time_s:float|None
    interval_previous_reference_hz:float|None
    interval_following_reference_time_s:float|None
    interval_following_reference_hz:float|None
    interval_previous_blocked_by_interruption:bool
    interval_following_blocked_by_interruption:bool
    temporal_prev_same_note:bool
    temporal_next_same_note:bool
    temporal_persistence_count:int
    selected_by_initializer:bool


def _historical_interval_context(range_reference_rows):
    """Return trusted references and explicit V13 interruption timestamps.

    Historical V22 policy:
      * references are only rows marked as
        acoustic_period_supported_source_unverified + diagnostic_settled +
        usable_as_next_reference;
      * an explicit interruption is any V13 row with low_energy_flag OR
        decaying_flag; unresolved F0 alone is NOT treated as silence.
    """
    refs=[]
    interruptions=[]
    for row in (range_reference_rows or ()): 
        try:
            t=float(row.time_s)
        except Exception:
            continue
        if bool(getattr(row, "low_energy_flag", False)) or bool(getattr(row, "decaying_flag", False)):
            interruptions.append(t)
        if (
            getattr(row, "evidence_status", None) == "acoustic_period_supported_source_unverified"
            and bool(getattr(row, "diagnostic_settled", False))
            and bool(getattr(row, "usable_as_next_reference", False))
        ):
            hz=getattr(row, "selected_measured_hz", None)
            try:
                hz=float(hz)
            except Exception:
                continue
            if math.isfinite(hz) and hz > 0.0:
                refs.append((t,hz))
    refs.sort()
    return refs, sorted(set(interruptions))


def _adjacent_interval_references(refs, interruptions, t, max_gap_s=INTERVAL_MAX_ADJACENT_GAP_S):
    """Historical independent past/future lookup with interruption barrier."""
    if not refs:
        return None, None, False, False
    times=[x[0] for x in refs]
    i=bisect_left(times, t)
    j=bisect_right(times, t)
    prev=refs[i-1] if i>0 and 0.0 < t-refs[i-1][0] <= max_gap_s else None
    nxt=refs[j] if j<len(refs) and 0.0 < refs[j][0]-t <= max_gap_s else None
    prev_blocked=False
    next_blocked=False
    if prev is not None:
        k=bisect_right(interruptions, prev[0])
        if k < len(interruptions) and interruptions[k] <= t:
            prev=None
            prev_blocked=True
    if nxt is not None:
        k=bisect_left(interruptions, t)
        if k < len(interruptions) and interruptions[k] < nxt[0]:
            nxt=None
            next_blocked=True
    return prev,nxt,prev_blocked,next_blocked


def build_juror_evidence(pitch_candidates, supported_reference_hz, range_reference_rows=None) -> tuple[tuple[JurorEvidenceRow,...], RangeCalibration]:
    rows=list(pitch_candidates)
    if not rows:
        return tuple(), derive_v22_range_calibration(supported_reference_hz)

    calibration=derive_v22_range_calibration(supported_reference_hz)

    by_frame=defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_index)].append(r)
    frame_ids=sorted(by_frame)
    idxpos={f:i for i,f in enumerate(frame_ids)}

    # V12 refined same-note policy:
    #   * STRICT abs(cents) < 49.0 groups by nearest equal-tempered note.
    #   * candidates at >=49 cents are NOT silently assigned and NOT discarded;
    #     each remains a singleton contestant.
    #   * within a same-note group, the representative is the strongest ACF
    #     candidate; deterministic tie-breakers only make source export stable.
    reps={}
    for fi,candidates in by_frame.items():
        groups=defaultdict(list)
        outside_serial=0
        for r in candidates:
            cents=float(r.cents_from_nearest_midi)
            midi=int(r.nearest_midi)
            if math.isfinite(cents) and abs(cents) < 49.0:
                key=f"note:{midi}"
            else:
                key=f"outside:{outside_serial}"
                outside_serial += 1
            groups[key].append(r)
        frame_reps={}
        for gid,grp in groups.items():
            # Historical V12 final refinement: representative by ACF support.
            rep=max(grp,key=lambda r:(float(r.measured_period_acf),
                                      -float(r.measured_period_cmndf),
                                      r.source=="waveform_periodicity",
                                      -r.candidate_hz))
            frame_reps[gid]=(rep,grp)
        reps[fi]=frame_reps

    # Historical V22 interval context is based on the mature V13 trusted
    # reference stream, NOT merely the initializer-selected candidate in the
    # immediately adjacent candidate-bearing frame.
    interval_refs, interruption_times = _historical_interval_context(range_reference_rows)

    out=[]
    for fi in frame_ids:
        pos=idxpos[fi]
        prev_f=frame_ids[pos-1] if pos>0 else None
        next_f=frame_ids[pos+1] if pos+1<len(frame_ids) else None
        for gid,(rep,grp) in sorted(reps[fi].items(),key=lambda kv:(int(kv[1][0].nearest_midi),kv[0])):
            midi=int(rep.nearest_midi)
            cents=float(rep.cents_from_nearest_midi)
            within=bool(math.isfinite(cents) and abs(cents)<49.0 and gid.startswith("note:"))

            # Persistence is note-group persistence only for genuine <49c note groups.
            prev_same=bool(within and prev_f is not None and f"note:{midi}" in reps.get(prev_f,{}))
            next_same=bool(within and next_f is not None and f"note:{midi}" in reps.get(next_f,{}))

            prev_comp=next_comp=None
            prev_ref,next_ref,prev_blocked,next_blocked = _adjacent_interval_references(
                interval_refs, interruption_times, float(rep.time_s)
            )
            if prev_ref is not None:
                rt,rhz=prev_ref
                dt=max(0.0,(float(rep.time_s)-float(rt))*1000.0)
                if dt>0:
                    penalty=float(_juror_quadratic_transition_penalty(12.0*math.log2(float(rep.candidate_hz)/float(rhz)),dt))
                    prev_comp=-min(1.0,max(0.0,penalty/2.5))
            if next_ref is not None:
                rt,rhz=next_ref
                dt=max(0.0,(float(rt)-float(rep.time_s))*1000.0)
                if dt>0:
                    penalty=float(_juror_quadratic_transition_penalty(12.0*math.log2(float(rep.candidate_hz)/float(rhz)),dt))
                    next_comp=-min(1.0,max(0.0,penalty/2.5))

            out.append(JurorEvidenceRow(
                frame_index=fi,time_s=float(rep.time_s),group_id=gid,
                note_group_midi=midi,within_49c=within,cents_from_note=cents,
                representative_hz=float(rep.candidate_hz),representative_source=rep.source,
                candidate_count_in_group=len(grp),
                acf_median=_median([x.measured_period_acf for x in grp]),
                cmndf_median=_median([x.measured_period_cmndf for x in grp]),
                harmonic_count_median=_median([x.spectral_harmonic_count for x in grp]),
                component_amplitude_median=_median([x.measured_component_amplitude for x in grp]),
                range_confidence=_range_confidence(rep.candidate_hz,calibration),
                interval_previous_component=prev_comp,interval_following_component=next_comp,
                interval_previous_reference_time_s=(None if prev_ref is None else float(prev_ref[0])),
                interval_previous_reference_hz=(None if prev_ref is None else float(prev_ref[1])),
                interval_following_reference_time_s=(None if next_ref is None else float(next_ref[0])),
                interval_following_reference_hz=(None if next_ref is None else float(next_ref[1])),
                interval_previous_blocked_by_interruption=bool(prev_blocked),
                interval_following_blocked_by_interruption=bool(next_blocked),
                temporal_prev_same_note=prev_same,temporal_next_same_note=next_same,
                temporal_persistence_count=int(prev_same)+1+int(next_same),
                selected_by_initializer=any(x.selected_by_initializer for x in grp),
            ))
    return tuple(out),calibration

