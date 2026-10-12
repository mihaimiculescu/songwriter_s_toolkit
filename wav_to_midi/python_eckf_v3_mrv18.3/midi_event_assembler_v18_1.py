from __future__ import annotations

"""MIDI Renderer V18.1 -- calibrated acoustic adjudication.

V18.1 is a calibration pass on top of V18, not a new ownership architecture.
It keeps V17.1/V18 ownership and zero-unowned-valid guarantees, then applies
five narrow post-calibrations:

  * weak-start primary-onset trimming for long breath/noise backdating missed by
    V18's accepted-attack-only route;
  * weak short trusted-island heads folded into the following pitch curve;
  * terminal-decay island collapse when a false restart is envelope-decay shaped;
  * decay-aware subharmonic octave correction;
  * fast alternating PB residual compression around a local linear trend.

It also restores only acoustically certified same-pitch/repetition attacks that
V16's intentionally conservative bilateral restart gate can suppress.

The pitch-modulation contract is explicit: the local trend is the least-squares
line over +/-75 ms.  "Fast" means an alternating residual-extrema run with at
least four extrema, adjacent opposite-sign extrema 30--130 ms apart (roughly
3.8--16.7 Hz full-cycle), and residual excursion <=1.2 semitones.  Only that
residual is multiplied by 0.90; the trend itself is unchanged.
"""

from copy import deepcopy
from pathlib import Path
import math
import numpy as np

from .midi_event_assembler_v18 import (
    AssemblyResult, MusicalEvent, AttackDecision,
    assemble_events as assemble_events_v18,
    _WavProbe, _finite_curve, _recompute_bend, _merge_curves,
    _physical_reattack_proven, _moderate_primary_candidates,
    st_to_hz,
)
from .midi_event_assembler_v16_1 import _physical_reattack_evidence
from .midi_event_assembler_v17_1 import _reclean_monophonic
from .midi_pitch_evidence_v10 import cents_between_hz, hz_to_st

# ---------------------------------------------------------------------------
# V18.1 calibrated constants
# ---------------------------------------------------------------------------

WEAK_BACKDATE_START_DBFS = -45.0
WEAK_BACKDATE_MIN_LAG_S = 0.070
WEAK_BACKDATE_PRIMARY_LOOKBACK_S = 0.125
WEAK_BACKDATE_PRIMARY_LOOKAHEAD_S = 0.025
WEAK_BACKDATE_MIN_ACF = 0.55
WEAK_BACKDATE_MAX_CMNDF = 0.40
WEAK_BACKDATE_MAX_ACF_CMNDF_CENTS = 120.0
WEAK_BACKDATE_MAX_FAMILY_GAP_ST = 1.50
WEAK_BACKDATE_MIN_LEVEL_GAIN_DB = 8.0

WEAK_ISLAND_MAX_S = 0.055
WEAK_ISLAND_START_DBFS = -45.0
WEAK_ISLAND_MAX_GAP_NEXT_S = 0.015
WEAK_ISLAND_MAX_NEXT_GAP_ST = 7.0
WEAK_ISLAND_MIN_NEXT_GAIN_DB = 8.0

TERMINAL_DECAY_MAX_S = 0.140
TERMINAL_DECAY_MAX_PREV_GAP_S = 0.080
TERMINAL_DECAY_MIN_NEXT_GAP_S = 0.150
TERMINAL_DECAY_MIN_LEFT_DEPTH_DB = 7.0
TERMINAL_DECAY_MAX_RIGHT_DEPTH_DB = 1.5
TERMINAL_DECAY_MIN_RMS_FALL_DB = 7.0
TERMINAL_DECAY_MAX_PITCH_GAP_ST = 6.0

# Repetition / ostinato recovery.  This is intentionally stronger than merely
# lowering V16's 5/5 dB bilateral restart threshold globally.
REPEAT_MIN_ATTACK_SCORE = 30.0
REPEAT_MIN_LEFT_DEPTH_DB = 1.8
REPEAT_MIN_RIGHT_DEPTH_DB = 6.0
REPEAT_MAX_PRE_POST_PITCH_GAP_ST = 0.80
REPEAT_PRE_PITCH_WINDOW_S = 0.060
REPEAT_POST_PITCH_WINDOW_S = 0.075
REPEAT_INNER_GUARD_S = 0.060
REPEAT_BRACKET_MIN_S = 0.100
REPEAT_BRACKET_MAX_S = 0.350
REPEAT_BRACKET_MAX_RATIO = 2.05
REPEAT_DEDUPE_S = 0.055
REPEAT_SEQUENCE_MIN_COUNT = 3
REPEAT_SEQUENCE_MIN_INTERVAL_S = 0.080
REPEAT_SEQUENCE_MAX_INTERVAL_S = 0.250
REPEAT_SEQUENCE_MAX_INTERVAL_RATIO = 1.25

# Decaying subharmonic: ordinary V18 certificate remains untouched; this only
# catches an isolated low family whose head strongly supports 2*f0 before the
# tail loses periodicity while the surrounding context also favors octave-up.
DECAY_SUBHARMONIC_MIN_EVENT_S = 0.070
DECAY_SUBHARMONIC_MAX_EVENT_S = 0.160
DECAY_SUBHARMONIC_CONTEXT_IMPROVEMENT_ST = 8.0
DECAY_SUBHARMONIC_MAX_NEIGHBOR_GAP_ST = 5.0
DECAY_SUBHARMONIC_HEAD1_DB = 30.0
DECAY_SUBHARMONIC_HEAD2_DB = 18.0
DECAY_SUBHARMONIC_MIN_RMS_FALL_DB = 12.0

# Frequency-tremolo calibration.
FAST_PB_TREND_HALF_WINDOW_S = 0.075
FAST_PB_MIN_RESIDUAL_ST = 0.060
FAST_PB_MAX_RESIDUAL_ST = 1.20
FAST_PB_MIN_HALF_CYCLE_S = 0.030
FAST_PB_MAX_HALF_CYCLE_S = 0.130
FAST_PB_MIN_EXTREMA = 4
FAST_PB_COMPRESSION = 0.90
FAST_PB_TAPER_S = 0.020


def _interp_pitch(event: MusicalEvent, t_s: float) -> float:
    t, p = _finite_curve(event)
    if not len(t):
        return float(event.base_st)
    return float(np.interp(float(t_s), t, p))


def _trim_weak_backdated_starts(result: AssemblyResult, probe: _WavProbe) -> int:
    """Trim weak starts using primary periodicity near the frozen trusted anchor.

    This is the fallback V18 intentionally lacked: no later accepted attack is
    required, but the early start must be genuinely weak and a moderate
    ACF/CMNDF-agreeing primary family near the trusted anchor must agree with the
    event's own family and be materially louder than the borrowed start.
    """
    changed = 0
    for e in result.events:
        anchor = float(e.expression_anchor_s)
        if not math.isfinite(anchor):
            continue
        if anchor - e.start_s < WEAK_BACKDATE_MIN_LAG_S:
            continue
        start_level = probe.rms_db(e.start_s)
        if start_level > WEAK_BACKDATE_START_DBFS:
            continue

        # A later accepted bilateral WAV reattack of the same family is an
        # even stronger replacement onset (canonical Damelo ~106.60).
        reattack_candidates = []
        for d in result.attack_decisions:
            if not d.accepted or d.source != "wav_rearticulation":
                continue
            if not (e.start_s + 1e-6 < float(d.time_s) < e.end_s + 1e-9):
                continue
            if d.reattack_status != "bilateral_envelope_restart":
                continue
            if not math.isfinite(d.pitch_anchor_st) or abs(float(d.pitch_anchor_st) - float(e.base_st)) > WEAK_BACKDATE_MAX_FAMILY_GAP_ST:
                continue
            if probe.rms_db(float(d.time_s)) < start_level + WEAK_BACKDATE_MIN_LEVEL_GAIN_DB:
                continue
            reattack_candidates.append(float(d.time_s))

        lo = max(e.start_s, anchor - WEAK_BACKDATE_PRIMARY_LOOKBACK_S)
        hi = min(e.end_s, anchor + WEAK_BACKDATE_PRIMARY_LOOKAHEAD_S)
        candidates = []
        for r in result.pitch.frame_rows:
            try:
                t = float(r.get("start_time_s", "nan"))
                acf = float(r.get("periodicity_acf_peak", "nan"))
                cm = float(r.get("periodicity_cmndf_minimum", "nan"))
                fa = float(r.get("periodicity_acf_frequency_hz", "nan"))
                fc = float(r.get("periodicity_cmndf_frequency_hz", "nan"))
            except Exception:
                continue
            if not (lo <= t <= hi):
                continue
            if not (math.isfinite(acf) and math.isfinite(cm) and acf >= WEAK_BACKDATE_MIN_ACF and cm <= WEAK_BACKDATE_MAX_CMNDF):
                continue
            if not (math.isfinite(fa) and fa > 0 and math.isfinite(fc) and fc > 0):
                continue
            if cents_between_hz(fa, fc) > WEAK_BACKDATE_MAX_ACF_CMNDF_CENTS:
                continue
            st = hz_to_st(math.sqrt(fa * fc))
            if abs(st - float(e.base_st)) > WEAK_BACKDATE_MAX_FAMILY_GAP_ST:
                continue
            if probe.rms_db(t) < start_level + WEAK_BACKDATE_MIN_LEVEL_GAIN_DB:
                continue
            candidates.append((t, st, acf, cm))
        primary_chosen = min(candidates, key=lambda x: x[0]) if candidates else None
        chosen_time = primary_chosen[0] if primary_chosen is not None else math.inf
        if reattack_candidates:
            chosen_time = min(chosen_time, min(reattack_candidates))
        if not math.isfinite(chosen_time):
            continue
        # If both exist, use the earliest defensible voiced onset.
        if primary_chosen is None or abs(chosen_time - primary_chosen[0]) > 1e-9:
            chosen = (chosen_time, float(e.base_st), math.nan, math.nan)
        else:
            chosen = primary_chosen
        old = float(e.start_s)
        # V17.1's zero-unowned-valid contract outranks acoustic trimming.
        # Never trim past the first frozen corrected-valid point already owned by
        # this event; the acoustic onset may be later by a few milliseconds, but
        # valid V3R9 pitch must remain rendered.
        vm = result.pitch.valid & (result.pitch.t >= old - 1e-9) & (result.pitch.t <= chosen_time + 1e-9)
        if np.any(vm):
            chosen_time = min(chosen_time, float(np.min(result.pitch.t[vm])))
        from .midi_event_assembler_v18 import _trim_event_start
        _trim_event_start(e, chosen_time)
        e.source_kind = f"{e.source_kind}+weak_start_trim_v18_1"
        result.attack_decisions.append(AttackDecision(
            old, False, "weak_start_trim_v18_1",
            "weak_borrowed_start_replaced_by_near_anchor_primary_v18_1",
            pitch_source=e.pitch_source, pitch_anchor_s=chosen[0], pitch_anchor_st=chosen[1],
            continuity_status=f"acf={chosen[2]:.3f};cmndf={chosen[3]:.3f};trimmed_to={chosen_time:.9f}",
        ))
        changed += 1
    result.events = _reclean_monophonic(result.events)
    return changed


def _collapse_weak_short_islands(result: AssemblyResult, probe: _WavProbe) -> int:
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    removed: set[int] = set()
    changed = 0
    for i, e in enumerate(events[:-1]):
        if i in removed:
            continue
        nxt = events[i + 1]
        dur = e.end_s - e.start_s
        if dur > WEAK_ISLAND_MAX_S or "trusted_island_owner" not in e.source_kind:
            continue
        if probe.rms_db(e.start_s) > WEAK_ISLAND_START_DBFS:
            continue
        if nxt.start_s - e.end_s > WEAK_ISLAND_MAX_GAP_NEXT_S:
            continue
        if abs(float(e.base_st) - float(nxt.base_st)) > WEAK_ISLAND_MAX_NEXT_GAP_ST:
            continue
        if probe.rms_db(nxt.start_s) < probe.rms_db(e.start_s) + WEAK_ISLAND_MIN_NEXT_GAIN_DB:
            continue
        # Preserve the real short pitch as a lead-in, but remove its independent
        # Note-On.  This changes articulation, not acoustic pitch content.
        merged = _merge_curves(e, nxt, keep_base="second")
        merged.source_kind = f"{nxt.source_kind}+weak_island_pb_v18_1"
        merged.pitch_source = nxt.pitch_source
        removed.add(i)
        result.attack_decisions.append(AttackDecision(
            e.start_s, False, "weak_short_island_v18_1",
            "weak_short_trusted_island_folded_into_following_pitch_curve_v18_1",
            pitch_source=merged.pitch_source, pitch_anchor_s=e.expression_anchor_s,
            pitch_anchor_st=e.base_st,
        ))
        changed += 1
    result.events = _reclean_monophonic([e for i, e in enumerate(events) if i not in removed])
    return changed


def _collapse_terminal_decay_islands(result: AssemblyResult, probe: _WavProbe) -> int:
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    removed: set[int] = set()
    changed = 0
    for i in range(1, len(events)):
        if i in removed:
            continue
        e = events[i]
        prev = events[i - 1]
        nxt = events[i + 1] if i + 1 < len(events) else None
        dur = e.end_s - e.start_s
        if dur > TERMINAL_DECAY_MAX_S or "trusted_island_owner" not in e.source_kind:
            continue
        if e.start_s - prev.end_s > TERMINAL_DECAY_MAX_PREV_GAP_S:
            continue
        if nxt is not None and nxt.start_s - e.end_s < TERMINAL_DECAY_MIN_NEXT_GAP_S:
            continue
        if abs(float(e.base_st) - float(prev.base_st)) > TERMINAL_DECAY_MAX_PITCH_GAP_ST:
            continue
        peak = result.attack_lane.best_peak(e.start_s - 0.010, e.start_s + 0.010)
        if peak is None:
            continue
        re = _physical_reattack_evidence(result.attack_lane, peak)
        if re.proven or not (math.isfinite(re.left_depth_db) and math.isfinite(re.right_depth_db)):
            continue
        if re.left_depth_db < TERMINAL_DECAY_MIN_LEFT_DEPTH_DB or re.right_depth_db > TERMINAL_DECAY_MAX_RIGHT_DEPTH_DB:
            continue
        mid = 0.5 * (e.start_s + e.end_s)
        if probe.rms_db(mid) - probe.rms_db(e.end_s - min(0.005, 0.1 * dur)) < TERMINAL_DECAY_MIN_RMS_FALL_DB:
            continue
        _merge_curves(prev, e, keep_base="first")
        prev.source_kind = f"{prev.source_kind}+terminal_decay_pb_v18_1"
        removed.add(i)
        result.attack_decisions.append(AttackDecision(
            e.start_s, False, "terminal_decay_island_v18_1",
            "short_decay_island_without_physical_restart_folded_into_tail_v18_1",
            pitch_source=prev.pitch_source, pitch_anchor_s=e.expression_anchor_s,
            pitch_anchor_st=e.base_st,
            continuity_status=f"left_db={re.left_depth_db:.3f};right_db={re.right_depth_db:.3f}",
        ))
        changed += 1
    result.events = _reclean_monophonic([e for i, e in enumerate(events) if i not in removed])
    return changed


def _local_pitch_pre_post(event: MusicalEvent, t_s: float) -> tuple[float, float] | None:
    t, p = _finite_curve(event)
    pre = p[(t >= t_s - REPEAT_PRE_PITCH_WINDOW_S) & (t <= t_s - 0.015)]
    post = p[(t >= t_s + 0.015) & (t <= t_s + REPEAT_POST_PITCH_WINDOW_S)]
    if len(pre) < 2 or len(post) < 2:
        return None
    return float(np.median(pre)), float(np.median(post))


def _slice_event(event: MusicalEvent, start_s: float, end_s: float, *, attack_score: float | None = None) -> MusicalEvent:
    out = deepcopy(event)
    start_s = float(start_s); end_s = float(end_s)
    t, p = _finite_curve(event)
    if len(t):
        p0 = float(np.interp(start_s, t, p)); p1 = float(np.interp(end_s, t, p))
        m = (t > start_s + 1e-9) & (t < end_s - 1e-9)
        out.pitch_times_s = np.concatenate([[start_s], t[m], [end_s]])
        out.pitch_st = np.concatenate([[p0], p[m], [p1]])
    out.start_s = start_s; out.end_s = end_s
    out.expression_anchor_s = start_s
    if attack_score is not None:
        out.attack_score = float(attack_score)
    _recompute_bend(out)
    return out


def _restore_repetition_attacks(result: AssemblyResult) -> int:
    """Restore one-sided repeated attacks without weakening V16 globally.

    Two certificates exist:
      A) a very regular run of >=3 same-pitch right-restart attacks (ostinato);
      B) an isolated candidate with some left dip, strong right restart, and
         strong bracketing attacks on both sides.
    """
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    strong_peaks = [p for p in result.attack_lane.peaks if (p.novelty_z + 0.5 * p.prominence_z) >= REPEAT_MIN_ATTACK_SCORE]
    raw: dict[int, list[tuple[float, float, float, float]]] = {}
    for d in result.attack_decisions:
        if d.accepted or d.source != "wav_attack" or d.reason != "no_defensible_pitch_owner":
            continue
        ts = float(d.time_s)
        idx = next((i for i, e in enumerate(events) if e.start_s + REPEAT_INNER_GUARD_S < ts < e.end_s - REPEAT_INNER_GUARD_S), None)
        if idx is None:
            continue
        e = events[idx]
        peak = result.attack_lane.best_peak(ts - 0.008, ts + 0.008)
        if peak is None:
            continue
        score = float(peak.novelty_z + 0.5 * peak.prominence_z)
        if score < REPEAT_MIN_ATTACK_SCORE:
            continue
        re = _physical_reattack_evidence(result.attack_lane, peak)
        if re.right_depth_db < REPEAT_MIN_RIGHT_DEPTH_DB:
            continue
        pp = _local_pitch_pre_post(e, ts)
        if pp is None or abs(pp[0] - pp[1]) > REPEAT_MAX_PRE_POST_PITCH_GAP_ST:
            continue
        raw.setdefault(idx, []).append((ts, score, re.left_depth_db, re.right_depth_db))

    proposals: dict[int, list[tuple[float, float]]] = {}
    for idx, xs in raw.items():
        xs = sorted(xs)
        # Certificate A: regular ostinato run can admit a weak left-side dip.
        for a in range(len(xs)):
            run=[xs[a]]
            for b in range(a+1, len(xs)):
                dt=xs[b][0]-run[-1][0]
                if REPEAT_SEQUENCE_MIN_INTERVAL_S <= dt <= REPEAT_SEQUENCE_MAX_INTERVAL_S:
                    run.append(xs[b])
                elif dt > REPEAT_SEQUENCE_MAX_INTERVAL_S:
                    break
            if len(run) >= REPEAT_SEQUENCE_MIN_COUNT:
                ints=np.diff([q[0] for q in run])
                if len(ints) and max(ints)/max(min(ints),1e-6) <= REPEAT_SEQUENCE_MAX_INTERVAL_RATIO:
                    for ts,sc,_,_ in run:
                        proposals.setdefault(idx,[]).append((ts,sc))
        # Certificate B: asymmetric but bracketed physical repetition.
        for ts, score, left, right in xs:
            if left < REPEAT_MIN_LEFT_DEPTH_DB:
                continue
            before=[p for p in strong_peaks if REPEAT_BRACKET_MIN_S <= ts-p.time_s <= REPEAT_BRACKET_MAX_S]
            after=[p for p in strong_peaks if REPEAT_BRACKET_MIN_S <= p.time_s-ts <= REPEAT_BRACKET_MAX_S]
            if not before or not after:
                continue
            db=ts-max(p.time_s for p in before); da=min(p.time_s for p in after)-ts
            if max(db,da)/max(min(db,da),1e-6) <= REPEAT_BRACKET_MAX_RATIO:
                proposals.setdefault(idx,[]).append((ts,score))

    if not proposals:
        return 0
    rebuilt=[]; changed=0
    for i,e in enumerate(events):
        pts=proposals.get(i,[])
        chosen=[]
        for ts,sc in sorted(pts,key=lambda x:(-x[1],x[0])):
            if all(abs(ts-q[0])>REPEAT_DEDUPE_S for q in chosen):
                chosen.append((ts,sc))
        chosen.sort()
        if not chosen:
            rebuilt.append(e); continue
        bounds=[e.start_s]+[x[0] for x in chosen]+[e.end_s]
        scores=[None]+[x[1] for x in chosen]
        for j in range(len(bounds)-1):
            seg=_slice_event(e,bounds[j],bounds[j+1],attack_score=scores[j])
            if j>0:
                seg.source_kind=f"{e.source_kind}+asymmetric_repeat_v18_1"
                seg.source_id=f"{e.source_id}:repeat{j}"
                result.attack_decisions.append(AttackDecision(
                    seg.start_s, True, "asymmetric_repeat_v18_1",
                    "calibrated_same_pitch_repetition_restart_v18_1",
                    acoustic_score=seg.attack_score, pitch_source=seg.pitch_source,
                    pitch_anchor_s=seg.start_s, pitch_anchor_st=_interp_pitch(e,seg.start_s),
                ))
                changed+=1
            rebuilt.append(seg)
    result.events=_reclean_monophonic(rebuilt)
    return changed


def _decay_subharmonic_corrections(result: AssemblyResult, probe: _WavProbe) -> int:
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    changed = 0
    for i, e in enumerate(events):
        if "subharmonic_corrected_v18" in e.source_kind:
            continue
        dur = e.end_s - e.start_s
        if not (DECAY_SUBHARMONIC_MIN_EVENT_S <= dur <= DECAY_SUBHARMONIC_MAX_EVENT_S):
            continue
        if i == 0 or i + 1 >= len(events):
            continue
        prev, nxt = events[i - 1], events[i + 1]
        corrected = float(e.base_st) + 12.0
        cur_ctx = abs(float(e.base_st)-float(prev.base_st)) + abs(float(e.base_st)-float(nxt.base_st))
        cor_ctx = abs(corrected-float(prev.base_st)) + abs(corrected-float(nxt.base_st))
        if cur_ctx - cor_ctx < DECAY_SUBHARMONIC_CONTEXT_IMPROVEMENT_ST:
            continue
        if max(abs(corrected-float(prev.base_st)), abs(corrected-float(nxt.base_st))) > DECAY_SUBHARMONIC_MAX_NEIGHBOR_GAP_ST:
            continue
        t, p = _finite_curve(e)
        base_pitch = float(np.median(p)) if len(p) else float(e.base_st)
        f0 = st_to_hz(base_pitch)
        times = np.linspace(e.start_s + min(0.015, 0.10*dur), e.end_s - min(0.015, 0.10*dur), 5)
        dom=[]; rms=[]
        for ts in times:
            lo, hi = probe.spectral_pair_db(float(ts), f0)
            dom.append(hi-lo); rms.append(probe.rms_db(float(ts)))
        if not (dom[0] >= DECAY_SUBHARMONIC_HEAD1_DB and dom[1] >= DECAY_SUBHARMONIC_HEAD2_DB):
            continue
        if rms[0] - rms[-1] < DECAY_SUBHARMONIC_MIN_RMS_FALL_DB:
            continue
        e.base_st += 12.0; e.base_midi = int(np.clip(e.base_midi + 12, 0, 127))
        e.pitch_st = np.asarray(e.pitch_st, dtype=float) + 12.0
        e.pitch_source = f"{e.pitch_source}+decay_subharmonic_octave_up_v18_1"
        e.source_kind = f"{e.source_kind}+decay_subharmonic_corrected_v18_1"
        _recompute_bend(e)
        result.attack_decisions.append(AttackDecision(
            e.start_s, False, "decay_subharmonic_v18_1",
            "octave_up_dominant_at_decaying_head_plus_context_v18_1",
            pitch_source=e.pitch_source, pitch_anchor_s=e.expression_anchor_s,
            pitch_anchor_st=e.base_st,
            continuity_status=f"head_db={dom[0]:.3f},{dom[1]:.3f};rms_fall_db={rms[0]-rms[-1]:.3f}",
        ))
        changed += 1
    return changed


def _local_linear_trend(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    trend = np.empty_like(p)
    for i, x in enumerate(t):
        m = np.abs(t - x) <= FAST_PB_TREND_HALF_WINDOW_S
        if int(np.sum(m)) < 4:
            trend[i] = p[i]; continue
        xx = t[m] - x; yy = p[m]
        A = np.column_stack([np.ones(len(xx)), xx])
        trend[i] = float(np.linalg.lstsq(A, yy, rcond=None)[0][0])
    return trend


def _fast_modulation_gain(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, int]:
    if len(t) < 10 or t[-1] - t[0] < 0.12:
        return np.ones(len(t)), 0
    trend = _local_linear_trend(t, p)
    residual = p - trend
    d = np.diff(residual)
    extrema=[]
    for i in range(1, len(residual)-1):
        turn = (d[i-1] > 0 and d[i] <= 0) or (d[i-1] < 0 and d[i] >= 0)
        if turn and abs(residual[i]) >= FAST_PB_MIN_RESIDUAL_ST:
            extrema.append(i)
    runs=[]; cur=[]
    for idx in extrema:
        if not cur:
            cur=[idx]; continue
        dt=t[idx]-t[cur[-1]]
        ok=(residual[idx]*residual[cur[-1]] < 0 and FAST_PB_MIN_HALF_CYCLE_S <= dt <= FAST_PB_MAX_HALF_CYCLE_S
            and max(abs(residual[idx]),abs(residual[cur[-1]])) <= FAST_PB_MAX_RESIDUAL_ST)
        if ok:
            cur.append(idx)
        else:
            if len(cur) >= FAST_PB_MIN_EXTREMA: runs.append(cur)
            cur=[idx]
    if len(cur) >= FAST_PB_MIN_EXTREMA: runs.append(cur)
    gain=np.ones(len(t)); accepted=0
    for run in runs:
        half=np.diff(t[run])
        if not len(half) or not (FAST_PB_MIN_HALF_CYCLE_S <= float(np.median(half)) <= FAST_PB_MAX_HALF_CYCLE_S):
            continue
        a=float(t[run[0]]); b=float(t[run[-1]])
        m=(t>=a)&(t<=b)
        alpha=np.minimum(1.0, np.minimum((t-a)/FAST_PB_TAPER_S, (b-t)/FAST_PB_TAPER_S))
        alpha=np.clip(alpha,0.0,1.0)
        local=1.0-(1.0-FAST_PB_COMPRESSION)*alpha
        gain[m]=np.minimum(gain[m],local[m])
        accepted += 1
    return gain, accepted


def _compress_fast_pitch_modulation(result: AssemblyResult) -> tuple[int,int]:
    changed_events=0; changed_points=0
    for e in result.events:
        t,p=_finite_curve(e)
        if len(t)<10: continue
        trend=_local_linear_trend(t,p)
        gain,runs=_fast_modulation_gain(t,p)
        m=gain<0.999999
        if not runs or not np.any(m): continue
        residual=p-trend
        newp=trend+gain*residual
        e.pitch_times_s=t; e.pitch_st=newp
        e.pitch_source=f"{e.pitch_source}+fast_pb_residual_x{FAST_PB_COMPRESSION:.2f}_v18_1"
        _recompute_bend(e)
        changed_events += 1; changed_points += int(np.sum(m))
    return changed_events, changed_points


def assemble_events(prefix: Path, wav_path: Path) -> AssemblyResult:
    result = assemble_events_v18(prefix, wav_path)
    probe = _WavProbe(wav_path)

    weak_trim = _trim_weak_backdated_starts(result, probe)
    weak_islands = _collapse_weak_short_islands(result, probe)
    terminal_decay = _collapse_terminal_decay_islands(result, probe)
    repeats = _restore_repetition_attacks(result)
    decay_octaves = _decay_subharmonic_corrections(result, probe)
    pb_events, pb_points = _compress_fast_pitch_modulation(result)

    result.events = _reclean_monophonic(result.events)
    result.attack_decisions.append(AttackDecision(
        -1.0, False, "v18_1_summary",
        (
            f"weak_start_trimmed={weak_trim};"
            f"weak_short_islands_folded={weak_islands};"
            f"terminal_decay_islands_folded={terminal_decay};"
            f"asymmetric_repeats_restored={repeats};"
            f"decay_subharmonic_corrected={decay_octaves};"
            f"fast_pb_events={pb_events};fast_pb_points={pb_points};"
            f"fast_pb_factor={FAST_PB_COMPRESSION:.3f}"
        ),
        pitch_source="renderer_calibration_v18_1",
    ))
    return result
