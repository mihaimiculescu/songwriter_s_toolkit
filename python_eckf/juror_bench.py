from __future__ import annotations

"""V2 juror bench with strict V12 <49-cent same-note grouping upstream."""
from dataclasses import dataclass
from collections import defaultdict
import itertools, math
import numpy as np

from .temporal_persistence_v21 import measure_v21_temporal_pair, measure_v21_candidate_shift_acf

W_SPECTRAL=0.60; W_TEMPORAL=0.20; W_INTERVAL=0.15; W_RANGE=0.50
MIN_SCORE_MARGIN=0.20; MIN_ABSOLUTE_SCORE=-0.10; MIN_ACF=0.72

def _finite(v): return v is not None and math.isfinite(float(v))
def _clip(v,lo=-1.0,hi=1.0): return max(lo,min(hi,float(v)))
def _norm_diff(a,b,scale):
    if not (_finite(a) and _finite(b)) or scale<=0:return 0.0
    return _clip((float(a)-float(b))/scale)
def _range_component(conf): return None if not _finite(conf) else -(1.0-float(conf))
def _interval_component(row):
    vals=[v for v in (row.interval_previous_component,row.interval_following_component) if _finite(v)]
    return sum(map(float,vals))/len(vals) if vals else None
def _temporal_pair_legacy(a,b):
    pa,pb=int(a.temporal_persistence_count),int(b.temporal_persistence_count)
    if pa==pb:return 0.0,0.0
    d=_clip((pa-pb)/2.0); return d,-d

def _temporal_pair_v21(a,b,audio,sample_rate):
    """Use exact V21 shifted persistence only for a decisive direction.

    The existing V2 note-persistence component remains the production fallback.
    V21 may override it only for the two asymmetric historical patterns:
      * lower_period_persistent_higher_not
      * higher_period_persistent_lower_not

    For both-persistent, mixed/weak, insufficient measurements, or when no
    waveform is available, the legacy V2 temporal component is preserved.
    This keeps V21 as an expert directional test rather than erasing useful
    temporal evidence whenever the shifted-window test abstains.
    """
    legacy_a,legacy_b=_temporal_pair_legacy(a,b)
    if audio is None or sample_rate is None:
        return legacy_a,legacy_b,None

    result=measure_v21_temporal_pair(
        np.asarray(audio,dtype=np.float64),int(sample_rate),float(a.time_s),
        float(a.representative_hz),float(b.representative_hz),
    )

    decisive=result.pattern in {
        "lower_period_persistent_higher_not",
        "higher_period_persistent_lower_not",
    }
    if not decisive:
        return legacy_a,legacy_b,result

    if float(a.representative_hz) <= float(b.representative_hz):
        return result.low_component,result.high_component,result
    return result.high_component,result.low_component,result
def _spectral_pair(a,b):
    acf=_norm_diff(a.acf_median,b.acf_median,0.12)
    cmndf=-_norm_diff(a.cmndf_median,b.cmndf_median,0.20)
    harm=_norm_diff(a.harmonic_count_median,b.harmonic_count_median,4.0)
    amp=0.0
    if _finite(a.component_amplitude_median) and _finite(b.component_amplitude_median):
        aa=max(float(a.component_amplitude_median),1e-12); bb=max(float(b.component_amplitude_median),1e-12)
        amp=_clip(math.log2(aa/bb)/2.0)
    score=_clip(.55*acf+.20*cmndf+.15*harm+.10*amp)
    return score,-score,{"acf":acf,"cmndf":cmndf,"harmonic_count":harm,"component_ratio":amp}
def _admissible(r):
    if not _finite(r.range_confidence) or float(r.range_confidence)<=0:return False,"range_zero_or_unavailable"
    if not _finite(r.acf_median):return False,"acf_unavailable"
    if float(r.acf_median)<MIN_ACF:return False,f"acf_below_{MIN_ACF:.2f}"
    return True,"acf_supported"

@dataclass(frozen=True)
class JurorPairVerdict:
    frame_index:int; time_s:float
    a_group_id:str; a_midi:int; a_hz:float; b_group_id:str; b_midi:int; b_hz:float
    a_range_admitted:bool; b_range_admitted:bool; a_admissible:bool; b_admissible:bool
    a_score:float|None; b_score:float|None; a_spectral:float|None; b_spectral:float|None
    a_temporal:float|None; b_temporal:float|None; temporal_pattern:str; temporal_paired_measurements:int; temporal_low_supported_shift_count:int; temporal_high_supported_shift_count:int; temporal_low_acf_median:float|None; temporal_high_acf_median:float|None; a_interval:float|None; b_interval:float|None
    a_range_component:float|None; b_range_component:float|None
    winner_group_id:str|None; winner_midi:int|None; winner_hz:float|None
    outcome:str; diagnostic:str
@dataclass(frozen=True)
class JurorBenchVerdict:
    frame_index:int; time_s:float; candidate_count:int; status:str
    winner_group_id:str|None; winner_midi:int|None; winner_hz:float|None
    decisive_edges:int; undecided_pairs:int; abstention_category:str

@dataclass(frozen=True)
class DirectEdgeMeasurement:
    frame_index:int; time_s:float
    a_group_id:str; a_midi:int; a_hz:float
    b_group_id:str; b_midi:int; b_hz:float
    a_testable_views:int; b_testable_views:int
    a_supported_views:int; b_supported_views:int
    a_acf_median:float|None; b_acf_median:float|None
    temporal_pattern:str
    original_outcome:str; direct_outcome:str
    direct_winner_group_id:str|None
    direct_winner_midi:int|None; direct_winner_hz:float|None
    used_to_complete_edge:bool
    diagnostic:str

def _pair(a,b,audio=None,sample_rate=None):
    ar=_finite(a.range_confidence) and float(a.range_confidence)>0; br=_finite(b.range_confidence) and float(b.range_confidence)>0
    aok,areason=_admissible(a); bok,breason=_admissible(b)
    sa,sb,sd=_spectral_pair(a,b); ta,tb,tr=_temporal_pair_v21(a,b,audio,sample_rate); ia,ib=_interval_component(a),_interval_component(b)
    ra,rb=_range_component(a.range_confidence),_range_component(b.range_confidence)
    if ar!=br: sa=sb=0.0
    def score(ok,spec,temp,inter,rng):
        if not ok:return None
        v=W_SPECTRAL*spec+W_TEMPORAL*temp
        if inter is not None:v+=W_INTERVAL*inter
        if rng is not None:v+=W_RANGE*rng
        return float(v)
    ascore,bscore=score(aok,sa,ta,ia,ra),score(bok,sb,tb,ib,rb)
    winner=None
    if not ar and not br: outcome="abstain_all_candidates_out_of_range"
    elif not aok and not bok: outcome="abstain_no_admissible_candidate"
    elif aok and not bok:
        if ascore is not None and ascore>=MIN_ABSOLUTE_SCORE:winner=a;outcome="provisional_a_only"
        else:outcome="abstain_a_only_weak"
    elif bok and not aok:
        if bscore is not None and bscore>=MIN_ABSOLUTE_SCORE:winner=b;outcome="provisional_b_only"
        else:outcome="abstain_b_only_weak"
    else:
        diff=float(bscore-ascore)
        if abs(diff)<MIN_SCORE_MARGIN: outcome="abstain_insufficient_separation"
        else:
            winner=b if diff>0 else a; ws=bscore if diff>0 else ascore
            if ws is None or ws<MIN_ABSOLUTE_SCORE: outcome="abstain_winner_weak";winner=None
            else: outcome="provisional_b" if diff>0 else "provisional_a"
    tdiag=("legacy" if tr is None else f"{tr.pattern}:paired={tr.paired_measurements}:low={tr.low_supported_shift_count}:high={tr.high_supported_shift_count}")
    diag=(f"a:{areason};b:{breason};spec(acf={sd['acf']:.4f},cmndf={sd['cmndf']:.4f},harm={sd['harmonic_count']:.4f},amp={sd['component_ratio']:.4f});temporal({tdiag})")
    return JurorPairVerdict(int(a.frame_index),float(a.time_s),a.group_id,int(a.note_group_midi),float(a.representative_hz),b.group_id,int(b.note_group_midi),float(b.representative_hz),bool(ar),bool(br),bool(aok),bool(bok),ascore,bscore,sa,sb,ta,tb,("legacy_note_persistence" if tr is None else tr.pattern),(0 if tr is None else tr.paired_measurements),(0 if tr is None else tr.low_supported_shift_count),(0 if tr is None else tr.high_supported_shift_count),(None if tr is None else tr.low_acf_median),(None if tr is None else tr.high_acf_median),ia,ib,ra,rb,None if winner is None else winner.group_id,None if winner is None else int(winner.note_group_midi),None if winner is None else float(winner.representative_hz),outcome,diag)


DIRECT_MIN_SUPPORTED_MEASUREMENTS = 2


def _direct_acoustic_admissible(row, stats):
    if not (_finite(row.range_confidence) and float(row.range_confidence) > 0):
        return False
    return (
        stats.testable_views >= DIRECT_MIN_SUPPORTED_MEASUREMENTS
        and stats.supported_views >= DIRECT_MIN_SUPPORTED_MEASUREMENTS
        and stats.acf_median is not None
    )


def _direct_missing_edge_pair(a, b, original_pair, audio, sample_rate):
    """Directly remeasure an unresolved admitted edge from the original WAV.

    Historical V22 completed missing tournament edges from the waveform rather
    than synthesizing a comparison from neighboring results.  In V2 every
    representative pair already has a row, so an edge is considered *missing*
    when that row has no decisive winner.  Only such edges are remeasured.

    The remeasurement uses the restored V21 24/40/64 ms, -8/0/+8 ms shifted
    ACF field directly from the WAV.  Direct ACF medians refresh the ACF part of
    the spectral juror, while CMNDF/harmonic/component summaries, interval and
    range terms remain the existing V2 evidence.  A direct result may fill an
    absent edge; it never overturns an existing decisive edge.
    """
    ast = measure_v21_candidate_shift_acf(
        np.asarray(audio, dtype=np.float64), int(sample_rate),
        float(a.time_s), float(a.representative_hz)
    )
    bst = measure_v21_candidate_shift_acf(
        np.asarray(audio, dtype=np.float64), int(sample_rate),
        float(b.time_s), float(b.representative_hz)
    )
    tr = measure_v21_temporal_pair(
        np.asarray(audio, dtype=np.float64), int(sample_rate), float(a.time_s),
        float(a.representative_hz), float(b.representative_hz)
    )

    aok = _direct_acoustic_admissible(a, ast)
    bok = _direct_acoustic_admissible(b, bst)
    ar = _finite(a.range_confidence) and float(a.range_confidence) > 0
    br = _finite(b.range_confidence) and float(b.range_confidence) > 0

    # Reuse all non-ACF spectral evidence, but replace the ACF term by a fresh
    # direct-WAV shifted-window median comparison.
    acf = _norm_diff(ast.acf_median, bst.acf_median, 0.12)
    cmndf = -_norm_diff(a.cmndf_median, b.cmndf_median, 0.20)
    harm = _norm_diff(a.harmonic_count_median, b.harmonic_count_median, 4.0)
    amp = 0.0
    if _finite(a.component_amplitude_median) and _finite(b.component_amplitude_median):
        aa = max(float(a.component_amplitude_median), 1e-12)
        bb = max(float(b.component_amplitude_median), 1e-12)
        amp = _clip(math.log2(aa / bb) / 2.0)
    sa = _clip(.55 * acf + .20 * cmndf + .15 * harm + .10 * amp)
    sb = -sa
    if ar != br:
        sa = sb = 0.0

    # Decisive V21 direction is authoritative for temporal evidence; otherwise
    # keep the existing V2 temporal persistence component, exactly as V21 V2.
    legacy_a, legacy_b = _temporal_pair_legacy(a, b)
    if tr.pattern in {
        'lower_period_persistent_higher_not',
        'higher_period_persistent_lower_not',
    }:
        if float(a.representative_hz) <= float(b.representative_hz):
            ta, tb = tr.low_component, tr.high_component
        else:
            ta, tb = tr.high_component, tr.low_component
    else:
        ta, tb = legacy_a, legacy_b

    ia, ib = _interval_component(a), _interval_component(b)
    ra, rb = _range_component(a.range_confidence), _range_component(b.range_confidence)

    def score(ok, spec, temp, inter, rng):
        if not ok:
            return None
        v = W_SPECTRAL * spec + W_TEMPORAL * temp
        if inter is not None:
            v += W_INTERVAL * inter
        if rng is not None:
            v += W_RANGE * rng
        return float(v)

    ascore = score(aok, sa, ta, ia, ra)
    bscore = score(bok, sb, tb, ib, rb)
    winner = None
    if not ar and not br:
        outcome = 'direct_abstain_all_candidates_out_of_range'
    elif not aok and not bok:
        outcome = 'direct_abstain_no_admissible_candidate'
    elif aok and not bok:
        if ascore is not None and ascore >= MIN_ABSOLUTE_SCORE:
            winner, outcome = a, 'direct_provisional_a_only'
        else:
            outcome = 'direct_abstain_a_only_weak'
    elif bok and not aok:
        if bscore is not None and bscore >= MIN_ABSOLUTE_SCORE:
            winner, outcome = b, 'direct_provisional_b_only'
        else:
            outcome = 'direct_abstain_b_only_weak'
    else:
        diff = float(bscore - ascore)
        if abs(diff) < MIN_SCORE_MARGIN:
            outcome = 'direct_abstain_insufficient_separation'
        else:
            winner = b if diff > 0 else a
            ws = bscore if diff > 0 else ascore
            if ws is None or ws < MIN_ABSOLUTE_SCORE:
                winner = None
                outcome = 'direct_abstain_winner_weak'
            else:
                outcome = 'direct_provisional_b' if diff > 0 else 'direct_provisional_a'

    diag = (
        f'direct_wav_v21_shifted_acf;'
        f'a_views={ast.supported_views}/{ast.testable_views};'
        f'b_views={bst.supported_views}/{bst.testable_views};'
        f'a_acf_med={ast.acf_median};b_acf_med={bst.acf_median};'
        f'temporal={tr.pattern};'
        f'spec(acf={acf:.4f},cmndf={cmndf:.4f},harm={harm:.4f},amp={amp:.4f})'
    )
    pair = JurorPairVerdict(
        int(a.frame_index), float(a.time_s),
        a.group_id, int(a.note_group_midi), float(a.representative_hz),
        b.group_id, int(b.note_group_midi), float(b.representative_hz),
        bool(ar), bool(br), bool(aok), bool(bok),
        ascore, bscore, sa, sb, ta, tb,
        tr.pattern, tr.paired_measurements,
        tr.low_supported_shift_count, tr.high_supported_shift_count,
        tr.low_acf_median, tr.high_acf_median,
        ia, ib, ra, rb,
        None if winner is None else winner.group_id,
        None if winner is None else int(winner.note_group_midi),
        None if winner is None else float(winner.representative_hz),
        outcome, diag,
    )
    audit = DirectEdgeMeasurement(
        int(a.frame_index), float(a.time_s),
        a.group_id, int(a.note_group_midi), float(a.representative_hz),
        b.group_id, int(b.note_group_midi), float(b.representative_hz),
        ast.testable_views, bst.testable_views,
        ast.supported_views, bst.supported_views,
        ast.acf_median, bst.acf_median,
        tr.pattern, original_pair.outcome, outcome,
        None if winner is None else winner.group_id,
        None if winner is None else int(winner.note_group_midi),
        None if winner is None else float(winner.representative_hz),
        bool(winner is not None), diag,
    )
    return pair, audit


def _abstention_category(s):
    return {"provisional_tournament_champion":"provisional_champion","abstain_no_decisive_pair":"no_decisive_acoustic_pair","abstain_tournament_no_unique_champion":"multiple_undefeated_candidates","abstain_tournament_cycle":"tournament_cycle","abstain_tournament_champion_ambiguous":"unresolved_champion_direct_comparison","abstain_tournament_incomplete_comparisons":"incomplete_comparisons","abstain_all_candidates_out_of_range":"all_candidates_out_of_range","abstain_no_admissible_candidate":"no_admissible_candidate","abstain_single_candidate_no_acoustic_support":"single_candidate_no_acoustic_support"}.get(s,"other_abstention")

def _tournament(rows,pairs):
    nodes={r.group_id:r for r in rows if _finite(r.range_confidence) and float(r.range_confidence)>0}
    if not nodes:return "abstain_all_candidates_out_of_range",None,0,0
    if len(nodes)==1:
        only=next(iter(nodes.values()))
        if not _admissible(only)[0]:return "abstain_single_candidate_no_acoustic_support",None,0,0
        return "provisional_tournament_champion",only,0,0
    edges=set();undecided=set()
    for p in pairs:
        if p.a_group_id not in nodes or p.b_group_id not in nodes:continue
        key=frozenset((p.a_group_id,p.b_group_id))
        if p.winner_group_id is not None:
            loser=p.b_group_id if p.winner_group_id==p.a_group_id else p.a_group_id
            edges.add((p.winner_group_id,loser))
        elif p.a_admissible and p.b_admissible:undecided.add(key)
    if not edges:return "abstain_no_decisive_pair",None,0,len(undecided)
    graph={g:set() for g in nodes}
    for w,l in edges:graph[w].add(l)
    visited=set();active=set()
    def cyc(v):
        if v in active:return True
        if v in visited:return False
        active.add(v)
        if any(cyc(w) for w in graph[v]):return True
        active.remove(v);visited.add(v);return False
    if any(cyc(g) for g in graph):return "abstain_tournament_cycle",None,len(edges),len(undecided)
    incoming={l for _,l in edges}; champs=[g for g in nodes if g not in incoming]
    if len(champs)!=1:return "abstain_tournament_no_unique_champion",None,len(edges),len(undecided)
    c=champs[0];reached={c};stack=[c]
    while stack:
        v=stack.pop()
        for w in graph[v]-reached:reached.add(w);stack.append(w)
    if len(reached)!=len(nodes):return "abstain_tournament_incomplete_comparisons",None,len(edges),len(undecided)
    if any(c in u for u in undecided):return "abstain_tournament_champion_ambiguous",None,len(edges),len(undecided)
    return "provisional_tournament_champion",nodes[c],len(edges),len(undecided)

def build_juror_bench(evidence_rows,audio=None,sample_rate=None):
    by=defaultdict(list)
    for r in evidence_rows:
        by[int(r.frame_index)].append(r)
    allpairs=[]; verdicts=[]; direct_audit=[]
    for fi in sorted(by):
        rows=sorted(by[fi],key=lambda r:(r.note_group_midi,r.group_id))
        original=[_pair(a,b,audio=audio,sample_rate=sample_rate) for a,b in itertools.combinations(rows,2)]

        # First run the tournament exactly as before.  Direct WAV completion is
        # invoked only for hung/incomplete frames and only on unresolved admitted
        # edges. Existing decisive edges are never remeasured or overturned.
        status,winner,edges,undecided=_tournament(rows,original)
        effective=list(original)
        if (
            audio is not None and sample_rate is not None
            and status != 'provisional_tournament_champion'
            and len(rows) >= 2
        ):
            row_by_id={r.group_id:r for r in rows}
            replaced=[]
            for p in effective:
                if (
                    p.winner_group_id is None
                    and p.a_range_admitted and p.b_range_admitted
                    and p.a_group_id in row_by_id and p.b_group_id in row_by_id
                ):
                    dp, audit = _direct_missing_edge_pair(
                        row_by_id[p.a_group_id], row_by_id[p.b_group_id], p,
                        audio, sample_rate,
                    )
                    direct_audit.append(audit)
                    # Fill only a genuinely missing edge. If direct measurement
                    # also abstains, preserve the original unresolved row.
                    replaced.append(dp if dp.winner_group_id is not None else p)
                else:
                    replaced.append(p)
            effective=replaced
            status,winner,edges,undecided=_tournament(rows,effective)

        allpairs.extend(effective)
        verdicts.append(JurorBenchVerdict(
            fi,float(rows[0].time_s),len(rows),status,
            None if winner is None else winner.group_id,
            None if winner is None else int(winner.note_group_midi),
            None if winner is None else float(winner.representative_hz),
            edges,undecided,_abstention_category(status)
        ))
    return tuple(allpairs),tuple(verdicts),tuple(direct_audit)
