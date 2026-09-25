from __future__ import annotations

"""V2 juror bench with strict V12 <49-cent same-note grouping upstream."""
from dataclasses import dataclass
from collections import defaultdict
import itertools, math

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
def _temporal_pair(a,b):
    pa,pb=int(a.temporal_persistence_count),int(b.temporal_persistence_count)
    if pa==pb:return 0.0,0.0
    d=_clip((pa-pb)/2.0); return d,-d
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
    a_temporal:float|None; b_temporal:float|None; a_interval:float|None; b_interval:float|None
    a_range_component:float|None; b_range_component:float|None
    winner_group_id:str|None; winner_midi:int|None; winner_hz:float|None
    outcome:str; diagnostic:str
@dataclass(frozen=True)
class JurorBenchVerdict:
    frame_index:int; time_s:float; candidate_count:int; status:str
    winner_group_id:str|None; winner_midi:int|None; winner_hz:float|None
    decisive_edges:int; undecided_pairs:int; abstention_category:str

def _pair(a,b):
    ar=_finite(a.range_confidence) and float(a.range_confidence)>0; br=_finite(b.range_confidence) and float(b.range_confidence)>0
    aok,areason=_admissible(a); bok,breason=_admissible(b)
    sa,sb,sd=_spectral_pair(a,b); ta,tb=_temporal_pair(a,b); ia,ib=_interval_component(a),_interval_component(b)
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
    diag=(f"a:{areason};b:{breason};spec(acf={sd['acf']:.4f},cmndf={sd['cmndf']:.4f},harm={sd['harmonic_count']:.4f},amp={sd['component_ratio']:.4f})")
    return JurorPairVerdict(int(a.frame_index),float(a.time_s),a.group_id,int(a.note_group_midi),float(a.representative_hz),b.group_id,int(b.note_group_midi),float(b.representative_hz),bool(ar),bool(br),bool(aok),bool(bok),ascore,bscore,sa,sb,ta,tb,ia,ib,ra,rb,None if winner is None else winner.group_id,None if winner is None else int(winner.note_group_midi),None if winner is None else float(winner.representative_hz),outcome,diag)

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

def build_juror_bench(evidence_rows):
    by=defaultdict(list)
    for r in evidence_rows:by[int(r.frame_index)].append(r)
    allpairs=[];verdicts=[]
    for fi in sorted(by):
        rows=sorted(by[fi],key=lambda r:(r.note_group_midi,r.group_id))
        pairs=[_pair(a,b) for a,b in itertools.combinations(rows,2)];allpairs.extend(pairs)
        status,winner,edges,undecided=_tournament(rows,pairs)
        verdicts.append(JurorBenchVerdict(fi,float(rows[0].time_s),len(rows),status,None if winner is None else winner.group_id,None if winner is None else int(winner.note_group_midi),None if winner is None else float(winner.representative_hz),edges,undecided,_abstention_category(status)))
    return tuple(allpairs),tuple(verdicts)
