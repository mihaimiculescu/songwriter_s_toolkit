#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, sys
from dataclasses import dataclass
from pathlib import Path
import mido
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/"tests"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

# V1 is deliberately reused ONLY for independent alignment.
from diagnose_midi_wav_event_correspondence import (
    Config as V1Config, load_eckf_csv, load_midi_notes,
    extract_wav_events, estimate_global_offset, section_offsets,
)
from python_eckf.expressive_smoothing import ExpressiveSmoothingConfig,smooth_expressive_pitch
from python_eckf.gesture_features import GestureFeatureConfig,extract_gesture_features
from python_eckf.gesture_objects import GestureObjectConfig,construct_gesture_objects
from python_eckf.gesture_object_splitter import GestureObjectSplitterConfig,analyse_gesture_object_splits
from python_eckf.gesture_structural_splitter import StructuralSplitterConfig,apply_structural_splitting
from python_eckf.trajectory_interpreter import RegionKind,interpret_pitch_trajectory

TOLS=(50.,100.,150.,250.)
REFERENCE_BPM=120.0
TIMING_ALPHA=0.20

def timing_eyebrow_threshold_ms(bpm, reference_bpm=REFERENCE_BPM, alpha=TIMING_ALPHA):
    """
    Soft musical timing prior.

    Base = one eighth note.  The beat-relative allowance widens gently at
    higher tempi and tightens gently at lower tempi:
        threshold = (30000 / bpm) * (bpm / reference_bpm)**alpha

    This is diagnostic only.  It never changes structural decisions.
    """
    bpm=float(bpm)
    if not np.isfinite(bpm) or bpm <= 0:
        return float("nan")
    return (30000.0/bpm) * (bpm/reference_bpm)**alpha

def timing_flag(abs_delta_ms, threshold_ms):
    if not np.isfinite(abs_delta_ms) or not np.isfinite(threshold_ms):
        return "-"
    ratio=abs_delta_ms/threshold_ms
    if ratio < 1.0: return "ordinary"
    if ratio < 2.0: return "EYEBROW"
    if ratio < 4.0: return "LARGE"
    return "EXTREME"

def semitone_relation(delta_st):
    """Observational harmonic-family warning only; no correction."""
    if not np.isfinite(delta_st): return "-"
    relations=((12.0,"2x"),(-12.0,"1/2x"),(19.01955,"3x"),(-19.01955,"1/3x"))
    rel,err=min(relations,key=lambda x:abs(delta_st-x[0]))
    return rel if abs(delta_st-relations[[x[1] for x in relations].index(rel)][0]) <= .75 else "-"

class MidiMap:
    def __init__(self,path):
        mid=mido.MidiFile(path); self.tpb=mid.ticks_per_beat
        msgs=[]
        for tr in mid.tracks:
            tick=0
            for msg in tr:
                tick+=msg.time; msgs.append((tick,msg))
        tempos=[(0,500000)]; meters=[(0,4,4)]
        for tick,msg in msgs:
            if msg.type=="set_tempo": tempos.append((tick,int(msg.tempo)))
            elif msg.type=="time_signature": meters.append((tick,int(msg.numerator),int(msg.denominator)))
        self.tempos=self._collapse(tempos); self.meters=self._collapse(meters)
    @staticmethod
    def _collapse(xs):
        out=[]
        for x in sorted(xs):
            if out and out[-1][0]==x[0]: out[-1]=x
            else: out.append(x)
        return out
    def sec2tick(self,sec):
        elapsed=0.; prev=0.; tempo=self.tempos[0][1]
        for tick,newtempo in self.tempos[1:]:
            span=mido.tick2second(tick-prev,self.tpb,tempo)
            if elapsed+span>=sec:
                return prev+(sec-elapsed)/(tempo/1e6)*self.tpb
            elapsed+=span; prev=tick; tempo=newtempo
        return prev+(sec-elapsed)/(tempo/1e6)*self.tpb
    def bpm_at(self,sec):
        """Tempo active at a MIDI-time position, in quarter-note BPM."""
        tick=self.sec2tick(sec)
        tempo=self.tempos[0][1]
        for t,newtempo in self.tempos[1:]:
            if t > tick: break
            tempo=newtempo
        return 60_000_000.0/tempo

    def timing_metrics(self,delta_ms,sec):
        bpm=self.bpm_at(sec)
        beat_ms=60000.0/bpm
        eighth_ms=beat_ms/2.0
        eyebrow=timing_eyebrow_threshold_ms(bpm)
        return bpm, delta_ms/beat_ms, delta_ms/eighth_ms, eyebrow, timing_flag(abs(delta_ms),eyebrow)

    def pos(self,sec):
        tick=self.sec2tick(sec); bar=1; section=0.; active=self.meters[0]
        for i,m in enumerate(self.meters):
            if m[0]>tick: break
            if i:
                prevm=self.meters[i-1]; bt=self.tpb*4/prevm[2]; bart=bt*prevm[1]
                span=m[0]-section; whole=int(math.floor(span/bart+1e-9))
                bar+=whole
                if span-whole*bart>1e-6: bar+=1
                section=float(m[0])
            active=m
        bt=self.tpb*4/active[2]; bart=bt*active[1]
        local=max(0.,tick-section); bars=int(math.floor(local/bart+1e-12)); bar+=bars
        wb=local-bars*bart; beat=int(math.floor(wb/bt+1e-12))+1
        within=wb-(beat-1)*bt; six=self.tpb/4.; sf=within/six
        slot=int(math.floor(sf+1e-12))+1; frac=sf-(slot-1)
        return f"{bar}|{beat}|{slot}" if abs(frac)<.01 else f"{bar}|{beat}|{slot}+{frac:.2f}"

@dataclass(frozen=True)
class Anchor:
    obj:int; link:int; decision:str; region:int
    start:float; mid:float; end:float; pitch:float; dur:float
    connectors:str; boundaries:str; nc:int; nb:int

def flags(e):
    c=[]; b=[]
    if e.excursion_connector:c.append("EXCURSION")
    if e.direct_through_connector:c.append("DIRECT")
    if e.transition_similarity_connector:c.append("SIMILAR")
    if e.duration_discontinuity:b.append("DURATION")
    if e.path_discontinuity:b.append("PATH")
    if e.center_shape_discontinuity:b.append("CENTER/SHAPE")
    if e.residual_discontinuity:b.append("RESIDUAL")
    if e.topology_discontinuity:b.append("TOPOLOGY")
    return ",".join(c) or "-", ",".join(b) or "-"

def frozen(data):
    hz=1/float(np.median(np.diff(data.time_s)))
    interp=interpret_pitch_trajectory(clean_f0_hz=data.clean_f0_hz,valid=data.corrected_valid,analysis_hz=hz)
    sm=smooth_expressive_pitch(
        pitch_st=interp.pitch_st,valid=data.corrected_valid,analysis_hz=hz,
        config=ExpressiveSmoothingConfig(analysis_hz=hz,window_ms=50.,polyorder=2,center_window_ms=250.))
    feat=extract_gesture_features(interp,sm,GestureFeatureConfig())
    objs=construct_gesture_objects(interp,feat,GestureObjectConfig(short_target_max_ms=250.,max_chain_transitions=8))
    ev=analyse_gesture_object_splits(objs,feat,GestureObjectSplitterConfig(near_flat_st=.05))
    st=apply_structural_splitting(ev,StructuralSplitterConfig())
    rmap={int(r.index):r for r in interp.regions}; anchors=[]
    for o in st.objects:
        for d in o.link_decisions:
            r=rmap[int(d.stable_region_index)]
            if r.kind is not RegionKind.STABLE_TARGET: raise RuntimeError(f"S{r.index} is not STABLE_TARGET")
            c,b=flags(d.evidence); a=float(r.start_time_s); z=float(r.end_time_s)
            anchors.append(Anchor(int(d.object_index),int(d.link_index),d.decision.name,int(r.index),
                                  a,(a+z)/2,z,float(r.median_pitch_st),float(r.duration_ms),
                                  c,b,int(d.evidence.connector_family_count),int(d.evidence.boundary_family_count)))
    if len(anchors)!=st.total_links: raise RuntimeError("structural-link coverage failed")
    return interp,st,tuple(anchors),hz

def interval_dist(t,a):
    if a.start<=t<=a.end:return 0.,0.
    ds=a.start-t; de=a.end-t; x=ds if abs(ds)<=abs(de) else de
    return abs(1000*x),1000*x

def nearest_anchor(t,anchors):
    if not anchors:return None,float("nan"),float("nan")
    vals=[interval_dist(t,a) for a in anchors]; i=min(range(len(vals)),key=lambda k:vals[k][0])
    return anchors[i],vals[i][0],vals[i][1]

def nearest_point(t,points):
    if not points:return float("nan"),-1
    arr=np.asarray(points); i=int(np.argmin(abs(arr-t))); return 1000*(arr[i]-t),i

def bins(label,vals):
    a=np.asarray([x for x in vals if np.isfinite(x)])
    if not len(a): print(f"{label:<30} n=0"); return
    print(f"{label:<30} n={len(a):4d} median={np.median(a):7.1f}ms p90={np.percentile(a,90):7.1f}ms")
    for t in TOLS:
        n=int(np.sum(a<=t)); print(f"    within {t:3.0f} ms: {n:4d}/{len(a):4d} ({100*n/len(a):5.1f}%)")

def note_name(n):
    names=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    return f"{names[n%12]}{n//12-1}"


# =============================================================================
# V4: GT PITCH-CHANGE <-> FROZEN WAV BOUNDARY CORRESPONDENCE
# =============================================================================
#
# "Vicinity" finds candidates.  Pitch geometry chooses among them.
# Timing is measured only AFTER a credible structural correspondence is found.
#
# No frozen production threshold is changed by this diagnostic.
#

VICINITY_EYEBROW_MULT = 2.0   # search aperture = 2 * tempo-adaptive eyebrow
MIN_VICINITY_MS = 350.0       # do not make slow-song/local searches absurdly tiny
MAX_VICINITY_MS = 900.0       # do not let "vicinity" become a whole phrase

# Pitch-correspondence tolerances are diagnostic matching tolerances, not
# production/transcription thresholds.
PITCH_EXACT_ST = 1.25
PITCH_LOOSE_ST = 2.25
INTERVAL_EXACT_ST = 1.50
INTERVAL_LOOSE_ST = 3.00

@dataclass(frozen=True)
class WavBoundary:
    index:int
    left_region:int
    right_region:int
    time_s:float
    left_pitch:float
    right_pitch:float
    interval_st:float
    direction:int
    gap_ms:float
    validity_wall:bool

@dataclass(frozen=True)
class BoundaryCandidate:
    boundary:WavBoundary
    delta_ms:float
    source_err:float
    dest_err:float
    interval_err:float
    direction_match:bool
    harmonic_source:str
    harmonic_dest:str
    pitch_score:float
    correspondence_score:float
    credible:bool
    reason:str

@dataclass(frozen=True)
class GTMatch:
    gt_i:int
    status:str                    # MATCHED / AMBIGUOUS / NO_CORRESPONDENCE
    candidate:BoundaryCandidate|None
    runner_up:BoundaryCandidate|None
    vicinity_ms:float
    eyebrow_ms:float
    candidates_seen:int
    credible_candidates:int

def sgn(x, eps=.35):
    if x > eps: return 1
    if x < -eps: return -1
    return 0

def harmonic_relation_for_error(observed_pitch, expected_pitch):
    if not np.isfinite(observed_pitch) or not np.isfinite(expected_pitch):
        return "-"
    d=float(observed_pitch)-float(expected_pitch)
    for target,label in ((12.0,"2x"),(-12.0,"1/2x"),
                         (19.01955,"3x"),(-19.01955,"1/3x")):
        if abs(d-target)<=.75:
            return f"{label}({d:+.2f}st)"
    return "-"

def harmonic_adjusted_error(observed_pitch, expected_pitch):
    """
    Return (best absolute pitch error, relation label).
    Harmonic/subharmonic alternatives are OBSERVATIONAL allowances in matching;
    they do not rewrite the frozen trajectory.
    """
    raw=abs(float(observed_pitch)-float(expected_pitch))
    best=raw; label="-"
    d=float(observed_pitch)-float(expected_pitch)
    for target,lab in ((12.0,"2x"),(-12.0,"1/2x"),
                       (19.01955,"3x"),(-19.01955,"1/3x")):
        e=abs(d-target)
        if e < best:
            best=e; label=lab
    return best,label

def stable_boundaries(interp):
    """
    Build actual boundaries between consecutive frozen STABLE_TARGET regions.

    A boundary is admitted only when the two stable targets belong to the same
    contiguous valid island.  Intervening TRANSITION regions are allowed.
    Validity walls are NEVER bridged.

    The boundary time is the midpoint of the temporal gap between the end of
    the left stable target and the start of the right stable target.  This is
    an observational structural boundary, not a manufactured onset.
    """
    regs=list(interp.regions)
    stable_pos=[i for i,r in enumerate(regs) if r.kind is RegionKind.STABLE_TARGET]
    out=[]
    for bi,(lp,rp) in enumerate(zip(stable_pos,stable_pos[1:]),1):
        left=regs[lp]; right=regs[rp]

        # Every region between the targets must be sample-contiguous.  This is
        # the important V4 validity-wall guard.
        chain=regs[lp:rp+1]
        contiguous=True
        for a,b in zip(chain,chain[1:]):
            if int(b.start_index) != int(a.end_index)+1:
                contiguous=False
                break
        if not contiguous:
            continue

        t0=float(left.end_time_s)
        t1=float(right.start_time_s)
        mid=(t0+t1)/2.0
        li=float(left.median_pitch_st)
        ri=float(right.median_pitch_st)
        iv=ri-li
        out.append(WavBoundary(
            index=bi,
            left_region=int(left.index),
            right_region=int(right.index),
            time_s=mid,
            left_pitch=li,
            right_pitch=ri,
            interval_st=iv,
            direction=sgn(iv),
            gap_ms=max(0.0,1000.0*(t1-t0)),
            validity_wall=False,
        ))
    return tuple(out)

def vicinity_ms(mm, midi_sec):
    """
    Tempo-adaptive local search aperture.

    The eyebrow threshold is the human-timing prior.  Vicinity is deliberately
    wider: it is only a candidate-retrieval aperture, not a timing tolerance.
    """
    bpm=mm.bpm_at(midi_sec)
    eyebrow=timing_eyebrow_threshold_ms(bpm)
    v=VICINITY_EYEBROW_MULT*eyebrow
    return float(np.clip(v,MIN_VICINITY_MS,MAX_VICINITY_MS)),float(eyebrow)

def candidate_for_gt(boundary, gt_time, prev_pitch, cur_pitch, vicinity):
    dms=1000.0*(boundary.time_s-gt_time)
    if abs(dms)>vicinity:
        return None

    src_err,src_h=harmonic_adjusted_error(boundary.left_pitch,prev_pitch)
    dst_err,dst_h=harmonic_adjusted_error(boundary.right_pitch,cur_pitch)
    gt_iv=float(cur_pitch-prev_pitch)
    iv_err=abs(abs(boundary.interval_st)-abs(gt_iv))
    dir_match=(sgn(gt_iv)==boundary.direction) if sgn(gt_iv)!=0 else True

    # Pitch geometry dominates.  Timing only weakly breaks ties *inside*
    # vicinity; nearest timestamp cannot defeat a clearly better pitch match.
    endpoint=(src_err+dst_err)/2.0
    pitch_score=(
        0.45*np.clip(src_err/PITCH_LOOSE_ST,0,2) +
        0.45*np.clip(dst_err/PITCH_LOOSE_ST,0,2) +
        0.10*np.clip(iv_err/INTERVAL_LOOSE_ST,0,2)
    )
    direction_penalty=0.0 if dir_match else 0.55
    timing_tiebreak=0.10*min(1.0,abs(dms)/max(vicinity,1.0))
    score=float(pitch_score+direction_penalty+timing_tiebreak)

    exact=(src_err<=PITCH_EXACT_ST and dst_err<=PITCH_EXACT_ST and
           iv_err<=INTERVAL_EXACT_ST and dir_match)
    loose=(src_err<=PITCH_LOOSE_ST and dst_err<=PITCH_LOOSE_ST and
           iv_err<=INTERVAL_LOOSE_ST and dir_match)

    # If both endpoints only match via the SAME harmonic family, retain as a
    # credible tracking-family correspondence.  One-sided harmonic matches are
    # printed but do not by themselves make a boundary credible.
    harmonic_pair=(src_h!="-" and src_h==dst_h and
                   src_err<=PITCH_EXACT_ST and dst_err<=PITCH_EXACT_ST and
                   iv_err<=INTERVAL_LOOSE_ST and dir_match)

    credible=bool(exact or loose or harmonic_pair)
    why=("EXACT_GEOMETRY" if exact else
         "LOOSE_GEOMETRY" if loose else
         f"HARMONIC_PAIR_{src_h}" if harmonic_pair else
         "GEOMETRY_MISMATCH")

    return BoundaryCandidate(
        boundary=boundary,delta_ms=dms,
        source_err=float(src_err),dest_err=float(dst_err),
        interval_err=float(iv_err),direction_match=bool(dir_match),
        harmonic_source=src_h,harmonic_dest=dst_h,
        pitch_score=float(pitch_score),
        correspondence_score=score,
        credible=credible,reason=why,
    )

def match_gt_change(i, midi, gt, mm, boundaries):
    prev=midi.notes[i-1]; cur=midi.notes[i]
    gt_time=float(gt[i])
    vic,eyebrow=vicinity_ms(mm,cur.onset_s)

    candidates=[]
    for b in boundaries:
        c=candidate_for_gt(b,gt_time,prev.pitch,cur.pitch,vic)
        if c is not None:
            candidates.append(c)

    candidates.sort(key=lambda c:(not c.credible,c.correspondence_score,
                                  abs(c.delta_ms),c.boundary.index))
    credible=[c for c in candidates if c.credible]
    if not credible:
        return GTMatch(i,"NO_CORRESPONDENCE",None,
                       candidates[0] if candidates else None,
                       vic,eyebrow,len(candidates),0)

    best=credible[0]
    runner=credible[1] if len(credible)>1 else None

    # Ambiguous only when a second *credible* pitch-geometric solution is
    # genuinely competitive.  This is not based on timing alone.
    ambiguous=False
    if runner is not None:
        score_gap=runner.correspondence_score-best.correspondence_score
        pitch_gap=runner.pitch_score-best.pitch_score
        if score_gap<=0.18 and pitch_gap<=0.15:
            ambiguous=True

    return GTMatch(i,"AMBIGUOUS" if ambiguous else "MATCHED",
                   best,runner,vic,eyebrow,len(candidates),len(credible))

def boundary_to_structural_link(boundary, anchors):
    """
    Observational cross-reference only: nearest V3.1 link to this WAV boundary.
    Does NOT participate in GT matching.
    """
    if not anchors: return None,float("nan")
    a=min(anchors,key=lambda x:abs(x.mid-boundary.time_s))
    return a,1000.0*(a.mid-boundary.time_s)

def print_match(mm,midi,gt,m,anchors):
    i=m.gt_i; prev=midi.notes[i-1]; cur=midi.notes[i]
    gt_time=float(gt[i])
    print(f"GT{cur.index:03d} {note_name(prev.pitch)}->{note_name(cur.pitch)} "
          f"({cur.pitch-prev.pitch:+d}st) MIDI={cur.onset_s:8.3f}s "
          f"{mm.pos(cur.onset_s):<14s} WAVref={gt_time:8.3f}s")
    print(f"    status={m.status:<17s} vicinity=±{m.vicinity_ms:.1f}ms  "
          f"eyebrow={m.eyebrow_ms:.1f}ms  candidates={m.candidates_seen} "
          f"credible={m.credible_candidates}")

    if m.candidate is None:
        if m.runner_up is not None:
            c=m.runner_up; b=c.boundary
            print(f"    nearest/best rejected B{b.index}: "
                  f"R{b.left_region}({b.left_pitch:.2f}) -> "
                  f"R{b.right_region}({b.right_pitch:.2f}) "
                  f"Δ={c.delta_ms:+.1f}ms reason={c.reason}")
            print(f"    geometry: src_err={c.source_err:.2f}st "
                  f"dst_err={c.dest_err:.2f}st interval_err={c.interval_err:.2f}st "
                  f"direction={'YES' if c.direction_match else 'NO'} "
                  f"harmonic={c.harmonic_source}/{c.harmonic_dest}")
        return

    c=m.candidate; b=c.boundary
    print(f"    MATCH B{b.index}: R{b.left_region} {b.left_pitch:.2f}st -> "
          f"R{b.right_region} {b.right_pitch:.2f}st "
          f"(Δpitch={b.interval_st:+.2f}st) WAV={b.time_s:.3f}s "
          f"{mm.pos(b.time_s-(gt_time-cur.onset_s))}")
    print(f"    geometry: src_err={c.source_err:.2f}st "
          f"dst_err={c.dest_err:.2f}st interval_err={c.interval_err:.2f}st "
          f"direction={'YES' if c.direction_match else 'NO'} "
          f"reason={c.reason} harmonic={c.harmonic_source}/{c.harmonic_dest}")
    print_timing(mm,c.delta_ms,cur.onset_s)

    a,ad=boundary_to_structural_link(b,anchors)
    if a is None:
        print("    V3.1 cross-ref: none")
    else:
        print(f"    V3.1 cross-ref only: {a.decision} O{a.obj:02d} L{a.link+1} "
              f"S{a.region} link-mid minus boundary={ad:+.1f}ms")

    if m.runner_up is not None:
        r=m.runner_up; rb=r.boundary
        print(f"    runner-up B{rb.index}: R{rb.left_region}->{rb.right_region} "
              f"Δ={r.delta_ms:+.1f}ms score={r.correspondence_score:.3f} "
              f"geometry={r.source_err:.2f}/{r.dest_err:.2f}/{r.interval_err:.2f}st")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("eckf_csv",type=Path)
    ap.add_argument("midi",type=Path)
    ap.add_argument("--fixed-offset",type=float,default=None)
    ap.add_argument("--offset-min",type=float,default=-15.)
    ap.add_argument("--offset-max",type=float,default=15.)
    args=ap.parse_args()

    midi=load_midi_notes(args.midi)
    mm=MidiMap(args.midi)
    data=load_eckf_csv(args.eckf_csv)

    # Independent V1 alignment remains exactly conceptually separate.
    cfg=V1Config(global_offset_min_s=args.offset_min,
                 global_offset_max_s=args.offset_max)
    primitive=extract_wav_events(data,cfg)
    if args.fixed_offset is None:
        off,score=estimate_global_offset(midi,data,primitive,cfg)
        source="independent V1 estimate"
    else:
        off=float(args.fixed_offset); score=float("nan")
        source="FIXED independent V1 offset"
    sections=section_offsets(midi,data,primitive,off)

    # Frozen stack: unchanged.
    interp,st,anchors,hz=frozen(data)
    gt=np.asarray([n.onset_s+off for n in midi.notes],dtype=float)
    boundaries=stable_boundaries(interp)

    # Only genuine GT pitch changes. Rearticulation timing is a separate future
    # problem because identical-pitch note-on boundaries have no pitch geometry.
    change_indices=[i for i in range(1,len(midi.notes))
                    if midi.notes[i].pitch!=midi.notes[i-1].pitch]
    matches=[match_gt_change(i,midi,gt,mm,boundaries) for i in change_indices]

    print("="*148)
    print("MIDI <-> FROZEN WAV PITCH-BOUNDARY CORRESPONDENCE V4")
    print("="*148)
    print(f"MIDI: {args.midi}")
    print(f"ECKF: {args.eckf_csv}")
    print(f"MIDI notes={len(midi.notes)} pitch_changes={len(change_indices)} "
          f"rows={len(data.time_s)} analysis={hz:.3f}Hz")
    print(f"frozen regions={len(interp.regions)} WAV stable boundaries={len(boundaries)}")
    print(f"V3.1 objects={st.total_objects} links={st.total_links} "
          f"JOIN={st.join_links} SPLIT={st.split_links} UNRESOLVED={st.unresolved_links}")
    print(f"alignment={source}; WAV_time=MIDI_time+offset; offset={off:+.3f}s",end="")
    if np.isfinite(score): print(f" score={score:.4f}")
    else: print()
    for k in ("EARLY","MIDDLE","LATE"):
        if k in sections:
            print(f"  {k:<6} {sections[k][0]:+.3f}s score={sections[k][1]:.4f}")
    print("position=bar|beat|sixteenth-slot(+fraction)")
    print("V4 principle: VICINITY RETRIEVES; PITCH GEOMETRY MATCHES; TIMING IS MEASURED AFTER MATCHING.")
    print(f"vicinity = clip({VICINITY_EYEBROW_MULT:g} * adaptive-eyebrow, "
          f"{MIN_VICINITY_MS:.0f}..{MAX_VICINITY_MS:.0f}ms)")
    print("All matching thresholds are diagnostic only. Frozen production decisions are untouched.")

    print("\n"+"="*148)
    print("GT PITCH CHANGES -> FROZEN WAV STRUCTURAL PITCH BOUNDARIES")
    print("="*148)
    for m in matches:
        print_match(mm,midi,gt,m,anchors)

    matched=[m for m in matches if m.status=="MATCHED"]
    ambiguous=[m for m in matches if m.status=="AMBIGUOUS"]
    missing=[m for m in matches if m.status=="NO_CORRESPONDENCE"]

    print("\n"+"="*148)
    print("SUMMARY")
    print("="*148)
    print(f"GT pitch changes:       {len(matches)}")
    print(f"MATCHED:                {len(matched)}")
    print(f"AMBIGUOUS:              {len(ambiguous)}")
    print(f"NO_CORRESPONDENCE:      {len(missing)}")

    timed=matched+ambiguous
    abs_ms=[abs(m.candidate.delta_ms) for m in timed if m.candidate is not None]
    bins("credible match timing",abs_ms)

    ordinary=[]; eyebrow=[]; large=[]; extreme=[]
    for m in timed:
        if m.candidate is None: continue
        _,_,_,thr,flag=mm.timing_metrics(
            m.candidate.delta_ms,midi.notes[m.gt_i].onset_s)
        {"ordinary":ordinary,"EYEBROW":eyebrow,
         "LARGE":large,"EXTREME":extreme}[flag].append(m)
    print(f"timing flags on credible matches only: ordinary={len(ordinary)} "
          f"EYEBROW={len(eyebrow)} LARGE={len(large)} EXTREME={len(extreme)}")

    print("\n"+"="*148)
    print("INSPECTION QUEUE A — CREDIBLE CORRESPONDENCE WITH BPM-ADAPTIVE TIMING FLAG")
    print("="*148)
    flagged=eyebrow+large+extreme
    if not flagged: print("none")
    for m in flagged:
        c=m.candidate; n=midi.notes[m.gt_i]; p=midi.notes[m.gt_i-1]
        bpm,beats,eighths,thr,flag=mm.timing_metrics(c.delta_ms,n.onset_s)
        print(f"{flag:7s} GT{n.index:03d} {note_name(p.pitch)}->{note_name(n.pitch)} "
              f"{mm.pos(n.onset_s):<14s} B{c.boundary.index} "
              f"Δ={c.delta_ms:+.1f}ms ({beats:+.3f} beats, {eighths:+.3f} eighths) "
              f"threshold={thr:.1f}ms reason={c.reason}")

    print("\n"+"="*148)
    print("INSPECTION QUEUE B — AMBIGUOUS STRUCTURAL CORRESPONDENCE")
    print("="*148)
    if not ambiguous: print("none")
    for m in ambiguous:
        n=midi.notes[m.gt_i]; p=midi.notes[m.gt_i-1]
        a=m.candidate; r=m.runner_up
        print(f"GT{n.index:03d} {note_name(p.pitch)}->{note_name(n.pitch)} "
              f"{mm.pos(n.onset_s):<14s} "
              f"B{a.boundary.index} Δ={a.delta_ms:+.1f}ms score={a.correspondence_score:.3f} | "
              f"B{r.boundary.index} Δ={r.delta_ms:+.1f}ms score={r.correspondence_score:.3f}")

    print("\n"+"="*148)
    print("INSPECTION QUEUE C — NO CREDIBLE WAV PITCH-BOUNDARY CORRESPONDENCE IN VICINITY")
    print("="*148)
    if not missing: print("none")
    for m in missing:
        n=midi.notes[m.gt_i]; p=midi.notes[m.gt_i-1]
        print(f"GT{n.index:03d} {note_name(p.pitch)}->{note_name(n.pitch)} "
              f"MIDI={n.onset_s:.3f}s {mm.pos(n.onset_s):<14s} "
              f"WAVref={gt[m.gt_i]:.3f}s vicinity=±{m.vicinity_ms:.1f}ms "
              f"candidates={m.candidates_seen}")

    # Reverse coverage: which frozen WAV boundaries were never selected by a
    # credible GT match?  These are candidates for expressive/ornamental
    # structure, omitted GT detail, or tracking structure.  No label is assigned.
    used={m.candidate.boundary.index for m in timed if m.candidate is not None}
    unused=[b for b in boundaries if b.index not in used]
    print("\n"+"="*148)
    print("REVERSE COVERAGE — FROZEN WAV BOUNDARIES NOT USED BY A CREDIBLE GT MATCH")
    print("="*148)
    print(f"unused={len(unused)}/{len(boundaries)}")
    for b in unused:
        # nearest GT change only for context; not a timing judgment
        if change_indices:
            gi=min(change_indices,key=lambda i:abs(float(gt[i])-b.time_s))
            n=midi.notes[gi]; p=midi.notes[gi-1]
            d=1000*(b.time_s-float(gt[gi]))
            print(f"B{b.index:03d} R{b.left_region}->{b.right_region} "
                  f"{b.left_pitch:.2f}->{b.right_pitch:.2f}st "
                  f"WAV={b.time_s:.3f}s nearest-context GT{n.index:03d} "
                  f"{note_name(p.pitch)}->{note_name(n.pitch)} Δ={d:+.1f}ms")
        else:
            print(f"B{b.index:03d} R{b.left_region}->{b.right_region} "
                  f"{b.left_pitch:.2f}->{b.right_pitch:.2f}st WAV={b.time_s:.3f}s")

    print("\n"+"="*148)
    print("INVARIANTS / INTERPRETATION")
    print("="*148)
    print(f"V3.1 structural links preserved: {len(anchors)}/{st.total_links}")
    print("Validity walls are not bridged when constructing frozen WAV boundaries.")
    print("GT same-pitch rearticulations are intentionally excluded from V4 pitch-geometry matching.")
    print("NO_CORRESPONDENCE means no credible pitch-geometric counterpart IN VICINITY; it is NOT a timing error.")
    print("AMBIGUOUS means multiple credible pitch-geometric counterparts compete.")
    print("Timing flags are computed ONLY after credible structural correspondence.")
    print("No gesture labels. No frozen threshold changed.")

if __name__=="__main__":
    main()
