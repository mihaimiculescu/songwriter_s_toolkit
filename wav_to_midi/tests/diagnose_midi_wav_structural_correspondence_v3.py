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

def outer_stable_regions(interp, anchor):
    """
    Find the nearest frozen STABLE_TARGET before and after the link's central
    stable target.  Pure observation: no manufactured targets.
    """
    regs=interp.regions
    pos=next((i for i,r in enumerate(regs) if int(r.index)==anchor.region),None)
    if pos is None: return None,None
    left=next((r for r in reversed(regs[:pos]) if r.kind is RegionKind.STABLE_TARGET),None)
    right=next((r for r in regs[pos+1:] if r.kind is RegionKind.STABLE_TARGET),None)
    return left,right

def nearest_gt_index(t, gt_times, require_pitch_change=False, notes=None):
    candidates=[]
    for i,x in enumerate(gt_times):
        if require_pitch_change:
            if i==0 or notes[i].pitch==notes[i-1].pitch:
                continue
        candidates.append((abs(float(x)-t),i))
    if not candidates: return -1
    return min(candidates)[1]

def gt_context(notes,i):
    prev=notes[i-1] if i>0 else None
    cur=notes[i]
    interval=(cur.pitch-prev.pitch) if prev is not None else float("nan")
    return prev,cur,interval

def pitch_harmonic_warning(wav_pitch_st, midi_pitch):
    if not np.isfinite(wav_pitch_st): return "-"
    d=float(wav_pitch_st)-float(midi_pitch)
    for target,label in ((12.0,"WAV~2x"),(-12.0,"WAV~1/2x"),
                         (19.01955,"WAV~3x"),(-19.01955,"WAV~1/3x")):
        if abs(d-target)<=.75:
            return f"{label} ({d:+.2f}st)"
    return "-"

def print_timing(mm, delta_ms, midi_sec, indent="    "):
    bpm,beats,eighths,thr,flag=mm.timing_metrics(delta_ms,midi_sec)
    print(f"{indent}timing: Δ={delta_ms:+7.1f}ms  beats={beats:+.3f}  "
          f"eighths={eighths:+.3f}  BPM={bpm:.2f}  "
          f"eyebrow={thr:.1f}ms  flag={flag}")

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

    # ------------------------------------------------------------------
    # INDEPENDENT ALIGNMENT FIRST.  V3.1 gets no vote.
    # ------------------------------------------------------------------
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

    # Only after alignment is frozen do we construct frozen V3.1 structure.
    interp,st,anchors,hz=frozen(data)
    gt=np.asarray([n.onset_s+off for n in midi.notes],dtype=float)

    splits=tuple(a for a in anchors if a.decision=="SPLIT")
    joins=tuple(a for a in anchors if a.decision=="JOIN")
    unresolved=tuple(a for a in anchors if a.decision=="UNRESOLVED")

    print("="*144)
    print("MIDI <-> FROZEN WAV STRUCTURE CORRESPONDENCE V3")
    print("="*144)
    print(f"MIDI: {args.midi}")
    print(f"ECKF: {args.eckf_csv}")
    print(f"MIDI notes={len(midi.notes)}  rows={len(data.time_s)}  analysis={hz:.3f}Hz")
    print(f"regions={len(interp.regions)} objects={st.total_objects} links={st.total_links}")
    print(f"JOIN={st.join_links} SPLIT={st.split_links} UNRESOLVED={st.unresolved_links}")
    print(f"alignment={source}; WAV_time=MIDI_time+offset; offset={off:+.3f}s",end="")
    if np.isfinite(score): print(f" score={score:.4f}")
    else: print()
    for k in ("EARLY","MIDDLE","LATE"):
        if k in sections:
            print(f"  {k:<6} {sections[k][0]:+.3f}s score={sections[k][1]:.4f}")
    print("position=bar|beat|sixteenth-slot(+fraction)")
    print(f"soft timing prior: one-eighth base with sublinear tempo correction; "
          f"reference={REFERENCE_BPM:g} BPM alpha={TIMING_ALPHA:g}")
    print("flags are INSPECTION FLAGS ONLY; no structural decision is changed.")

    # ================================================================
    # 1. EVERY V3.1 LINK AGAINST NEAREST GT PITCH-CHANGE BOUNDARY
    # ================================================================
    print("\n"+"="*144)
    print("EVERY V3.1 LINK -> NEAREST GT PITCH-CHANGE BOUNDARY")
    print("="*144)

    decision_rows=[]
    for a in anchors:
        gi=nearest_gt_index(a.mid,gt,True,midi.notes)
        if gi<0: continue
        prev,cur,gt_interval=gt_context(midi.notes,gi)
        gt_wav=float(gt[gi])
        delta_ms=1000.0*(a.mid-gt_wav)
        bpm,beats,eighths,thr,tflag=mm.timing_metrics(delta_ms,cur.onset_s)
        left,right=outer_stable_regions(interp,a)

        # Central stable target compared with intended target after GT boundary.
        hw=pitch_harmonic_warning(a.pitch,cur.pitch)
        decision_rows.append((a,gi,delta_ms,thr,tflag,beats,eighths,hw))

        print(f"{a.decision:10s} O{a.obj:02d} L{a.link+1} S{a.region:<3d} "
              f"WAVmid={a.mid:8.3f}s {mm.pos(a.mid-off):<13s}")
        if prev is not None:
            print(f"    nearest GT pitch change: GT{cur.index:03d} "
                  f"{note_name(prev.pitch)}->{note_name(cur.pitch)} "
                  f"({gt_interval:+.0f}st) MIDI={cur.onset_s:8.3f}s "
                  f"{mm.pos(cur.onset_s):<13s} WAV={gt_wav:8.3f}s")
        print_timing(mm,delta_ms,cur.onset_s)
        print(f"    central stable: {a.pitch:6.2f}st dur={a.dur:.0f}ms  "
              f"harmonic-warning={hw}")
        if left is not None or right is not None:
            ltxt="-" if left is None else f"R{left.index}:{left.median_pitch_st:.2f}st"
            rtxt="-" if right is None else f"R{right.index}:{right.median_pitch_st:.2f}st"
            print(f"    WAV stable context: {ltxt} -> S{a.region}:{a.pitch:.2f}st -> {rtxt}")
        print(f"    connector={a.connectors:<24s} boundary={a.boundaries:<40s} "
              f"families C/B={a.nc}/{a.nb}")

    # ================================================================
    # 2. JOIN-SPECIFIC: GT BOUNDARY INSIDE / NEAR JOINED CONSTRUCTION
    # ================================================================
    print("\n"+"="*144)
    print("JOIN INSPECTION — DOES AN INTENDED GT PITCH CHANGE OCCUR INSIDE/NEAR THE JOIN?")
    print("="*144)
    if not joins: print("none")
    for a in joins:
        # Use central stable interval for containment; report nearest pitch-change too.
        pitch_changes=[i for i in range(1,len(midi.notes))
                       if midi.notes[i].pitch!=midi.notes[i-1].pitch]
        inside=[i for i in pitch_changes if a.start <= gt[i] <= a.end]
        gi=nearest_gt_index(a.mid,gt,True,midi.notes)
        n=midi.notes[gi]; prev=midi.notes[gi-1]
        d=1000*(a.mid-float(gt[gi]))
        print(f"JOIN O{a.obj:02d} L{a.link+1} S{a.region} "
              f"{a.start:.3f}-{a.end:.3f}s {mm.pos(a.mid-off)}")
        print(f"    GT pitch-change inside central stable interval: "
              f"{','.join('GT%03d'%midi.notes[i].index for i in inside) if inside else 'none'}")
        print(f"    nearest: GT{n.index:03d} {note_name(prev.pitch)}->{note_name(n.pitch)}")
        print_timing(mm,d,n.onset_s)
        print(f"    evidence: C={a.connectors} B={a.boundaries}")

    # ================================================================
    # 3. UNRESOLVED — PRESERVE AMBIGUITY
    # ================================================================
    print("\n"+"="*144)
    print("UNRESOLVED INSPECTION — AMBIGUITY PRESERVED")
    print("="*144)
    if not unresolved: print("none")
    for a in unresolved:
        gi=nearest_gt_index(a.mid,gt,True,midi.notes)
        n=midi.notes[gi]; prev=midi.notes[gi-1]
        d=1000*(a.mid-float(gt[gi]))
        print(f"UNRESOLVED O{a.obj:02d} L{a.link+1} S{a.region} "
              f"WAV={a.mid:.3f}s {mm.pos(a.mid-off)}")
        print(f"    nearest GT change GT{n.index:03d} "
              f"{note_name(prev.pitch)}->{note_name(n.pitch)}")
        print_timing(mm,d,n.onset_s)
        print(f"    evidence: C={a.connectors} B={a.boundaries}")

    # ================================================================
    # 4. GT PITCH CHANGES -> V3.1
    # ================================================================
    print("\n"+"="*144)
    print("GT PITCH CHANGES -> NEAREST V3.1 LINK")
    print("="*144)
    gt_rows=[]
    for i in range(1,len(midi.notes)):
        n=midi.notes[i]; prev=midi.notes[i-1]
        if n.pitch==prev.pitch: continue
        t=float(gt[i])
        a,dist,edge=nearest_anchor(t,anchors)
        if a is None: continue
        # Midpoint delta is timing diagnostic; interval distance remains structural proximity.
        mid_delta=1000*(a.mid-t)
        bpm,beats,eighths,thr,tflag=mm.timing_metrics(mid_delta,n.onset_s)
        hw=pitch_harmonic_warning(a.pitch,n.pitch)
        gt_rows.append((i,a,dist,mid_delta,thr,tflag,hw))
        print(f"GT{n.index:03d} {note_name(prev.pitch)}->{note_name(n.pitch)} "
              f"({n.pitch-prev.pitch:+d}st) MIDI={n.onset_s:.3f}s "
              f"{mm.pos(n.onset_s):<13s} WAV={t:.3f}s")
        print(f"    nearest link: {a.decision} O{a.obj:02d} L{a.link+1} S{a.region} "
              f"interval-dist={dist:.1f}ms midpointΔ={mid_delta:+.1f}ms")
        print_timing(mm,mid_delta,n.onset_s)
        print(f"    stable={a.pitch:.2f}st harmonic-warning={hw} "
              f"C={a.connectors} B={a.boundaries}")

    # ================================================================
    # 5. SOFT INSPECTION QUEUES — BPM ADAPTIVE
    # ================================================================
    print("\n"+"="*144)
    print("BPM-ADAPTIVE INSPECTION QUEUES")
    print("="*144)
    print("These are not error declarations. Artistic anticipation/posticipation remains possible.\n")

    print("A) SPLIT whose nearest GT pitch-change timing raises >= EYEBROW:")
    rows=[r for r in decision_rows if r[0].decision=="SPLIT" and abs(r[2])>=r[3]]
    if not rows: print("  none")
    for a,gi,d,thr,flag,beats,eighths,hw in rows:
        n=midi.notes[gi]
        print(f"  {flag:7s} O{a.obj:02d} L{a.link+1} S{a.region} "
              f"{a.mid:.3f}s {mm.pos(a.mid-off):<13s} "
              f"GT{n.index:03d} Δ={d:+.1f}ms ({beats:+.3f} beats, "
              f"{eighths:+.3f} eighths) threshold={thr:.1f}ms harmonic={hw}")

    print("\nB) JOIN close to an intended GT pitch change (within its adaptive eyebrow threshold):")
    jrows=[r for r in decision_rows if r[0].decision=="JOIN" and abs(r[2])<r[3]]
    if not jrows: print("  none")
    for a,gi,d,thr,flag,beats,eighths,hw in jrows:
        n=midi.notes[gi]
        print(f"  O{a.obj:02d} L{a.link+1} S{a.region} GT{n.index:03d} "
              f"Δ={d:+.1f}ms ({eighths:+.3f} eighths) threshold={thr:.1f}ms "
              f"C={a.connectors} B={a.boundaries}")

    print("\nC) GT pitch changes whose nearest V3.1 link midpoint raises >= EYEBROW:")
    grows=[r for r in gt_rows if abs(r[3])>=r[4]]
    if not grows: print("  none")
    for i,a,dist,d,thr,flag,hw in grows:
        n=midi.notes[i]; prev=midi.notes[i-1]
        bpm,beats,eighths,_,_=mm.timing_metrics(d,n.onset_s)
        print(f"  {flag:7s} GT{n.index:03d} {note_name(prev.pitch)}->{note_name(n.pitch)} "
              f"{mm.pos(n.onset_s):<13s} nearest={a.decision} O{a.obj:02d} L{a.link+1} "
              f"interval-dist={dist:.1f}ms midpointΔ={d:+.1f}ms "
              f"({eighths:+.3f} eighths) threshold={thr:.1f}ms harmonic={hw}")

    print("\nD) SPLIT with no GT pitch-change inside its central stable interval:")
    none_inside=[]
    for a in splits:
        inside=False
        for i in range(1,len(midi.notes)):
            if midi.notes[i].pitch==midi.notes[i-1].pitch: continue
            if a.start <= gt[i] <= a.end:
                inside=True; break
        if not inside: none_inside.append(a)
    if not none_inside: print("  none")
    for a in none_inside:
        gi=nearest_gt_index(a.mid,gt,True,midi.notes)
        n=midi.notes[gi]
        d=1000*(a.mid-float(gt[gi]))
        _,beats,eighths,thr,flag=mm.timing_metrics(d,n.onset_s)
        print(f"  O{a.obj:02d} L{a.link+1} S{a.region} "
              f"{a.start:.3f}-{a.end:.3f}s {mm.pos(a.mid-off):<13s} "
              f"nearestGT=GT{n.index:03d} midpointΔ={d:+.1f}ms "
              f"({eighths:+.3f} eighths) {flag}")

    print("\n"+"="*144)
    print("DESCRIPTIVE FIXED-MS BINS (retained only for cross-song comparability)")
    print("="*144)
    split_abs=[abs(r[2]) for r in decision_rows if r[0].decision=="SPLIT"]
    join_abs=[abs(r[2]) for r in decision_rows if r[0].decision=="JOIN"]
    unr_abs=[abs(r[2]) for r in decision_rows if r[0].decision=="UNRESOLVED"]
    bins("SPLIT midpoint -> GT change",split_abs)
    bins("JOIN midpoint -> GT change",join_abs)
    bins("UNRES midpoint -> GT change",unr_abs)

    print("\n"+"="*144)
    print("INVARIANTS")
    print("="*144)
    print(f"JOIN + SPLIT + UNRESOLVED = {st.join_links}+{st.split_links}+{st.unresolved_links}={st.total_links}")
    print(f"anchors represented exactly once: {len(anchors)}/{st.total_links}")
    print("V1 alignment independent. No gesture labels. No frozen threshold changed.")
    print("BPM-adaptive timing flags are observational only.")

if __name__=="__main__":
    main()
