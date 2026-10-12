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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("eckf_csv",type=Path); ap.add_argument("midi",type=Path)
    ap.add_argument("--fixed-offset",type=float,default=None)
    ap.add_argument("--offset-min",type=float,default=-15.); ap.add_argument("--offset-max",type=float,default=15.)
    args=ap.parse_args()
    midi=load_midi_notes(args.midi); mm=MidiMap(args.midi); data=load_eckf_csv(args.eckf_csv)
    cfg=V1Config(global_offset_min_s=args.offset_min,global_offset_max_s=args.offset_max)
    primitive=extract_wav_events(data,cfg)
    if args.fixed_offset is None:
        off,score=estimate_global_offset(midi,data,primitive,cfg); source="independent V1 estimate"
    else:
        off=float(args.fixed_offset); score=float("nan"); source="FIXED independent V1 offset"
    sections=section_offsets(midi,data,primitive,off)
    interp,st,anchors,hz=frozen(data)
    splits=tuple(a for a in anchors if a.decision=="SPLIT")
    joins=tuple(a for a in anchors if a.decision=="JOIN")
    unresolved=tuple(a for a in anchors if a.decision=="UNRESOLVED")
    gt=np.asarray([n.onset_s+off for n in midi.notes])
    # Frozen region boundaries (independent of V3.1 decisions).
    bounds=[]
    for l,r in zip(interp.regions[:-1],interp.regions[1:]):
        bounds.append((.5*(float(l.end_time_s)+float(r.start_time_s)),int(l.index),int(r.index),l.kind.name,r.kind.name))
    btimes=[x[0] for x in bounds]

    print("="*132); print("MIDI <-> FROZEN WAV STRUCTURE CORRESPONDENCE V2"); print("="*132)
    print(f"MIDI notes: {len(midi.notes)}  rows: {len(data.time_s)}  rate: {hz:.3f}Hz")
    print(f"regions: {len(interp.regions)}  objects: {st.total_objects}  links: {st.total_links}")
    print(f"JOIN={st.join_links} SPLIT={st.split_links} UNRESOLVED={st.unresolved_links}")
    print(f"alignment: {source}; WAV_time=MIDI_time+offset; offset={off:+.3f}s",end="")
    if np.isfinite(score):print(f" score={score:.4f}")
    else:print()
    for k in ("EARLY","MIDDLE","LATE"):
        if k in sections: print(f"  {k:<6} {sections[k][0]:+.3f}s score={sections[k][1]:.4f}")
    print("position format: bar|beat|sixteenth-slot; +fraction means between sixteenth slots")

    gta=[]; gts=[]; gtb=[]
    print("\n"+"="*132+"\nGT MIDI ONSETS -> FROZEN WAV STRUCTURE\n"+"="*132)
    for n,t in zip(midi.notes,gt):
        aa,ad,asd=nearest_anchor(float(t),anchors); sa,sd,ssd=nearest_anchor(float(t),splits)
        bd,bi=nearest_point(float(t),btimes); gta.append(ad); gts.append(sd); gtb.append(abs(bd))
        print(f"GT{n.index:03d} {note_name(n.pitch):4s} MIDI={n.onset_s:8.3f}s {mm.pos(n.onset_s):<13s} WAV={t:8.3f}s")
        if bi>=0:
            b=bounds[bi]; print(f"    region boundary {b[0]:8.3f}s {mm.pos(b[0]-off):<13s} Δ={bd:+7.1f}ms R{b[1]}({b[3]})->R{b[2]}({b[4]})")
        if aa:
            print(f"    V3.1 link      {aa.start:8.3f}-{aa.end:8.3f}s {mm.pos(aa.mid-off):<13s} dist={ad:6.1f}ms edgeΔ={asd:+7.1f} {aa.decision} O{aa.obj} L{aa.link+1} S{aa.region}")
        if sa:
            print(f"    nearest SPLIT  {sa.start:8.3f}-{sa.end:8.3f}s {mm.pos(sa.mid-off):<13s} dist={sd:6.1f}ms edgeΔ={ssd:+7.1f} O{sa.obj} L{sa.link+1}")

    dd={k:[] for k in ("SPLIT","JOIN","UNRESOLVED")}
    print("\n"+"="*132+"\nV3.1 LINKS -> NEAREST GT MIDI ONSET\n"+"="*132)
    for a in anchors:
        vals=[interval_dist(float(t),a) for t in gt]; i=min(range(len(vals)),key=lambda k:vals[k][0])
        dist,signed=vals[i]; n=midi.notes[i]; dd[a.decision].append(dist)
        print(f"{a.decision:10s} O{a.obj:02d} L{a.link+1} S{a.region:<3d} WAV={a.start:8.3f}-{a.end:8.3f}s MID={a.mid:8.3f}s {mm.pos(a.mid-off):<13s}")
        print(f"    stable={a.pitch:6.2f}st dur={a.dur:5.0f}ms nearestGT=GT{n.index:03d} {note_name(n.pitch):4s} {mm.pos(n.onset_s):<13s} dist={dist:6.1f}ms edgeΔ={signed:+7.1f}ms")
        print(f"    connector={a.connectors:<24s} boundary={a.boundaries:<40s} families C/B={a.nc}/{a.nb}")

    print("\n"+"="*132+"\nTEMPORAL CORRESPONDENCE SUMMARY\n"+"="*132)
    print("Bins are descriptive only; NOT correctness thresholds.\n")
    bins("GT -> frozen region boundary",gtb); bins("GT -> any V3.1 link",gta); bins("GT -> SPLIT",gts)
    print(); bins("SPLIT -> nearest GT",dd["SPLIT"]); bins("JOIN -> nearest GT",dd["JOIN"]); bins("UNRESOLVED -> nearest GT",dd["UNRESOLVED"])

    print("\n"+"="*132+"\nINSPECTION QUEUES\n"+"="*132)
    print("SPLIT farther than 250ms from every GT onset:")
    bad=[(a,d) for a,d in zip(splits,dd["SPLIT"]) if d>250]
    for a,d in bad: print(f"  O{a.obj:02d} L{a.link+1} S{a.region} {a.mid:.3f}s {mm.pos(a.mid-off)} nearestGT={d:.1f}ms C={a.connectors} B={a.boundaries}")
    if not bad:print("  none")
    print("\nJOIN within 150ms of a GT onset:")
    near=[(a,d) for a,d in zip(joins,dd["JOIN"]) if d<=150]
    for a,d in near: print(f"  O{a.obj:02d} L{a.link+1} S{a.region} {a.mid:.3f}s {mm.pos(a.mid-off)} nearestGT={d:.1f}ms C={a.connectors} B={a.boundaries}")
    if not near:print("  none")
    print("\nGT farther than 250ms from every V3.1 link interval:")
    far=[(n,t,d) for n,t,d in zip(midi.notes,gt,gta) if d>250]
    for n,t,d in far: print(f"  GT{n.index:03d} {note_name(n.pitch):4s} {mm.pos(n.onset_s):<13s} MIDI={n.onset_s:.3f}s WAV={t:.3f}s nearestLink={d:.1f}ms")
    if not far:print("  none")

    print("\n"+"="*132+"\nINVARIANTS\n"+"="*132)
    print(f"JOIN + SPLIT + UNRESOLVED = {st.join_links}+{st.split_links}+{st.unresolved_links}={st.total_links}")
    print(f"anchors represented exactly once: {len(anchors)}/{st.total_links}")
    print("No musical gesture labels. No frozen threshold changed.")

if __name__=="__main__": main()
