#!/usr/bin/env python3
"""Read-only raw-ECKF lock / shared-window audit. No project imports or production changes.
Usage: python tests/diagnose_eckf_lock.py ECKF.csv GroundTruth.mid --offset 0 --start 57.5 --end 60 --trace
Times: MIDI note seconds + --offset = CSV/WAV seconds. Offset is an assumption, not estimated here.
"""
import argparse,csv,math,collections
from pathlib import Path
import mido

def midi_notes(path):
    mf=mido.MidiFile(path); tempo=500000; tick=0; sec=0.; active=collections.defaultdict(list); out=[]
    for msg in mido.merge_tracks(mf.tracks):
        dt=msg.time; sec+=mido.tick2second(dt,mf.ticks_per_beat,tempo); tick+=dt
        if msg.type=='set_tempo': tempo=msg.tempo
        if msg.type=='note_on' and msg.velocity>0: active[(msg.channel,msg.note)].append((sec,tick))
        elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
            key=(msg.channel,msg.note)
            if active[key]:
                start,st=active[key].pop(0);out.append((start,sec,msg.note,msg.channel,st,tick))
    return sorted(out),mf.ticks_per_beat

def hz_pitch(hz): return 69+12*math.log2(hz/440) if hz>0 and math.isfinite(hz) else float('nan')
def name(p): return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][p%12]+str(p//12-1)
def load(path):
    with open(path,newline='') as f:
        r=csv.DictReader(f);need={'time_s','f0_hz','first_pass_valid','corrected_valid','first_pass_reason','validity_reason','correction_reason'}
        if not need.issubset(r.fieldnames or []):raise ValueError('Missing CSV fields: '+str(sorted(need-set(r.fieldnames or []))))
        rows=list(r)
    for row in rows:
        row['t']=float(row['time_s']);row['f']=float(row['f0_hz']);row['p']=hz_pitch(row['f'])
    if any(b['t']<=a['t'] for a,b in zip(rows,rows[1:])):raise ValueError('Non-increasing CSV timestamps')
    return rows

def median(xs):
    xs=sorted(xs);n=len(xs);return (xs[(n-1)//2]+xs[n//2])/2 if n else float('nan')
def fmt(x):return f'{x:.2f}' if math.isfinite(x) else '-'
def runs(rows, classify):
    out=[]
    for r in rows:
        k=classify(r)
        if not out or out[-1][0]!=k:out.append([k,r['t'],r['t'],[r]])
        else:out[-1][2]=r['t'];out[-1][3].append(r)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv');ap.add_argument('midi');ap.add_argument('--offset',type=float,default=0.,help='WAV time minus MIDI time; use 0 and 0.210 separately to test alignment')
    ap.add_argument('--start',type=float,default=None);ap.add_argument('--end',type=float,default=None)
    ap.add_argument('--trace',action='store_true');ap.add_argument('--min-run-ms',type=float,default=80.);ap.add_argument('--near-st',type=float,default=1.0)
    args=ap.parse_args();rr=load(args.csv);notes,tpb=midi_notes(args.midi)
    start=args.start if args.start is not None else rr[0]['t'];end=args.end if args.end is not None else rr[-1]['t']
    if end<=start:ap.error('end must exceed start')
    rows=[r for r in rr if start<=r['t']<=end]
    if not rows:ap.error('No CSV rows in selected interval')
    print('READ-ONLY ECKF LOCK AUDIT');print(f'CSV={args.csv} MIDI={args.midi} TPB={tpb} OFFSET={args.offset:+.3f}s (assumed; NOT estimated)')
    print(f'WINDOW={start:.3f}..{end:.3f}s rows={len(rows)}; source f0_hz is RAW; no inference of internal ECKF state from CSV')
    print('CODE QUESTIONS: 2048 role, Kalman state, reset and excursion-penalty placement require inspecting the actual source; CSV cannot answer them.')
    relevant=[n for n in notes if n[0]+args.offset<end and n[1]+args.offset>start]
    print('\nMIDI NOTE INTERVALS (mapped to WAV by assumed offset):')
    for a,b,p,ch,_,_ in relevant:print(f'  {name(p):4} MIDI {a:8.3f}..{b:8.3f} WAV {a+args.offset:8.3f}..{b+args.offset:8.3f} ch={ch}')
    print('\nNOTE-INTERIOR AUDIT (exclude 30ms around each note boundary; overlap flagged):')
    for a,b,p,ch,_,_ in relevant:
        left=max(start,a+args.offset+.030);right=min(end,b+args.offset-.030)
        subset=[r for r in rows if left<=r['t']<right]
        if not subset:continue
        overlapping=[n for n in relevant if n[2]!=p and n[0]+args.offset<right and n[1]+args.offset>left]
        pos=[r for r in subset if math.isfinite(r['p'])]
        near=sum(abs(r['p']-p)<=args.near_st for r in pos)
        ratios={label:sum(abs(r['p']-(p-shift))<=args.near_st for r in pos) for label,shift in [('fundamental',0),('half',12),('quarter',24),('third',12*math.log2(3)),('double',-12)]}
        reasons=collections.Counter((r['validity_reason'],r['correction_reason']) for r in subset)
        print(f'  {name(p):4} {left:.3f}..{right:.3f} n={len(subset)} raw_positive={len(pos)} median_raw_midi={fmt(median([r["p"] for r in pos]))} near={near} ratios={ratios} overlapping_other_pitch={bool(overlapping)}')
        print('    reasons:',reasons.most_common(4))
        if pos:
            steps=[abs(y['p']-x['p']) for x,y in zip(pos,pos[1:]) if y['t']-x['t']<=.021]
            print(f'    median_adjacent_raw_step={fmt(median(steps))}st; median_expected_error={fmt(median([r["p"]-p for r in pos]))}st')
    print('\nGLOBAL LOW-F0 RUNS (<120Hz; raw, independent of MIDI):')
    lowruns=runs(rows,lambda r:'LOW' if r['f']>0 and r['f']<120 else ('NO_F0' if r['f']<=0 or not math.isfinite(r['f']) else 'OTHER'))
    suspicious=[]
    for k,a,b,part in lowruns:
        dur=(b-a)*1000
        if k=='LOW' and dur>=args.min_run_ms:
            active=[name(p) for x,y,p,_,_,_ in relevant if x+args.offset<=b and y+args.offset>=a]
            suspicious.append((a,b,part,active))
            print(f'  {a:.3f}..{b:.3f} duration={dur:.0f}ms n={len(part)} median_f0={fmt(median([r["f"] for r in part]))}Hz median_pitch={fmt(median([r["p"] for r in part]))} MIDI_overlap={active}')
    if not suspicious:print('  none meeting duration threshold')
    print('\nSHARED START / OVERLAPPING PROBE WARNING:')
    print('  This script does NOT parse V4.5 probe windows or GT IDs. Equal probe starts in V4.5 can reuse the SAME CSV samples; do not count them as independent tracker failures.')
    print('  Low-F0 runs above are unique time intervals and are counted once.')
    print('\nEVIDENCE LIMITS: note-boundary timing depends on --offset; MIDI is a note skeleton, not phoneme timing; ratios are proximity tests, not proof of a subharmonic mechanism.')
    if args.trace:
        print('\nRAW TIME TRACE: time rawHz rawMIDI activeMIDI first_pass corrected first_reason validity_reason correction_reason')
        for r in rows:
            active=[name(p) for a,b,p,_,_,_ in relevant if a+args.offset<=r['t']<b+args.offset]
            print(f'{r["t"]:8.3f} {r["f"]:9.3f} {fmt(r["p"]):>6} {",".join(active) or "-":>10} {r["first_pass_valid"]:>2} {r["corrected_valid"]:>2} {r["first_pass_reason"]} | {r["validity_reason"]} | {r["correction_reason"]}')
if __name__=='__main__':main()
