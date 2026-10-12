#!/usr/bin/env python3
"""Blind, read-only validation of V12 vs Harmonic Detective V2.

Prepare:
 python tests/audit_harmonic_reference_validation_v2.py prepare \
   --detective-v2 tests/harmonic_detective_v2.zip --wav-root tests \
   --out tests/harmonic_reference_v2

Annotate ONLY the central 80 ms TARGET WAV in annotation_blind.csv. Context WAV is for orientation, never label all notes in it. If TARGET spans a transition or is unidentifiable, mark unknown.
Annotate ONLY annotation_blind.csv. Fill reference_midi (integer 0..127), or
reference_status=unknown/no_pitch; optionally condition=noisy/clean/unknown.
A MIDI note is justified only with independent acoustic/manual evidence. Do not
peek at predictions_frozen.csv before finalizing the labels.

Evaluate:
 python tests/audit_harmonic_reference_validation_v2.py evaluate \
   --out tests/harmonic_reference_v2

Requires numpy and soundfile for WAV snippet export. The report uses stdlib only.
No source data, production verdicts, or annotations are overwritten.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import zipfile, io

SONGS=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
FIELDS=['item_id','song','time_s','target_file','context_file','target_start_s','target_end_s','reference_status','reference_midi','reference_confidence','condition','transition','annotation_notes']

def rows_from(source, name):
    if source.is_dir():
        p=source/name
        if not p.is_file(): raise FileNotFoundError(p)
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    with zipfile.ZipFile(source) as z:
        matches=[n for n in z.namelist() if n==name or n.endswith('/'+name)]
        if len(matches)!=1:raise ValueError(f'Expected exactly one {name}, found {len(matches)}')
        return list(csv.DictReader(io.StringIO(z.read(matches[0]).decode('utf-8-sig'))))

def write_csv(path, records, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)

def num(x):
    try:
        v=float(x);return v if math.isfinite(v) else None
    except(TypeError,ValueError):return None

def midi(hz):
    return round(69+12*math.log2(hz/440)) if hz and hz>0 else None

def label(r):
    if r.get('diagnostic_proposed_hz','').strip():return 'proposal'
    if r.get('v12_champion_hz','').strip():return 'champion_control'
    return 'abstention_control'

def prepare(a):
    if a.seed<0 or a.cluster_seconds<=0 or a.clip_seconds<=0 or a.target_seconds<=0 or a.target_seconds>a.clip_seconds:raise ValueError('Invalid settings')
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError(f'Output already contains files: {a.out}; choose a new directory')
    a.out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(a.seed)
    pool={}
    for song in SONGS:
        rr=rows_from(a.detective_v2,f'{song}_hung_jury_audit.csv')
        times=set();pool[song]=[]
        for r in rr:
            t=num(r.get('time_s'))
            if t is None or t in times:raise ValueError(f'Invalid or duplicate {song} time {r.get("time_s")}')
            times.add(t);r['song']=song;r['_time']=t;r['_stratum']=label(r)
            pool[song].append(r)
    desired={'RATATA':(a.ratata_proposals,a.ratata_controls),
             'Ochiitai':(a.other_proposals,a.other_controls),
             'PREDESTINATI':(a.other_proposals,a.other_controls),
             'Trandafiri':(a.other_proposals,a.other_controls)}
    selected=[]; shortage=[]
    # Sampling *clusters* rather than individual 10 ms observations limits correlated examples.
    # One observation per time cluster (anchored to fixed time bins).
    for song, (nproposal,ncontrol) in desired.items():
        groups=defaultdict(list)
        for r in pool[song]:groups[int(r['_time']/a.cluster_seconds)].append(r)
        bins=list(groups);rng.shuffle(bins)
        # Independently shuffle within bins; no preference to high-margin predictions.
        for b in bins:rng.shuffle(groups[b])
        for stratum,target in [('proposal',nproposal),('champion_control',ncontrol),('abstention_control',ncontrol)]:
            remaining=[]
            for b in bins:
                if any(x['song']==song and int(x['_time']/a.cluster_seconds)==b for x in selected):continue
                options=[r for r in groups[b] if r['_stratum']==stratum]
                if options:remaining.append(options[0])
            rng.shuffle(remaining)
            choice=remaining[:target];selected.extend(choice)
            if len(choice)<target:shortage.append(f'{song} {stratum}: selected {len(choice)}/{target}; insufficient independent clusters')
    # Group neighboring clips in same dataset split; each sampled cluster is unique and
    # blocks of cluster indices stay in same split via deterministic block hash.
    selected.sort(key=lambda r:(SONGS.index(r['song']),r['_time']))
    try:import numpy as np;import soundfile as sf
    except ImportError as e:raise RuntimeError('Snippet export requires numpy and soundfile') from e
    waveforms={}; clips=a.out/'clips';clips.mkdir()
    annotations=[];frozen=[];counts=Counter()
    for i,r in enumerate(selected,1):
        song=r['song']; t=r['_time']; id_=f'HR{i:04d}';path=a.wav_root/f'{song}.wav'
        if not path.is_file():raise FileNotFoundError(f'Missing audio: {path}')
        if song not in waveforms:waveforms[song]=sf.read(path,always_2d=True,dtype='float32')
        data,sr=waveforms[song]; half=a.clip_seconds/2
        lo=max(0,int(round((t-half)*sr)));hi=min(len(data),int(round((t+half)*sr)))
        if hi<=lo:raise ValueError(f'Clip is empty: {song}@{t}')
        # Preserve native samplerate and channels; no processing, filters or normalization.
        context=f'clips/{id_}_context.wav';sf.write(a.out/context,data[lo:hi],sr,subtype='PCM_16')
        # Only the narrow, explicitly defined target is to be annotated.
        target_lo=max(0,int(round((t-a.target_seconds/2)*sr)))
        target_hi=min(len(data),int(round((t+a.target_seconds/2)*sr)))
        if target_hi<=target_lo:raise ValueError(f'Target is empty: {song}@{t}')
        target=f'clips/{id_}_TARGET.wav'
        sf.write(a.out/target,data[target_lo:target_hi],sr,subtype='PCM_16')
        # Keep dataset partition hidden from annotator; contiguous time blocks don't cross partitions.
        block=int(t//a.split_block_seconds)
        key=f'{a.seed}:{song}:{block}'.encode();u=int(hashlib.sha256(key).hexdigest()[:12],16)/float(16**12)
        split='calibration' if u<a.calibration_fraction else 'holdout'
        annotations.append({'item_id':id_,'song':song,'time_s':f'{t:.6f}','target_file':target,'context_file':context,
                            'target_start_s':f'{target_lo/sr:.6f}', 'target_end_s':f'{target_hi/sr:.6f}',
                            'reference_status':'','reference_midi':'','reference_confidence':'',
                            'condition':'','transition':'','annotation_notes':''})
        frozen.append({'item_id':id_,'song':song,'time_s':f'{t:.6f}','stratum':r['_stratum'],
                       'split':split,'v12_outcome':r.get('v12_outcome',''),
                       'v12_hz':r.get('v12_champion_hz',''),
                       'v12_midi':midi(num(r.get('v12_champion_hz'))),
                       'detective_proposal_hz':r.get('diagnostic_proposed_hz',''),
                       'detective_midi':midi(num(r.get('diagnostic_proposed_hz'))),
                       'detective_score':r.get('best_score',''),
                       'detective_margin':r.get('harmonic_margin',''),
                       'detective_reason':r.get('reason','')})
        counts[(song,r['_stratum'],split)]+=1
    write_csv(a.out/'annotation_blind.csv',annotations,FIELDS)
    ff=list(frozen[0]) if frozen else []
    write_csv(a.out/'predictions_frozen.csv',frozen,ff)
    manifest={'protocol':'harmonic_reference_validation_v2','source_v2':str(a.detective_v2),
              'source_zip_sha256':hashlib.sha256(a.detective_v2.read_bytes()).hexdigest() if a.detective_v2.is_file() else None,
              'seed':a.seed,'cluster_seconds':a.cluster_seconds,'split_block_seconds':a.split_block_seconds,
              'calibration_fraction':a.calibration_fraction,'clip_seconds':a.clip_seconds,
              'target_seconds':a.target_seconds,'annotation_scope':'TARGET WAV only; context is navigation only',
              'requested':desired,'selection_counts':{'|'.join(k):v for k,v in sorted(counts.items())},
              'shortages':shortage,'annotation_file':'annotation_blind.csv',
              'blinding':'Do not open predictions_frozen.csv or reports before locking annotation_blind.csv',
              'notes':'Manual labels are not automatically ground truth. Independent annotations recommended.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared {len(selected)} excerpts across {len(SONGS)} WAVs. Output: {a.out}')
    for s in shortage:print('SAMPLING LIMIT:',s)
    print('Annotate annotation_blind.csv without opening predictions_frozen.csv; then run evaluate.')

def evaluate(a):
    d=a.out;ann=list(csv.DictReader((d/'annotation_blind.csv').open(encoding='utf-8-sig',newline='')))
    pred=list(csv.DictReader((d/'predictions_frozen.csv').open(encoding='utf-8-sig',newline='')))
    amap={r['item_id']:r for r in ann};pmap={r['item_id']:r for r in pred}
    if len(amap)!=len(ann) or set(amap)!=set(pmap) or len(pmap)!=len(pred):raise ValueError('IDs missing/duplicated/mismatched')
    scores=[];bad=[];counters=defaultdict(Counter)
    for id_,p in pmap.items():
        r=amap[id_];st=r['reference_status'].strip().lower();confidence=r['reference_confidence'].strip().lower()
        if st not in ('','pitched','no_pitch','unknown'):bad.append(f'{id_}: invalid reference_status');continue
        if st=='':continue
        if st=='pitched':
            raw=r['reference_midi'].strip()
            if not re.fullmatch(r'\d{1,3}',raw) or not 0<=int(raw)<=127:bad.append(f'{id_}: pitched requires MIDI integer 0..127');continue
            ref=int(raw)
        else:
            if r['reference_midi'].strip():bad.append(f'{id_}: non-pitched must have blank reference_midi');continue
            ref=None
        if confidence not in ('high','medium','low','unknown'):bad.append(f'{id_}: confidence must be high/medium/low/unknown');continue
        condition=r['condition'].strip().lower()
        if condition not in ('noisy','clean','unknown'):bad.append(f'{id_}: condition must be noisy/clean/unknown');continue
        transition=r['transition'].strip().lower()
        if transition not in ('yes','no','unknown'):bad.append(f'{id_}: transition must be yes/no/unknown');continue
        dm=p['detective_midi'];vm=p['v12_midi']
        candidate=int(dm) if dm not in ('','None') else None
        v12=int(vm) if vm not in ('','None') else None
        # Primary evaluation excludes ambiguous labels, uncertain voice, and transitions.
        eligible=st=='pitched' and confidence=='high' and transition=='no'
        error=candidate-ref if eligible and candidate is not None else None
        row={'item_id':id_,'song':p['song'],'split':p['split'],'stratum':p['stratum'],
             'reference_status':st,'reference_midi':ref if ref is not None else '',
             'reference_confidence':confidence,'condition':condition,'transition':transition,
             'v12_midi':v12 if v12 is not None else '', 'detective_midi':candidate if candidate is not None else '',
             'primary_eligible':int(eligible),'note_error_semitones':error if error is not None else '',
             'exact_note':int(error==0) if error is not None else '',
             'octave_error':int(error!=0 and error%12==0) if error is not None else '',
             'other_note_error':int(error!=0 and error%12!=0) if error is not None else ''}
        scores.append(row)
        for k in [('overall',),('song',p['song']),('split',p['split']),('stratum',p['stratum']),
                  ('song_condition',p['song'],condition),('split_stratum',p['split'],p['stratum'])]:
            c=counters[k];c['annotated']+=1
            if eligible:c['eligible']+=1
            if eligible and candidate is not None:
                c['proposed_evaluable']+=1
                c['correct']+=error==0;c['octave_errors']+=error!=0 and error%12==0
                c['other_errors']+=error!=0 and error%12!=0
            if st=='no_pitch':c['no_pitch_labels']+=1
            if st=='unknown':c['unknown_labels']+=1
    if bad:raise ValueError('Invalid annotations:\n'+'\n'.join(bad[:30]))
    if not scores:raise ValueError('No completed reference annotations. Fill annotation_blind.csv first.')
    report=[]
    for k,c in sorted(counters.items()):
        v=c['proposed_evaluable'];row={'slice':'/'.join(k),**c,'accuracy_if_proposed':round(c['correct']/v,4) if v else '',
                                     'note':'Descriptive only; clustered samples, no independence claim'}
        report.append(row)
    write_csv(d/'evaluation_per_item.csv',scores,list(scores[0]))
    fields=['slice','annotated','eligible','proposed_evaluable','correct','octave_errors','other_errors','no_pitch_labels','unknown_labels','accuracy_if_proposed','note']
    write_csv(d/'evaluation_summary.csv',report,fields)
    print('Annotation rows evaluated:',len(scores))
    for r in report:
        if r['slice'] in ('overall','song/RATATA','split/holdout','split/calibration'):
            print(r['slice'],'eligible=',r.get('eligible',0),'proposals=',r.get('proposed_evaluable',0),
                  'correct=',r.get('correct',0),'octave errors=',r.get('octave_errors',0),
                  'accuracy=',r['accuracy_if_proposed'])
    print('Reports:',d/'evaluation_summary.csv',d/'evaluation_per_item.csv')
    print('Do not choose thresholds using holdout labels; inspect calibration first.')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sp=p.add_subparsers(dest='command',required=True)
    a=sp.add_parser('prepare');a.add_argument('--detective-v2',type=Path,default=Path('tests/harmonic_detective_v2.zip'))
    a.add_argument('--wav-root',type=Path,default=Path('tests'));a.add_argument('--out',type=Path,default=Path('tests/harmonic_reference_v2'))
    a.add_argument('--seed',type=int,default=20260921);a.add_argument('--ratata-proposals',type=int,default=80)
    a.add_argument('--ratata-controls',type=int,default=20);a.add_argument('--other-proposals',type=int,default=35)
    a.add_argument('--other-controls',type=int,default=10);a.add_argument('--cluster-seconds',type=float,default=.30)
    a.add_argument('--split-block-seconds',type=float,default=3.0);a.add_argument('--calibration-fraction',type=float,default=.60)
    a.add_argument('--clip-seconds',type=float,default=1.2)
    a.add_argument('--target-seconds',type=float,default=0.08)
    b=sp.add_parser('evaluate');b.add_argument('--out',type=Path,default=Path('tests/harmonic_reference_v2'))
    args=p.parse_args()
    if args.command=='prepare':
        if not 0<args.calibration_fraction<1 or args.split_block_seconds<args.cluster_seconds: p.error('Invalid sampling/splitting fractions')
        prepare(args)
    else:evaluate(args)
if __name__=='__main__':main()
