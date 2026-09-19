#!/usr/bin/env python3
"""Read-only past/center/future temporal-support audit of ALL V6 review observations.
Run from repo root: python tests/audit_temporal_views_v7.py --case all
Requires tests/diagnose_adaptive_evidence.py and original mono WAVs in tests/.
Never treats production as truth; no F0/voicing decisions or production modifications.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from diagnose_adaptive_evidence import analyze, credible

NAMES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
FOCUS={'remote_periodicity','testable_no_local_support','block_timing_ambiguity','candidate_during_unvoiced_block'}
FIELD=('case','region','sample','time_s','comparison','primary_mechanism','focus','production_voiced_samples',
       'production_median_hz','adaptive_observational_hz','aperture_ms','view_ms','period_hz','period_origin',
       'past_json','center_json','future_json','geometry','frequency_relation','cycle_warning',
       'interpretation','status_at_instant','status_source','f0_applied','voicing_applied','tracker_reset_requested')

def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None

def read_rows(path):
    with path.open(newline='') as f:
        yield from csv.DictReader(f)

def window(audio,fs,sample,n,position):
    if position=='past': lo=sample-n
    elif position=='center':lo=sample-n//2
    else:lo=sample
    hi=lo+n
    return (audio[lo:hi] if lo>=0 and hi<=len(audio) else None,lo,hi)

def periodic_support(y,fs,hz):
    if y is None or hz is None or hz<=0:return {'testability':'not_available','cycles':None,'acf_at_period':None,'fundamental_fraction':None}
    n=len(y);cycles=n*hz/fs
    y=np.asarray(y,dtype=np.float64);y=y-y.mean()
    p=float(y@y)
    if p<1e-18:return {'testability':'silent','cycles':cycles,'acf_at_period':None,'fundamental_fraction':None}
    lag=round(fs/hz)
    if lag<2 or lag>=n-2:return {'testability':'lag_unavailable','cycles':cycles,'acf_at_period':None,'fundamental_fraction':None}
    a,b=y[:-lag],y[lag:];den=math.sqrt(float(a@a)*float(b@b))
    acf=float(a@b)/den if den>1e-15 else None
    angle=2*np.pi*hz*np.arange(n)/fs
    basis=np.column_stack((np.cos(angle),np.sin(angle)))
    fitted=basis@np.linalg.lstsq(basis,y,rcond=None)[0]
    frac=float(np.clip(float(fitted@fitted)/p,0,1))
    return {'testability':'testable' if cycles>=3 else 'insufficient_cycles','cycles':cycles,
            'acf_at_period':acf,'fundamental_fraction':frac}

def measure(audio,fs,sample,view_ms,position,reference_hz):
    n=max(64,round(fs*view_ms/1000))
    y,lo,hi=window(audio,fs,sample,n,position)
    o={'start_sample':lo,'end_sample':hi,'start_s':lo/fs,'end_s':hi/fs,'duration_ms':1000*n/fs,
       'rms':None,'raw_candidates':[],'eligible_candidates':[],'reference_support':None,'status':'missing_at_boundary'}
    if y is None:return o
    rms,candidates=analyze(y,fs,max_candidates=8)
    o['rms']=rms;o['raw_candidates']=candidates
    o['eligible_candidates']=[c for c in candidates if credible(c) and n*c['hz']/fs>=3]
    o['reference_support']=periodic_support(y,fs,reference_hz)
    o['status']='measured'
    return o

def strength(view):
    s=view['reference_support']
    return (s is not None and s['testability']=='testable' and s['acf_at_period'] is not None
            and s['acf_at_period']>=.70 and s['fundamental_fraction'] is not None
            and s['fundamental_fraction']>=.03)

def reference(row):
    # Only frequencies ALREADY independently measured by the experimental WAV analyser.
    hz=number(row.get('adaptive_observational_hz'))
    if hz and hz>0:return hz,'adaptive_waveform_observation'
    try: stages=json.loads(row.get('measured_stage_evidence_json') or '[]')
    except (TypeError,ValueError):stages=[]
    for stage in stages:
        for candidate in stage.get('eligible_local_candidates',[]):
            h=number(candidate.get('hz'))
            if h and h>0:return h,'v6_measured_local_period'
    # No synthesized candidate; unanchored view raw candidates are still reported independently.
    return None,'no_preexisting_measured_period'

def relation(views):
    vals=[]
    for name,v in views.items():
        cc=v['eligible_candidates']
        if len(cc)==1:vals.append((name,cc[0]['hz']))
    if len(vals)<2:return 'insufficient_unique_view_candidates'
    if max(h for _,h in vals)/min(h for _,h in vals)<=2**(70/1200):return 'view_frequencies_agree_70c'
    return 'view_frequencies_disagree_70c'

def classify(views,hz):
    if any(v['status']!='measured' for v in views.values()):return 'boundary_missing','inconclusive_missing_window'
    if hz is None:
        strong=[k for k,v in views.items() if len(v['eligible_candidates'])>0]
        if not strong:return 'no_measured_period_in_views','unresolved_no_period'
        return 'unanchored_independent_view_candidates','investigate_competing_periods_not_voicing'
    s={k:strength(v) for k,v in views.items()}
    if s['past'] and not s['center'] and not s['future']:return 'past_only_period_support','possible_earlier_audio_leakage'
    if s['future'] and not s['center'] and not s['past']:return 'future_only_period_support','possible_lookahead_or_onset'
    if s['center'] and s['past'] and s['future']:return 'all_three_support_period','periodic_across_timestamp_not_proven_vocal'
    if s['center']:return 'center_support_with_asymmetry','local_periodic_fragment_review'
    if not any(s.values()):
        if any(v['reference_support']['testability']=='insufficient_cycles' for v in views.values()):
            return 'insufficient_cycles','inconclusive_cycle_count'
        return 'no_view_support_for_reference','reference_period_not_locally_supported'
    return 'flanking_only_or_asymmetric','possible_temporal_attribution_problem'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*NAMES),default='all')
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--v6-dir',type=Path,default=Path('tests/acoustic_review_v6'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/temporal_views_v7'))
    p.add_argument('--view-ms',type=float,default=40.,help='Equal length past, centered, future windows (default 40 ms)')
    p.add_argument('--all-four-apertures',action='store_true',help='Audit 24, 40, 64, 96 ms equal-view lengths (more expensive)')
    args=p.parse_args()
    if not 12<=args.view_ms<=128:p.error('--view-ms must be between 12 and 128')
    names=NAMES if args.case=='all' else (args.case,)
    # Verify inputs before writing any output; never silently reduce all to an available subset.
    for name in names:
        if not (args.v6_dir/f'{name}_review.csv').is_file():p.error(f'missing {args.v6_dir/name}_review.csv')
        if not (args.wav_dir/f'{name}.wav').is_file():p.error(f'missing {args.wav_dir/name}.wav')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'temporal_views_v7','read_only':True,'production_is_not_ground_truth':True,
              'no_pitch_or_voicing_applied':True,'focus_mechanisms':sorted(FOCUS),
              'view_lengths_ms':[24,40,64,96] if args.all_four_apertures else [args.view_ms],
              'correct_ratata_djuvvs':[[36.923,37.435],[41.025,41.538],[49.230,49.743]],
              'trandafiri':'steady i between consonants; not a glissando','cases':{}}
    for name in names:
        audio,fs=sf.read(args.wav_dir/f'{name}.wav',dtype='float64')
        if audio.ndim!=1:p.error(f'{name}: expected mono WAV; got shape {audio.shape}')
        counts=Counter();comparisons=Counter();focus=Counter();rows=0
        out=args.out_dir/f'{name}_temporal_views.csv'
        with out.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELD);writer.writeheader()
            for r in read_rows(args.v6_dir/f'{name}_review.csv'):
                sample=int(r['sample']);hz,origin=reference(r)
                for ms in manifest['view_lengths_ms']:
                    views={pos:measure(audio,fs,sample,ms,pos,hz) for pos in ('past','center','future')}
                    geometry,interpretation=classify(views,hz)
                    status='block_level_only_not_exact_instant'
                    writer.writerow({'case':name,'region':r.get('region',''),'sample':sample,'time_s':sample/fs,
                        'comparison':r['comparison'],'primary_mechanism':r['primary_mechanism'],
                        'focus':r['primary_mechanism'] in FOCUS,'production_voiced_samples':r.get('production_voiced_samples',''),
                        'production_median_hz':r.get('production_median_hz',''),
                        'adaptive_observational_hz':r.get('adaptive_observational_hz',''),
                        'aperture_ms':r.get('first_local_eligible_aperture_ms',''),
                        'view_ms':ms,'period_hz':hz,'period_origin':origin,
                        **{pos+'_json':json.dumps(views[pos],allow_nan=False) for pos in views},
                        'geometry':geometry,'frequency_relation':relation(views),
                        'cycle_warning':any(v['reference_support'] and v['reference_support']['testability']=='insufficient_cycles' for v in views.values()),
                        'interpretation':interpretation,'status_at_instant':'unknown','status_source':status,
                        'f0_applied':False,'voicing_applied':False,'tracker_reset_requested':False})
                    rows+=1;counts[geometry]+=1;comparisons[r['comparison']]+=1
                    if r['primary_mechanism'] in FOCUS:focus[geometry]+=1
        manifest['cases'][name]={'observations':rows//len(manifest['view_lengths_ms']),
            'output_rows':rows,'sample_rate':fs,'geometry_counts':dict(counts),
            'comparison_counts_per_view':dict(comparisons),'focus_geometry_counts':dict(focus),'output':str(out)}
        print(name,'observations',manifest['cases'][name]['observations'],'views',rows,'geometry',dict(counts),flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('READ ONLY. Interpret geometry as evidence hypotheses, never final voicing or F0.')
if __name__=='__main__':main()
