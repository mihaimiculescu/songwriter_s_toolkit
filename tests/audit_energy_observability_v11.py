#!/usr/bin/env python3
"""V11: read-only, four-song energy-conditioned acoustic observability audit.

Inputs: original mono WAVs and V6, V9, V10 CSVs. No MIDI or production edits.
Energy NEVER labels a sample unvoiced or confirms a pitch. The unchanged, existing
vocal_transition_penalty is evaluated for *measured* candidate comparisons only,
when a prior independently supported acoustic anchor can be found. Neither the
penalty nor a frequency range creates or replaces a candidate.

Run: python tests/audit_energy_observability_v11.py --case all
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={'RATATA': [('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),
                    ('djuvv_3',49.230,49.743),('submultiple',42.90,43.10)],
         'Ochiitai':[('transition',42.19,42.40)],
         'Trandafiri':[('steady_i_between_consonants',10.70,10.95)],
         'PREDESTINATI':[('control_38',37.97,38.13),('control_53',53.52,53.73)]}

def num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None

def cents(a,b):return abs(1200*math.log2(a/b))

def load(path):
    with path.open(newline='') as f:yield from csv.DictReader(f)

def segment(audio,fs,center_s,duration_ms):
    n=max(16,round(fs*duration_ms/1000));c=round(center_s*fs)
    lo=c-n//2;hi=lo+n
    if lo<0 or hi>len(audio):return None
    return audio[lo:hi]

def rms(y):return float(np.sqrt(np.mean(np.square(y,dtype=np.float64)))) if y is not None and len(y) else None

def features(y,fs):
    if y is None or len(y)<32:return dict(rms=None,flatness=None,high_fraction=None)
    x=np.asarray(y,dtype=np.float64); r=rms(x)
    x=x-x.mean(); win=np.hanning(len(x));p=np.abs(np.fft.rfft(x*win))**2
    f=np.fft.rfftfreq(len(x),1/fs); valid=(f>=80)&(f<=min(10000,fs*.48))
    a=p[valid]
    if not len(a) or np.sum(a)<=1e-20:flat=None;high=None
    else:
        flat=float(np.exp(np.mean(np.log(a+1e-20)))/(np.mean(a)+1e-20))
        high=float(np.sum(p[(f>=3000)&valid])/(np.sum(a)+1e-20))
    return dict(rms=r,flatness=flat,high_fraction=high)

def context_curve(audio,fs,hop_ms=10.,win_ms=12.):
    hop=max(1,round(hop_ms*fs/1000));n=max(16,round(win_ms*fs/1000))
    # Integral squared waveform; local RMS computed without huge sliding copies.
    sq=np.r_[0.,np.cumsum(np.square(audio,dtype=np.float64))]
    starts=np.arange(0,max(0,len(audio)-n+1),hop)
    times=(starts+n/2)/fs
    curve=np.sqrt(np.maximum((sq[starts+n]-sq[starts])/n,0))
    return times,curve

def compare_energy(t,ts,curve,window=.30):
    a=np.searchsorted(ts,t-window);b=np.searchsorted(ts,t+window,side='right')
    local=curve[a:b]
    if not len(local):return None,None,None
    ref=float(np.quantile(local,.80));floor=float(np.quantile(local,.20))
    at=np.searchsorted(ts,t);at=min(len(curve)-1,at)
    return ref,floor,float(curve[at])

def v9_candidates(row):
    try:raw=json.loads(row.get('candidates_json') or '[]')
    except (ValueError,TypeError):return []
    chosen=[]
    for c in raw:
        if c.get('view')!='center':continue
        hz=num(c.get('measured_hz'))
        if not hz or hz<=0:continue
        if not any(cents(hz,q['hz'])<35 for q in chosen):
            chosen.append(dict(hz=hz,eligible=bool(c.get('eligible')),acf=num(c.get('acf')),
                               cmndf=num(c.get('cmndf')),fundamental_fraction=num(c.get('fundamental_fraction')),
                               harmonic_fraction=num(c.get('harmonic_fraction'))))
    return chosen

def shift_for(candidates,shift_rows,view_ms):
    relevant=[r for r in shift_rows if int(float(r['view_ms']))==int(view_ms)]
    for c in candidates:
        a=[r for r in relevant if num(r.get('anchor_hz')) and cents(c['hz'],float(r['anchor_hz']))<=35]
        if a:
            a.sort(key=lambda r:abs(cents(c['hz'],float(r['anchor_hz']))))
            x=a[0];c['shift_category']=x['category'];c['shift_supported']=int(x['supported_shift_count'])
            c['shift_spread_cents']=num(x['measured_spread_cents'])
    return candidates

def stable(c):
    return c.get('eligible') and c.get('shift_category') in ('shift_stable_acoustic_candidate',
                                                              'mostly_shift_stable_candidate')

def prior_anchor(history,t,max_age=.30):
    # Only PRIOR acoustic measurements; never the ECKF's currently disputed state.
    for old_t,old_hz in reversed(history):
        if old_t<t-1e-7 and t-old_t<=max_age:return old_t,old_hz
        if t-old_t>max_age:break
    return None,None

def calculate_penalties(func,prior_hz,elapsed_ms,candidates,prod_hz):
    if func is None:return {'status':'existing_curve_unavailable'}
    if prior_hz is None or elapsed_ms is None or elapsed_ms<=0:
        return {'status':'no_reliable_prior_acoustic_anchor'}
    vals=[]
    for c in candidates:
        if c['eligible']:
            try:p=float(func(12*math.log2(c['hz']/prior_hz),elapsed_ms))
            except Exception as exc:return {'status':'curve_error','detail':str(exc)}
            vals.append({'measured_hz':c['hz'],'penalty':p})
    prod=None
    if prod_hz and prod_hz>0:
        try:prod=float(func(12*math.log2(prod_hz/prior_hz),elapsed_ms))
        except Exception as exc:return {'status':'curve_error','detail':str(exc)}
    return {'status':'reported_only','prior_hz':prior_hz,'elapsed_ms':elapsed_ms,
            'candidate_penalties':vals,'production_frequency_hz':prod_hz,'production_penalty':prod,
            'does_not_select_pitch':True}

def energy_pattern(center,left,right,reference):
    if any(x is None for x in (center,left,right,reference)) or reference<=1e-10:return 'insufficient_energy_context'
    # Ratios are descriptive experimental bins, NOT thresholds for vocal presence.
    rel=center/reference
    if left>center*1.6 and center>=right*1.15:return 'falling_energy_relative_to_past'
    if right>center*1.6 and center>=left*1.15:return 'rising_energy_toward_future'
    if rel<.25:return 'locally_low_energy_without_direction'
    if max(left,right,center)<=1e-9:return 'near_numerical_silence'
    return 'no_strong_energy_asymmetry'

def mechanism(row,v9,candidates,pattern):
    eligible=[c for c in candidates if c['eligible']]
    stable_candidates=[c for c in eligible if stable(c)]
    if v9 is None:return 'no_v9_aperture_data'
    if v9.get('temporal_attribution') in ('past_only','future_only','distant_only_periodicity'):
        return 'check_temporal_attribution'
    if not eligible:return 'period_not_eligible_energy_cannot_rescue'
    if not stable_candidates:return 'period_shift_robustness_not_established'
    if len(eligible)>1:return 'acoustically_competing_candidates'
    if pattern in ('falling_energy_relative_to_past','locally_low_energy_without_direction'):
        return 'supported_period_in_low_or_decaying_energy_inspect_source'
    if pattern=='rising_energy_toward_future':return 'supported_period_near_possible_onset_inspect_source'
    return 'supported_period_source_identity_unresolved'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*CASES),default='all')
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--v6-dir',type=Path,default=Path('tests/acoustic_review_v6'))
    p.add_argument('--v9-dir',type=Path,default=Path('tests/measured_period_families_v9'))
    p.add_argument('--v10-dir',type=Path,default=Path('tests/shift_robustness_v10'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/energy_observability_v11'))
    args=p.parse_args()
    names=CASES if args.case=='all' else (args.case,)
    for name in names:
        for path in (args.wav_dir/f'{name}.wav',args.v6_dir/f'{name}_review.csv',
                     args.v9_dir/f'{name}_measured_families.csv',args.v10_dir/f'{name}_shift_summary.csv'):
            if not path.is_file():p.error(f'missing required input: {path}')
    try:
        from python_eckf.trajectory_resolver import vocal_transition_penalty
        curve=vocal_transition_penalty;curve_status='existing_function_imported_unchanged'
    except (ImportError,AttributeError) as exc:
        curve=None;curve_status='unavailable: '+str(exc)
        print('WARNING: existing penalty unavailable; its fields will say unavailable. No substitute curve.',file=sys.stderr)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'version':'v11','cases':{},'curve_status':curve_status,
              'energy_does_not_decide_voicing':True,'curve_does_not_decide_pitch':True,
              'production_modified':False,'no_midi':True,
              'correct_ratata_regions':REGIONS['RATATA']}
    for name in names:
        audio,fs=sf.read(args.wav_dir/f'{name}.wav',dtype='float64')
        if audio.ndim!=1:p.error(f'{name}: expected mono WAV')
        ts,curve_rms=context_curve(audio,fs)
        v9={(int(x['sample']),int(float(x['view_ms']))):x for x in load(args.v9_dir/f'{name}_measured_families.csv')}
        shift=defaultdict(list)
        for x in load(args.v10_dir/f'{name}_shift_summary.csv'):
            shift[int(x['sample'])].append(x)
        review=sorted(load(args.v6_dir/f'{name}_review.csv'),key=lambda r:int(r['sample']))
        out=[];counts=Counter();history=[];seen_history=set();region_stats=defaultdict(Counter)
        for r in review:
            sample=int(r['sample']);t=sample/fs
            reference,floor,at=compare_energy(t,ts,curve_rms)
            cur=features(segment(audio,fs,t,12),fs)
            past=features(segment(audio,fs,t-.018,12),fs)
            future=features(segment(audio,fs,t+.018,12),fs)
            wider=features(segment(audio,fs,t,40),fs)
            left,right=past['rms'],future['rms']
            pattern=energy_pattern(cur['rms'],left,right,reference)
            ratio=cur['rms']/reference if cur['rms'] is not None and reference and reference>1e-12 else None
            # 12ms-to-12ms ratio, and log energy gradient; zero protected, diagnostic only.
            slope_db=20*math.log10((right+1e-12)/(left+1e-12)) if left is not None and right is not None else None
            entry={'case':name,'sample':sample,'time_s':t,'review_comparison':r['comparison'],
                   'primary_mechanism_v6':r['primary_mechanism'],
                   'production_block_voiced_samples':r['production_voiced_samples'],
                   'production_median_hz':r['production_median_hz'],
                   'rms_present_12ms':cur['rms'],'rms_past_12ms':left,'rms_future_12ms':right,
                   'rms_centered_40ms':wider['rms'],'rms_context_p80_600ms':reference,
                   'rms_context_p20_600ms':floor,'rms_present_vs_context_p80':ratio,
                   'energy_slope_past_to_future_db':slope_db,'energy_pattern_diagnostic':pattern,
                   'spectral_flatness_present':cur['flatness'],
                   'high_band_fraction_present':cur['high_fraction'],
                   'spectral_flatness_past':past['flatness'],
                   'spectral_flatness_future':future['flatness'],
                   'high_band_fraction_past':past['high_fraction'],
                   'high_band_fraction_future':future['high_fraction'],
                   'no_voicing_or_pitch_decision':True}
            per_aperture=[];reference_candidates=[]
            for ms in (24,40,64,96):
                row=v9.get((sample,ms))
                if row is None:continue
                candidates=shift_for(v9_candidates(row),shift.get(sample,[]),ms)
                per_aperture.append({'aperture_ms':ms,'actual_start_sample':row['center_start_sample'],
                       'actual_end_sample':row['center_end_sample'],
                       'temporal_attribution':row['temporal_attribution'],
                       'acoustic_comparison':row['acoustic_comparison'],'candidates':candidates})
                if ms==40:reference_candidates=candidates
            # A reliable historical acoustic anchor requires a unique eligible candidate,
            # V10 shift stability, and no contradictory eligible candidates at 40 ms.
            elig=[c for c in reference_candidates if c['eligible']]
            accepted_anchor=elig[0]['hz'] if len(elig)==1 and stable(elig[0]) else None
            prior_t,prior_hz=prior_anchor(history,t)
            elapsed=(t-prior_t)*1000 if prior_t is not None else None
            penalty=calculate_penalties(curve,prior_hz,elapsed,reference_candidates,num(r.get('production_median_hz')))
            if accepted_anchor is not None and sample not in seen_history:
                history.append((t,accepted_anchor));seen_history.add(sample)
            primary=v9.get((sample,40))
            classification=mechanism(r,primary,reference_candidates,pattern)
            entry.update(aperture_evidence_json=json.dumps(per_aperture),
                         acoustic_prior_hz=prior_hz,acoustic_prior_time_s=prior_t,
                         interval_penalty_json=json.dumps(penalty),
                         investigation_class=classification,
                         region_tags=';'.join(tag for tag,a,b in REGIONS.get(name,[]) if a<=t<=b))
            out.append(entry);counts[classification]+=1;counts['pattern:'+pattern]+=1
            for tag in entry['region_tags'].split(';') if entry['region_tags'] else ():
                region_stats[tag][classification]+=1
        path=args.out_dir/f'{name}_energy_observability.csv'
        if not out:raise RuntimeError(f'{name}: empty review')
        with path.open('w',newline='') as file:
            w=csv.DictWriter(file,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
        manifest['cases'][name]={'review_observations':len(out),'investigation_classes':dict(counts),
                                  'annotated_regions':{k:dict(v) for k,v in region_stats.items()},
                                  'sample_rate':fs,'output':str(path)}
        print(f'{name}: {len(out)} review observations; {dict(Counter(x["investigation_class"] for x in out))}',flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('READ-ONLY AUDIT. Low energy never sets UNVOICED; transition penalty never sets F0.')
if __name__=='__main__':main()
