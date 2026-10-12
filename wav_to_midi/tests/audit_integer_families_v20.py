#!/usr/bin/env python3
"""V20: adaptive-only, observational integer-period discrimination and range audit.

Consumes V19 waveform CSVs; remeasures original WAV at candidate-specific apertures
for integer families. Neither song-range nor musical pitch is used to select F0.
No production, MIDI, continuity or interval penalty. Read-only shadow diagnostic.
Run: python tests/audit_integer_families_v20.py --case all
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
WIDTHS=(24,40,64)
def cents(a,b):
    return abs(1200*math.log2(a/b)) if a>0 and b>0 else float('inf')
def number(v):
    try:
        x=float(v);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None
def measure(x,sr,hz):
    """Direct normalized period coherence and spectral energy NOT a vocal-source test."""
    if x is None or len(x)<64 or hz<=0:return None
    x=np.asarray(x,dtype=np.float64);x=x-x.mean();n=len(x)
    if n*hz/sr<3 or float(x@x)<1e-18:return None
    p=sr/hz;lo=max(2,int(round(p*.975)));hi=min(n//2,int(round(p*1.025)))
    if hi<lo:return None
    lag_scores=[]
    for lag in range(lo,hi+1):
        a=x[:-lag];b=x[lag:];d=math.sqrt(float(a@a)*float(b@b))
        lag_scores.append((float(a@b)/d if d>1e-18 else 0.,lag))
    ac,lag=max(lag_scores)
    z=np.abs(np.fft.rfft(x*np.hanning(n)))**2;total=float(z[1:].sum());df=sr/n
    if total<=1e-20:return None
    def band(freq):
        k=int(round(freq/df));rad=max(1,int(round(max(10.,.013*freq)/df)))
        return set(range(max(1,k-rad),min(len(z),k+rad+1)))
    bins=set();fund=0.
    for k in range(1,min(12,int((sr/2)/hz))+1):
        bb=band(k*hz)
        if k==1:fund=float(sum(z[j] for j in bb))/total
        bins.update(bb)
    harm=float(sum(z[j] for j in bins))/total
    return {'acf':round(ac,6),'hz_measured':round(sr/lag,5),'cycles':round(n*hz/sr,3),
            'fundamental_fraction':round(fund,7),'harmonic_fraction':round(harm,6)}
def seg(y,sr,t,width_ms):
    half=width_ms/2000.;a=int(round((t-half)*sr));b=int(round((t+half)*sr))
    return y[a:b] if a>=0 and b<=len(y) and b-a>=64 else None
def sibling_residual(x,sr,low,high):
    """Power on low-fundamental harmonics EXCLUDING bins attributable to high F0.
    Descriptive evidence: finite resolution and overlapping bins may make it untestable.
    """
    if x is None:return None
    x=np.asarray(x,dtype=np.float64);x=x-x.mean();n=len(x)
    z=np.abs(np.fft.rfft(x*np.hanning(n)))**2;total=float(z[1:].sum());df=sr/n
    if total<=1e-20:return None
    def band(hz):
        k=int(round(hz/df));r=max(1,int(round(max(10.,.013*hz)/df)))
        return set(range(max(1,k-r),min(len(z),k+r+1)))
    high_bins=set()
    for k in range(1,min(16,int((sr/2)/high))+1):high_bins.update(band(k*high))
    low_only=set()
    for k in range(1,min(16,int((sr/2)/low))+1):low_only.update(band(k*low)-high_bins)
    # If resolution doesn't permit separating low-only lines, report untestable.
    return {'low_exclusive_harmonic_fraction':round(float(sum(z[k] for k in low_only))/total,7) if low_only else None,
            'resolvable_bins':len(low_only),'bin_width_hz':round(df,3)}
def hypotheses(row):
    raw=json.loads(row.get('hypothesis_tests_json') or '[]')
    result=[]
    for h in raw:
        hz=number(h.get('hz'))
        if hz is None or not 65<=hz<=950:continue
        if any(cents(hz,z['hz'])<=35 for z in result):continue
        result.append({'hz':hz,'present_support_views':int(h.get('present_support_views') or 0),
                       'from_v18':bool(h.get('from_v18')),
                       'independently_discovered':bool(h.get('independently_discovered'))})
    return result
def pairs(hs):
    out=[]
    for i,a in enumerate(hs):
        for b in hs[i+1:]:
            low,high=sorted((a,b),key=lambda x:x['hz']);ratio=high['hz']/low['hz'];k=round(ratio)
            if 2<=k<=5 and abs(1200*math.log2(ratio/k))<=85:
                out.append((low,high,k,round(1200*math.log2(ratio/k),2)))
    return out
def stable_anchor(row,hs):
    """Conservative control-only anchor; cannot certify active voice or octave correctness."""
    if row.get('population')!='supported_control':return None
    supported=[h for h in hs if h['present_support_views']>=2]
    if len(supported)!=1:return None
    if any(2<=round(max(h['hz'],supported[0]['hz'])/min(h['hz'],supported[0]['hz']))<=5 and
           cents(max(h['hz'],supported[0]['hz']),round(max(h['hz'],supported[0]['hz'])/min(h['hz'],supported[0]['hz']))*min(h['hz'],supported[0]['hz']))<85
           for h in hs if h is not supported[0] and h['present_support_views']>=1):return None
    return supported[0]['hz']
def classify(mlo,mhi,exclusive):
    if mlo is None or mhi is None:return 'insufficient_direct_measurement'
    # These flags highlight evidence to inspect, not select a winner.
    if mlo['acf']<.72 and mhi['acf']>=.72:return 'higher_period_supported_lower_weak'
    if mhi['acf']<.72 and mlo['acf']>=.72:return 'lower_period_supported_higher_weak'
    if exclusive is None or exclusive['low_exclusive_harmonic_fraction'] is None:return 'both_periods_measurable_spectral_resolution_limited'
    if exclusive['low_exclusive_harmonic_fraction']>=.035:return 'both_periods_measurable_low_exclusive_energy_present'
    return 'both_periods_measurable_low_exclusive_energy_weak'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*CASES),default='all')
    p.add_argument('--v19-dir',type=Path,default=Path('tests/adaptive_waveform_v19'))
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/integer_families_v20'))
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'v20_adaptive_only_integer_family_waveform_and_range_audit',
              'production_used':False,'midi_used':False,'new_f0_selected':False,
              'source_identity_verified':False,'range_prior_used_for_decisions':False,
              'range_estimated_from_sparse_v19_supported_controls_only':True,
              'acoustic_thresholds_are_diagnostic_not_adjudication':True,'cases':{}}
    for case in CASES if args.case=='all' else (args.case,):
        src=args.v19_dir/f'{case}_waveform.csv';wav=args.wav_dir/f'{case}.wav'
        if not src.is_file():raise FileNotFoundError(src)
        if not wav.is_file():raise FileNotFoundError(wav)
        y,sr=sf.read(wav,dtype='float64',always_2d=False)
        if y.ndim==2:y=y.mean(axis=1)
        anchors=[];total=Counter()
        # Two sequential streaming passes: first reads EXISTING records, no WAV remeasurement;
        # second measures waveform. Range data are diagnostic and never affect the second pass.
        with src.open(newline='') as f:
            for row in csv.DictReader(f):
                total[row['population']]+=1
                h=stable_anchor(row,hypotheses(row))
                if h:anchors.append(h)
        sufficient=len(anchors)>=40
        bounds=[float(np.percentile(anchors,q)) for q in (1,10,50,90,99)] if sufficient else None
        range_report={'anchor_count':len(anchors),'minimum_for_report':40,
                      'robust_percentiles_hz':dict(zip(('p01','p10','p50','p90','p99'),bounds)) if bounds else None,
                      'status':'provisional_diagnostic_only' if sufficient else 'insufficient_unambiguous_controls',
                      'not_hard_vocal_limits':True,'anchors_not_voice_verified':True}
        counts=Counter();family_count=0;pair_rows=0
        rec_path=args.out_dir/f'{case}_observations.csv'
        pair_path=args.out_dir/f'{case}_integer_pairs.csv'
        with src.open(newline='') as inp,rec_path.open('w',newline='') as outf,pair_path.open('w',newline='') as pf:
            ow=csv.DictWriter(outf,fieldnames=['case','sample','time_s','population','v19_screen','hypotheses_count',
              'integer_pairs','family_diagnostic','range_context_json','final_f0_selected','source_verified'])
            pw=csv.DictWriter(pf,fieldnames=['case','sample','time_s','population','low_hz','high_hz','integer_ratio',
              'ratio_offset_cents','low_v19_present_support','high_v19_present_support',
              'apertures_json','family_diagnostic','range_context_json','final_f0_selected'])
            ow.writeheader();pw.writeheader()
            for row in csv.DictReader(inp):
                hs=hypotheses(row);pp=pairs(hs);t=float(row['time_s'])
                if abs(int(row['sample'])/sr-t)>.003:raise ValueError(f'{case}: sample/time mismatch at {t}')
                labels=[];family_count+=bool(pp)
                for low,high,k,offset in pp:
                    widths={}
                    for ms in WIDTHS:
                        x=seg(y,sr,t,ms)
                        widths[str(ms)]={'low':measure(x,sr,low['hz']),
                                         'high':measure(x,sr,high['hz']),
                                         'low_exclusive':sibling_residual(x,sr,low['hz'],high['hz'])}
                    e=widths['64'];label=classify(e['low'],e['high'],e['low_exclusive'])
                    labels.append(label);counts['pair:'+label]+=1;pair_rows+=1
                    context={'available':sufficient,'low_below_p01':low['hz']<bounds[0] if bounds else None,
                             'high_below_p01':high['hz']<bounds[0] if bounds else None,
                             'low_above_p99':low['hz']>bounds[4] if bounds else None,
                             'high_above_p99':high['hz']>bounds[4] if bounds else None}
                    pw.writerow({'case':case,'sample':row['sample'],'time_s':row['time_s'],
                      'population':row['population'],'low_hz':low['hz'],'high_hz':high['hz'],
                      'integer_ratio':k,'ratio_offset_cents':offset,
                      'low_v19_present_support':low['present_support_views'],
                      'high_v19_present_support':high['present_support_views'],
                      'apertures_json':json.dumps(widths,separators=(',',':')),
                      'family_diagnostic':label,'range_context_json':json.dumps(context,separators=(',',':')),
                      'final_f0_selected':False})
                label=('no_integer_pair' if not labels else
                       labels[0] if len(set(labels))==1 else 'multiple_integer_pairs_mixed_evidence')
                counts['observation:'+label]+=1
                ow.writerow({'case':case,'sample':row['sample'],'time_s':row['time_s'],
                  'population':row['population'],'v19_screen':row['diagnostic_screen'],
                  'hypotheses_count':len(hs),'integer_pairs':len(pp),'family_diagnostic':label,
                  'range_context_json':json.dumps({'range_available':sufficient,'anchor_count':len(anchors)},separators=(',',':')),
                  'final_f0_selected':False,'source_verified':False})
        manifest['cases'][case]={'v19_observations':dict(total),'observations_with_integer_pairs':family_count,
                                'integer_pairs_tested':pair_rows,'range':range_report,'diagnostics':dict(counts)}
        print(f'{case}: observations={sum(total.values())} integer-family observations={family_count} pairs={pair_rows} range anchors={len(anchors)}',flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('V20 is diagnostic ONLY; no candidate chosen and no range-based exclusions.',flush=True)
if __name__=='__main__':main()
