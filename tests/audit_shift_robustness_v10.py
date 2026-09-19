#!/usr/bin/env python3
"""V10 read-only aperture-shift robustness for ALL four recordings.

Re-extracts actual ACF peaks and CMNDF minima in independently shifted windows;
V9's measured periods are comparison anchors, never forced candidates.
No production changes, MIDI, prior-pitch preference, continuity, or pitch decisions.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
SHIFTS=(-4.,-2.,0.,2.,4.)

def number(v):
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError): return None

def cents(a,b): return abs(1200*math.log2(a/b))

def bands(y,fs,hz):
    """Separate LS fundamental and joint 1..8 harmonic variance explained."""
    y=np.asarray(y,dtype=np.float64);y=y-y.mean(); power=float(y@y)
    k=min(8,int((fs*.475)//hz))
    if power<1e-16 or k<1:return None,None,0
    t=np.arange(len(y),dtype=np.float64)/fs
    p=2*np.pi*np.outer(t,hz*np.arange(1,k+1))
    mat=np.empty((len(y),2*k));mat[:,0::2]=np.cos(p);mat[:,1::2]=np.sin(p)
    try:
        fundamental=mat[:,:2]@np.linalg.lstsq(mat[:,:2],y,rcond=None)[0]
        harmonic=mat@np.linalg.lstsq(mat,y,rcond=None)[0]
    except np.linalg.LinAlgError:return None,None,0
    projections=np.hypot(*(mat.T@y).reshape(k,2).T)
    count=int(np.count_nonzero(projections>=np.max(projections)*.1)) if np.max(projections)>0 else 0
    return float(np.clip(fundamental@fundamental/power,0,1)),float(np.clip(harmonic@harmonic/power,0,1)),count

def independent_peaks(y,fs,min_hz=70.,max_hz=1000.):
    """All genuine discrete local peaks from normalized ACF. No anchor-driven search."""
    y=np.asarray(y,dtype=np.float64);n=len(y);y=y-y.mean()
    if n<32 or y@y<1e-16:return []
    lo=max(2,int(math.ceil(fs/max_hz)))
    hi=min(n//2,int(math.floor(fs/min_hz)))
    if hi<=lo+2:return []
    # FFT performs linear autocorrelation; zero-padding is computational only.
    fft_n=1 << (2*n-1).bit_length()
    Y=np.fft.rfft(y,n=fft_n)
    corr=np.fft.irfft(Y*np.conjugate(Y),n=fft_n)[:hi+2]
    sq=np.r_[0.,np.cumsum(y*y)]
    lags=np.arange(hi+2)
    left=sq[n-lags]
    right=sq[n]-sq[lags]
    den=np.sqrt(np.maximum(left*right,1e-30))
    acf=corr/den
    # Difference and cumulative-mean-normalized difference (YIN style).
    difference=np.maximum(left+right-2*corr,0.)
    running=np.cumsum(difference[1:])
    cmndf=np.ones(hi+2)
    valid=running>1e-20
    cmndf[1:][valid]=difference[1:][valid]*np.arange(1,hi+2)[valid]/running[valid]
    peaks=[]
    for lag in range(lo,hi+1):
        if acf[lag]<acf[lag-1] or acf[lag]<acf[lag+1]:continue
        hz=fs/lag
        peaks.append({'hz':hz,'lag':lag,'acf':float(acf[lag]),'cmndf':float(cmndf[lag]),
                      'cycles':n/lag})
    return sorted(peaks,key=lambda r:(-r['acf'],r['cmndf']))

def read_v9(path):
    with path.open(newline='') as file:
        reader=csv.DictReader(file)
        required={'sample','time_s','view_ms','center_start_sample','center_end_sample','candidates_json'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing {sorted(required-set(reader.fieldnames or []))}')
        yield from reader

def reference_candidates(row):
    raw=json.loads(row['candidates_json'])
    # Cluster duplicate measured peaks around a local peak; preserve separate rivals.
    selected=[]
    for p in sorted((x for x in raw if x['view']=='center' and x['eligible']),
                    key=lambda x: (-(number(x.get('acf')) or 0),number(x.get('cmndf')) or 9)):
        hz=number(p.get('measured_hz'))
        if hz and not any(cents(hz,x['hz'])<=35 for x in selected):
            selected.append({'hz':hz,'acf':number(p.get('acf')),
                             'cmndf':number(p.get('cmndf'))})
    return selected

def analyze(row,audio,fs,shifts,tolerance):
    lo=int(row['center_start_sample']);hi=int(row['center_end_sample'])
    if not (0<=lo<hi<=len(audio)):return [],'invalid_center_window'
    anchors=reference_candidates(row)
    if not anchors:return [],'no_eligible_v9_center_candidate'
    out=[]
    for shift in shifts:
        delta=round(shift*fs/1000)
        start=lo+delta;end=hi+delta
        if start<0 or end>len(audio):
            for anchor in anchors:
                out.append(dict(shift_ms=shift,anchor_hz=anchor['hz'],status='out_of_bounds',
                                start_sample=start,end_sample=end))
            continue
        y=audio[start:end]
        peaks=independent_peaks(y,fs)
        for anchor in anchors:
            matches=[p for p in peaks if cents(p['hz'],anchor['hz'])<=tolerance]
            # The anchor is NOT used to create/refine a frequency. Only measured peaks count.
            winner=max(matches,key=lambda p:(p['acf'],-p['cmndf'])) if matches else None
            entry={'shift_ms':shift,'anchor_hz':anchor['hz'],'start_sample':start,
                   'end_sample':end,'raw_peak_count':len(peaks),
                   'status':'independently_measured_match' if winner else 'no_independent_match',
                   'measured_hz':winner['hz'] if winner else None,
                   'acf':winner['acf'] if winner else None,
                   'cmndf':winner['cmndf'] if winner else None,
                   'cycles':winner['cycles'] if winner else None,
                   'fundamental_fraction':None,'harmonic_fraction':None,'harmonic_count':None,
                   'strongest_other_measured_hz':None,'strongest_other_acf':None,
                   'strongest_other_cmndf':None,'strongest_other_fundamental_fraction':None,
                   'strongest_other_harmonic_fraction':None}
            if winner:
                entry['fundamental_fraction'],entry['harmonic_fraction'],entry['harmonic_count']=bands(y,fs,winner['hz'])
            # Competing independently measured peaks. Compare actual energy on same waveform.
            others=[p for p in peaks if not winner or cents(p['hz'],winner['hz'])>tolerance]
            if others:
                other=others[0]
                entry.update(strongest_other_measured_hz=other['hz'],strongest_other_acf=other['acf'],
                             strongest_other_cmndf=other['cmndf'])
                f,h,_=bands(y,fs,other['hz'])
                entry.update(strongest_other_fundamental_fraction=f,strongest_other_harmonic_fraction=h)
            out.append(entry)
    return out,None

def verdict(anchor,rows,offsets,acft=.70,cmndft=.25):
    records=[r for r in rows if r['anchor_hz']==anchor['hz']]
    good=[r for r in records if r['status']=='independently_measured_match' and
          r['acf']>=acft and r['cmndf']<=cmndft and r['cycles']>=3]
    center=next((r for r in records if r['shift_ms']==0),None)
    # No musical-prior confidence: stability means repeated *measured* peaks only.
    observed=[r for r in records if r['status']=='independently_measured_match']
    freqspread=max((cents(x['measured_hz'],y['measured_hz']) for x in observed for y in observed),default=None)
    n=len(offsets); supported=len(good)
    if len(observed)<2:category='insufficient_shift_measurements'
    elif supported==n and freqspread<=70:category='shift_stable_acoustic_candidate'
    elif supported>=max(2,n-1) and freqspread<=70:category='mostly_shift_stable_candidate'
    elif supported==0:category='measured_but_no_shift_has_sufficient_evidence'
    elif freqspread>70:category='peak_location_sensitive'
    else:category='shift_sensitive_acoustic_support'
    return {'anchor_hz':anchor['hz'],'shift_count':n,'measured_shift_count':len(observed),
            'supported_shift_count':supported,'measured_spread_cents':freqspread,
            'center_acf':center.get('acf') if center else None,
            'center_cmndf':center.get('cmndf') if center else None,
            'category':category,'not_a_pitch_or_voicing_decision':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*CASES),default='all')
    p.add_argument('--v9-dir',type=Path,default=Path('tests/measured_period_families_v9'))
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/shift_robustness_v10'))
    p.add_argument('--shifts-ms',nargs='+',type=float,default=list(SHIFTS))
    p.add_argument('--match-cents',type=float,default=70.)
    args=p.parse_args()
    if 0. not in args.shifts_ms:p.error('shifts must include 0 ms')
    if len(set(args.shifts_ms))!=len(args.shifts_ms):p.error('shifts must be unique')
    if any(abs(s)>12 for s in args.shifts_ms):p.error('shift must be within +/-12 ms')
    if not 10<=args.match_cents<=150:p.error('match-cents must be 10..150')
    names=CASES if args.case=='all' else (args.case,)
    for name in names:
        for path in (args.v9_dir/f'{name}_measured_families.csv',args.wav_dir/f'{name}.wav'):
            if not path.is_file():p.error(f'missing {path}')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'version':'v10','shifts_ms':args.shifts_ms,'match_cents':args.match_cents,
              'wav_only':True,'reference_used_only_for_comparison':True,
              'production_unchanged':True,'no_pitch_or_voicing_applied':True,'cases':{}}
    for name in names:
        audio,fs=sf.read(args.wav_dir/f'{name}.wav',dtype='float64')
        if audio.ndim!=1:p.error(f'{name}: mono WAV required')
        summaries=[];detailed=[];skips=Counter();categories=Counter();nobs=set()
        for row in read_v9(args.v9_dir/f'{name}_measured_families.csv'):
            key=(row['sample'],row['view_ms']);nobs.add(row['sample'])
            result,reason=analyze(row,audio,fs,args.shifts_ms,args.match_cents)
            if reason:skips[reason]+=1;continue
            anchors=reference_candidates(row)
            for a in anchors:
                r=verdict(a,result,args.shifts_ms)
                r.update(case=name,sample=row['sample'],time_s=row['time_s'],
                         view_ms=row['view_ms'],region=row.get('region',''),
                         prior_v9_comparison=row.get('acoustic_comparison',''))
                summaries.append(r);categories[r['category']]+=1
            for r in result:
                r.update(case=name,sample=row['sample'],time_s=row['time_s'],
                         view_ms=row['view_ms'],region=row.get('region',''))
                detailed.append(r)
        def write(name_suffix,rows):
            path=args.out_dir/f'{name}_{name_suffix}.csv'
            if rows:
                with path.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                    writer.writeheader();writer.writerows(rows)
            else:path.write_text('no_results\n')
        write('shift_candidates',detailed);write('shift_summary',summaries)
        manifest['cases'][name]={'review_observations':len(nobs),'candidate_shift_rows':len(detailed),
                                  'candidate_summaries':len(summaries),'categories':dict(categories),
                                  'skipped_aperture_rows':dict(skips),'sample_rate':fs}
        print(name,'review',len(nobs),'candidates',len(summaries),'categories',dict(categories),flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: no F0, voiced decisions, ECKF resets, BPM, MIDI, or temporal smoothing.')
if __name__=='__main__':main()
