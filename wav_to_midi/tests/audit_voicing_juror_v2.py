#!/usr/bin/env python3
"""Read-only V12 voicing-juror audition: per-observation evidence and cross-tabs.

Run at repository root: python tests/audit_voicing_juror_v2.py --wav-root tests --out tests/voicing_juror_v2
Supply --v12-zip PATH if V12 ZIP is not located under --wav-root or tests/.
Requires numpy and soundfile. Does NOT alter V12 verdicts or produce MIDI.
Labels describe *acoustic periodicity*, not verified human vocal identity or MIDI rests.
"""
from __future__ import annotations
import argparse, csv, io, json, math, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')

def read_results(path):
    with zipfile.ZipFile(path) as z:
        for case in CASES:
            match=[x for x in z.namelist() if x.endswith(f'/{case}_observation_summary.csv')]
            if len(match)!=1: raise ValueError(f'Expected exactly one {case} observation CSV, found {match}')
            yield case,list(csv.DictReader(io.StringIO(z.read(match[0]).decode('utf-8-sig'))))

def csv_write(path, rows, fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def acoustic_frame(x,fs,t,duration,low_hz,high_hz):
    n=int(round(duration*fs)); c=int(round(t*fs)); a=c-n//2; b=a+n
    if a<0 or b>len(x) or n<16: return {'valid':False,'reason':'waveform_boundary'}
    v=x[a:b].astype(np.float64,copy=False)
    v=v-np.mean(v); rms=float(np.sqrt(np.mean(v*v)))
    if not np.isfinite(rms): return {'valid':False,'reason':'nonfinite_audio'}
    if rms<1e-9: return {'valid':True,'rms':rms,'peak_acf':0.,'peak_hz':None,'flatness':1.,'periodicity_contrast':0.}
    v=v*np.hanning(n)
    # FFT linear autocorrelation, normalized by zero-lag energy. This is an
    # observation about periodicity, not a reuse of V11's candidate ACF votes.
    size=1 << (2*n-1).bit_length(); ft=np.fft.rfft(v,size)
    ac=np.fft.irfft(ft*np.conj(ft),size)[:n]
    lag_lo=max(2,int(math.ceil(fs/high_hz)));lag_hi=min(n//2,int(math.floor(fs/low_hz)))
    if lag_lo>=lag_hi: return {'valid':False,'reason':'insufficient_lag_range'}
    # Unbiased overlap normalization is deliberately not used: it exaggerates
    # short-window large-lag peaks. Use zero-lag normalized, windowed correlation.
    values=ac[lag_lo:lag_hi+1]/max(ac[0],1e-20)
    ix=int(np.argmax(values)); peak=float(values[ix]); lag=lag_lo+ix
    # Spectrum noise-like vs peaky, with DC suppressed; secondary corroborator.
    power=np.abs(np.fft.rfft(v))**2
    freqs=np.fft.rfftfreq(n,1/fs)
    band=power[(freqs>=100)&(freqs<=min(5000,fs/2-1))]
    flat=float(np.exp(np.mean(np.log(band+1e-20)))/(np.mean(band)+1e-20)) if len(band) else 1.
    return {'valid':True,'rms':rms,'peak_acf':peak,'peak_hz':float(fs/lag),'flatness':flat,
            'periodicity_contrast':float(peak-np.median(values))}

def classify(frames,rel_db, args):
    if len(frames)!=3 or any(not f['valid'] for f in frames):return 'uncertain','incomplete_waveform_evidence'
    acfs=[f['peak_acf'] for f in frames]
    flats=[f['flatness'] for f in frames]
    strong=sum(a>=args.strong_acf for a in acfs)
    weak=sum(a<=args.weak_acf for a in acfs)
    quiet=(rel_db is not None and rel_db<=args.quiet_db)
    noisy=sum(f>=args.noise_flatness for f in flats)>=2
    # A voiced note can be quiet; only call unvoiced-like with 3 weak ACF
    # measurements AND an independent low-energy or noise-like indication.
    if strong>=2 and not quiet and sum(f['periodicity_contrast']>=args.min_contrast for f in frames)>=2:
        return 'voiced_like','repeatable_periodic_waveform'
    if weak==3 and (quiet or noisy):
        return 'unvoiced_like','weak_periodicity_plus_'+('low_energy' if quiet else 'noise_like_spectrum')
    return 'uncertain','mixed_or_insufficient_evidence'

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v12-zip',type=Path,default=None)
    p.add_argument('--v11-zip',type=Path,default=None,help='Optional baseline V11_04 archive, for observation-by-observation comparison')
    p.add_argument('--wav-root',type=Path,default=Path('tests'))
    p.add_argument('--out',type=Path,default=Path('tests/voicing_juror_v2'))
    p.add_argument('--short-ms',type=float,default=40)
    p.add_argument('--middle-ms',type=float,default=64)
    p.add_argument('--long-ms',type=float,default=96)
    p.add_argument('--min-hz',type=float,default=75)
    p.add_argument('--max-hz',type=float,default=850)
    p.add_argument('--strong-acf',type=float,default=.62)
    p.add_argument('--weak-acf',type=float,default=.27)
    p.add_argument('--min-contrast',type=float,default=.12)
    p.add_argument('--noise-flatness',type=float,default=.45)
    p.add_argument('--quiet-db',type=float,default=-28)
    args=p.parse_args()
    if not (0<args.weak_acf<args.strong_acf<1 and 0<args.min_hz<args.max_hz and
            0<=args.noise_flatness<=1 and 0<args.short_ms<args.middle_ms<args.long_ms):
        p.error('Invalid thresholds, pitch limits or window order')
    source=args.v12_zip
    if source is None:
        options=[args.wav_root/'shadow_adjudicator_v22_fixed_v12.zip',Path('tests/shadow_adjudicator_v22_fixed_v12.zip'),Path('shadow_adjudicator_v22_fixed_v12.zip')]
        source=next((x for x in options if x.is_file()),None)
        if source is None:p.error('Supply --v12-zip pointing to shadow_adjudicator_v22_fixed_v12.zip')
    if not source.is_file():p.error(f'Cannot find V12 archive: {source}')
    baseline={}
    if args.v11_zip is not None:
        if not args.v11_zip.is_file():p.error(f'Missing V11 baseline: {args.v11_zip}')
        baseline={(case,round(float(r['time_s']),6)):r for case, rows in read_results(args.v11_zip) for r in rows}
    args.out.mkdir(parents=True,exist_ok=True)
    allrows=[]; totals={}; cross=[]; examples=[]
    duration_ms=(args.short_ms,args.middle_ms,args.long_ms)
    for case, observations in read_results(source):
        candidates=[args.wav_root/f'{case}.wav',Path('tests')/f'{case}.wav',Path(f'{case}.wav')]
        wav=next((q for q in candidates if q.is_file()),None)
        if wav is None:p.error(f'Missing {case}.wav: place WAV in --wav-root')
        audio,sr=sf.read(str(wav),dtype='float32',always_2d=False)
        if audio.ndim==2:audio=audio.mean(axis=1)
        # Case-relative energy estimated only from ACTIVE V12 observation times;
        # this avoids using RATATA's excluded tail as the noise floor.
        times=[float(row['time_s']) for row in observations]
        frames_short=[acoustic_frame(audio,sr,t,args.short_ms/1000,args.min_hz,args.max_hz) for t in times]
        db_values=[20*math.log10(max(f['rms'],1e-12)) for f in frames_short if f['valid']]
        reference_db=float(np.percentile(db_values,90)) if db_values else None
        counts=Counter(); outcomes=Counter(); matrix=Counter(); rows=[]
        for ob,t,short in zip(observations,times,frames_short):
            frames=[short]+[acoustic_frame(audio,sr,t,d/1000,args.min_hz,args.max_hz) for d in duration_ms[1:]]
            db=20*math.log10(max(short['rms'],1e-12)) if short['valid'] else None
            rel=db-reference_db if db is not None and reference_db is not None else None
            label,reason=classify(frames,rel,args)
            outcome='champion' if ob.get('tournament_champion_frequency_hz','').strip() else 'unresolved'
            counts[label]+=1; outcomes[outcome]+=1;matrix[(outcome,label)]+=1
            previous=baseline.get((case,round(t,6)))
            result={'case':case,'time_s':ob['time_s'],
                    'v11_prior_outcome':previous['observation_outcome'] if previous else '',
                    'v11_prior_champion_hz':previous.get('tournament_champion_frequency_hz','') if previous else '',
                    'v11_to_v12_transition':('champion_to_champion' if previous.get('tournament_champion_frequency_hz','') and outcome=='champion' else
                        'champion_to_unresolved' if previous.get('tournament_champion_frequency_hz','') else
                        'unresolved_to_champion' if outcome=='champion' else 'unresolved_to_unresolved') if previous else 'baseline_not_supplied','v12_status':outcome,
                    'v12_outcome':ob['observation_outcome'],'v12_abstention_reason':ob.get('abstention_category',''),
                    'v12_champion_hz':ob.get('tournament_champion_frequency_hz',''),
                    'voicing_class':label,'voicing_reason':reason,'rms_dbfs':db,
                    'relative_to_case_p90_db':rel,'case_p90_rms_dbfs':reference_db}
            for d,fr in zip(duration_ms,frames):
                name=f'{int(d)}ms'
                for k in ('valid','reason','rms','peak_acf','peak_hz','flatness','periodicity_contrast'):
                    result[f'{name}_{k}']=fr.get(k)
            rows.append(result);allrows.append(result)
        csv_write(args.out/f'{case}_voicing_observations.csv',rows,list(rows[0]) if rows else [])
        totals[case]={'observations':len(rows),'v12':dict(outcomes),'voicing':dict(counts),
                      'case_p90_rms_dbfs':reference_db,'cross_tab':{v:{l:matrix[(v,l)] for l in ('voiced_like','unvoiced_like','uncertain')} for v in ('champion','unresolved')}}
        for outcome in ('champion','unresolved'):
            for label in ('voiced_like','unvoiced_like','uncertain'):
                n=matrix[(outcome,label)]
                cross.append({'case':case,'v12_status':outcome,'voicing_class':label,'count':n,
                              'pct_of_case':round(100*n/len(rows),3) if rows else 0,
                              'pct_of_v12_status':round(100*n/outcomes[outcome],3) if outcomes[outcome] else 0})
        for outcome in ('champion','unresolved'):
            for label in ('voiced_like','unvoiced_like','uncertain'):
                matches=[r for r in rows if r['v12_status']==outcome and r['voicing_class']==label]
                # Reproducible examples distributed across timeline, not cherry-picked extremes.
                for ix in (np.linspace(0,len(matches)-1,min(5,len(matches))).astype(int) if matches else []):
                    examples.append(matches[int(ix)])
        print(f'{case}: {len(rows)} observations; '+', '.join(f'{k}: {v}' for k,v in counts.items()))
    csv_write(args.out/'cross_tab_by_song.csv',cross,list(cross[0]))
    csv_write(args.out/'example_observations.csv',examples,list(allrows[0]) if allrows else [])
    grand=Counter(); aggregate=Counter()
    for row in allrows:grand[row['voicing_class']]+=1;aggregate[(row['v12_status'],row['voicing_class'])]+=1
    summary={'method':'observational acoustic periodicity heuristic, NOT a trained human-voice detector',
             'reference_v12_zip':str(source),'reference_v12_manifest':'V12 observation_summary.csv',
             'waveform_windows_ms':list(duration_ms),'thresholds':{k:getattr(args,k) for k in
                ('min_hz','max_hz','strong_acf','weak_acf','min_contrast','noise_flatness','quiet_db')},
             'labels_not_ground_truth':True,'no_midi_rest_inference':True,
             'total_observations':len(allrows),'voicing_totals':dict(grand),
             'aggregate_cross_tab':{o:{v:aggregate[(o,v)] for v in ('voiced_like','unvoiced_like','uncertain')} for o in ('champion','unresolved')},
             'by_song':totals,'baseline_v11_supplied':bool(baseline),
             'v11_to_v12_transitions':dict(Counter(r['v11_to_v12_transition'] for r in allrows))}
    (args.out/'manifest.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('TOTAL',len(allrows),'cross-tab',summary['aggregate_cross_tab'])
    print('Wrote',args.out)

if __name__=='__main__':main()
