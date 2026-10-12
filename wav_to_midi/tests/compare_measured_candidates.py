#!/usr/bin/env python3
"""READ-ONLY four-WAV comparison of independently MEASURED acoustic periods.
Run from the songwriter_s_toolkit repository root:
    python tests/compare_measured_candidates.py --case all
    python tests/compare_measured_candidates.py --case Ochiitai --case Trandafiri
No MIDI. No production mutations. Never replaces F0, resets or generates notes.
This report compares different window supports; their evidence is correlated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

CASES = {
    'Ochiitai': (42.214, 42.260, 42.307, 42.353),
    'PREDESTINATI': (38.034, 53.592),
    'RATATA': (36.923, 41.026, 42.965, 43.008, 49.231),
    'Trandafiri': (10.774, 10.821),
}
WINDOWS_MS = (24., 40., 64., 96.)
FIELDNAMES = ('case','anchor_s','frame_start','frame_time_s','window_ms','window_start_s',
              'window_end_s','rms','window_position','candidate_hz','lag_samples',
              'observed_cycles','acf','cmndf','spectral_fundamental_amp',
              'spectral_amp_relative','spectral_support_harmonics','spectrum_resolution_hz',
              'fft_nearest_peak_hz','candidate_kind','production_periodicity_hz',
              'production_choice_before_hz','production_choice_after_hz',
              'production_reset_hz')

def finite(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError): return None

def frame_candidates(x,fs):
    """Calculate local ACF peaks AND local CMNDF values from actual samples.
    Preserve measured alternatives even below production's acceptance bounds.
    FFT peaks are corroborative, not independent statistical votes.
    """
    x=np.asarray(x,dtype=np.float64)
    x=x-np.mean(x)
    n=len(x)
    rms=float(np.sqrt(np.mean(x*x)))
    first=max(1,int(np.floor(fs/1000.)))
    last=min(n-1,int(np.ceil(fs/70.)))
    if last<=first or rms==0: return rms,[]
    differences=np.zeros(last+1,dtype=np.float64)
    acf=np.full(last+1,-1.,dtype=np.float64)
    for lag in range(1,last+1):
        a=x[:-lag]; b=x[lag:]
        denom=float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
        if denom>0: acf[lag]=np.dot(a,b)/denom
        differences[lag]=np.dot(a-b,a-b)
    cum=np.cumsum(differences[1:])
    cmndf=np.ones(last+1,dtype=np.float64)
    cmndf[1:]=np.divide(differences[1:]*np.arange(1,last+1),cum,
                         out=np.ones(last),where=cum>0)
    peaks,_=find_peaks(acf)
    peaks=peaks[(peaks>=first)&(peaks<=last)].tolist()
    if acf[first]>acf[first+1]: peaks.append(first)
    if acf[last]>acf[last-1]: peaks.append(last)
    # Zero padding improves peak location, NOT the intrinsic frequency resolution.
    nfft=max(4096,1<<(4*n-1).bit_length())
    spectrum=np.abs(np.fft.rfft(x*np.hanning(n),n=nfft))
    fbins=np.fft.rfftfreq(nfft,1/fs)
    spectral_peaks,_=find_peaks(spectrum)
    spectral_peaks=spectral_peaks[(fbins[spectral_peaks]>=70)&(fbins[spectral_peaks]<min(fs/2,6000))]
    peak_max=float(np.max(spectrum[spectral_peaks])) if len(spectral_peaks) else 0.
    # Explicit evidence columns, no choice/score/correction.
    result=[]
    for lag in sorted(set(peaks)):
        if acf[lag]<.20 or cmndf[lag]>.65: continue
        hz=float(fs/lag)
        width=max(fs/n, hz*(2**(35/1200)-1))
        close=spectral_peaks[np.abs(fbins[spectral_peaks]-hz)<=width]
        nearest=int(close[np.argmax(spectrum[close])]) if len(close) else None
        amp=float(spectrum[nearest]) if nearest is not None else 0.
        relatives=amp/peak_max if peak_max else 0.
        harmonics=[]
        for k in range(1,13):
            target=k*hz
            if target>=min(fs/2,6000): break
            hits=spectral_peaks[np.abs(fbins[spectral_peaks]-target)<=max(fs/n,target*(2**(35/1200)-1))]
            if len(hits): harmonics.append(k)
        result.append(dict(hz=hz,lag=int(lag),cycles=n/lag,acf=float(acf[lag]),
                           cmndf=float(cmndf[lag]),amp=amp,amp_relative=relatives,
                           harmonic_indices=harmonics,nearest_hz=float(fbins[nearest]) if nearest is not None else None,
                           resolution_hz=fs/n))
    result.sort(key=lambda z:(-z['acf'],z['cmndf']))
    return rms,result

def trace_at(path,anchor,fs,block):
    if not path.is_file(): return None,{}
    events=[]
    with path.open() as f:
        for line in f:
            if line.strip():
                event=json.loads(line)
                if abs(float(event['time_s'])-anchor)<.11: events.append(event)
    frame=int(round(anchor*fs/block))*block
    starts=sorted({int(e.get('trace_values',{}).get('frame_start',e['sample'])) for e in events
                   if e['event']=='PERIODICITY_DECISION'})
    if starts: frame=min(starts,key=lambda s:abs(s/fs-anchor))
    values={}
    for e in events:
        if int(e.get('trace_values',{}).get('frame_start',e['sample']))!=frame: continue
        v=e.get('trace_values',{})
        key={'PERIODICITY_DECISION':('production_periodicity_hz','acf_frequency_hz'),
             'INITIALIZATION_CHOICE_BEFORE_OCTAVE':('production_choice_before_hz','choice_hz'),
             'INITIALIZATION_CHOICE_AFTER_OCTAVE':('production_choice_after_hz','final_choice_hz'),
             'RESET_COMPLETED':('production_reset_hz','init_f0_hz')}.get(e['event'])
        if key: values[key[0]]=finite(v.get(key[1]))
    return frame,values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=tuple(CASES)+('all',),action='append',default=[])
    p.add_argument('--tests-dir',type=Path,default=Path('tests'))
    p.add_argument('--traces-dir',type=Path,default=Path('tests/production_decision_trace'))
    p.add_argument('--output-dir',type=Path,default=Path('tests/measured_candidate_comparison'))
    p.add_argument('--block-size',type=int,default=2048)
    p.add_argument('--step-ms',type=float,default=10.)
    args=p.parse_args()
    if args.block_size<32 or not 2<=args.step_ms<=25: p.error('invalid block size or step')
    names=list(CASES) if not args.case or 'all' in args.case else list(dict.fromkeys(args.case))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'midi_used':False,'production_modified':False,'pitch_modified':False,
              'comparisons_are_observational':True,'cases':{}}
    for name in names:
        wavs=[args.tests_dir/f'{name}{extension}' for extension in ('.wav','.wa') if (args.tests_dir/f'{name}{extension}').is_file()]
        if len(wavs)!=1: p.error(f'{name}: expected exactly one WAV/WA: {wavs}')
        path=wavs[0]
        audio,fs=sf.read(path,dtype='float64')
        if audio.ndim!=1: p.error(f'{path}: mono WAV required (no downmix)')
        trace=args.traces_dir/f'{name}_decision_path.jsonl'
        csv_path=args.output_dir/f'{name}_candidates.csv'
        rows=[]; events=[]
        for anchor in CASES[name]:
            frame,production=trace_at(trace,anchor,fs,args.block_size)
            if frame is None: frame=int(np.floor(anchor*fs/args.block_size))*args.block_size
            center=frame+args.block_size//2
            # Three center positions expose transition mixing without altering any tracking.
            positions=[('frame_center',center),('marker_center',int(round(anchor*fs))),
                       ('marker_plus_10ms',int(round((anchor+.010)*fs)))]
            for label,c in positions:
                for ms in (*WINDOWS_MS,0.):
                    n=args.block_size if ms==0 else round(fs*ms/1000)
                    left=c-n//2
                    right=left+n
                    if left<0 or right>len(audio): continue
                    rms,candidates=frame_candidates(audio[left:right],fs)
                    for candidate in candidates:
                        rows.append(dict(case=name,anchor_s=anchor,frame_start=frame,frame_time_s=frame/fs,
                          window_ms='production_2048' if ms==0 else ms,window_start_s=left/fs,window_end_s=right/fs,
                          rms=rms,window_position=label,candidate_hz=candidate['hz'],lag_samples=candidate['lag'],
                          observed_cycles=candidate['cycles'],acf=candidate['acf'],cmndf=candidate['cmndf'],
                          spectral_fundamental_amp=candidate['amp'],spectral_amp_relative=candidate['amp_relative'],
                          spectral_support_harmonics=','.join(map(str,candidate['harmonic_indices'])),
                          spectrum_resolution_hz=candidate['resolution_hz'],fft_nearest_peak_hz=candidate['nearest_hz'],
                          candidate_kind='local_measured_acf_peak',**production))
                    events.append({'anchor_s':anchor,'position':label,'window_ms': 'production_2048' if ms==0 else ms,
                                   'start_s':left/fs,'end_s':right/fs,'rms':rms,
                                   'candidates_meeting_exploratory_display_floor':len(candidates),
                                   'production':production})
        with csv_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDNAMES,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
        details=args.output_dir/f'{name}_windows.json'
        details.write_text(json.dumps(events,indent=2)+'\n')
        manifest['cases'][name]={'sample_rate':fs,'wav_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'trace_present':trace.is_file(),'candidate_rows':len(rows),
                                 'candidate_csv':str(csv_path),'window_summary':str(details)}
        print(f'{name}: {len(rows)} candidate rows; trace={trace.is_file()}; {csv_path}',flush=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('No production or WAV files modified. Comparisons are observations, not F0 decisions.')

if __name__=='__main__': main()
