#!/usr/bin/env python3
"""V21: adaptive-only, read-only onset/persistence/integer-family waveform audit.

Inputs: V19 waveform observations, V20 integer pair reports, ORIGINAL isolated-vocal WAVs.
Outputs are EVIDENCE, never F0/noise/voice decisions. No production, MIDI, penalty,
range exclusions, pitch continuity, or ground-truth annotations enter computation.
Run: python tests/audit_temporal_integer_families_v21.py --case all
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
# These regions are report tags ONLY, never affect sampling, scores, or classification.
REGIONS = {
 'Ochiitai': [('ku',3.465,3.692),('early_reacquisition',42.19,42.40)],
 'PREDESTINATI': [('control_38',37.97,38.13),('control_53',53.52,53.73)],
 'RATATA': [('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),
            ('djuvv_3',49.230,49.743),('submultiple_tail',42.90,43.10)],
 'Trandafiri': [('steady_i_not_glissando',10.70,10.95)]
}
WIDTHS = (24,40,64)
SHIFTS_MS = (-8.,0.,8.)

def fnum(x):
 try:
  v=float(x)
  return v if math.isfinite(v) else None
 except (ValueError,TypeError): return None

def cents(a,b): return abs(1200*math.log2(a/b)) if a>0 and b>0 else float('inf')

def wav_segment(y,sr,center,width_ms):
 a=round((center-width_ms/2000)*sr);b=round((center+width_ms/2000)*sr)
 return None if a<0 or b>len(y) or b-a<64 else y[a:b]

def coherence(x,sr,hz):
 """Overlap-normalized lag coherence. Always report cycles and lag; no F0 verdict."""
 if x is None or hz<=0: return {'testable':False,'reason':'outside_wav'}
 z=np.asarray(x,dtype=np.float64);z=z-z.mean();n=len(z)
 cycles=n*hz/sr
 if cycles<3.:return {'testable':False,'reason':'under_three_cycles','cycles':round(cycles,3)}
 if float(z@z)<1e-18:return {'testable':False,'reason':'negligible_signal','cycles':round(cycles,3)}
 lo=max(2,round(sr/hz*.975));hi=min(n//2,round(sr/hz*1.025))
 if hi<lo:return {'testable':False,'reason':'lag_unavailable','cycles':round(cycles,3)}
 scores=[]
 for lag in range(lo,hi+1):
  a,b=z[:-lag],z[lag:];den=math.sqrt(float(a@a)*float(b@b))
  scores.append((float(a@b)/den if den>1e-18 else 0.,lag))
 ac,lag=max(scores)
 return {'testable':True,'acf':round(ac,6),'measured_hz':round(sr/lag,5),
         'cycles':round(cycles,3),'lag':lag}

def spectral_pair(x,sr,low,high):
 """Exclusive low-harmonic power AND whether spectral resolution permits a comparison.

 FFT bins are shared and wide; these are descriptive fractions, never proof of
 independent oscillators, singer identity, or true fundamental.
 """
 if x is None:return {'testable':False,'reason':'outside_wav'}
 a=np.asarray(x,dtype=np.float64);a=a-a.mean();n=len(a)
 p=np.abs(np.fft.rfft(a*np.hanning(n)))**2;total=float(p[1:].sum())
 if total<=1e-20:return {'testable':False,'reason':'negligible_signal'}
 df=sr/n;ny=sr/2
 def bins(freq):
  k=round(freq/df);r=max(1,round(max(10.,freq*.013)/df))
  return set(range(max(1,k-r),min(len(p),k+r+1)))
 high_set=set();low_set=set()
 for j in range(1,min(16,int(ny/high))+1):high_set.update(bins(j*high))
 for j in range(1,min(16,int(ny/low))+1):low_set.update(bins(j*low))
 only=low_set-high_set
 # Harmonic-only bins must be separated by >1 FFT bin from high's harmonic centers.
 distinct_centers=[j*low for j in range(1,min(16,int(ny/low))+1)
                   if min(abs(j*low-k*high) for k in range(1,min(16,int(ny/high))+1))>2*df]
 return {'testable':True,'bin_width_hz':round(df,4),
         'low_exclusive_power_fraction':round(float(sum(p[k] for k in only))/total,7) if only else None,
         'low_exclusive_bins':len(only),'distinct_low_harmonic_centers':len(distinct_centers),
         'low_harmonic_power_fraction':round(float(sum(p[k] for k in low_set))/total,6),
         'high_harmonic_power_fraction':round(float(sum(p[k] for k in high_set))/total,6)}

def scene(x,sr):
 if x is None:return None
 a=np.asarray(x,dtype=np.float64);n=len(a)
 rms=math.sqrt(float(a@a)/n)
 if rms<1e-12:return {'rms':rms,'flatness':None,'high_band_fraction':None}
 z=np.abs(np.fft.rfft((a-a.mean())*np.hanning(n)))**2
 z=z[1:];tot=float(z.sum());freq=np.fft.rfftfreq(n,1/sr)[1:]
 if tot<=1e-20:return {'rms':rms,'flatness':None,'high_band_fraction':None}
 return {'rms':rms,'flatness':float(np.exp(np.mean(np.log(z+1e-20)))/(np.mean(z)+1e-20)),
         'high_band_fraction':float(z[freq>=2500].sum()/tot)}

def read_csv(path):
 if not path.is_file():raise FileNotFoundError(path)
 with path.open(newline='') as f:return list(csv.DictReader(f))

def row_quality(r):
 # V20 preexisting support determines which pair to test; no frequency chosen.
 return int(r.get('low_v19_present_support') or 0)+int(r.get('high_v19_present_support') or 0)

def audit_pair(y,sr,t,low,high):
 # Centre views shifted only a few milliseconds: NOT a trajectory or smoothing.
 windows={};acf_low=[];acf_high=[];excl=[]
 for ms in WIDTHS:
  for shift in SHIFTS_MS:
   key=f'{ms}ms_{shift:+g}ms';w=wav_segment(y,sr,t+shift/1000,ms)
   l=coherence(w,sr,low);h=coherence(w,sr,high)
   spec=spectral_pair(w,sr,low,high)
   windows[key]={'low':l,'high':h,'spectral':spec}
   if l.get('testable') and h.get('testable'):
    acf_low.append(l['acf']);acf_high.append(h['acf'])
    if spec.get('low_exclusive_power_fraction') is not None:
     excl.append(spec['low_exclusive_power_fraction'])
 # Non-overlapping past/future probes help distinguish persistence from leakage.
 flank={}
 for direction,tt in [('past',t-.020),('future',t+.020)]:
  x=wav_segment(y,sr,tt,24)
  flank[direction]={'low':coherence(x,sr,low),'high':coherence(x,sr,high),
                    'spectral':spectral_pair(x,sr,low,high),'scene':scene(x,sr)}
 sc={name:scene(wav_segment(y,sr,t+delta,12),sr) for name,delta in
     [('past',-.012),('present',0.),('future',.012)]}
 past=sc['past'];future=sc['future'];cur=sc['present']
 slope=20*math.log10(max(future['rms'],1e-15)/max(past['rms'],1e-15)) if past and future else None
 flat_change=(future['flatness']-past['flatness'] if past and future and
              past['flatness'] is not None and future['flatness'] is not None else None)
 high_change=(future['high_band_fraction']-past['high_band_fraction'] if past and future and
              past['high_band_fraction'] is not None and future['high_band_fraction'] is not None else None)
 # The flags describe only repeated measurements; DO NOT assign oscillator identity.
 low_repeat=sum(v>=.72 for v in acf_low);high_repeat=sum(v>=.72 for v in acf_high)
 paired=len(acf_low)
 if paired<3:label='insufficient_paired_shift_measurements'
 elif low_repeat>=6 and high_repeat>=6:label='both_integer_periods_shift_persistent'
 elif high_repeat>=6 and low_repeat<=2:label='higher_period_persistent_lower_not'
 elif low_repeat>=6 and high_repeat<=2:label='lower_period_persistent_higher_not'
 else:label='temporal_support_mixed_or_weak'
 return {'shift_windows':windows,'nonoverlapping_flanks':flank,'scene_12ms':sc,
         'past_to_future_energy_db':slope,'past_to_future_flatness_change':flat_change,
         'past_to_future_high_band_change':high_change,
         'paired_measurements':paired,'low_supported_shift_count':low_repeat,
         'high_supported_shift_count':high_repeat,
         'low_exclusive_power_min':min(excl) if excl else None,
         'low_exclusive_power_max':max(excl) if excl else None,
         'temporal_pattern':label,
         'potential_attack_or_decay_context':'inspect_energy_and_spectrum_not_voice_verdict'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--case',choices=('all',*CASES),default='all')
 p.add_argument('--v19-dir',type=Path,default=Path('tests/adaptive_waveform_v19'))
 p.add_argument('--v20-dir',type=Path,default=Path('tests/integer_families_v20'))
 p.add_argument('--wav-dir',type=Path,default=Path('tests'))
 p.add_argument('--out-dir',type=Path,default=Path('tests/temporal_integer_families_v21'))
 p.add_argument('--max-pairs-per-observation',type=int,default=2,
                help='Highest V19 combined support, deterministic; 0=all pairs (slow)')
 args=p.parse_args()
 if args.max_pairs_per_observation<0:p.error('max-pairs-per-observation must be nonnegative')
 args.out_dir.mkdir(parents=True,exist_ok=True)
 manifest={'schema':'v21_onset_persistence_integer_family_adaptive_only',
           'input':'original_WAV_plus_V19_and_V20', 'production_used':False,
           'midi_used':False,'penalty_used':False,'vocal_range_used':False,
           'selected_f0':False,'active_voice_verified':False,'noise_declared':False,
           'decisions_changed':False,'shift_ms':SHIFTS_MS,'apertures_ms':WIDTHS,
           'max_pairs_per_observation':args.max_pairs_per_observation,'cases':{}}
 for case in (CASES if args.case=='all' else (args.case,)):
  wav=args.wav_dir/f'{case}.wav'
  if not wav.is_file():raise FileNotFoundError(wav)
  y,sr=sf.read(wav,dtype='float64',always_2d=False)
  if y.ndim==2:y=y.mean(axis=1)
  observations=read_csv(args.v19_dir/f'{case}_waveform.csv')
  oldpairs=read_csv(args.v20_dir/f'{case}_integer_pairs.csv')
  bysample=defaultdict(list)
  for r in oldpairs:bysample[int(r['sample'])].append(r)
  counts=Counter();picked=0
  out=args.out_dir/f'{case}_temporal_pairs.csv'
  cols=['case','sample','time_s','population','region_tags','low_hz','high_hz',
        'integer_ratio','v20_family_diagnostic','v19_low_support','v19_high_support',
        'temporal_pattern','paired_measurements','low_supported_shift_count',
        'high_supported_shift_count','past_to_future_energy_db',
        'past_to_future_flatness_change','past_to_future_high_band_change',
        'low_exclusive_power_min','low_exclusive_power_max','evidence_json',
        'f0_selected','voice_identity_verified']
  with out.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
   for ob in observations:
    sample=int(ob['sample']);t=float(ob['time_s'])
    if abs(sample/sr-t)>.003:raise ValueError(f'{case}: sample/time mismatch at {t}')
    counts['observations_'+ob['population']]+=1
    rr=sorted(bysample.get(sample,[]),key=lambda r:(-row_quality(r),float(r['low_hz']),float(r['high_hz'])))
    if rr:counts['observations_with_integer_pairs']+=1
    if args.max_pairs_per_observation:rr=rr[:args.max_pairs_per_observation]
    for r in rr:
     low=float(r['low_hz']);high=float(r['high_hz'])
     result=audit_pair(y,sr,t,low,high);picked+=1
     counts['pattern:'+result['temporal_pattern']]+=1
     tags=[name for name,start,end in REGIONS.get(case,[]) if start<=t<=end]
     writer.writerow({'case':case,'sample':sample,'time_s':t,
      'population':ob['population'],'region_tags':'|'.join(tags),
      'low_hz':low,'high_hz':high,'integer_ratio':r['integer_ratio'],
      'v20_family_diagnostic':r['family_diagnostic'],
      'v19_low_support':r['low_v19_present_support'],
      'v19_high_support':r['high_v19_present_support'],
      'temporal_pattern':result['temporal_pattern'],
      'paired_measurements':result['paired_measurements'],
      'low_supported_shift_count':result['low_supported_shift_count'],
      'high_supported_shift_count':result['high_supported_shift_count'],
      'past_to_future_energy_db':result['past_to_future_energy_db'],
      'past_to_future_flatness_change':result['past_to_future_flatness_change'],
      'past_to_future_high_band_change':result['past_to_future_high_band_change'],
      'low_exclusive_power_min':result['low_exclusive_power_min'],
      'low_exclusive_power_max':result['low_exclusive_power_max'],
      'evidence_json':json.dumps(result,separators=(',',':')),
      'f0_selected':False,'voice_identity_verified':False})
  manifest['cases'][case]={'input_observations':len(observations),
                            'input_integer_pairs':len(oldpairs),
                            'examined_pairs':picked,'counts':dict(counts),
                            'output':str(out)}
  print(f'{case}: observations={len(observations)} pairs={picked} patterns={dict(counts)}',flush=True)
 (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('V21: observational only. No F0 decisions, source verdicts, or production modifications.',flush=True)

if __name__=='__main__':main()
