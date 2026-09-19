#!/usr/bin/env python3
"""V19: read-only adaptive-only INDEPENDENT waveform evidence audit.

Input: V18 *_dark.csv and *_supported_source.csv, original unmodified WAVs.
Re-detects periods from waveform (not V3 candidates), then tests both new
periods and old measured families in synchronized centered / non-overlapping
past / future views. No MIDI, production fields, pitch selection, voice/noise
verdict, interval penalty, propagation, or production modifications.

Run from repository root:
 python tests/audit_adaptive_waveform_v19.py --case all
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={'Ochiitai':[('ku_k_attack',3.465,3.692),('transition',42.19,42.40)],
 'PREDESTINATI':[('control38',37.97,38.13),('control53',53.52,53.73)],
 'RATATA':[('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.90,43.10)],
 'Trandafiri':[('steady_i_NOT_glissando',10.70,10.95)]}
WIDTHS=(24,40,64)

def num(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None

def cents(a,b):return abs(1200*math.log2(a/b)) if a and b and a>0 and b>0 else float('inf')

def read(path):
 if not path.is_file():raise FileNotFoundError(path)
 with path.open(newline='') as f:return list(csv.DictReader(f))

def segment(y,sr,lo,hi):
 a=round(lo*sr);b=round(hi*sr)
 return None if a<0 or b>len(y) or b-a<64 else y[a:b]

def spectrum_metrics(x,sr,hz):
 n=len(x);z=np.abs(np.fft.rfft(x*np.hanning(n)))**2
 total=float(z[1:].sum());df=sr/n;nyq=sr/2
 if total<=1e-21:return (0.,0.)
 occupied=np.zeros(len(z),dtype=bool);fund=0.
 for j in range(1,min(10,int(nyq/hz))+1):
  freq=j*hz;k=round(freq/df);radius=max(1,round(max(12.,.015*freq)/df))
  lo=max(1,k-radius);hi=min(len(z),k+radius+1)
  if j==1:fund=float(z[lo:hi].sum())/total
  occupied[lo:hi]=True
 return fund,min(1.,float(z[occupied].sum())/total)

def analyze_window(x,sr,lo_hz=65.,hi_hz=950.,limit=7):
 """Independent ACF peak discovery with overlap-energy normalized lags."""
 if x is None:return {'status':'outside_wav','peaks':[]}
 x=np.asarray(x,dtype=np.float64);x=x-x.mean();n=len(x)
 power=float(x@x);rms=math.sqrt(power/n)
 if power<1e-18:return {'status':'near_zero_signal','rms':rms,'peaks':[]}
 fftlen=1<<(2*n-1).bit_length()
 X=np.fft.rfft(x,fftlen)
 ac=np.fft.irfft(X*X.conj(),fftlen)[:n]
 sq=np.r_[0.,np.cumsum(x*x)]
 first=max(2,int(math.floor(sr/hi_hz)));last=min(n//2,int(math.ceil(sr/lo_hz)))
 if first>last:return {'status':'untestable_lags','rms':rms,'peaks':[]}
 lags=np.arange(first,last+1)
 denom=np.sqrt(np.maximum((sq[n-lags]-sq[0])*(sq[n]-sq[lags]),0.))
 vals=ac[lags]/np.maximum(denom,1e-20)
 loc=[]
 for i in range(len(lags)):
  if vals[i]<.20:continue
  if (i>0 and vals[i]<vals[i-1]) or (i+1<len(lags) and vals[i]<vals[i+1]):continue
  hz=sr/int(lags[i]);cycles=n*hz/sr
  if cycles<3.:continue
  fund,harm=spectrum_metrics(x,sr,hz)
  loc.append({'hz':hz,'lag':int(lags[i]),'acf':float(vals[i]),'cycles':cycles,
              'fundamental_fraction':fund,'harmonic_fraction':harm})
 loc.sort(key=lambda p:(-p['acf'],-p['harmonic_fraction']))
 distinct=[]
 for p in loc:
  if all(cents(p['hz'],q['hz'])>45 for q in distinct):distinct.append(p)
  if len(distinct)>=limit:break
 return {'status':'measured','rms':rms,'peaks':distinct}

def probe(x,sr,hz):
 """Test a named frequency without granting it F0 status."""
 if x is None:return {'testable':False,'reason':'outside_wav'}
 n=len(x);cycles=n*hz/sr
 if cycles<3:return {'testable':False,'reason':'fewer_than_three_cycles','cycles':cycles}
 centered=np.asarray(x,dtype=np.float64)-float(np.mean(x));power=float(centered@centered)
 if power<1e-18:return {'testable':False,'reason':'near_zero_signal','cycles':cycles}
 lo=max(2,round(sr/hz*.975));hi=min(n//2,round(sr/hz*1.025))
 if lo>hi:return {'testable':False,'reason':'lag_not_available','cycles':cycles}
 v=[]
 for lag in range(lo,hi+1):
  a=centered[:-lag];b=centered[lag:];den=math.sqrt(float(a@a)*float(b@b))
  v.append((float(a@b)/den if den>1e-18 else 0.,lag))
 ac,lag=max(v);fund,harm=spectrum_metrics(centered,sr,hz)
 return {'testable':True,'acf':ac,'peak_hz':sr/lag,'cycles':cycles,
         'fundamental_fraction':fund,'harmonic_fraction':harm,
         'rms':math.sqrt(power/n)}

def select_control(rows,max_count):
 if max_count<=0:return []
 if len(rows)<=max_count:return rows
 if max_count==1:return [rows[len(rows)//2]]
 idx=np.linspace(0,len(rows)-1,max_count,dtype=int)
 return [rows[int(i)] for i in idx]

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--case',choices=('all',*CASES),default='all')
 ap.add_argument('--v18-dir',type=Path,default=Path('tests/adaptive_criteria_v18'))
 ap.add_argument('--wav-dir',type=Path,default=Path('tests'))
 ap.add_argument('--out-dir',type=Path,default=Path('tests/adaptive_waveform_v19'))
 ap.add_argument('--supported-controls',type=int,default=400,
                 help='Uniform time-sampled controls per song; 0=none')
 args=ap.parse_args()
 if args.supported_controls<0:ap.error('--supported-controls must be >=0')
 args.out_dir.mkdir(parents=True,exist_ok=True)
 manifest={'schema':'v19_adaptive_only_independent_waveform_measurement',
  'wav_remeasured':True,'production_used':False,'midi_used':False,
  'new_f0_selected':False,'adjudication_changed':False,'source_identity_verified':False,
  'note':'All classification flags are diagnostic only. ACF peaks can be integer submultiples or residual periodic material.',
  'widths_ms':WIDTHS,'supported_controls_per_song_limit':args.supported_controls,'cases':{}}
 for case in CASES if args.case=='all' else (args.case,):
  dark=read(args.v18_dir/f'{case}_dark.csv')
  supported=read(args.v18_dir/f'{case}_supported_source.csv')
  controls=select_control(supported,args.supported_controls)
  wav=args.wav_dir/f'{case}.wav'
  if not wav.is_file():raise FileNotFoundError(f'Original WAV required: {wav}')
  y,sr=sf.read(wav,dtype='float64',always_2d=False)
  if y.ndim==2:y=y.mean(axis=1)
  out=[];counts=Counter()
  for pop,rows in [('dark',dark),('supported_control',controls)]:
   for r in rows:
    sample=int(r['sample']);t=float(r['time_s'])
    if abs(sample/sr-t)>.003:raise ValueError(f'{case}: sample/time/WAV mismatch at {sample}')
    families=json.loads(r.get('families_json') or '[]')
    # The frequencies below are *hypotheses only*, discovered earlier from waveform.
    old=[num(f.get('representative_hz')) for f in families]
    old=[h for h in old if h and 65<=h<=950]
    views={}; discovered=[]
    for ms in WIDTHS:
     half=ms/2000.
     for side,lo,hi in [('present',t-half,t+half),
                        ('past',t-3*half,t-half),('future',t+half,t+3*half)]:
      result=analyze_window(segment(y,sr,lo,hi),sr)
      views[f'{ms}_{side}']=result
      if side=='present':
       discovered.extend((ms,p) for p in result['peaks'])
    # Deduplicate peaks across newly and independently measured apertures.
    families_new=[]
    for ms,p in sorted(discovered,key=lambda q:-q[1]['acf']):
     match=next((q for q in families_new if cents(q['hz'],p['hz'])<=45),None)
     if match is None:
      match={'hz':p['hz'],'discovery_ms':[],'best_acf':p['acf']}
      families_new.append(match)
     match['discovery_ms'].append(ms)
     match['best_acf']=max(match['best_acf'],p['acf'])
    # Always re-probe original hypotheses as well as newly measured peaks.
    hypotheses=[]
    for hz in old+[q['hz'] for q in families_new]:
     if any(cents(hz,z)<=45 for z in hypotheses):continue
     hypotheses.append(hz)
    tests=[]
    for hz in hypotheses:
     evidence={}
     for ms in WIDTHS:
      half=ms/2000.
      for side,lo,hi in [('present',t-half,t+half),('past',t-3*half,t-half),('future',t+half,t+3*half)]:
       evidence[f'{ms}_{side}']=probe(segment(y,sr,lo,hi),sr,hz)
     # Screening thresholds are descriptive experimental flags, not an F0 rule.
     present=[evidence[f'{ms}_present'] for ms in WIDTHS]
     flank=[evidence[f'{ms}_{s}'] for ms in WIDTHS for s in ('past','future')]
     good=lambda e:e.get('testable') and e.get('acf',0)>=.72 and e.get('harmonic_fraction',0)>=.15
     ng=sum(bool(good(e)) for e in present);nf=sum(bool(good(e)) for e in flank)
     tests.append({'hz':hz,'from_v18':any(cents(hz,h)<=45 for h in old),
      'independently_discovered':any(cents(hz,q['hz'])<=45 for q in families_new),
      'present_support_views':ng,'flank_support_views':nf,
      'screen':'multi_present_period_not_voice_proof' if ng>=2 else
               'flank_only_possible_leakage' if nf and not ng else
               'insufficient_or_ambiguous_present_evidence',
      'views':evidence})
    multi=[h for h in tests if h['present_support_views']>=2]
    new=[h for h in multi if not h['from_v18']]
    # 12ms adjacent windows straddle the instant; do not force quiet notes to noise.
    past=segment(y,sr,t-.018,t-.006);now=segment(y,sr,t-.006,t+.006)
    future=segment(y,sr,t+.006,t+.018)
    def rms(a):return float(np.sqrt(np.mean(a*a))) if a is not None else None
    energies=[rms(z) for z in (past,now,future)]
    db=(20*math.log10(max(energies[2],1e-12)/max(energies[0],1e-12))
        if energies[0] is not None and energies[2] is not None else None)
    tags=[tag for tag,lo,hi in REGIONS.get(case,[]) if lo<=t<=hi]
    status=('new_present_period_hypothesis_not_f0' if new else
     'competing_present_periods_not_f0' if len(multi)>1 else
     'one_present_period_source_unverified' if len(multi)==1 else
     'no_multi_aperture_present_support_not_noise_proof')
    record={'case':case,'sample':sample,'time_s':t,'population':pop,
      'v18_priority':r['investigation_priority'],'diagnostic_screen':status,
      'new_present_period_hypotheses':len(new),'multi_present_period_hypotheses':len(multi),
      'v18_hypotheses':len(old),'energy_past_present_future_12ms_json':json.dumps(energies),
      'energy_past_to_future_db':db,'region_tags':';'.join(tags),
      'independent_discoveries_json':json.dumps(families_new,separators=(',',':')),
      'hypothesis_tests_json':json.dumps(tests,separators=(',',':')),
      'views_json':json.dumps(views,separators=(',',':')),
      'active_source_verified':False,'final_f0_selected':False,'production_used':False}
    out.append(record);counts[(pop,status)]+=1
  target=args.out_dir/f'{case}_waveform.csv'
  if out:
   with target.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
  manifest['cases'][case]={'sample_rate':sr,'dark_processed':len(dark),
     'supported_controls_processed':len(controls),'screens':{f'{p}:{s}':n for (p,s),n in counts.items()},
     'new_f0_validated':0,'source_identity_unverified':True}
  print(f'{case}: dark={len(dark)} supported_controls={len(controls)}',flush=True)
  for (population,status),count in sorted(counts.items()):print(f'  {population} {status}: {count}',flush=True)
 (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('V19 is acoustic discovery/measurement ONLY; not a source classifier or revised F0 tracker.',flush=True)
if __name__=='__main__':main()
