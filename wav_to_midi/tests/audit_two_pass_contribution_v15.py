#!/usr/bin/env python3
"""V15 read-only, waveform-based two-pass contribution audit.

Input: V14 *_dark.csv (NOT a MIDI truth file), and corresponding original WAVs.
Production median is a BLOCK SUMMARY: its exact-frame F0 is NOT independently known.
Measures production frequency anew in independent centered, left and right waveform views.
No production/tracker changes, no pitch imputation, no claims of vocal-source identity.
"""
from __future__ import annotations
import argparse, csv, json, math, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={'RATATA':[(36.923,37.435,'djuvv_1'),(41.025,41.538,'djuvv_2'),(49.230,49.743,'djuvv_3'),(42.915,43.065,'submultiple')],
'Ochiitai':[(42.19,42.40,'transition')], 'PREDESTINATI':[(37.97,38.13,'control_38'),(53.52,53.73,'control_53')],
'Trandafiri':[(10.70,10.95,'steady_i_not_glissando')]}
def num(x):
 try:
  y=float(x);return y if math.isfinite(y) else None
 except (ValueError,TypeError):return None
def cents(x,y):return abs(1200*math.log2(x/y)) if x and y and x>0 and y>0 else None
def load_rows(path,zip_obj,name):
 if zip_obj:
  keys=[k for k in zip_obj.namelist() if k.endswith('/'+name+'_dark.csv') or k==name+'_dark.csv']
  if len(keys)!=1:raise ValueError(f'Expected exactly one {name}_dark.csv in ZIP, got {keys}')
  import io
  with zip_obj.open(keys[0]) as f:return list(csv.DictReader(io.TextIOWrapper(f)))
 p=path/f'{name}_dark.csv'
 with p.open(newline='') as f:return list(csv.DictReader(f))
def segment(audio,sr,t0,t1):
 a=max(0,round(t0*sr));b=min(len(audio),round(t1*sr))
 return audio[a:b] if b>a else np.empty(0)
def periodicity(x,sr,hz):
 """Independent normalized ACF at predicted period and local ±2.5% lags.
 A high ACF alone is not proof of current active singing or correct fundamental.
 """
 if len(x)<64 or hz<=0:return {'testable':False,'reason':'insufficient_window'}
 x=np.asarray(x,dtype=np.float64);x=x-np.mean(x)
 rms=float(np.sqrt(np.mean(x*x)))
 if rms<1e-10:return {'testable':False,'reason':'near_zero_signal','rms':rms}
 period=sr/hz
 if len(x)/period<3:return {'testable':False,'reason':'fewer_than_three_cycles','cycles':len(x)/period,'rms':rms}
 lagmin=max(2,int(round(period*.975)));lagmax=min(len(x)//2,int(round(period*1.025)))
 if lagmax<lagmin:return {'testable':False,'reason':'lag_outside_window','rms':rms}
 val=[]
 for lag in range(lagmin,lagmax+1):
  a=x[:-lag];b=x[lag:];den=float(np.linalg.norm(a)*np.linalg.norm(b))
  val.append((float(a@b/den) if den>1e-16 else 0.0,lag))
 acf,lag=max(val)
 # Independent harmonic spectral concentration centered on predicted multiples.
 w=np.hanning(len(x)); power=np.abs(np.fft.rfft(x*w))**2
 df=sr/len(x); nyquist=sr/2
 fundamental=0.; harmonics=0.; total=float(power[1:].sum())
 for harmonic in range(1,min(10,int(nyquist/hz))+1):
  f=hz*harmonic; k=round(f/df);halfwidth=max(1,round(max(12.,f*.015)/df))
  lo=max(1,k-halfwidth);hi=min(len(power),k+halfwidth+1)
  band=float(power[lo:hi].sum()) if hi>lo else 0.
  harmonics+=band
  if harmonic==1:fundamental=band
 return {'testable':True,'rms':rms,'cycles':len(x)/period,'acf':acf,'acf_peak_hz':sr/lag,
         'fundamental_fraction':fundamental/max(total,1e-20),'harmonic_fraction':min(1.,harmonics/max(total,1e-20))}
def view_evidence(audio,sr,t,hz):
 out={}
 for ms in (24,40,64):
  h=ms/2000
  for side,start,end in (('present',t-h,t+h),('earlier',t-3*h,t-h),('later',t+h,t+3*h)):
   out[f'{ms}_{side}']=periodicity(segment(audio,sr,start,end),sr,hz)
 return out
def summarize_views(views):
 present=[views[f'{ms}_present'] for ms in (24,40,64)]
 flank=[views[f'{ms}_{side}'] for ms in (24,40,64) for side in ('earlier','later')]
 testable=[v for v in present if v.get('testable')]
 # Conservative *screening* thresholds, not tuned F0 acceptance criteria.
 good=[v for v in testable if v.get('acf',0)>=.72 and v.get('harmonic_fraction',0)>=.15]
 bad=[v for v in testable if v.get('acf',0)<.50]
 flank_good=[v for v in flank if v.get('testable') and v.get('acf',0)>=.72 and v.get('harmonic_fraction',0)>=.15]
 if len(good)>=2: status='independent_present_period_support_not_voice_proof'
 elif not testable:status='not_testable_present'
 elif not good and len(bad)>=2:status='present_period_not_supported'
 elif flank_good and not good:status='flank_only_possible_temporal_leakage'
 else:status='ambiguous_present_support'
 return status,len(testable),len(good),len(flank_good)
def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--case',choices=('all',*CASES),default='all')
 a.add_argument('--v14-dir',type=Path,default=Path('tests/failure_mechanisms_v14'))
 a.add_argument('--v14-zip',type=Path,help='Optional alternative to --v14-dir')
 a.add_argument('--wav-dir',type=Path,default=Path('tests'))
 a.add_argument('--out-dir',type=Path,default=Path('tests/two_pass_contribution_v15'))
 args=a.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
 z=zipfile.ZipFile(args.v14_zip) if args.v14_zip else None
 report={'schema':'v15_read_only_independent_two_pass_screen','production_block_median_is_not_instantaneous_pitch':True,
 'acoustic_screen_not_validated_f0_or_active_voice':True,'no_automatic_rescue':True,'production_modified':False,
 'criteria':{'minimum_present_acf':.72,'minimum_present_harmonic_fraction':.15,'present_views_required':2,'minimum_cycles':3},'cases':{}}
 for name in (CASES if args.case=='all' else (args.case,)):
  dark=load_rows(args.v14_dir,z,name);wav=args.wav_dir/f'{name}.wav'
  if not wav.is_file():raise FileNotFoundError(f'Original waveform needed for independent test: {wav}')
  audio,sr=sf.read(wav,dtype='float64',always_2d=False)
  if audio.ndim==2:audio=audio.mean(axis=1)
  counter=Counter();out=[];last=-1
  for r in dark:
   sample=int(r['sample']);t=float(r['time_s'])
   if sample<=last:raise ValueError(f'{name} V14 rows non-increasing samples')
   last=sample
   # v14 uses song-local original samples, confirm time alignment.
   if abs(sample/sr-t)>.003:raise ValueError(f'{name} WAV sample-rate mismatch at {sample}: {sample/sr} vs {t}')
   hz=num(r['production_median_hz']);voiced=num(r['production_voiced_samples']);available=bool(hz and hz>0 and voiced and voiced>0)
   record={k:r[k] for k in ('case','sample','time_s','mechanism','v13_evidence_status','production_block_status','production_voiced_samples','production_median_hz','v13_measured_hz','annotation_tags')}
   record.update(production_pitch_available=available,production_is_block_median=True,production_waveform_evidence='',screen='',
                 present_views_testable='',present_views_supported='',flank_views_supported='',
                 same_aperture_candidates_json=r['all_final_raw_candidates_json'],source_identity_verified=False,
                 decisive_rescue=False,decision_applied=False)
   if available:
    views=view_evidence(audio,sr,t,hz)
    status,nt,ng,nf=summarize_views(views)
    record.update(production_waveform_evidence=json.dumps(views,separators=(',',':')),
                  screen=status,present_views_testable=nt,present_views_supported=ng,flank_views_supported=nf)
    # V14 local eligible measured candidates: compare frequency families, not claim independent new information.
    try:eligible=json.loads(r['eligible_candidates_json'] or '[]')
    except json.JSONDecodeError:eligible=[]
    matching=[c for c in eligible if num(c.get('hz')) and cents(hz,num(c['hz']))<=70]
    try: raw=json.loads(r['all_final_raw_candidates_json'] or '[]')
    except json.JSONDecodeError:raw=[]
    raw_matching=[c for c in raw if num((c.get('candidate') or {}).get('hz')) and cents(hz,num(c['candidate']['hz']))<=70]
    if status=='independent_present_period_support_not_voice_proof':
     if matching:record['screen']='present_support_already_in_adaptive_eligible_candidates'
     elif raw_matching:record['screen']='production_frequency_already_measured_but_adaptive_rejected_or_remote'
     else:record['screen']='production_period_adds_testable_present_support'
   else:
    record['screen']='neither_pass_has_available_pitch_no_pitch_inferred'
    # Explore what the existing inspected apertures actually observed. A WAV window alone cannot conjure a pitch.
    record['production_waveform_evidence']=''
   counter[record['screen']]+=1;out.append(record)
  if len(out)!=len(dark):raise AssertionError('Lost records')
  outfile=args.out_dir/f'{name}_two_pass.csv'
  with outfile.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
  manifest={'dark':len(out),'production_pitch_available':sum(r['production_pitch_available'] for r in out),
    'no_production_pitch':sum(not r['production_pitch_available'] for r in out),'screens':dict(counter),
    'potential_new_present_period_evidence_not_rescue':counter['production_period_adds_testable_present_support'],
    'validated_rescues':0,'original_wav_sample_rate':sr}
  report['cases'][name]=manifest
  print(name, json.dumps(manifest,separators=(',',':')),flush=True)
 (args.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('No verdict on whether two passes improve final F0 until source identity and exact-frame production pitch are verified.')
if __name__=='__main__':main()
