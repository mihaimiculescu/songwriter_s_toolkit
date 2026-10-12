#!/usr/bin/env python3
"""Read-only V12 harmonic-pattern juror: FFT/STFT candidate templates, no changed verdicts.

Run at repo root:
 python tests/audit_harmonic_detective_v1.py --v12-zip tests/shadow_adjudicator_v22_fixed_v12.zip --wav-root tests --out tests/harmonic_detective_v1
Dependencies: numpy, soundfile. No scipy, no model download.
"""
from __future__ import annotations
import argparse, csv, io, json, math, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')

def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def observations(source):
    if source.is_dir():
        for case in CASES:
            p=source/f'{case}_observation_summary.csv'
            if not p.is_file(): raise FileNotFoundError(p)
            with p.open(newline='',encoding='utf-8-sig') as f: yield case,list(csv.DictReader(f))
    else:
        with zipfile.ZipFile(source) as z:
            for case in CASES:
                matches=[n for n in z.namelist() if n.endswith('/'+case+'_observation_summary.csv') or n==case+'_observation_summary.csv']
                if len(matches)!=1: raise ValueError(f'{case}: expected one observation summary, found {matches}')
                yield case,list(csv.DictReader(io.StringIO(z.read(matches[0]).decode('utf-8-sig'))))

def candidates_from_groups(row):
    groups=json.loads(row.get('note_groups_json') or '[]')
    seen={}
    for group in groups:
        for member in group.get('members',[]):
            hz=float(member['hz'])
            if hz<=0 or not np.isfinite(hz): continue
            # The diagnostic includes rejected candidates as controls, but explicitly marks admission.
            key=round(hz,7)
            if key not in seen:
                seen[key]={'hz':hz,'note':member.get('nearest_midi'), 'admitted':bool(member.get('acoustically_admissible',False)),
                           'within_49c':bool(member.get('within_49c',False)), 'v12_acf':member.get('acf_median'),
                           'group_reason':group.get('reason','')}
    return list(seen.values())

def spectrum(x,sr,time_s,window_ms,nfft_factor):
    n=max(64,int(round(sr*window_ms/1000))); center=int(round(time_s*sr)); start=center-n//2
    if start<0 or start+n>len(x): return None,'wav_boundary'
    seg=np.asarray(x[start:start+n],dtype=np.float64)
    if not np.all(np.isfinite(seg)): return None,'nonfinite_audio'
    seg-=seg.mean(); rms=float(np.sqrt(np.mean(seg*seg)))
    if rms<1e-9: return None,'near_silence'
    seg*=np.hanning(n)
    nfft=1<<(int(np.ceil(np.log2(n*nfft_factor))))
    mag=np.abs(np.fft.rfft(seg,n=nfft)); freqs=np.fft.rfftfreq(nfft,1/sr)
    # Widening FFT with zero padding interpolates, but DOES NOT increase physical resolution.
    return (freqs,mag,rms,float(sr/n)),''

def harmonic_score(spec,f0,args):
    frequencies,mag,rms,resolution=spec
    nyquist=frequencies[-1]
    max_h=min(args.harmonics,int(min(args.max_hz,nyquist-20)/f0))
    if max_h<args.min_harmonics: return None
    logmag=np.log(np.maximum(mag,1e-12))
    # Spectral floor around each harmonic. Exclude the peak neighborhood to prevent
    # the peak itself artificially raising the local floor.
    prominences=[]; valid=[]; peak_hz=[]; backgrounds=[]
    for h in range(1,max_h+1):
        target=h*f0
        tol=max(args.tolerance_hz, target*args.tolerance_cents*math.log(2)/1200, resolution*.75)
        core=(frequencies>=target-tol)&(frequencies<=target+tol)
        outer=max(args.background_hz, 3*tol)
        flank=(frequencies>=target-outer)&(frequencies<=target+outer)&(~((frequencies>=target-1.4*tol)&(frequencies<=target+1.4*tol)))
        if not np.any(core) or np.count_nonzero(flank)<4: continue
        ix=np.flatnonzero(core); best=ix[int(np.argmax(logmag[ix]))]
        baseline=float(np.median(logmag[flank])); delta=float(logmag[best]-baseline)
        # Clamp the effect of a single huge peak: require a pattern, not one partial.
        prominences.append(max(0.,min(args.max_prominence_db,delta*20/math.log(10))))
        valid.append(h);peak_hz.append(round(float(frequencies[best]),3));backgrounds.append(baseline)
    if len(valid)<args.min_harmonics: return None
    a=np.array(prominences,dtype=float)
    weights=1/np.sqrt(np.array(valid,dtype=float))
    support=float(np.sum(weights*(a>=args.support_db))/sum(weights))
    # Mean capped prominence, tempered by fraction of harmonics that clear the floor.
    strength=float(np.sum(weights*a)/sum(weights)/args.max_prominence_db)
    score=float(.55*support+.45*strength)
    return {'score':round(score,6),'support':round(support,6),'strength':round(strength,6),
            'harmonics_tested':len(valid),'harmonics_supported':int(sum(a>=args.support_db)),
            'prominence_db':json.dumps([round(v,2) for v in a]),
            'peak_hz':json.dumps(peak_hz),'rms':round(rms,8),'physical_resolution_hz':round(resolution,3)}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v12-zip',type=Path,default=Path('tests/shadow_adjudicator_v22_fixed_v12.zip'))
    p.add_argument('--wav-root',type=Path,default=Path('tests'))
    p.add_argument('--out',type=Path,default=Path('tests/harmonic_detective_v1'))
    p.add_argument('--windows-ms',type=float,nargs='+',default=[64.,96.])
    p.add_argument('--harmonics',type=int,default=10)
    p.add_argument('--min-harmonics',type=int,default=3)
    p.add_argument('--max-hz',type=float,default=4500.)
    p.add_argument('--nfft-factor',type=int,default=4)
    p.add_argument('--tolerance-hz',type=float,default=5.)
    p.add_argument('--tolerance-cents',type=float,default=22.)
    p.add_argument('--background-hz',type=float,default=90.)
    p.add_argument('--support-db',type=float,default=5.)
    p.add_argument('--max-prominence-db',type=float,default=24.)
    p.add_argument('--diagnostic-margin',type=float,default=.08)
    args=p.parse_args()
    if (not args.windows_ms or min(args.windows_ms)<=0 or args.min_harmonics<2 or args.harmonics<args.min_harmonics or
        args.nfft_factor<1 or args.support_db<0 or args.max_prominence_db<=0 or args.diagnostic_margin<0): p.error('Invalid parameters')
    if not args.v12_zip.exists(): p.error(f'Missing V12 source: {args.v12_zip}')
    args.out.mkdir(parents=True,exist_ok=True)
    all_candidates=[]; all_obs=[]; cross=[]; manifest={}
    for case,rows in observations(args.v12_zip):
        wav=args.wav_root/(case+'.wav')
        if not wav.is_file(): p.error(f'Missing WAV: {wav}')
        x,sr=sf.read(wav,dtype='float32'); x=x.mean(axis=1) if x.ndim==2 else x
        obs_out=[]; cand_out=[]
        for row in rows:
            t=float(row['time_s']); original=row.get('selected_provisional_hz',''); has_champion=original not in ('',None)
            candidates=candidates_from_groups(row)
            specs={duration:spectrum(x,sr,t,duration,args.nfft_factor) for duration in args.windows_ms}
            scored=[]
            for c in candidates:
                frames=[]; failures=[]
                for duration,(spec,reason) in specs.items():
                    if spec is None: failures.append(f'{duration}ms:{reason}');continue
                    result=harmonic_score(spec,c['hz'],args)
                    if result is None: failures.append(f'{duration}ms:insufficient_harmonic_band');continue
                    frames.append(result)
                out={'case':case,'time_s':t,'candidate_hz':c['hz'],'nearest_midi':c['note'],
                     'range_and_acoustic_admitted':int(c['admitted']), 'within_49c':int(c['within_49c']),
                     'v12_acf_median':c['v12_acf'] if c['v12_acf'] is not None else '',
                     'valid_windows':len(frames),'invalid_reasons':'|'.join(failures),
                     'harmonic_score':round(float(np.median([f['score'] for f in frames])),6) if frames else '',
                     'harmonic_support':round(float(np.median([f['support'] for f in frames])),6) if frames else '',
                     'harmonic_strength':round(float(np.median([f['strength'] for f in frames])),6) if frames else '',
                     'window_details_json':json.dumps(dict(zip([str(k) for k,(spec,_) in specs.items() if spec is not None],frames))) if frames else '[]'}
                cand_out.append(out)
                if c['admitted'] and c['within_49c'] and frames: scored.append(out)
            scored.sort(key=lambda r:r['harmonic_score'],reverse=True)
            if not candidates: diagnosis='no_candidates_in_v12_groups'; top=None; margin=None
            elif not scored: diagnosis='no_admissible_scored_candidate';top=None;margin=None
            elif len(scored)==1: diagnosis='only_one_admissible_scored_candidate';top=scored[0];margin=None
            else:
                top=scored[0];margin=round(scored[0]['harmonic_score']-scored[1]['harmonic_score'],6)
                diagnosis='detective_prefers_one' if margin>=args.diagnostic_margin else 'detective_inconclusive'
            top_hz=top['candidate_hz'] if top else None
            agrees=bool(has_champion and top_hz is not None and abs(top_hz-float(original))<.02)
            record={'case':case,'time_s':t,'v12_outcome':row['observation_outcome'],
                    'v12_champion_hz':original,'v12_abstention_category':row.get('abstention_category',''),
                    'candidate_count':len(candidates),'scored_admissible_count':len(scored),
                    'detective_status':diagnosis,'detective_best_hz':top_hz if top else '',
                    'detective_best_score':top['harmonic_score'] if top else '',
                    'detective_runnerup_hz':scored[1]['candidate_hz'] if len(scored)>1 else '',
                    'detective_runnerup_score':scored[1]['harmonic_score'] if len(scored)>1 else '',
                    'detective_margin':margin if margin is not None else '',
                    'detective_agrees_with_champion':int(agrees) if has_champion and top else '',
                    'NOTE':'OBSERVATIONAL_ONLY_NO_VERDICT_CHANGE'}
            obs_out.append(record)
        write_csv(args.out/f'{case}_harmonic_candidates.csv',cand_out,
                  ['case','time_s','candidate_hz','nearest_midi','range_and_acoustic_admitted','within_49c','v12_acf_median','valid_windows','invalid_reasons','harmonic_score','harmonic_support','harmonic_strength','window_details_json'])
        write_csv(args.out/f'{case}_harmonic_observations.csv',obs_out,
                  ['case','time_s','v12_outcome','v12_champion_hz','v12_abstention_category','candidate_count','scored_admissible_count','detective_status','detective_best_hz','detective_best_score','detective_runnerup_hz','detective_runnerup_score','detective_margin','detective_agrees_with_champion','NOTE'])
        all_candidates+=cand_out; all_obs+=obs_out
        for outcome in ('champion','unresolved'):
            part=[r for r in obs_out if (r['v12_champion_hz']!='')==(outcome=='champion')]
            for status,n in Counter(r['detective_status'] for r in part).items(): cross.append({'case':case,'v12_status':outcome,'detective_status':status,'observations':n})
        manifest[case]={'observations':len(rows),'champions':sum(bool(r['v12_champion_hz']) for r in obs_out),'unresolved':sum(not bool(r['v12_champion_hz']) for r in obs_out)}
        print(case,manifest[case])
    write_csv(args.out/'cross_tab_by_song.csv',cross,['case','v12_status','detective_status','observations'])
    manifest['total']={'observations':len(all_obs),'champions':sum(bool(r['v12_champion_hz']) for r in all_obs),
                       'unresolved':sum(not bool(r['v12_champion_hz']) for r in all_obs),'candidate_records':len(all_candidates)}
    manifest['parameters']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest['warning']='No new verdicts; spectral evidence overlaps with existing juror; FFT zero padding is not increased physical resolution; no ground truth.'
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('TOTAL',manifest['total']);print('Diagnostic only: NO V12 verdicts modified.')

if __name__=='__main__': main()
