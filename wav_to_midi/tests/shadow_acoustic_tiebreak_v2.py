#!/usr/bin/env python3
"""Read-only four-WAV shadow adjudication; acoustic only vs optional transition tie-break.

Run from repository root:
 python tests/shadow_acoustic_tiebreak_v2.py --case all

Only WAV-derived measured periods are candidates. The previous independently accepted
acoustic measurement is used solely to compare *already acoustically tied* candidates;
no MIDI, no F0 injection, no modification to WAV, tracker, status or production files.
Thresholds here are experimental diagnostics, NOT validated production settings.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

CASES = {'Ochiitai':(42.214,42.353), 'PREDESTINATI':(38.034,53.592),
         'RATATA':(36.923,41.026,42.965,43.008,49.231),
         'Trandafiri':(10.774,10.821)}

def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError): return None

def cents(a,b): return abs(1200*math.log2(a/b))
def related(a,b):
    ratio=max(a,b)/min(a,b); multiple=round(ratio)
    return 2<=multiple<=5 and cents(ratio,multiple)<=55

def amplitude(x,fs,hz):
    # Actual measured fundamental: least-squares projection including constant.
    n=np.arange(len(x),dtype=np.float64)
    c=np.cos(2*np.pi*hz*n/fs); s=np.sin(2*np.pi*hz*n/fs)
    matrix=np.column_stack((c,s,np.ones(len(x))))
    beta=np.linalg.lstsq(matrix,x,rcond=None)[0]
    return float(np.hypot(beta[0],beta[1]))

def select_acoustically(row,audio,fs):
    info={'reason':'','candidates':[],'winner':None,'frequency':None,'near_tie':[]}
    if row.get('nonstationary')=='1': info['reason']='nonstationary';return info
    if row.get('status')!='observed_candidate':info['reason']='adaptive_unresolved';return info
    center=number(row.get('time_s'));duration=number(row.get('window_ms'))
    if center is None or duration is None:info['reason']='missing_window';return info
    # Reproduce prototype window convention: its observation time is the CENTER.
    length=max(64,round(duration*fs/1000));mid=round(center*fs)
    lo=mid-length//2;hi=lo+length
    if lo<0 or hi>len(audio):info['reason']='window_out_of_bounds';return info
    x=np.asarray(audio[lo:hi],dtype=np.float64)
    x=x-x.mean()
    try: raw=json.loads(row.get('candidates') or '[]')
    except json.JSONDecodeError:info['reason']='invalid_candidate_json';return info
    seen=set()
    for entry in raw:
        hz=number(entry.get('hz'));acf=number(entry.get('acf'));cm=number(entry.get('cmndf'))
        if hz is None or acf is None or cm is None or hz<=0:continue
        key=round(hz,5)
        if key in seen:continue
        seen.add(key)
        item={'hz':hz,'acf':acf,'cmndf':cm,'coverage':int(entry.get('coverage') or 0),
              'cycles':duration*hz/1000,'fundamental_amplitude':amplitude(x,fs,hz)}
        item['eligible']=(acf>=.70 and cm<=.25 and item['coverage']>=3 and item['cycles']>=3)
        info['candidates'].append(item)
    eligible=[c for c in info['candidates'] if c['eligible']]
    if not eligible:info['reason']='no_eligible_measured_period';return info
    # Sort by purely acoustic evidence, never by the baseline tracker or pitch history.
    eligible.sort(key=lambda c:(-c['acf'],c['cmndf'],-c['fundamental_amplitude']))
    rivals=list(eligible)
    # A lower submultiple may be defeated only by a separately OBSERVED shorter
    # period with strong own fundamental and comparable/better ACF/CMNDF.
    defeated=set()
    for low in rivals:
        for high in rivals:
            if high is low or high['hz']<=low['hz'] or not related(high['hz'],low['hz']):continue
            ratio=high['fundamental_amplitude']/max(low['fundamental_amplitude'],1e-12)
            period_ok=(high['acf']>=low['acf']-.025 and high['cmndf']<=low['cmndf']+.025)
            if ratio>=5 and period_ok: defeated.add(low['hz'])
    survivors=[c for c in rivals if c['hz'] not in defeated]
    if len(survivors)==1:
        chosen=survivors[0]; info.update(reason='unique_acoustic_candidate',winner=chosen,frequency=chosen['hz']);return info
    # Comparability refers to actual measured quantities, including oscillator
    # strength. ACF peaks at period multiples are not automatically equal rivals.
    top=survivors[0]
    competitive=[c for c in survivors if
                 abs(c['acf']-top['acf'])<=.035 and
                 abs(c['cmndf']-top['cmndf'])<=.035 and
                 max(c['fundamental_amplitude'],top['fundamental_amplitude']) /
                 max(min(c['fundamental_amplitude'],top['fundamental_amplitude']),1e-12)<=2]
    if len(competitive)>=2:
        info.update(reason='acoustically_tied',near_tie=competitive)
        return info
    # Do not call one candidate decisive merely because it is first lexicographically;
    # require a material measured advantage over *every* surviving alternative.
    beats_all=all(c is top or
        ((top['acf']>=c['acf']+.04 and top['cmndf']<=c['cmndf']+.03) or
         (top['fundamental_amplitude']>=5*max(c['fundamental_amplitude'],1e-12)
          and top['acf']>=c['acf']-.025 and top['cmndf']<=c['cmndf']+.025))
        for c in survivors)
    if beats_all:info.update(reason='acoustic_advantage',winner=top,frequency=top['hz'])
    else:info.update(reason='unresolved_competing_acoustics',near_tie=survivors)
    return info

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['all',*CASES],default='all')
    parser.add_argument('--adaptive-dir',type=Path,default=Path('tests/adaptive_evidence'))
    parser.add_argument('--baseline-dir',type=Path,default=Path('tests/robustness_v3'))
    parser.add_argument('--out-dir',type=Path,default=Path('tests/shadow_acoustic_tiebreak_v2'))
    a=parser.parse_args()
    # Running `python tests/script.py` places tests/, not the repository root, on sys.path.
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "python_eckf").is_dir():
        parser.error(f"Expected python_eckf/ alongside tests/ under {repo_root}")
    sys.path.insert(0, str(repo_root))
    # Use the EXISTING implementation, rather than approximating its curve.
    try:
        from python_eckf.trajectory_resolver import vocal_transition_penalty
    except ImportError as exc:
        parser.error('Existing vocal_transition_penalty unavailable; run from repository root: '+str(exc))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'midi_used':False,'production_modified':False,'decisions_applied':False,
              'tiebreak_is_only_for_acoustically_tied_measured_candidates':True,
              'parameters_are_experimental':True,'cases':{}}
    for name,sites in (CASES.items() if a.case=='all' else [(a.case,CASES[a.case])]):
        wavs=[p for p in (Path('tests')/f'{name}.wav',Path('tests')/f'{name}.wa') if p.is_file()]
        if len(wavs)!=1:parser.error(f'{name}: expected one tests/{name}.wav or .wa')
        wav=wavs[0];audio,fs=sf.read(wav,dtype='float64',always_2d=False)
        if audio.ndim!=1:parser.error(f'{wav}: mono WAV required')
        input_csv=a.adaptive_dir/f'{name}_adaptive.csv'
        if not input_csv.is_file():parser.error(f'missing {input_csv}')
        with input_csv.open(newline='') as f:input_rows=list(csv.DictReader(f))
        baseline=a.baseline_dir/f'{name}_v3_frames.csv'
        with baseline.open(newline='') as f:base=list(csv.DictReader(f)) if baseline.is_file() else []
        bi=0;prior=None;prior_time=None;counts=Counter();rows=[];hot=[];started=time.perf_counter()
        for raw in input_rows:
            t=number(raw.get('time_s'))
            if t is None:continue
            while bi+1<len(base) and number(base[bi+1].get('time_s')) is not None and float(base[bi+1]['time_s'])<=t:bi+=1
            b=base[bi] if base else {}
            baseline_hz=number(b.get('median_f0_hz'))
            voiced=(int(b.get('voiced_samples') or 0)>0) if b else None
            acoustic=select_acoustically(raw,audio,fs)
            acoustic_hz=acoustic['frequency'];off_hz=acoustic_hz;on_hz=off_hz
            penalties=[];prior_valid=False;changed=False
            # A baseline estimate is NEVER a prior for the curve. Only a previously
            # unambiguous waveform-only result can provide context. Even that
            # context cannot invent a candidate or override clear acoustic evidence.
            if acoustic['reason']=='acoustically_tied' and prior is not None and prior_time is not None:
                elapsed_ms=(t-prior_time)*1000
                if 0<elapsed_ms<=30:
                    prior_valid=True
                    for c in acoustic['near_tie']:
                        interval=12*math.log2(c['hz']/prior)
                        p=float(vocal_transition_penalty(interval,elapsed_ms))
                        penalties.append({'hz':c['hz'],'interval_st':interval,'elapsed_ms':elapsed_ms,'penalty':p})
                    penalties.sort(key=lambda q:q['penalty'])
                    # Curve can report a preference; it does NOT turn an acoustic
                    # tie into a confident F0 decision. Remain unresolved.
                    if len(penalties)>=2 and math.isfinite(penalties[0]['penalty']) and penalties[0]['penalty']<penalties[1]['penalty']:
                        on_hz=penalties[0]['hz'];changed=True
            result={'case':name,'time_s':t,'baseline_voiced':voiced,'baseline_hz':baseline_hz,
                    'adaptive_observed_hz':number(raw.get('observed_hz')),
                    'window_ms':number(raw.get('window_ms')),'probe_rms':number(raw.get('probe_rms')),
                    'acoustic_only_hz':off_hz,'acoustic_reason':acoustic['reason'],
                    'curve_preference_hz':on_hz if changed else None,
                    'curve_changed_preference':changed,'curve_prior_hz':prior if prior_valid else None,
                    'curve_prior_age_ms':(t-prior_time)*1000 if prior_valid else None,
                    'curve_penalties':json.dumps(penalties,separators=(',',':')),
                    'measured_candidates':json.dumps(acoustic['candidates'],separators=(',',':')),
                    'state_disagreement_review':(voiced is True and acoustic_hz is not None and baseline_hz is not None
                        and cents(acoustic_hz,baseline_hz)>=100),
                    'baseline_unvoiced_review':voiced is False and acoustic_hz is not None,
                    'decision_applied':False}
            rows.append(result);counts[acoustic['reason']]+=1
            if changed:counts['curve_preference_reported']+=1
            if result['state_disagreement_review']:counts['state_disagreement_review']+=1
            if result['baseline_unvoiced_review']:counts['baseline_unvoiced_review']+=1
            if any(abs(t-s)<=.15 for s in sites):hot.append(result)
            if acoustic_hz is not None:
                prior=acoustic_hz;prior_time=t
            else:
                # Never bridge unvoiced/ambiguous material with a historical pitch.
                prior=None;prior_time=None
        fields=list(rows[0]) if rows else []
        output=a.out_dir/f'{name}_comparison.csv'
        with output.open('w',newline='') as f:
            if fields:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        (a.out_dir/f'{name}_hotspots.json').write_text(json.dumps(hot,indent=2,allow_nan=False)+'\n')
        manifest['cases'][name]={'rows':len(rows),'runtime_s':round(time.perf_counter()-started,3),
             'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
             'adaptive_sha256':hashlib.sha256(input_csv.read_bytes()).hexdigest(),
             'counts':dict(counts)}
        print(f'{name}: {len(rows)} observations; {dict(counts)}',flush=True)
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: no F0 decisions applied; curve preference is NOT a pitch estimate.')
if __name__=='__main__':main()
