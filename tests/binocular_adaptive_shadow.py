#!/usr/bin/env python3
"""Read-only, waveform-only binocular/aperture experiment for four piggies.

Look-ahead suggests an aperture; a fresh measurement and a separate present-time
probe assess the CURRENT instant. None of these measurements changes ECKF state,
produces final voiced status, reads MIDI, or supplies a pitch from continuity.

Run from repository root: python tests/binocular_adaptive_shadow.py --case all
Requires existing tests/diagnose_adaptive_evidence.py (for identical acoustic
measurement primitives), numpy, scipy, soundfile. No other reports required.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_adaptive_evidence import analyze, credible, select, NAMES

REGIONS = {
 'RATATA': [('djuvv_1',32.307,32.380),('djuvv_2',36.410,36.923),
            ('djuvv_3',44.615,45.128),('submultiple',42.915,43.065)],
 'Ochiitai':[('transition',42.174,42.413)],
 'Trandafiri':[('tsis',10.724,10.921)],
 'PREDESTINATI':[('control_38',37.984,38.144),('control_53',53.542,53.752)],
}
WINDOWS=(24.,40.,64.,96.)

def piece(audio,fs,t,ms):
    n=max(8,round(ms*fs/1000)); center=round(t*fs); start=center-n//2; end=start+n
    if start<0 or end>len(audio): return None,None,None
    return audio[start:end], start/fs, end/fs

def view(audio,fs,t,ms,cycles):
    x,start,end=piece(audio,fs,t,ms)
    if x is None:return dict(start_s=None,end_s=None,rms=None,candidates=[],hz=None,reason='edge')
    rms,candidates=analyze(x,fs)
    winner,reason=select(candidates,fs,len(x),cycles)
    return dict(start_s=start,end_s=end,rms=rms,candidates=candidates,
                hz=winner['hz'] if winner else None,reason=reason)

def evidence(audio,fs,t,ms,hz):
    """Test EXISTING measured Hz; don't infer/multiply a new pitch."""
    x,start,end=piece(audio,fs,t,ms)
    d=dict(start_s=start,end_s=end,ms=ms,hz_tested=hz,cycles=None,
           rms=None,acf=None,fundamental_fraction=None,period_testable=False)
    if x is None or hz is None or hz<=0 or hz>=fs/2:return d
    z=np.asarray(x,dtype=np.float64);z-=z.mean()
    n=len(z); energy=float(z@z); lag=round(fs/hz)
    d['cycles']=n*hz/fs;d['rms']=math.sqrt(energy/n)
    if energy<1e-20:return d
    if 1<=lag<n-2:
        a,b=z[:-lag],z[lag:];den=math.sqrt(float(a@a)*float(b@b))
        if den>1e-20:d['acf']=float(a@b)/den;d['period_testable']=True
    phases=2*np.pi*hz*np.arange(n)/fs
    basis=np.column_stack((np.cos(phases),np.sin(phases)))
    coef,*_=np.linalg.lstsq(basis,z,rcond=None)
    fit=basis@coef
    d['fundamental_fraction']=float((fit@fit)/energy)
    return d

def supported(e,min_cycles=3.):
    """EXPLORATORY diagnostic, not a voiced classifier."""
    return bool(e['cycles'] is not None and e['cycles']>=min_cycles and
                e['acf'] is not None and e['acf']>=.70 and
                e['fundamental_fraction'] is not None and e['fundamental_fraction']>=.03)

def evaluate(audio,fs,t,cycles,look_ms,look_offset_ms,local_ms):
    # Binoculars observe the FUTURE, but are not the final pitch estimator.
    future_t=t+look_offset_ms/1000
    future=view(audio,fs,future_t,look_ms,3.)
    future_period=future['hz']
    # Absent credible forward periodicity: investigate with 24 ms, do not treat
    # high-frequency noise as a high note or blindly jump to 96 ms.
    if future_period is None:
        target=WINDOWS[0]; aperture_reason='lookahead_no_credible_period_short_probe'
    else:
        required=1000*cycles/future_period
        target=next((ms for ms in WINDOWS if ms>=required),WINDOWS[-1])
        aperture_reason='measured_lookahead_period_sets_cycle_aperture'
    # Centered measurement is independent: not a copy of the future frequency.
    measure=view(audio,fs,t,target,cycles)
    measured=measure['hz']; selected_ms=target
    # Escalation is merely investigative; no 'pitch' created when future is noise.
    if measured is None:
        for ms in WINDOWS:
            if ms<=target:continue
            trial=view(audio,fs,t,ms,cycles)
            selected_ms=ms;measure=trial;measured=trial['hz']
            if measured is not None:break
        if selected_ms>target:aperture_reason+='|expanded_for_measurement_only'
    # At this point check the INSTANT with independently bounded support.
    # A very short probe with <3 cycles cannot disprove periodicity.
    central=evidence(audio,fs,t,local_ms,measured)
    left=evidence(audio,fs,t-local_ms/1000,local_ms,measured)
    right=evidence(audio,fs,t+local_ms/1000,local_ms,measured)
    whole=evidence(audio,fs,t,selected_ms,measured)
    local_ok=supported(central); outer_ok=supported(left) or supported(right)
    if measured is None: verdict='no_measured_candidate'
    elif central['cycles'] is None or central['cycles']<3 or not central['period_testable']:
        verdict='present_probe_insufficient_cycles'
    elif local_ok:verdict='candidate_locally_supported_NOT_voicing_proof'
    elif outer_ok and supported(whole):verdict='possible_temporal_leakage'
    elif supported(whole):verdict='whole_only_local_unconfirmed'
    else:verdict='measured_candidate_local_evidence_weak'
    # A disagreement between lookahead and measurement is descriptive only.
    divergence=None
    if future_period and measured:divergence=1200*math.log2(measured/future_period)
    return dict(time_s=t,lookahead_center_s=future_t,
       lookahead_start_s=future['start_s'],lookahead_end_s=future['end_s'],
       lookahead_rms=future['rms'],lookahead_hz=future_period,
       lookahead_reason=future['reason'],lookahead_candidates_json=json.dumps(future['candidates'][:6]),
       first_aperture_ms=target,actual_aperture_ms=selected_ms,
       aperture_selection_reason=aperture_reason,
       actual_start_s=measure['start_s'],actual_end_s=measure['end_s'],
       independent_measured_hz=measured,measurement_reason=measure['reason'],
       measurement_candidates_json=json.dumps(measure['candidates'][:6]),
       lookahead_measurement_difference_cents=divergence,
       local_evidence_status=verdict,center_json=json.dumps(central),
       left_json=json.dumps(left),right_json=json.dumps(right),whole_json=json.dumps(whole),
       f0_applied=False,voicing_applied=False,tracker_reset_requested=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*NAMES),default='all')
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/binocular_adaptive_shadow'))
    p.add_argument('--hop-ms',type=float,default=10.)
    p.add_argument('--lookahead-ms',type=float,default=24.)
    p.add_argument('--lookahead-offset-ms',type=float,default=12.)
    p.add_argument('--local-ms',type=float,default=24.)
    p.add_argument('--cycles',type=float,default=6.)
    p.add_argument('--regions-only',action='store_true',help='Quick audit; NOT full-song regression')
    a=p.parse_args()
    if not 2<=a.hop_ms<=25 or not 12<=a.lookahead_ms<=40 or not 0<=a.lookahead_offset_ms<=48 or not 12<=a.local_ms<=40 or not 3<=a.cycles<=12:p.error('Invalid probe/hop/cycle settings')
    names=NAMES if a.case=='all' else (a.case,)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=dict(version=1,read_only=True,midi_used=False,production_modified=False,
       lookahead_is_not_pitch_estimate=True,local_labels_are_not_voicing_truth=True,
       parameters=dict(hop_ms=a.hop_ms,lookahead_ms=a.lookahead_ms,
          lookahead_offset_ms=a.lookahead_offset_ms,local_ms=a.local_ms,
          cycles=a.cycles,regions_only=a.regions_only),cases={})
    for name in names:
        wav=a.wav_dir/f'{name}.wav'
        if not wav.is_file():p.error(f'Missing mono WAV: {wav}')
        audio,fs=sf.read(wav,dtype='float64',always_2d=False)
        if audio.ndim!=1:p.error(f'{wav}: mono required, refusing downmix')
        hop=max(1,round(fs*a.hop_ms/1000));margin=round(fs*.12)
        centers=range(margin,len(audio)-margin,hop)
        rows=[];counts=Counter();beg=time.perf_counter()
        for center in centers:
            t=center/fs
            if a.regions_only and not any(lo-.05<=t<=hi+.05 for _,lo,hi in REGIONS[name]):continue
            r=evaluate(audio,fs,t,a.cycles,a.lookahead_ms,a.lookahead_offset_ms,a.local_ms)
            labels=[label for label,lo,hi in REGIONS[name] if lo<=t<=hi]
            r['annotated_regions']=';'.join(labels)
            counts[r['local_evidence_status']]+=1;rows.append(r)
        if not rows:raise RuntimeError(f'{name}: no observations')
        path=a.out_dir/f'{name}_binocular.csv'
        with path.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        manifest['cases'][name]=dict(wav=str(wav),wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
             sample_rate=fs,observations=len(rows),counts=dict(counts),elapsed_s=round(time.perf_counter()-beg,3),
             output=str(path),regions=REGIONS[name])
        print(name,len(rows),dict(counts),'elapsed_s',manifest['cases'][name]['elapsed_s'])
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out_dir/'README.txt').write_text(
       'Binocular lookahead selects aperture only; measurement independently reanalyzes waveform.\n'
       'No musical pitch inference from spectral brightness. No lookahead Hz used as final Hz.\n'
       'Central 24ms probe may be insufficient below 125Hz: explicitly marked.\n'
       'center/left/right probes do not overlap each other, but all center on nominal timestamps.\n'
       'Long window selection may itself straddle transitions; temporal attribution is exploratory.\n'
       'No actual production sample-status in this experiment. Labels are NOT voice/source truth.\n'
       'No MIDI, production changes, oscillator resets, or final F0 output.\n')
    print('Output:',a.out_dir)
if __name__=='__main__':main()
