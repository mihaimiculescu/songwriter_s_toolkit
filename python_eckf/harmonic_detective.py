from __future__ import annotations

"""Conditional Harmonic Detective for unresolved juror-bench frames.

Uses only already-admitted <49-cent note-group representatives. Existing jury
champions are never overridden. Out-of-range or acoustically unsupported fields
are never rescued. FFT/STFT evidence is computed directly from the current WAV.
"""
from dataclasses import dataclass
from collections import defaultdict
import json, math
import numpy as np

WINDOWS_MS=(64.0,96.0)
HARMONICS=10
MIN_HARMONICS=3
MAX_HZ=4500.0
NFFT_FACTOR=4
TOLERANCE_HZ=5.0
TOLERANCE_CENTS=22.0
BACKGROUND_HZ=90.0
SUPPORT_DB=5.0
MAX_PROMINENCE_DB=24.0
MIN_VALID_WINDOWS=2
MIN_SUPPORT=0.50
MIN_SCORE=0.45
MIN_MARGIN=0.08
# High-confidence expert route: retain the historical ordinary margin rule,
# but allow a smaller separation when the absolute harmonic evidence is strong.
HIGH_CONF_MIN_SCORE=0.80
HIGH_CONF_MIN_SUPPORT=0.90
HIGH_CONF_MIN_MARGIN=0.02
HIGH_CONF_MIN_VALID_WINDOWS=2

@dataclass(frozen=True)
class DetectiveCandidate:
    frame_index:int; time_s:float; group_id:str; midi:int; hz:float
    valid_windows:int; harmonic_score:float|None; harmonic_support:float|None
    harmonic_strength:float|None; window_details_json:str; failure_reasons:str

@dataclass(frozen=True)
class DetectiveVerdict:
    frame_index:int; time_s:float; jury_status:str; detective_called:bool
    eligible_representatives:int; best_group_id:str|None; best_midi:int|None; best_hz:float|None
    best_score:float|None; best_support:float|None; runner_group_id:str|None
    runner_hz:float|None; runner_score:float|None; harmonic_margin:float|None
    detective_winner_group_id:str|None; detective_winner_midi:int|None; detective_winner_hz:float|None
    reason:str

@dataclass(frozen=True)
class FinalAdjudicationVerdict:
    frame_index:int; time_s:float; provenance:str; status:str
    winner_group_id:str|None; winner_midi:int|None; winner_hz:float|None
    original_jury_status:str; original_abstention_category:str

def _spectrum(audio,sr,time_s,window_ms):
    n=max(64,int(round(sr*window_ms/1000.0)))
    center=int(round(time_s*sr)); start=center-n//2
    if start<0 or start+n>len(audio): return None,'wav_boundary'
    seg=np.asarray(audio[start:start+n],dtype=np.float64)
    if not np.all(np.isfinite(seg)): return None,'nonfinite_audio'
    seg=seg-seg.mean(); rms=float(np.sqrt(np.mean(seg*seg)))
    if rms<1e-9:return None,'near_silence'
    seg=seg*np.hanning(n)
    nfft=1<<(int(np.ceil(np.log2(n*NFFT_FACTOR))))
    mag=np.abs(np.fft.rfft(seg,n=nfft)); freqs=np.fft.rfftfreq(nfft,1.0/sr)
    return (freqs,mag,rms,float(sr/n)),''

def _harmonic_score(spec,f0):
    freqs,mag,rms,resolution=spec; nyquist=freqs[-1]
    max_h=min(HARMONICS,int(min(MAX_HZ,nyquist-20.0)/f0))
    if max_h<MIN_HARMONICS:return None
    logmag=np.log(np.maximum(mag,1e-12)); prom=[]; hs=[]; peaks=[]
    for h in range(1,max_h+1):
        target=h*f0
        tol=max(TOLERANCE_HZ,target*TOLERANCE_CENTS*math.log(2)/1200.0,resolution*.75)
        core=(freqs>=target-tol)&(freqs<=target+tol)
        outer=max(BACKGROUND_HZ,3*tol)
        flank=(freqs>=target-outer)&(freqs<=target+outer)&(~((freqs>=target-1.4*tol)&(freqs<=target+1.4*tol)))
        if not np.any(core) or np.count_nonzero(flank)<4:continue
        ix=np.flatnonzero(core); best=ix[int(np.argmax(logmag[ix]))]
        baseline=float(np.median(logmag[flank])); delta=float(logmag[best]-baseline)
        prom.append(max(0.0,min(MAX_PROMINENCE_DB,delta*20/math.log(10))))
        hs.append(h);peaks.append(round(float(freqs[best]),3))
    if len(hs)<MIN_HARMONICS:return None
    a=np.asarray(prom); weights=1/np.sqrt(np.asarray(hs,dtype=float))
    support=float(np.sum(weights*(a>=SUPPORT_DB))/np.sum(weights))
    strength=float(np.sum(weights*a)/np.sum(weights)/MAX_PROMINENCE_DB)
    score=float(.55*support+.45*strength)
    return {'score':score,'support':support,'strength':strength,'harmonics_tested':len(hs),
            'harmonics_supported':int(np.sum(a>=SUPPORT_DB)),'prominence_db':[round(x,2) for x in a],
            'peak_hz':peaks,'rms':rms,'physical_resolution_hz':resolution}

def _acoustic_admissible(r):
    return (r.range_confidence is not None and math.isfinite(float(r.range_confidence)) and float(r.range_confidence)>0
            and r.acf_median is not None and math.isfinite(float(r.acf_median)) and float(r.acf_median)>=0.72)

def build_harmonic_detective(audio,sr,evidence_rows,bench_rows):
    by=defaultdict(list)
    for r in evidence_rows:by[int(r.frame_index)].append(r)
    candidates=[]; verdicts=[]; finals=[]
    for b in bench_rows:
        fi=int(b.frame_index); rows=by.get(fi,[])
        if b.status=='provisional_tournament_champion':
            verdicts.append(DetectiveVerdict(fi,float(b.time_s),b.status,False,0,None,None,None,None,None,None,None,None,None,None,None,None,'not_called_jury_already_has_champion'))
            finals.append(FinalAdjudicationVerdict(fi,float(b.time_s),'jury','resolved',b.winner_group_id,b.winner_midi,b.winner_hz,b.status,b.abstention_category))
            continue
        # Hard-gate abstentions are not detective business.
        if b.abstention_category in {'all_candidates_out_of_range','no_admissible_candidate','single_candidate_no_acoustic_support'}:
            verdicts.append(DetectiveVerdict(fi,float(b.time_s),b.status,False,0,None,None,None,None,None,None,None,None,None,None,None,None,'not_called_hard_gate_abstention'))
            finals.append(FinalAdjudicationVerdict(fi,float(b.time_s),'abstention','unresolved',None,None,None,b.status,b.abstention_category))
            continue
        # Score *all* acoustically admissible contestants that survived the hard
        # range gate.  True <49-cent note groups have already been reduced to one
        # representative upstream; >=49-cent candidates are intentional singleton
        # contestants and must remain visible to the expert witness.
        field=[r for r in rows if _acoustic_admissible(r)]
        scored=[]
        for r in field:
            frames=[];fails=[]
            for w in WINDOWS_MS:
                spec,reason=_spectrum(audio,sr,float(r.time_s),w)
                if spec is None:fails.append(f'{w:g}ms:{reason}');continue
                z=_harmonic_score(spec,float(r.representative_hz))
                if z is None:fails.append(f'{w:g}ms:insufficient_harmonic_band');continue
                frames.append(z)
            dc=DetectiveCandidate(fi,float(r.time_s),r.group_id,int(r.note_group_midi),float(r.representative_hz),len(frames),
                float(np.median([x['score'] for x in frames])) if frames else None,
                float(np.median([x['support'] for x in frames])) if frames else None,
                float(np.median([x['strength'] for x in frames])) if frames else None,
                json.dumps({str(w):f for w,f in zip(WINDOWS_MS,frames)},separators=(',',':')) if frames else '{}','|'.join(fails))
            candidates.append(dc)
            if dc.valid_windows>=MIN_VALID_WINDOWS and dc.harmonic_score is not None and dc.harmonic_support is not None:scored.append(dc)
        scored.sort(key=lambda x:(-x.harmonic_score,x.hz))
        best=scored[0] if scored else None; runner=scored[1] if len(scored)>1 else None
        margin=(best.harmonic_score-runner.harmonic_score) if best and runner else None
        winner=None
        if len(field)<2:
            reason='no_cross_note_contest_to_break'
        elif len(scored)!=len(field):
            reason='incomplete_competitor_field'
        elif best is None:
            reason='no_supported_group_representative'
        elif best.harmonic_support<MIN_SUPPORT or best.harmonic_score<MIN_SCORE:
            reason='harmonic_evidence_too_weak'
        elif runner is not None and best.midi==runner.midi:
            reason='same_note_representatives_unexpected'
        elif margin is not None and margin>=MIN_MARGIN:
            reason='harmonic_detective_tiebreak'; winner=best
        elif (margin is not None
              and best.valid_windows>=HIGH_CONF_MIN_VALID_WINDOWS
              and best.harmonic_score>=HIGH_CONF_MIN_SCORE
              and best.harmonic_support>=HIGH_CONF_MIN_SUPPORT
              and margin>=HIGH_CONF_MIN_MARGIN):
            reason='harmonic_detective_high_confidence_tiebreak'; winner=best
        else:
            reason='harmonic_margin_insufficient'
        verdicts.append(DetectiveVerdict(fi,float(b.time_s),b.status,True,len(field),
            None if best is None else best.group_id,None if best is None else best.midi,None if best is None else best.hz,
            None if best is None else best.harmonic_score,None if best is None else best.harmonic_support,
            None if runner is None else runner.group_id,None if runner is None else runner.hz,None if runner is None else runner.harmonic_score,
            margin,None if winner is None else winner.group_id,None if winner is None else winner.midi,None if winner is None else winner.hz,reason))
        if winner is not None:
            finals.append(FinalAdjudicationVerdict(fi,float(b.time_s),'harmonic_detective','resolved',winner.group_id,winner.midi,winner.hz,b.status,b.abstention_category))
        else:
            finals.append(FinalAdjudicationVerdict(fi,float(b.time_s),'abstention','unresolved',None,None,None,b.status,b.abstention_category))
    return tuple(candidates),tuple(verdicts),tuple(finals)
