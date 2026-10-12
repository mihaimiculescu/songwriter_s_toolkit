#!/usr/bin/env python3
"""Read-only WAV acoustic voicing and ACTIVE-STATE disagreement audit, four piggies.

Inputs: tests/<case>.wav and tests/shadow_acoustic_tiebreak_v2/<case>_comparison.csv.
NO MIDI, modifications, implied note boundaries, persistence-based pitch selection,
or corrections. Baseline 'unvoiced' is NOT ground truth. Neither a high spectral
amplitude nor agreement of overlapping windows establishes a direct vocal source.

Run from repo root: python tests/audit_voicing_and_recovery_shadow.py --case all
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

SITES={'Ochiitai':(42.214,42.353),'PREDESTINATI':(38.034,53.592),
       'RATATA':(36.923,41.026,42.965,43.008,49.231),
       'Trandafiri':(10.774,10.821)}


def num(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError): return None


def rms(x):
    return float(np.sqrt(np.mean(x*x))) if len(x) else None


def projection(x,fs,hz):
    """Measured single-frequency energy; NOT proof of a direct vocal source."""
    if hz is None or hz<=0 or len(x)<8:return None
    x=np.asarray(x,dtype=np.float64); x=x-x.mean()
    n=np.arange(len(x),dtype=np.float64)
    a=2*np.pi*hz*n/fs
    design=np.column_stack((np.cos(a),np.sin(a),np.ones(len(x))))
    coeff=np.linalg.lstsq(design,x,rcond=None)[0]
    fit=design@coeff
    energy=float(np.dot(x,x))
    return {'amplitude':float(np.hypot(coeff[0],coeff[1])),
            'variance_fraction':float(np.dot(fit,fit)/energy) if energy>0 else None}


def period_evidence(x,fs,hz):
    """Check specified measured period on a subset; do NOT discover/fabricate F0."""
    if hz is None or hz<=0 or len(x)<8:return None
    lag=round(fs/hz)
    if lag<1 or lag>=len(x):return None
    z=np.asarray(x,dtype=np.float64);z=z-z.mean()
    a=z[:-lag];b=z[lag:];den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    diff=np.dot(a-b,a-b)
    return {'lag':lag,'cycles':len(z)*hz/fs,
            'acf_at_measured_period':float(np.dot(a,b)/den) if den>0 else None,
            'difference_over_total_energy':float(diff/max(np.dot(z,z),1e-30))}


def evidence(x,fs,hz):
    if hz is None:return None
    parts=np.array_split(x,3)
    whole=period_evidence(x,fs,hz)
    projection_whole=projection(x,fs,hz)
    chunks=[{'rms':rms(part),'period':period_evidence(part,fs,hz),
             'projection':projection(part,fs,hz)} for part in parts]
    return {'whole':whole,'projection':projection_whole,'thirds':chunks,
            'rms':rms(x),'third_rms_ratio':(
                max(rms(p) for p in parts)/max(min(rms(p) for p in parts),1e-12)),
            'thirds_with_period_test':sum(c['period'] is not None for c in chunks)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['all',*SITES],default='all')
    ap.add_argument('--shadow-dir',type=Path,default=Path('tests/shadow_acoustic_tiebreak_v2'))
    ap.add_argument('--out-dir',type=Path,default=Path('tests/voicing_recovery_shadow'))
    ap.add_argument('--hotspot-radius-ms',type=float,default=180)
    args=ap.parse_args()
    if args.hotspot_radius_ms<=0:ap.error('radius must be > 0')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'method':'read_only_observational','midi_used':False,'pitch_modified':False,
              'recovery_applied':False,'cases':{},'note':'Overlapping windows are dependent; labels mean REVIEW, not truth.'}
    for name,sites in SITES.items():
        if args.case!='all' and name!=args.case:continue
        wav=Path('tests')/(name+'.wav')
        if not wav.exists():wav=Path('tests')/(name+'.wa')
        source=args.shadow_dir/(name+'_comparison.csv')
        if not wav.is_file() or not source.is_file():ap.error(f'missing required input {wav} or {source}')
        begun=time.perf_counter()
        audio,fs=sf.read(wav,dtype='float64',always_2d=False)
        if audio.ndim!=1:ap.error(f'{wav}: mono required; never silently downmix')
        with source.open(newline='') as fh:rows=list(csv.DictReader(fh))
        out=[];detail=[];counts=Counter()
        for i,row in enumerate(rows):
            t=num(row.get('time_s'));duration=num(row.get('window_ms'))
            if t is None or duration is None:continue
            baseline=num(row.get('baseline_hz'))
            candidate=num(row.get('acoustic_only_hz'))
            voiced=str(row.get('baseline_voiced')).lower()=='true'
            unvoiced=str(row.get('baseline_voiced')).lower()=='false'
            # Evaluate only relevant locations, with identical support for state & candidate.
            is_hot=any(abs(t-s)<=args.hotspot_radius_ms/1000 for s in sites)
            is_unvoiced=unvoiced and candidate is not None
            is_disagreement=(voiced and candidate is not None and baseline is not None
                 and abs(1200*math.log2(candidate/baseline))>=100)
            if not (is_hot or is_unvoiced or is_disagreement):continue
            count=max(64,round(duration*fs/1000));middle=round(t*fs)
            lo=middle-count//2;hi=lo+count
            if lo<0 or hi>len(audio):continue
            x=audio[lo:hi]
            c_evidence=evidence(x,fs,candidate)
            state_evidence=evidence(x,fs,baseline) if is_disagreement else None
            # Local RMS context only.  Median is a reference, NOT a classification threshold.
            ctx_lo=max(0,round((t-.25)*fs));ctx_hi=min(len(audio),round((t+.25)*fs))
            context=audio[ctx_lo:ctx_hi]
            chunks=np.array_split(context,max(1,int(len(context)/(fs*.025))))
            local_rms=float(np.median([rms(z) for z in chunks if len(z)])) if chunks else None
            ratio=c_evidence['rms']/max(local_rms,1e-12) if c_evidence and local_rms is not None else None
            # Distinguish finding a state mismatch from deciding to reset and selecting F0.
            # These flags are review queues, not actions or reclassification.
            state_unsupported=None
            if is_disagreement and state_evidence:
                cproj=c_evidence['projection'] if c_evidence else None
                sproj=state_evidence['projection']
                cp=c_evidence['whole'] if c_evidence else None
                sp=state_evidence['whole']
                state_unsupported=(cproj is not None and sproj is not None and cp is not None and sp is not None
                    and cproj['variance_fraction'] is not None and sproj['variance_fraction'] is not None
                    and cp['acf_at_measured_period'] is not None and sp['acf_at_measured_period'] is not None
                    and cproj['variance_fraction']>sproj['variance_fraction']
                    and cp['acf_at_measured_period']>sp['acf_at_measured_period'])
            category=('baseline_unvoiced_review' if is_unvoiced else
                      'state_disagreement_review' if is_disagreement else 'hotspot_context')
            counts[category]+=1
            if state_unsupported:counts['state_weaker_on_two_measures_review']+=1
            record={'case':name,'time_s':t,'window_ms':duration,'window_start_s':lo/fs,
                'window_end_s':hi/fs,'category':category,'baseline_voiced':voiced,
                'baseline_f0_hz':baseline,'measured_candidate_hz':candidate,
                'acoustic_reason':row.get('acoustic_reason'),
                'context_median_25ms_rms':local_rms,'window_to_context_rms_ratio':ratio,
                'candidate_evidence':c_evidence,'state_evidence':state_evidence,
                'state_weaker_on_two_measures_review':state_unsupported,
                'trigger_only':bool(is_disagreement), 'reset_requested':False,
                'replacement_selected':False,'decision_applied':False,
                'source_row_index':i}
            detail.append(record)
            out.append({k:v if not isinstance(v,(dict,list)) else json.dumps(v,separators=(',',':'))
                        for k,v in record.items()})
        if out:
            with (args.out_dir/f'{name}_review.csv').open('w',newline='') as fh:
                writer=csv.DictWriter(fh,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
        (args.out_dir/f'{name}_hotspots.json').write_text(json.dumps(
            [v for v in detail if any(abs(v['time_s']-s)<=args.hotspot_radius_ms/1000 for s in sites)],
            indent=2,allow_nan=False)+'\n')
        manifest['cases'][name]={'source_rows':len(rows),'review_rows':len(out),'counts':dict(counts),
            'sample_rate':fs,'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
            'shadow_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'runtime_s':round(time.perf_counter()-begun,3)}
        print(f'{name}: {len(out)} reviews; {dict(counts)}; {time.perf_counter()-begun:.2f}s',flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: trigger review is NOT a reset; no new F0 chosen; no production modifications.')
if __name__=='__main__':main()
