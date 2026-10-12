#!/usr/bin/env python3
"""V13: full-timeline observational adjudication, four songs, no production changes.

Requires V3 FULL-SONG CSVs, V11 review-only CSVs, original WAVs, and the
UNCHANGED python_eckf.trajectory_resolver.vocal_transition_penalty.

'Acoustically supported' is NOT confirmed active singing, source identity, or
final F0. 'No local period' is NOT proven noise or silence. Reports these
separately; never advertises an inferred noise label as settled.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CASES=('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
REGIONS={'RATATA':[('djuvv_1',36.923,37.435),('djuvv_2',41.025,41.538),('djuvv_3',49.230,49.743),('submultiple',42.90,43.10)],'Ochiitai':[('transition',42.19,42.40)],'Trandafiri':[('steady_i_NOT_glissando',10.70,10.95)],'PREDESTINATI':[('control38',37.97,38.13),('control53',53.52,53.73)]}

def num(x):
    try:
        v=float(x);return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None

def load(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def js(x,default):
    try:return json.loads(x)
    except (ValueError,TypeError):return default

def cents(a,b):return abs(1200*math.log2(a/b))

def rms_window(integral,center,n):
    lo=center-n//2;hi=lo+n
    if lo<0 or hi>=len(integral):return None
    return float(math.sqrt(max(0.,(integral[hi]-integral[lo])/n)))

def energy_at(audio,fs,integral,sample,context_s=.30):
    probe=max(32,round(.012*fs));hop=max(1,round(.010*fs))
    now=rms_window(integral,sample,probe)
    left=rms_window(integral,sample-probe,probe)
    right=rms_window(integral,sample+probe,probe)
    centers=np.arange(max(probe,len(audio)*0+sample-round(context_s*fs)),min(len(audio)-probe,sample+round(context_s*fs))+1,hop,dtype=np.int64)
    # Explicit time-local reference; no fixed vocal-volume floor.
    vals=np.sqrt(np.maximum((integral[centers+probe//2]-integral[centers-probe//2])/probe,0)) if len(centers) else np.array([])
    ref=float(np.quantile(vals,.80)) if len(vals) else None
    ratio=now/ref if now is not None and ref and ref>1e-12 else None
    slope=20*math.log10((right+1e-12)/(left+1e-12)) if left is not None and right is not None else None
    return dict(rms_present=now,rms_left=left,rms_right=right,relative_energy=ratio,slope_db=slope,context_p80=ref)

def raw_candidates(row):
    stages=js(row.get('stages_json'),[])
    if not stages:return [],[],None
    stage=stages[-1]
    all_raw=stage.get('raw_candidates',[])
    local=[]
    for item in all_raw:
        c=item.get('candidate') or {}; hz=num(c.get('hz'))
        if not hz or hz<=0:continue
        p=item.get('present') or {}
        if item.get('eligible') and item.get('attribution')=='local_support_exploratory' and p.get('testability')=='testable':
            local.append(dict(hz=hz,acf=num(c.get('acf')),cmndf=num(c.get('cmndf')),
                              fundamental_fraction=num(p.get('fundamental_fraction')),present_acf=num(p.get('acf')),
                              cycles=num(p.get('cycles'))))
    return all_raw,local,stage

def competing(local,chosen):
    if chosen is None:return True
    others=[c for c in local if cents(c['hz'],chosen['hz'])>70]
    # Candidate separation must come from the SAME measured frame; where
    # rivals have comparable periodic evidence, abstain, not a frequency prior.
    for other in others:
        if (other['present_acf'] is not None and chosen['present_acf'] is not None and
            other['present_acf']>=chosen['present_acf']-.07 and
            (other['fundamental_fraction'] or 0)>=(chosen['fundamental_fraction'] or 0)*.5):
            return True
    return False

def penalty_call(func,prior,hz,elapsed):
    if func is None:return None,'curve_unavailable'
    if prior is None:return None,'no_immediately_preceding_acoustic_reference'
    try:
        p=float(func(12*math.log2(hz/prior),elapsed))
        return (p,'evaluated_unchanged') if math.isfinite(p) else (None,'nonfinite')
    except Exception as exc:return None,'curve_error:'+repr(exc)

def run_case(name,v3rows,v11rows,audio,fs,curve,hop_ms):
    review={int(r['sample']):r for r in v11rows}
    assert len(review)==len(v11rows),'duplicate V11 review samples'
    v3rows=sorted(v3rows,key=lambda r:int(r['sample']))
    assert len({int(r['sample']) for r in v3rows})==len(v3rows),'duplicate V3 samples'
    sq=np.r_[0.,np.cumsum(np.square(audio,dtype=np.float64))]
    out=[];counts=Counter(); previous=None; regions=Counter()
    for row in v3rows:
        sample=int(row['sample']);t=float(row['time_s'])
        raw,local,stage=raw_candidates(row)
        chosen_hz=num(row.get('observational_hz'))
        selected=min((c for c in local if chosen_hz and cents(c['hz'],chosen_hz)<=35),
                     key=lambda c:cents(c['hz'],chosen_hz),default=None)
        review_row=review.get(sample)
        energy=energy_at(audio,fs,sq,sample)
        # Review-only V11 features override no full-song judgments; log both.
        ratio=energy['relative_energy'];slope=energy['slope_db']
        low=ratio is not None and ratio<.25
        decay=slope is not None and slope< -4.
        if not raw:acoustic='no_raw_period_observed'
        elif selected is None:acoustic='no_unique_locally_supported_period'
        elif competing(local,selected):acoustic='competing_locally_supported_periods'
        else:acoustic='unique_locally_supported_period'
        adjacent=(previous is not None and sample>previous['sample'] and
                  abs((sample-previous['sample'])/fs*1000-hop_ms)<=1.5)
        prior=previous['hz'] if adjacent and previous['reference_ok'] else None
        dt=(sample-previous['sample'])/fs*1000 if prior is not None else None
        tests=[]
        for c in local:
            p,status=penalty_call(curve,prior,c['hz'],dt)
            tests.append(dict(measured_hz=c['hz'],penalty=p,status=status))
        chosen_penalty=next((x['penalty'] for x in tests if selected and cents(x['measured_hz'],selected['hz'])<=35),None)
        joint_challenge=(acoustic=='unique_locally_supported_period' and chosen_penalty is not None and
                         chosen_penalty>0 and low and decay)
        # A reportable locally supported period is not a certified active source.
        # A jointly challenged interval cannot be a subsequent curve reference.
        if joint_challenge:category='interval_energy_joint_challenge_unresolved'
        elif acoustic=='unique_locally_supported_period':category='acoustic_period_supported_source_unverified'
        elif acoustic=='no_raw_period_observed':category='no_measured_period_source_unverified'
        else:category='acoustic_ambiguity_unresolved'
        reference_ok=(category=='acoustic_period_supported_source_unverified')
        # Never carry a frequency across an unresolved observation.
        previous={'sample':sample,'hz':selected['hz'] if reference_ok else None,
                  'reference_ok':reference_ok}
        tags=[tag for tag,start,end in REGIONS.get(name,[]) if start<=t<=end]
        for tag in tags:regions[(tag,category)]+=1
        result=dict(case=name,sample=sample,time_s=t,
            evidence_status=category,acoustic_measurement=acoustic,
            diagnostic_settled=category in ('acoustic_period_supported_source_unverified','no_measured_period_source_unverified'),
            final_f0_settled=False,active_source_confirmed=False,
            selected_measured_hz=selected['hz'] if selected else None,
            candidate_count=len(local),raw_candidate_count=len(raw),
            selected_acf=selected['acf'] if selected else None,
            selected_present_acf=selected['present_acf'] if selected else None,
            selected_fundamental_fraction=selected['fundamental_fraction'] if selected else None,
            relative_energy=ratio,energy_slope_db=slope,low_energy_flag=low,decaying_flag=decay,
            prior_hz=prior,penalty_elapsed_ms=dt,penalty=chosen_penalty,
            penalty_candidates_json=json.dumps(tests,separators=(',',':')),
            interval_energy_joint_challenge=joint_challenge,
            usable_as_next_reference=reference_ok,
            review_population=sample in review,
            review_comparison=review_row.get('review_comparison') if review_row else '',
            original_v3_status=row.get('status'),original_v3_stop=row.get('stop_reason'),
            aperture_ms=stage.get('aperture_ms') if stage else None,
            annotation_tags=';'.join(tags),output_f0_modified=False,production_modified=False)
        out.append(result);counts[category]+=1
    assert len(out)==len(v3rows)
    assert sum(counts.values())==len(out)
    return out,counts,regions,len(review)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('all',*CASES),default='all')
    ap.add_argument('--v3-dir',type=Path,default=Path('tests/binocular_evidence_search_v3'))
    ap.add_argument('--v11-dir',type=Path,default=Path('tests/energy_observability_v11'))
    ap.add_argument('--wav-dir',type=Path,default=Path('tests'))
    ap.add_argument('--out-dir',type=Path,default=Path('tests/full_timeline_v13'))
    ap.add_argument('--hop-ms',type=float,default=10.)
    a=ap.parse_args()
    try:
        from python_eckf.trajectory_resolver import vocal_transition_penalty
        curve=vocal_transition_penalty;status='imported_unchanged'
    except (ImportError,AttributeError) as exc:
        curve=None;status='unavailable:'+str(exc)
        print('WARNING: original penalty unavailable, no substitute used',file=sys.stderr)
    v3manifest=json.loads((a.v3_dir/'manifest.json').read_text())
    v11manifest=json.loads((a.v11_dir/'manifest.json').read_text())
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema='v13_full_timeline_shadow',curve_status=status,definitions={
        'settled_diagnostic':'unique locally supported measured period OR no raw period observed; neither proves source identity',
        'in_the_dark':'locally competing/unsupported periods OR interval-energy challenge',
        'settled_final_f0':'NOT AVAILABLE: active singing vs tail not established by a mono WAV using these measures'},cases={})
    totals=Counter()
    for name in CASES if a.case=='all' else (a.case,):
        v3file=a.v3_dir/f'{name}.csv';v11file=a.v11_dir/f'{name}_energy_observability.csv'
        wavfile=a.wav_dir/f'{name}.wav'
        for p in (v3file,v11file,wavfile):
            if not p.exists():ap.error(f'missing required file: {p}')
        audio,fs=sf.read(wavfile,dtype='float64',always_2d=False)
        if audio.ndim==2:audio=audio.mean(axis=1)
        expected=num(v3manifest['cases'][name]['sample_rate'])
        if not expected or round(expected)!=fs:ap.error(f'sample rate mismatch {name}: {fs} != {expected}')
        rows,counts,regions,nreview=run_case(name,load(v3file),load(v11file),audio,fs,curve,a.hop_ms)
        if nreview!=sum(x['review_population'] for x in rows):
            ap.error(f'V11 review rows not wholly represented in full V3 timeline for {name}')
        outfile=a.out_dir/f'{name}_full_timeline.csv'
        with outfile.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        settled=sum(x['diagnostic_settled'] for x in rows);dark=len(rows)-settled
        counts_case=dict(total=len(rows),diagnostically_settled=settled,in_the_dark=dark,
            final_f0_settled=0,review_observations=nreview,penalty_evaluated=sum(x['penalty'] is not None for x in rows),
            joint_challenges=counts['interval_energy_joint_challenge_unresolved'],categories=dict(counts),
            annotated_regions={tag:dict((k,v) for (t,k),v in regions.items() if t==tag) for tag,_,_ in REGIONS.get(name,[])},
            output=str(outfile))
        manifest['cases'][name]=counts_case;totals.update(total=len(rows),settled=settled,dark=dark)
        print(f'{name}: TOTAL {len(rows)} | DIAGNOSTICALLY SETTLED {settled} | IN THE DARK {dark} | FINAL F0 SETTLED not established',flush=True)
    manifest['totals']=dict(totals)
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY. Diagnostic classification != verified pitch or vocal-source identity.')
if __name__=='__main__':main()
