#!/usr/bin/env python3
"""V9: read-only, full-song measured-period FAMILY audit.

Inputs: V7 all-four-apertures temporal-view CSVs and original mono WAVs.
Run from the repository root:
  python tests/audit_measured_period_families_v9.py --case all

NO MIDI, BPM/grid changes, prior-pitch preference, pitch/voicing decisions,
continuity enforcement, tracker resets, or production edits.

A candidate is ONLY an actual independently measured V7 period peak. Integer
relations are used to GROUP measured peaks, NEVER to generate new pitches.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
VIEWS = ('past', 'center', 'future')
DURATIONS = (24., 40., 64., 96.)
FIELDS = ('case','sample','time_s','region','comparison','primary_mechanism',
          'view_ms','center_start_sample','center_end_sample','center_rms',
          'center_candidate_count','center_eligible_count','family_count',
          'eligible_family_count','integer_pair_count','family_geometry',
          'acoustic_comparison','temporal_attribution','investigation_route',
          'best_measured_center_hz_diagnostic_only','best_measured_center_acf',
          'best_measured_center_cmndf','best_center_fundamental_fraction',
          'candidate_families_json','candidates_json','peak_tracking_json',
          'no_pitch_applied','no_voicing_applied','no_tracker_reset')

def num(x):
    try:
        f=float(x)
        return f if math.isfinite(f) else None
    except (TypeError,ValueError):
        return None

def cent(a,b):
    return abs(1200*math.log2(a/b))

def integer_relation(a,b,tolerance=70.,max_integer=5):
    if not a or not b or a<=0 or b<=0:return None
    ratio=max(a,b)/min(a,b)
    options=[(cent(ratio,float(n)),n) for n in range(2,max_integer+1)]
    err,n=min(options)
    return {'integer':n,'cents_error':round(err,3)} if err<=tolerance else None

def parse(row):
    views={v:json.loads(row[v+'_json']) for v in VIEWS}
    rec=[]
    for v in VIEWS:
        eligible={num(c.get('lag')) for c in (views[v].get('eligible_candidates') or [])}
        for k,c in enumerate(views[v].get('raw_candidates') or []):
            hz=num(c.get('hz'))
            if hz is None or hz<=0:continue
            rec.append({'id':len(rec),'view':v,'measured_hz':hz,
                        'lag':num(c.get('lag')),'acf':num(c.get('acf')),
                        'cmndf':num(c.get('cmndf')),'coverage':c.get('coverage'),
                        'eligible':num(c.get('lag')) in eligible,
                        'fundamental_fraction':None,'harmonic_fraction':None,
                        'harmonic_count':None,'cycles':None})
    return views,rec

def same_frequency(a,b,tol):
    return cent(a,b)<=tol

def components(items, connected):
    parent=list(range(len(items)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if connected(items[i],items[j]):parent[find(j)]=find(i)
    out=defaultdict(list)
    for i,it in enumerate(items):out[find(i)].append(it)
    return list(out.values())

def bands(y,fs,hz,max_harmonics=8):
    """Exact-frequency sinusoidal projection, diagnostics only.

    Fundamental energy uses a two-coefficient least-squares fit to the centered
    waveform. Multi-harmonic energy uses a joint least-squares fit, avoiding
    double-counting nonorthogonal harmonics in short windows. Harmonic count is
    a descriptive >=10%-of-strongest projection count, not voicing eligibility.
    """
    n=len(y)
    if n<16 or hz<=0 or hz>=fs/2:return None,None,0
    y=np.asarray(y,dtype=np.float64);y=y-y.mean()
    power=float(y@y)
    if power<=1e-18:return None,None,0
    harmonics=min(max_harmonics,int((fs/2*0.95)//hz))
    if harmonics<1:return None,None,0
    t=np.arange(n,dtype=np.float64)/fs
    frequencies=hz*np.arange(1,harmonics+1)
    phase=2*np.pi*np.outer(t,frequencies)
    mat=np.empty((n,2*harmonics))
    mat[:,0::2]=np.cos(phase);mat[:,1::2]=np.sin(phase)
    # The fundamental occupies first two columns. Both fits use SAME waveform.
    f0_fit=mat[:,:2] @ np.linalg.lstsq(mat[:,:2],y,rcond=None)[0]
    all_fit=mat @ np.linalg.lstsq(mat,y,rcond=None)[0]
    # Magnitudes here are used only to describe detectable spectral structure.
    coeff=mat.T@y
    magnitudes=np.hypot(coeff[0::2],coeff[1::2])
    strong=float(max(magnitudes))
    count=int(np.sum(magnitudes>=strong*.1)) if strong>0 else 0
    return (round(float(np.clip(f0_fit@f0_fit/power,0,1)),7),
            round(float(np.clip(all_fit@all_fit/power,0,1)),7),count)

def enrich_center(audio,fs,view,records):
    lo=int(view['start_sample']);hi=int(view['end_sample'])
    if lo<0 or hi>len(audio) or lo>=hi:return
    y=audio[lo:hi]
    for r in records:
        if r['view']!='center':continue
        r['cycles']=round((hi-lo)*r['measured_hz']/fs,3)
        f,h,k=bands(y,fs,r['measured_hz'])
        r['fundamental_fraction']=f;r['harmonic_fraction']=h;r['harmonic_count']=k

def family_report(records,tolerance):
    # First cluster SAME measured frequency across independent temporal views.
    clusters=components(records,lambda a,b:same_frequency(a['measured_hz'],b['measured_hz'],tolerance))
    # Then connect clusters only if EXISTING measured members are integer-related.
    families=components(clusters,lambda a,b:any(integer_relation(x['measured_hz'],y['measured_hz'],tolerance)
                                for x in a for y in b))
    reports=[]; integer_pairs=[]
    for i,group in enumerate(families):
        peaks=[r for component in group for r in component]
        center=[r for r in peaks if r['view']=='center']
        eligible=[r for r in center if r['eligible']]
        representatives=[]
        for component in group:
            cc=[r for r in component if r['view']=='center']
            eligible_cc=[r for r in cc if r['eligible']]
            chosen=min(eligible_cc or cc or component,
                       key=lambda r:(r['cmndf'] if r['cmndf'] is not None else 2.,
                                     -(r['acf'] if r['acf'] is not None else -1.)))
            representatives.append(chosen)
        relations=[]
        for a in range(len(representatives)):
            for b in range(a+1,len(representatives)):
                relation=integer_relation(representatives[a]['measured_hz'],representatives[b]['measured_hz'],tolerance)
                if relation:
                    pair={'a_hz':representatives[a]['measured_hz'],
                          'b_hz':representatives[b]['measured_hz'],**relation}
                    relations.append(pair);integer_pairs.append(pair)
        family={'family_id':i,'measured_members_hz':[r['measured_hz'] for r in representatives],
                'center_eligible_members_hz':[r['measured_hz'] for r in eligible],
                'present_eligible':bool(eligible),
                'past_eligible':any(r['eligible'] and r['view']=='past' for r in peaks),
                'future_eligible':any(r['eligible'] and r['view']=='future' for r in peaks),
                'integer_relations':relations,
                'member_evidence':[{'measured_hz':r['measured_hz'],'view':r['view'],
                         'eligible':r['eligible'],'acf':r['acf'],'cmndf':r['cmndf'],
                         'fundamental_fraction':r['fundamental_fraction'],
                         'harmonic_fraction':r['harmonic_fraction'],
                         'harmonic_count':r['harmonic_count'],'cycles':r['cycles']} for r in peaks]}
        reports.append(family)
    return reports,integer_pairs

def assess(views,rec,families,tol):
    eligible=[r for r in rec if r['view']=='center' and r['eligible']]
    center=[r for r in rec if r['view']=='center']
    live=[f for f in families if f['present_eligible']]
    flank=[f for f in families if not f['present_eligible'] and (f['past_eligible'] or f['future_eligible'])]
    if not center:
        geometry='no_measured_center_period';comparison='no_center_candidate'
        route='inspect_observability_and_boundary_without_assigning_pitch'
    elif not eligible:
        geometry='raw_center_periods_ineligible';comparison='no_eligible_center_candidate'
        route='inspect_raw_peak_rejection_and_cycle_sufficiency'
    elif len(live)>1:
        geometry='distinct_present_families';comparison='multiple_distinct_measured_families'
        route='inspect_noninteger_competing_peaks_and_local_acoustics'
    elif len(eligible)>1:
        geometry='multiple_measured_periods_one_family';comparison='member_evidence_needed'
        route='compare_actual_periods_acf_cmndf_fundamental_and_harmonic_support'
    else:
        geometry='one_eligible_center_period';comparison='single_eligible_measured_period'
        route='audit_temporal_observability_separately'
    temporal=('flank_only_eligible_family' if not live and flank else
              'center_and_flank_families' if live and flank else
              'center_present_in_all_views' if live and all(f['past_eligible'] and f['future_eligible'] for f in live) else
              'center_with_view_asymmetry' if live else 'no_eligible_family_any_view')
    # ACF, CMNDF, fraction are separate axes. No winner if axes disagree.
    # Candidate ranking is descriptive; in particular NO preference for higher Hz.
    best=None
    if eligible:
        dominant=[]
        for candidate in eligible:
            other=[r for r in eligible if r is not candidate and not same_frequency(r['measured_hz'],candidate['measured_hz'],tol)]
            if not other:continue
            def advantage(a,b):
                if None in (a['acf'],a['cmndf'],a['fundamental_fraction'],
                            b['acf'],b['cmndf'],b['fundamental_fraction']):return False
                # Explicit evidence margins: two axes improve and none materially worse.
                gains=(a['acf']>=b['acf']+.05)+(a['cmndf']<=b['cmndf']-.04)+\
                      (a['fundamental_fraction']>=b['fundamental_fraction']+.05)
                worse=(a['acf']<b['acf']-.04 or a['cmndf']>b['cmndf']+.04 or
                       a['fundamental_fraction']<b['fundamental_fraction']-.05)
                return gains>=2 and not worse
            if all(advantage(candidate,r) for r in other):dominant.append(candidate)
        if len(dominant)==1:
            best=dominant[0]
            comparison='acoustically_advantaged_measured_member_DIAGNOSTIC_ONLY'
        elif len(eligible)>1:
            comparison='unresolved_competing_measured_members'
    if best is not None:
        route='audit_advantaged_measured_member_against_local_voicing_and_eckf_state; do_not_apply_pitch'
    return geometry,comparison,temporal,route,best

def inspect(row,audio,fs,tol):
    views,rec=parse(row)
    enrich_center(audio,fs,views['center'],rec)
    families,integer_pairs=family_report(rec,tol)
    geometry,comparison,temporal,route,best=assess(views,rec,families,tol)
    center=[r for r in rec if r['view']=='center']
    result={'case':row['case'],'sample':row['sample'],'time_s':row['time_s'],
        'region':row.get('region',''),'comparison':row.get('comparison',''),
        'primary_mechanism':row.get('primary_mechanism',''),'view_ms':row['view_ms'],
        'center_start_sample':views['center'].get('start_sample'),
        'center_end_sample':views['center'].get('end_sample'),
        'center_rms':views['center'].get('rms'),
        'center_candidate_count':len(center),
        'center_eligible_count':sum(r['eligible'] for r in center),
        'family_count':len(families),
        'eligible_family_count':sum(f['present_eligible'] for f in families),
        'integer_pair_count':len(integer_pairs),'family_geometry':geometry,
        'acoustic_comparison':comparison,'temporal_attribution':temporal,
        'investigation_route':route,
        'best_measured_center_hz_diagnostic_only':best['measured_hz'] if best else '',
        'best_measured_center_acf':best['acf'] if best else '',
        'best_measured_center_cmndf':best['cmndf'] if best else '',
        'best_center_fundamental_fraction':best['fundamental_fraction'] if best else '',
        'candidate_families_json':json.dumps(families,allow_nan=False),
        'candidates_json':json.dumps(rec,allow_nan=False),
        'peak_tracking_json':'',
        'no_pitch_applied':True,'no_voicing_applied':True,'no_tracker_reset':True}
    return result

def track(rows,tol):
    """Observational nearest measured peak across adjacent GRID observations only.

    Does not fill gaps, select pitches, or use prior estimate to decide current F0.
    """
    previous={}
    for r in rows:
        aperture=r['view_ms']; sample=int(r['sample'])
        current=[x for x in json.loads(r['candidates_json']) if x['view']=='center' and x['eligible']]
        former=previous.get(aperture)
        out=[]
        if former is not None:
            prior_sample,old=former
            delta_ms=1000*(float(r['time_s'])-prior_sample)
            if 0<delta_ms<=15.1:
                for x in current:
                    nearest=min(old,key=lambda y:cent(x['measured_hz'],y['measured_hz'])) if old else None
                    diff=cent(x['measured_hz'],nearest['measured_hz']) if nearest else None
                    out.append({'current_measured_hz':x['measured_hz'],
                        'previous_measured_hz':nearest['measured_hz'] if nearest else None,
                        'distance_cents':round(diff,2) if diff is not None else None,
                        'same_peak_neighborhood':bool(diff is not None and diff<=tol),
                        'integer_related_switch':integer_relation(x['measured_hz'],nearest['measured_hz'],tol) if nearest and diff>tol else None,
                        'unmatched_measured_period':nearest is None or diff>tol})
        r['peak_tracking_json']=json.dumps(out,allow_nan=False)
        previous[aperture]=(float(r['time_s']),current)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('all',*CASES),default='all')
    p.add_argument('--v7-dir',type=Path,default=Path('tests/temporal_views_v7'))
    p.add_argument('--wav-dir',type=Path,default=Path('tests'))
    p.add_argument('--out-dir',type=Path,default=Path('tests/measured_period_families_v9'))
    p.add_argument('--tolerance-cents',type=float,default=70.)
    args=p.parse_args()
    if not 10<=args.tolerance_cents<=150:p.error('tolerance must be between 10 and 150 cents')
    names=CASES if args.case=='all' else (args.case,)
    for name in names:
        for path in (args.v7_dir/f'{name}_temporal_views.csv',args.wav_dir/f'{name}.wav'):
            if not path.is_file():p.error(f'missing {path}')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'measured_period_families_v9','read_only':True,
        'complete_four_case_standard':args.case=='all','tolerance_cents':args.tolerance_cents,
        'production_is_not_ground_truth':True,'no_pitch_or_voicing_applied':True,
        'correct_ratata_djuvvs':[[36.923,37.435],[41.025,41.538],[49.230,49.743]],
        'trandafiri':'steady i between consonants; frequency drift is an estimator question, not glissando',
        'cases':{}}
    for name in names:
        audio,fs=sf.read(args.wav_dir/f'{name}.wav',dtype='float64')
        if audio.ndim!=1:p.error(f'{name}: requires mono WAV, shape={audio.shape}')
        data=[];counts=Counter();ms_counts=defaultdict(Counter);samples=defaultdict(set)
        with (args.v7_dir/f'{name}_temporal_views.csv').open(newline='') as f:
            reader=csv.DictReader(f)
            needed={'sample','view_ms','past_json','center_json','future_json','time_s'}
            missing=needed-set(reader.fieldnames or [])
            if missing:p.error(f'{name}: missing V7 fields {sorted(missing)}')
            for row in reader:
                row['case']=name
                report=inspect(row,audio,fs,args.tolerance_cents)
                data.append(report)
                counts[report['family_geometry']]+=1
                ms_counts[str(float(report['view_ms']))][report['family_geometry']]+=1
                samples[report['sample']].add(float(report['view_ms']))
        data.sort(key=lambda r:(float(r['time_s']),float(r['view_ms'])))
        track(data,args.tolerance_cents)
        out=args.out_dir/f'{name}_measured_families.csv'
        with out.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(data)
        incomplete=sum(set(DURATIONS)-dur!=set() for dur in samples.values())
        manifest['cases'][name]={'review_observations':len(samples),'rows':len(data),
             'missing_aperture_sets':incomplete,'sample_rate':fs,
             'geometry_counts_all_apertures':dict(counts),
             'geometry_counts_by_aperture':{k:dict(v) for k,v in ms_counts.items()},
             'output':str(out)}
        print(name,'review observations',len(samples),'rows',len(data),
              '40ms',dict(ms_counts['40.0']),flush=True)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SHADOW ONLY: NO pitch, voicing, tracker-state, MIDI, BPM, or grid changes.')
if __name__=='__main__':main()
