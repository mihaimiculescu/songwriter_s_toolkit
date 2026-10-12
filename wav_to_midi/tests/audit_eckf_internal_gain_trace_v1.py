#!/usr/bin/env python3
"""Compare two observational ECKF internal traces from the same waveform at different gain.

Expected input is produced by python_eckf V2 internal-gain-trace instrumentation with
ECKF_TRACE_CSV set. This tool does not modify tracking decisions.
"""
from __future__ import annotations
import argparse, csv, math, statistics
from pathlib import Path

NUM = [
    'f0_pre_hz','f0_hz','innovation_abs','innovation_after_abs','q','gain_norm','covariance_norm',
    'sample_value_real','predicted_measurement_abs','kalman_denominator_real',
    'k0_abs','k1_abs','k2_abs','p00_abs','p11_abs','p22_abs','x1_pre_abs','x2_pre_abs','x3_pre_abs',
    'x1_post_abs','x2_post_abs','x3_post_abs'
]

def f(x):
    try: return float(x)
    except: return math.nan

def cents(a,b):
    if a>0 and b>0 and math.isfinite(a) and math.isfinite(b): return 1200*math.log2(b/a)
    return math.nan

def ratio(a,b):
    if a!=0 and math.isfinite(a) and math.isfinite(b): return b/a
    return math.nan

def load(path):
    out={}
    with open(path,newline='',encoding='utf-8') as h:
        for r in csv.DictReader(h):
            if r.get('event')!='KALMAN_SAMPLE': continue
            out[int(r['sample'])]=r
    return out

def med(vals):
    vals=[x for x in vals if math.isfinite(x)]
    return statistics.median(vals) if vals else math.nan

def pct(vals,p):
    vals=sorted(x for x in vals if math.isfinite(x))
    if not vals: return math.nan
    i=(len(vals)-1)*p
    lo=int(math.floor(i)); hi=int(math.ceil(i))
    if lo==hi:return vals[lo]
    return vals[lo]*(hi-i)+vals[hi]*(i-lo)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',required=True)
    ap.add_argument('--test',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--material-cents',type=float,default=1.0)
    args=ap.parse_args()
    a=load(args.reference); b=load(args.test)
    keys=sorted(set(a)&set(b))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    gain_ratios=[]
    for n in keys:
        av=f(a[n].get('sample_value_real','')); bv=f(b[n].get('sample_value_real',''))
        if abs(av)>1e-5 and math.isfinite(bv): gain_ratios.append(bv/av)
    g=med(gain_ratios)
    rows=[]
    for n in keys:
        ar=a[n]; br=b[n]
        row={'sample':n,'time_s':f(ar.get('time_s','')),'estimated_audio_gain':g}
        for col in NUM:
            x=f(ar.get(col,'')); y=f(br.get(col,''))
            row['ref_'+col]=x; row['test_'+col]=y
            row['delta_'+col]=y-x if math.isfinite(x) and math.isfinite(y) else math.nan
            row['ratio_'+col]=ratio(x,y)
        row['delta_f0_pre_cents']=cents(row['ref_f0_pre_hz'],row['test_f0_pre_hz'])
        row['delta_f0_post_cents']=cents(row['ref_f0_hz'],row['test_f0_hz'])
        # Scale-compensated measurement-model quantities: divide test amplitudes by estimated global gain.
        if math.isfinite(g) and g!=0:
            row['test_sample_gain_comp']=row['test_sample_value_real']/g
            row['sample_gain_comp_error']=row['test_sample_gain_comp']-row['ref_sample_value_real']
            row['test_predicted_measurement_gain_comp']=row['test_predicted_measurement_abs']/abs(g)
            row['predicted_measurement_gain_comp_error']=row['test_predicted_measurement_gain_comp']-row['ref_predicted_measurement_abs']
        rows.append(row)
    fieldnames=list(rows[0]) if rows else []
    with open(out/'matched_internal_trace.csv','w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    divergent=[r for r in rows if math.isfinite(r['delta_f0_post_cents']) and abs(r['delta_f0_post_cents'])>=args.material_cents]
    with open(out/'material_f0_divergences.csv','w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fieldnames); w.writeheader(); w.writerows(divergent)
    pre=[abs(r['delta_f0_pre_cents']) for r in rows if math.isfinite(r['delta_f0_pre_cents'])]
    post=[abs(r['delta_f0_post_cents']) for r in rows if math.isfinite(r['delta_f0_post_cents'])]
    first=divergent[0] if divergent else None
    lines=[
        'ECKF INTERNAL GAIN-SENSITIVITY AUDIT V1',
        f'Matched traced samples: {len(rows)}',
        f'Estimated test/reference waveform gain: {g:.9f} ({20*math.log10(abs(g)):.6f} dB)' if math.isfinite(g) and g else 'Estimated gain: unavailable',
        f'Material F0 threshold: {args.material_cents:.3f} cents',
        f'Material post-update divergences: {len(divergent)}',
        f'Pre-update |F0 drift| median/p95/max: {med(pre):.6f} / {pct(pre,.95):.6f} / {max(pre) if pre else math.nan:.6f} cents',
        f'Post-update |F0 drift| median/p95/max: {med(post):.6f} / {pct(post,.95):.6f} / {max(post) if post else math.nan:.6f} cents',
    ]
    if first:
        lines += [
            '', 'FIRST MATERIAL POST-UPDATE DIVERGENCE',
            f"time_s: {first['time_s']:.9f}",
            f"pre-update F0 ref/test/delta_cents: {first['ref_f0_pre_hz']:.9f} / {first['test_f0_pre_hz']:.9f} / {first['delta_f0_pre_cents']:.6f}",
            f"post-update F0 ref/test/delta_cents: {first['ref_f0_hz']:.9f} / {first['test_f0_hz']:.9f} / {first['delta_f0_post_cents']:.6f}",
            f"innovation_abs ref/test/ratio: {first['ref_innovation_abs']:.9g} / {first['test_innovation_abs']:.9g} / {first['ratio_innovation_abs']:.9g}",
            f"innovation_after_abs ref/test/ratio: {first['ref_innovation_after_abs']:.9g} / {first['test_innovation_after_abs']:.9g} / {first['ratio_innovation_after_abs']:.9g}",
            f"q ref/test/ratio: {first['ref_q']:.9g} / {first['test_q']:.9g} / {first['ratio_q']:.9g}",
            f"Kalman gain norm ref/test/ratio: {first['ref_gain_norm']:.9g} / {first['test_gain_norm']:.9g} / {first['ratio_gain_norm']:.9g}",
            f"covariance norm ref/test/ratio: {first['ref_covariance_norm']:.9g} / {first['test_covariance_norm']:.9g} / {first['ratio_covariance_norm']:.9g}",
            f"denominator ref/test/ratio: {first['ref_kalman_denominator_real']:.9g} / {first['test_kalman_denominator_real']:.9g} / {first['ratio_kalman_denominator_real']:.9g}",
        ]
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(f'Reports: {out}')
if __name__=='__main__': main()
