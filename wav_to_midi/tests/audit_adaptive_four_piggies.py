#!/usr/bin/env python3
"""READ-ONLY four-WAV adaptive evidence / djuvv / reset-path audit.

Inputs: existing adaptive_evidence CSVs and synchronized v3 diagnostic CSV/TXT.
Does not import tracker, change code, inspect MIDI, or generate/correct F0.
Reports disagreement and acoustically observable features; NOT correctness labels.
Run: python tests/audit_adaptive_four_piggies.py
"""
from __future__ import annotations
import argparse, csv, json, math, re, statistics
from collections import Counter
from pathlib import Path

NAMES = ('Ochiitai','PREDESTINATI','RATATA','Trandafiri')
# RATATA WAV times from the three previously annotated djuvv events; context only.
SITES = {
 'RATATA': [('DJUVV_1',36.923),('DJUVV_2',41.026),('RATATA_SUBMULTIPLE',42.965),('DJUVV_3',49.231)],
 'Ochiitai': [('OCHIITAI_TRANSITION',42.214),('OCHIITAI_HIGHER',42.353)],
 'Trandafiri': [('TRANDAFIRI_RESET',10.774),('TRANDAFIRI_VOWEL',10.821)],
 'PREDESTINATI': [('PREDESTINATI_GLISS_END',38.034),('PREDESTINATI_ONSET',53.592)],
}
EVENT_RE = re.compile(r'^\s*(\d+(?:\.\d+)?)s\s+([A-Z_]+)\s+(.*)$')
FIELD_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)')

def num(x):
 try:
  y=float(x)
  return y if math.isfinite(y) else None
 except (TypeError,ValueError): return None

def load_csv(path, expected):
 if not path.is_file(): raise SystemExit(f'MISSING: {path}')
 with path.open(newline='') as f:
  reader=csv.DictReader(f)
  if not expected.issubset(reader.fieldnames or []):
   raise SystemExit(f'BAD SCHEMA: {path}; columns={reader.fieldnames}')
  return list(reader)

def parse_events(path):
 if not path.is_file(): raise SystemExit(f'MISSING: {path}')
 events=[]
 for line in path.read_text(errors='replace').splitlines():
  match=EVENT_RE.match(line)
  if match:
   t, event, rest=match.groups()
   events.append((float(t),event,dict(FIELD_RE.findall(rest)),line.strip()))
 return events

def near(rows,t,margin):
 return [r for r in rows if abs(float(r['time_s'])-t)<=margin]

def pitch_str(r):
 hz=num(r.get('observed_hz'))
 return 'UNRESOLVED' if hz is None else f'{hz:.2f}Hz'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--adaptive-dir',type=Path,default=Path('tests/adaptive_evidence'))
 p.add_argument('--baseline-dir',type=Path,default=Path('tests/robustness_v3'))
 p.add_argument('--out-dir',type=Path,default=Path('tests/adaptive_audit'))
 p.add_argument('--site-radius-ms',type=float,default=140.)
 args=p.parse_args()
 if args.site_radius_ms<=0: p.error('site radius must be positive')
 args.out_dir.mkdir(parents=True,exist_ok=True)
 overview=[]; flags=[]; report=[]
 report.append('FOUR-PIGGY OBSERVATIONAL AUDIT — WAV-only existing outputs; no MIDI or F0 alteration')
 report.append('"baseline unvoiced" is NOT ground-truth silence; candidate = observational hypothesis only.')
 report.append('24/40/64/96ms acoustic windows overlap. No assumption of statistical independence.')
 report.append('Djuvv times annotate acoustic regions; they do NOT force F0 or voice status.\n')
 for name in NAMES:
  a=load_csv(args.adaptive_dir/f'{name}_adaptive.csv',{'time_s','status','observed_hz','probe_rms','window_ms','reason','comparison','candidates','baseline_voiced'})
  b=load_csv(args.baseline_dir/f'{name}_v3_frames.csv',{'time_s','sample','voiced_samples','median_f0_hz'})
  ev=parse_events(args.baseline_dir/f'{name}_v3.txt')
  t0=float(a[0]['time_s']); t1=float(a[-1]['time_s'])
  counted=Counter(r['comparison'] for r in a)
  selected=[r for r in a if r['comparison']=='baseline_unvoiced']
  unresolved=[r for r in a if r['status']!='observed_candidate']
  # Recording-specific probe floor is descriptive only; NEVER used for pitch decisions.
  rms_sorted=sorted(v for r in a if (v:=num(r['probe_rms'])) is not None)
  median_rms=statistics.median(rms_sorted) if rms_sorted else None
  percentile=lambda q: rms_sorted[min(len(rms_sorted)-1,int(q*(len(rms_sorted)-1)))] if rms_sorted else None
  record=dict(name=name,observations=len(a),first_s=t0,last_s=t1,
              baseline_unvoiced_candidate=len(selected),adaptive_unresolved=len(unresolved),
              probe_rms_median=median_rms,probe_rms_p10=percentile(.1),probe_rms_p90=percentile(.9),
              expansion_counts=dict(Counter(str(r['window_ms']) for r in a)),comparisons=dict(counted))
  overview.append(record)
  report.extend([f'=== {name}: full song ===',json.dumps(record,indent=2)])
  # Every baseline-unvoiced adaptive candidate: expose metrics and nearby baseline,
  # NOT label as false positive automatically.
  for r in selected:
   t=float(r['time_s'])
   close=min(b,key=lambda x:abs(float(x['time_s'])-t))
   flag=dict(name=name,time_s=t,adaptive_hz=num(r['observed_hz']),
             probe_rms=num(r['probe_rms']),chosen_window_ms=num(r['window_ms']),
             acf=num(r.get('acf')),cmndf=num(r.get('cmndf')),
             spectral_coverage=num(r.get('spectral_coverage')),
             reason=r['reason'],baseline_frame_time_s=float(close['time_s']),
             baseline_voiced_samples=int(close['voiced_samples']),
             baseline_median_hz=num(close['median_f0_hz']),
             candidate_count=int(r['candidate_count']),candidates=r['candidates'])
   flags.append(flag)
  for label, t in SITES[name]:
   sample=near(a,t,args.site_radius_ms/1000.)
   baseline=near(b,t,args.site_radius_ms/1000.)
   trace=[e for e in ev if abs(e[0]-t)<=args.site_radius_ms/1000.]
   report.append(f'\n--- {label} at {t:.3f}s (±{args.site_radius_ms:g}ms) ---')
   report.append('Adaptive: center  candidate  window  RMS  ACF  CMNDF  coverage  reason  baseline')
   for r in sample:
    report.append(f"{r['time_s']:>11} {pitch_str(r):>12} {r['window_ms']:>6} "
                  f"{r['probe_rms']:>12} {r['acf']:>8} {r['cmndf']:>8} "
                  f"{r['spectral_coverage']:>3} {r['reason']} baseline={r['baseline_hz'] or 'UNVOICED'}")
   report.append('Actual baseline frames (no interpolated grid):')
   for r in baseline:
    report.append(f"{r['time_s']} voiced_samples={r['voiced_samples']} "
                  f"median={r['median_f0_hz'] or '--'} first={r['first_f0_hz'] or '--'} "
                  f"last={r['last_f0_hz'] or '--'}")
   report.append('Actual baseline events (existing live trace, NOT standalone replay):')
   report.extend(e[3] for e in trace)
   if not trace: report.append('(no events in interval)')
  # Diagnostic control-flow integrity, across ALL reset events: proposal and reset must
  # be shown side by side, never assert they should match without examining reconciliation.
  proposals={}
  for t,event,fields,line in ev:
   frame=fields.get('frame_start')
   if event=='INITIALIZATION_PROPOSED' and frame is not None:
    proposals[frame]=(t,fields,line)
   if event=='RESET_COMPLETED' and frame is not None:
    match=proposals.get(frame)
    if match:
     f_proposed=num(match[1].get('init_f0_hz'))
     f_reset=num(fields.get('init_f0_hz'))
     if f_proposed and f_reset and abs(1200*math.log2(f_proposed/f_reset)) >= 50:
      report.append(f'\nPROPOSAL/RESET DISAGREEMENT {name} frame={frame} time={t:.6f}s '
                    f'proposed={f_proposed:.5f} actual={f_reset:.5f} '
                    f'delta_st={1200*math.log2(f_reset/f_proposed)/100:.3f}')
      report.append(match[2]);report.append(line)
      report.append('NOTE: location of change is NOT known from existing events; inspect call flow, do not patch by assumption.')
 report.append('\nCONCLUSION: None of these observations constitutes an accuracy score or a license to fill, smooth, or correct F0.')
 (args.out_dir/'four_piggy_audit.txt').write_text('\n'.join(report)+'\n')
 with (args.out_dir/'baseline_unvoiced_candidates.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flags[0]) if flags else ['name','time_s']);w.writeheader();w.writerows(flags)
 (args.out_dir/'overview.json').write_text(json.dumps(dict(midi_used=False,production_modified=False,
  corrected_pitch=False,summary=overview,baseline_unvoiced_candidates=len(flags),
  djuvv_wav_times_s=[36.923,41.026,49.231]),indent=2)+'\n')
 print(f'Wrote {args.out_dir}/four_piggy_audit.txt, baseline_unvoiced_candidates.csv, overview.json')
 print(f'Four recordings: {len(overview)}; baseline-unvoiced candidate observations (NOT accuracy labels): {len(flags)}')

if __name__=='__main__': main()
