#!/usr/bin/env python3
"""Read-only production ECKF decision-path instrumentation. Run from repo root.

Uses existing diagnostic driver and subclasses its RecordingTrace sink, preserving
its original call and tracker numerical behavior. No MIDI, candidate replacement, or code edits.
Captures real caller locals at emitted production events, including initialization
and reset source objects, detector evidence, gate variables and state frequency.

python tests/trace_production_decisions.py --case all
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import inspect
import json
import math
import runpy
import sys
import time
from pathlib import Path

SITES = {
    'Ochiitai': (42.214, 42.353),
    'PREDESTINATI': (38.034, 53.592),
    'RATATA': (36.923, 41.026, 42.965, 49.231),
    'Trandafiri': (10.774, 10.821),
}
EVENTS = {
    'SILENCE','FRAME_ANALYSIS','RESET_TRIGGER','LOOKAHEAD_FRAME',
    'PERIODICITY_DECISION','INITIALIZATION_GATE',
    'INITIALIZATION_CHOICE_BEFORE_OCTAVE','INITIALIZATION_CHOICE_AFTER_OCTAVE',
    'OCTAVE_INITIALIZATION_AUDIT',
    'INITIALIZATION_CANDIDATES','INITIALIZATION_PROPOSED',
    'RESET_DECISION','RESET_COMPLETED','KALMAN_SAMPLE',
}

def scalar(value):
    try:
        import numpy as np
        if isinstance(value,np.generic): value=value.item()
    except ImportError: pass
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float): return value if math.isfinite(value) else str(value)
    if isinstance(value,complex): return {'real':scalar(value.real),'imag':scalar(value.imag)}
    if isinstance(value,dict): return {str(k):scalar(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)) and len(value)<16: return [scalar(v) for v in value]
    if hasattr(value,'__dict__'):
        return {k:scalar(v) for k,v in vars(value).items() if not k.startswith('_') and k not in ('audio','signal','samples')}
    if hasattr(value,'shape'): return {'shape':list(value.shape),'dtype':str(getattr(value,'dtype','unknown'))}
    return str(type(value).__name__)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['all',*SITES],default='all')
    ap.add_argument('--radius-ms',type=float,default=220)
    ap.add_argument('--out-dir',type=Path,default=Path('tests/production_decision_trace'))
    args=ap.parse_args()
    root=Path.cwd()
    driver=root/'tests'/'diagnose_synchronized_eckf.py'
    if not driver.is_file(): ap.error(f'missing diagnostic driver: {driver}')
    if args.radius_ms<=0: ap.error('radius must be positive')
    sys.path.insert(0,str(root))
    driver_source=driver.read_text(encoding='utf-8')
    needle='tracker_module.ECKFTrace = RecordingTrace'
    if driver_source.count(needle)!=1:
        raise RuntimeError('Diagnostic driver sink assignment changed; refuse unsafe instrumentation')
    instrumented_source=driver_source.replace(
        needle, 'tracker_module.ECKFTrace = __decision_trace_factory__(RecordingTrace)', 1)
    compiled_driver=compile(instrumented_source,str(driver),'exec')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    names=SITES if args.case=='all' else {args.case:SITES[args.case]}
    manifest={'script':'trace_production_decisions.py','ground_truth_used':False,'production_modified':False,
              'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),'cases':{}}
    for name,sites in names.items():
        choices=[p for p in (root/'tests'/f'{name}.wav',root/'tests'/f'{name}.wa') if p.is_file()]
        if len(choices)!=1: ap.error(f'expected exactly one WAV/WA for {name}, found {len(choices)}')
        wav=choices[0]
        import soundfile as sf
        info=sf.info(str(wav))
        dest=args.out_dir/f'{name}_decision_path.jsonl'
        logfile=args.out_dir/f'{name}_driver_stdout.txt'
        records=[0]
        def capture(self,event,sample,**values):
            # The driver recorder is called with identical arguments after capture.
            try:
                t=float(sample)/float(self.sr)
                if event in EVENTS and any(abs(t-center)<=args.radius_ms/1000 for center in sites):
                    caller=inspect.currentframe().f_back.f_back
                    loc=caller.f_locals
                    wanted=('start','analysis_start','n','count','flag','silent_prev','silent_cur',
                            'harm_prev','harm_cur','f1','a1','phi1','gain_min_abs',
                            'reset_accepted','periodicity','initialization')
                    snapshot={k:scalar(loc[k]) for k in wanted if k in loc}
                    # Read oscillator state, never write to it.
                    if 'x_last' in loc and loc['x_last'] is not None:
                        try:
                            import numpy as np
                            first=complex(loc['x_last'][0,0]); fs=float(self.sr)
                            snapshot['x_last_frequency_hz']=scalar(abs(np.log(first)/(1j*2*np.pi/fs)))
                        except (TypeError, ValueError, IndexError, ArithmeticError) as exc:
                            snapshot['x_last_frequency_error']=type(exc).__name__
                    row={'event':event,'time_s':t,'sample':int(sample),'trace_values':scalar(values),
                         'caller_file':str(Path(caller.f_code.co_filename).resolve()),
                         'caller_line':caller.f_lineno,'caller_function':caller.f_code.co_name,
                         'live_locals':snapshot}
                    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                    records[0]+=1
            except Exception as exc:
                # A diagnostic must not interrupt tracking.
                print(f'TRACE_CAPTURE_ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        def trace_factory(recording_class):
            class InstrumentedRecordingTrace(recording_class):
                def emit(self,event_name,sample,**fields):
                    capture(self,event_name,sample,**fields)
                    return super().emit(event_name,sample,**fields)
            return InstrumentedRecordingTrace
        prior_argv=sys.argv[:]
        sys.argv=[str(driver),'--wav',str(wav),'--start','0','--end',str(info.duration),
                  '--output',str(args.out_dir/f'{name}_diagnostic.txt')]
        begun=time.perf_counter()
        try:
            with dest.open('w') as stream, logfile.open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                # Run the driver with only its trace-sink assignment substituted.
                # Its original RecordingTrace still records all diagnostic events.
                namespace={'__name__':'__main__','__file__':str(driver),
                           '__decision_trace_factory__':trace_factory}
                exec(compiled_driver,namespace)
        finally:
            sys.argv=prior_argv
        elapsed=time.perf_counter()-begun
        manifest['cases'][name]={'wav':str(wav),'samplerate':info.samplerate,
            'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
            'sites_s':sites,'records':records[0],'runtime_s':elapsed,
            'trace':str(dest),'diagnostic':str(args.out_dir/f'{name}_diagnostic.txt')}
        print(f'{name}: {records[0]} live events near {sites}; {elapsed:.2f}s')
        if not records[0]: print('WARNING: no events captured. Check diagnostic driver and tracker trace implementation.')
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Outputs:',args.out_dir)

if __name__=='__main__': main()
