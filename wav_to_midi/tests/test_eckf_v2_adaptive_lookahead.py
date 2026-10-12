"""Run from repo root: python tests/test_eckf_v2_adaptive_lookahead.py"""
from dataclasses import replace
import numpy as np
import python_eckf.tracker as tracker
from python_eckf.config import ECKFConfig
from python_eckf.initialization_candidates import InitializationChoice

FS = 16000
B = 2048

def sine(hz, start_block=0):
    t = (np.arange(B) + start_block * B) / FS
    return .2 * np.sin(2*np.pi*hz*t)

assert tracker._same_note(228.5,238.38)
assert not tracker._same_note(220.,238.38)
assert not tracker._same_note(440.,466.164)

orig = tracker.choose_initialization

def fail_first(detector, frame, fs, start, *rest, **kwargs):
    if start == 0:
        return InitializationChoice(None,None,None,'test','forced_failure',0,None)
    return orig(detector,frame,fs,start,*rest,**kwargs)

try:
    tracker.choose_initialization = fail_first
    signal = np.concatenate([sine(230,i) for i in range(3)])
    baseline = tracker.track_pitch(signal, FS, ECKFConfig(mode='offline',num_buf_to_wait=0))
    assisted = tracker.track_pitch(signal, FS, ECKFConfig(mode='offline',num_buf_to_wait=2))
    assert baseline.frame_decision[0] == 'INITIALIZATION_UNRESOLVED'
    assert assisted.frame_decision[0] == 'VOICED_TRACKED_LOOKAHEAD'
    assert assisted.onset_samples.tolist() == [0]  # no onset shifted to future frame
    assert assisted.f0_hz[0] > 0  # filter actually revisited original audio
    print('PASS: lookahead recovers failed initialization; onset remains at original sample')
finally:
    tracker.choose_initialization = orig

# Low-energy frames remain absolute vetoes and are never filled by lookahead.
signal = np.concatenate([np.zeros(B),sine(230,1),sine(230,2),np.zeros(B)])
r = tracker.track_pitch(signal,FS,ECKFConfig(mode='offline',num_buf_to_wait=4))
for index in (0,3):
    sl=slice(index*B,(index+1)*B)
    assert r.frame_decision[index]=='LOW_ENERGY_SILENCE'
    assert not r.f0_hz[sl].any()
print('PASS: silence is authoritative; no frequency is assigned to silent frames')

# No borrowing from a new note after a forced failure in the initial frame.
try:
    tracker.choose_initialization=fail_first
    changing=np.concatenate([sine(230,0),sine(300,1),sine(300,2)])
    result=tracker.track_pitch(changing,FS,ECKFConfig(mode='offline',num_buf_to_wait=2))
    assert result.frame_decision[0]=='INITIALIZATION_UNRESOLVED', result.frame_decision
    assert not result.f0_hz[:B].any()
    print('PASS: cannot borrow a later, different musical note')
finally:
    tracker.choose_initialization=orig

print('ALL ADAPTIVE LOOKAHEAD TESTS PASS')
