#!/usr/bin/env python3
"""Read-only, phrase-aware independent F0 review queue for V22 exclusive-energy calibration.

Input: tests/exclusive_energy_calibration/pair_evidence.csv (previous audit).
Optional original WAVs: --wav-root DIR (files CASE.wav), clips in output/clips.
No F0 is inferred, no review label is auto-filled, and no scoring weights are changed.
The V13 proxy is used ONLY to select interesting cases; never as a truth label.
"""
import argparse
import csv
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path

CASES = ('Ochiitai', 'PREDESTINATI', 'RATATA', 'Trandafiri')
HOLDOUTS = {'Ochiitai': 3.550, 'PREDESTINATI': 4.780}
FIELDS = ['review_id','case','time_s','low_hz','high_hz','low_exclusive_percent',
          'acf_advantage_high_minus_low','v5_outcome','v13_proxy_closer',
          'selection_reason','block_id','phrase_group','regression_holdout',
          'clip_path','review_label','review_confidence','reviewer','notes']

def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}. Run audit_v22_exclusive_energy_calibration.py first.')
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

def audio_clip(wav, dest, center, before, after):
    with wave.open(str(wav), 'rb') as src:
        rate = src.getframerate()
        start = max(0, int((center-before)*rate))
        end = min(src.getnframes(), int((center+after)*rate))
        if end <= start:
            raise ValueError(f'Invalid clip bounds for {wav} at {center}')
        src.setpos(start)
        audio = src.readframes(end-start)
        with wave.open(str(dest), 'wb') as dst:
            dst.setnchannels(src.getnchannels())
            dst.setsampwidth(src.getsampwidth())
            dst.setframerate(rate)
            dst.writeframes(audio)
    return start/rate, end/rate

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='tests/exclusive_energy_calibration/pair_evidence.csv')
    ap.add_argument('--out', default='tests/exclusive_energy_targeted_controls')
    ap.add_argument('--wav-root', default=None, help='Directory with Ochiitai.wav etc; optional.')
    ap.add_argument('--block-s', type=float, default=0.5)
    ap.add_argument('--phrase-s', type=float, default=2.0, help='Grouping unit for review and future split; not a detected musical phrase.')
    ap.add_argument('--per-stratum', type=int, default=12)
    ap.add_argument('--clip-before', type=float, default=0.7)
    ap.add_argument('--clip-after', type=float, default=0.9)
    args = ap.parse_args()
    if min(args.block_s,args.phrase_s,args.clip_before,args.clip_after) <= 0 or args.per_stratum < 1:
        ap.error('All duration parameters and --per-stratum must be positive.')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw = read_csv(Path(args.input))
    if not raw: raise ValueError('Input pair evidence is empty.')
    mandatory = {'case','time_s','low_hz','high_hz','low_exclusive_fraction',
                 'acf_advantage_high_minus_low','v13_proxy_closer','v5_outcome'}
    if mandatory - set(raw[0]): raise ValueError(f'Missing input columns: {sorted(mandatory-set(raw[0]))}')
    candidates = []
    dropped = Counter()
    for r in raw:
        case = r['case']
        t, lo, hi, ex = (number(r.get(k)) for k in ('time_s','low_hz','high_hz','low_exclusive_fraction'))
        if case not in CASES or None in (t,lo,hi,ex) or not (0<=ex<=1) or not (0<lo<hi):
            dropped['invalid_or_missing_fields'] += 1
            continue
        holdout = case in HOLDOUTS and abs(t-HOLDOUTS[case]) <= .025 + 1e-8
        block = math.floor(t/args.block_s)
        phrase = math.floor(t/args.phrase_s)
        proxy = r.get('v13_proxy_closer','unavailable')
        if proxy not in ('high','low','unavailable'): proxy='unavailable'
        acf = number(r.get('acf_advantage_high_minus_low'))
        # Strata are descriptive, NOT F0 labels. This explicitly includes both
        # directions of surprising proxy evidence and unlabeled genuine-low candidates.
        strata = []
        if proxy == 'high' and ex >= .08: strata.append('high_proxy_high_exclusive_counterexample')
        if proxy == 'high' and ex >= .02 and acf is not None and acf < 0:
            strata.append('high_proxy_lower_acf_advantage')
        if proxy == 'low': strata.append('rare_low_proxy_any_energy')
        if proxy == 'low' and ex < .08: strata.append('low_proxy_weak_exclusive_counterexample')
        if proxy == 'unavailable' and ex >= .08: strata.append('unresolved_strong_exclusive')
        if proxy == 'unavailable' and ex < .002: strata.append('unresolved_near_zero_exclusive')
        if proxy == 'unavailable' and .002 <= ex < .08: strata.append('unresolved_middle_exclusive')
        if proxy == 'high' and ex < .002: strata.append('high_proxy_near_zero_control')
        if holdout: strata = ['regression_holdout_review_only']
        if not strata: continue
        candidates.append(dict(r, _t=t,_lo=lo,_hi=hi,_ex=ex,_acf=acf,
                               _block=block,_phrase=phrase,_proxy=proxy,
                               _holdout=holdout,_strata=strata))
    # Select at most one pair per case+0.5s block across all ordinary strata;
    # holdouts are kept separately. Prioritize rare counterexamples/low controls.
    priority = ['low_proxy_weak_exclusive_counterexample',
                'high_proxy_high_exclusive_counterexample',
                'high_proxy_lower_acf_advantage',
                'rare_low_proxy_any_energy',
                'unresolved_strong_exclusive',
                'unresolved_near_zero_exclusive',
                'unresolved_middle_exclusive',
                'high_proxy_near_zero_control']
    quotas = Counter()
    used_blocks = set()
    selected = []
    # Select diverse cases and temporal positions, not multiple adjacent frames.
    for category in priority:
        for case in CASES:
            pool = [r for r in candidates if case==r['case'] and category in r['_strata'] and not r['_holdout']]
            by_block = defaultdict(list)
            for r in pool: by_block[(case,r['_block'])].append(r)
            for key in sorted(by_block):
                if quotas[(case,category)] >= args.per_stratum: break
                if key in used_blocks: continue
                group = sorted(by_block[key], key=lambda r:(r['_t'],r['_lo'],r['_hi']))
                item = group[len(group)//2]
                item = dict(item, _reason=category)
                selected.append(item)
                used_blocks.add(key)
                quotas[(case,category)] += 1
    # Preserve regression observations as review-only; exclude them from fitting.
    for case, center in HOLDOUTS.items():
        matches = [r for r in candidates if r['case']==case and abs(r['_t']-center)<1e-6]
        for r in sorted(matches, key=lambda x:(x['_lo'],x['_hi'])):
            selected.append(dict(r,_reason='regression_holdout_review_only'))
    selected.sort(key=lambda r:(r['case'],r['_t'],r['_lo'],r['_hi']))
    queue = []
    clip_index = []
    wavroot = Path(args.wav_root) if args.wav_root else None
    if wavroot is not None: (root/'clips').mkdir(exist_ok=True)
    for index,r in enumerate(selected, 1):
        ident = f'R{index:04d}'
        clip = ''
        if wavroot is not None:
            source = wavroot/f"{r['case']}.wav"
            if not source.is_file(): raise FileNotFoundError(f'Missing requested WAV: {source}')
            clip = f'clips/{ident}_{r["case"]}_{int(round(r["_t"]*1000)):07d}ms.wav'
            start,end = audio_clip(source,root/clip,r['_t'],args.clip_before,args.clip_after)
            clip_index.append({'review_id':ident,'case':r['case'],'time_s':r['_t'],
                               'clip_path':clip,'clip_start_s':start,'clip_end_s':end})
        queue.append({'review_id':ident,'case':r['case'],'time_s':r['_t'],
                      'low_hz':r['_lo'],'high_hz':r['_hi'],
                      'low_exclusive_percent':round(r['_ex']*100,7),
                      'acf_advantage_high_minus_low':r['_acf'],
                      'v5_outcome':r['v5_outcome'],'v13_proxy_closer':r['_proxy'],
                      'selection_reason':r['_reason'],
                      'block_id':f'{r["case"]}:{r["_block"]}',
                      'phrase_group':f'{r["case"]}:segment_{r["_phrase"]}',
                      'regression_holdout':r['_holdout'], 'clip_path':clip,
                      'review_label':'', 'review_confidence':'', 'reviewer':'', 'notes':''})
    write_csv(root/'targeted_review_queue.csv',queue,FIELDS)
    write_csv(root/'clip_index.csv',clip_index,
              ['review_id','case','time_s','clip_path','clip_start_s','clip_end_s'])
    strata_summary = []
    all_strata = sorted({k for r in candidates for k in r['_strata']})
    for case in CASES:
        for stratum in all_strata:
            source=[r for r in candidates if r['case']==case and stratum in r['_strata']]
            picked=[r for r in queue if r['case']==case and r['selection_reason']==stratum]
            strata_summary.append({'case':case,'stratum':stratum,
                                   'source_pairs':len(source),
                                   'source_distinct_blocks':len({r['_block'] for r in source}),
                                   'selected_pairs':len(picked)})
    write_csv(root/'selection_accounting.csv',strata_summary,
              ['case','stratum','source_pairs','source_distinct_blocks','selected_pairs'])
    readme = '''INDEPENDENT F0 REVIEW INSTRUCTIONS\n\nListen to the clip (if exported), preferably compare with a spectrogram/longer\nphrase and a trusted external pitch reference. Review the ACTUAL fundamental,\nnot merely which candidate has more energy or agrees with V13.\n\nSet review_label to low, high, neither, or uncertain; do not leave a guess.\nSet review_confidence to high, medium, or low and provide notes identifying\nindependent evidence (e.g. audible octave, harmonic spacing, sustained vowel).\nDo NOT derive labels from v13_proxy_closer, V5 scores, the sampling stratum,\nor either known regression pitch. Keep review-only holdouts out of calibration.\n\nImportant: Multiple selected pairs in the same phrase_group are correlated.\nUse phrase_group as a splitting / deduplication unit for subsequent calibration.\nThe phrase_group is a fixed-duration segment, NOT an inferred musical phrase.\nIf the true F0 is between candidates or differs from both, mark neither.\nThe CSV has intentionally BLANK labels: no independent judgments exist yet.\nNo thresholds or weights are fitted or deployed by this tool.\n'''
    (root/'REVIEW_INSTRUCTIONS.txt').write_text(readme,encoding='utf-8')
    report = {'schema':'v22_targeted_independent_review_queue_v1',
              'input':str(args.input),'input_pairs':len(raw),
              'valid_eligible_pairs':len(candidates), 'invalid_rows':dict(dropped),
              'review_rows':len(queue),'regression_holdout_rows':sum(x['regression_holdout'] for x in queue),
              'independent_labels_created':0,'calibration_fitted':False,
              'production_modified':False,'v5_modified':False,
              'selection_is_not_ground_truth':True,
              'wav_clips_exported':len(clip_index),
              'sampling':'at most one ordinary pair per case+block across strata; fixed-duration phrase groups for review',
              'outputs':['targeted_review_queue.csv','selection_accounting.csv','clip_index.csv','REVIEW_INSTRUCTIONS.txt','manifest.json']}
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
