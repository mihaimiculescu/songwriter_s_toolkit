#!/usr/bin/env python3
"""
Target-formation deep dive for ECKF V2.

Read-only audit of *.target_formation.csv files produced by the active V2
trajectory pipeline. It does NOT alter tracking, validity, or target formation.

It groups contiguous corrected-valid points rejected from stable targets,
quantifies how coherent each rejected run actually is, and probes whether the
current fixed 120 ms local aperture is the main reason for rejection.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REJECT_REASONS = {"LOCAL_SPREAD_TOO_WIDE", "STABLE_RUN_TOO_SHORT", "TOO_FEW_LOCAL_POINTS", "NOT_PROMOTED"}
CURRENT_WINDOW_MS = 120.0
CURRENT_RANGE_ST = 0.65
CURRENT_MIN_STABLE_MS = 80.0
PROBE_WINDOWS_MS = (40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0, 240.0)
PROBE_MIN_DURATIONS_MS = (40.0, 60.0, 80.0, 100.0, 120.0)


def hz_to_midi_float(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    x = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in x.reshape(-1, 2)]


def odd_steps(ms: float, analysis_hz: float, minimum: int = 3) -> int:
    steps = max(minimum, int(round(ms * analysis_hz / 1000.0)))
    if steps % 2 == 0:
        steps += 1
    return steps


def robust_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    q10, q90 = np.percentile(values, [10, 90])
    return float(q90 - q10)


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def load_inputs(source: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            for name in zf.namelist():
                if not name.endswith(".target_formation.csv"):
                    continue
                base = Path(name).name
                song = base.split("_v2_")[0]
                with zf.open(name) as f:
                    out[song] = pd.read_csv(f)
    elif source.is_dir():
        for p in sorted(source.rglob("*.target_formation.csv")):
            song = p.name.split("_v2_")[0]
            out[song] = pd.read_csv(p)
    else:
        raise SystemExit(f"Input not found or unsupported: {source}")
    if not out:
        raise SystemExit("No *.target_formation.csv files found")
    return out


def infer_analysis_hz(df: pd.DataFrame) -> float:
    t = pd.to_numeric(df["time_s"], errors="coerce").dropna().to_numpy()
    d = np.diff(t)
    d = d[d > 1e-9]
    if not len(d):
        return 100.0
    return 1.0 / float(np.median(d))


def nearest_stable_context(df: pd.DataFrame, start: int, end: int) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    stable = df["final_stable"].fillna(0).astype(int).to_numpy().astype(bool)
    pitch = pd.to_numeric(df["structural_pitch_st"], errors="coerce").to_numpy(float)
    before = np.flatnonzero(stable[:start])
    after = np.flatnonzero(stable[end:])
    bi = int(before[-1]) if len(before) else None
    ai = int(end + after[0]) if len(after) else None
    return bi, ai, (float(pitch[bi]) if bi is not None and np.isfinite(pitch[bi]) else None), (float(pitch[ai]) if ai is not None and np.isfinite(pitch[ai]) else None)


def classify_suspicion(duration_ms: float, rr: float, note_fraction: float, core_fraction: float, reason_mix: str) -> str:
    # This is diagnostic triage, not a correctness verdict.
    if duration_ms >= 80 and note_fraction >= 0.80 and core_fraction >= 0.70:
        return "HIGH_GT_LIKE"
    if duration_ms >= 60 and note_fraction >= 0.70 and core_fraction >= 0.55:
        return "MEDIUM_GT_LIKE"
    if "STABLE_RUN_TOO_SHORT" in reason_mix and note_fraction >= 0.80:
        return "SHORT_COHERENT"
    return "LOW_OR_DECORATIVE"


def analyze_song(song: str, df: pd.DataFrame):
    required = {"time_s","corrected_valid","clean_f0_hz","structural_pitch_st","final_stable","target_formation_reason"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{song}: missing columns {missing}")

    hz = pd.to_numeric(df["clean_f0_hz"], errors="coerce").to_numpy(float)
    st = pd.to_numeric(df["structural_pitch_st"], errors="coerce").to_numpy(float)
    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    trusted = df["corrected_valid"].fillna(0).astype(int).to_numpy().astype(bool)
    stable = df["final_stable"].fillna(0).astype(int).to_numpy().astype(bool)
    reason = df["target_formation_reason"].fillna("").astype(str).to_numpy()
    rejected = trusted & ~stable
    analysis_hz = infer_analysis_hz(df)
    step_ms = 1000.0 / analysis_hz

    rows = []
    run_id = 0
    for start, end in contiguous_runs(rejected):
        run_id += 1
        vals = st[start:end]
        vals = vals[np.isfinite(vals)]
        hzvals = hz[start:end]
        hzvals = hzvals[np.isfinite(hzvals)]
        duration_ms = (end - start) * step_ms
        if len(vals):
            med_st = float(np.median(vals)); rr = robust_range(vals); md = mad(vals)
            nearest = np.rint(vals).astype(int)
            uniques, counts = np.unique(nearest, return_counts=True)
            j = int(np.argmax(counts)); dom_midi = int(uniques[j]); dom_frac = float(counts[j]/len(nearest))
            # Stable core around the run median: points within +/- 32.5 cents.
            core_frac = float(np.mean(np.abs(vals - med_st) <= 0.325))
            p05,p95=np.percentile(vals,[5,95]) if len(vals)>=2 else (vals[0],vals[0])
        else:
            med_st=rr=md=core_frac=dom_frac=float('nan'); dom_midi=None; p05=p95=float('nan')
        rs = reason[start:end]
        vals_reason, cnt_reason = np.unique(rs, return_counts=True)
        reason_mix = ";".join(f"{a}:{b}" for a,b in sorted(zip(vals_reason,cnt_reason), key=lambda x:-x[1]))
        bi, ai, bst, ast = nearest_stable_context(df,start,end)
        prev_gap_ms = (time_s[start]-time_s[bi])*1000.0 if bi is not None else None
        next_gap_ms = (time_s[ai]-time_s[end-1])*1000.0 if ai is not None else None
        same_prev = (bst is not None and np.isfinite(med_st) and abs(bst-med_st)<=0.5)
        same_next = (ast is not None and np.isfinite(med_st) and abs(ast-med_st)<=0.5)
        rows.append({
            "song":song,"run_id":run_id,"start_index":start,"end_index_exclusive":end,
            "start_time_s":float(time_s[start]),"end_time_s":float(time_s[end-1]+1/analysis_hz),
            "duration_ms":duration_ms,"point_count":end-start,"reason_mix":reason_mix,
            "median_structural_st":med_st,"median_hz":float(np.median(hzvals)) if len(hzvals) else np.nan,
            "robust_range_st_run":rr,"mad_st":md,"p05_p95_range_st":float(p95-p05) if np.isfinite(p95) else np.nan,
            "dominant_midi":dom_midi,"dominant_note":midi_name(dom_midi) if dom_midi is not None else "",
            "dominant_note_fraction":dom_frac,"median_core_fraction_32_5c":core_frac,
            "previous_stable_time_s":float(time_s[bi]) if bi is not None else np.nan,
            "previous_stable_pitch_st":bst if bst is not None else np.nan,"previous_gap_ms":prev_gap_ms if prev_gap_ms is not None else np.nan,
            "next_stable_time_s":float(time_s[ai]) if ai is not None else np.nan,
            "next_stable_pitch_st":ast if ast is not None else np.nan,"next_gap_ms":next_gap_ms if next_gap_ms is not None else np.nan,
            "same_target_as_previous":int(bool(same_prev)),"same_target_as_next":int(bool(same_next)),
            "triage":classify_suspicion(duration_ms,rr,dom_frac,core_frac,reason_mix),
        })
    runs = pd.DataFrame(rows)

    # Counterfactual aperture probe: same current structural pitch and threshold,
    # only vary local window size. Read-only diagnostic.
    probe_rows=[]
    current_rejected_idx=np.flatnonzero(rejected)
    valid_runs=contiguous_runs(trusted)
    for wms in PROBE_WINDOWS_MS:
        win=odd_steps(wms,analysis_hz,3); radius=win//2
        passes=np.zeros(len(df),bool)
        ranges=np.full(len(df),np.nan)
        for a,b in valid_runs:
            for i in range(a,b):
                lo=max(a,i-radius); hi=min(b,i+radius+1)
                local=st[lo:hi]
                local=local[np.isfinite(local)]
                if len(local)<3: continue
                r=robust_range(local); ranges[i]=r
                if r<=CURRENT_RANGE_ST: passes[i]=True
        regained = rejected & passes
        probe_rows.append({
            "song":song,"window_ms":wms,"window_steps":win,
            "currently_rejected_points":int(rejected.sum()),
            "rejected_points_passing_spread_at_window":int(regained.sum()),
            "fraction_of_rejected_points":float(regained.sum()/max(1,rejected.sum())),
            "local_spread_rejections_regained":int(np.sum(regained & (reason=="LOCAL_SPREAD_TOO_WIDE"))),
            "short_run_rejections_with_spread_pass":int(np.sum(regained & (reason=="STABLE_RUN_TOO_SHORT"))),
        })
    aperture=pd.DataFrame(probe_rows)

    # Duration sensitivity is computed from current local_stability_pass mask.
    local_pass = df.get("local_stability_pass", pd.Series(np.zeros(len(df)))).fillna(0).astype(int).to_numpy().astype(bool)
    duration_rows=[]
    for min_ms in PROBE_MIN_DURATIONS_MS:
        min_steps=max(1,int(math.ceil(min_ms*analysis_hz/1000.0-1e-9)))
        accepted=np.zeros(len(df),bool)
        for a,b in contiguous_runs(local_pass):
            if b-a>=min_steps: accepted[a:b]=True
        newly = trusted & accepted & ~stable
        duration_rows.append({
            "song":song,"min_stable_ms":min_ms,"min_steps":min_steps,
            "currently_rejected_points":int(rejected.sum()),
            "rejected_points_passing_duration_rule":int(newly.sum()),
            "fraction_of_rejected_points":float(newly.sum()/max(1,rejected.sum())),
            "short_run_rejections_regained":int(np.sum(newly & (reason=="STABLE_RUN_TOO_SHORT"))),
        })
    duration=pd.DataFrame(duration_rows)

    summary={
        "song":song,"analysis_hz":analysis_hz,"trusted_points":int(trusted.sum()),"stable_points":int((trusted&stable).sum()),
        "rejected_points":int(rejected.sum()),"rejected_runs":len(runs),
        "high_gt_like_runs":int((runs["triage"]=="HIGH_GT_LIKE").sum()) if len(runs) else 0,
        "medium_gt_like_runs":int((runs["triage"]=="MEDIUM_GT_LIKE").sum()) if len(runs) else 0,
        "short_coherent_runs":int((runs["triage"]=="SHORT_COHERENT").sum()) if len(runs) else 0,
        "local_spread_rejected_points":int(np.sum(rejected & (reason=="LOCAL_SPREAD_TOO_WIDE"))),
        "short_run_rejected_points":int(np.sum(rejected & (reason=="STABLE_RUN_TOO_SHORT"))),
    }
    return runs, aperture, duration, summary


def main():
    ap=argparse.ArgumentParser(description="Deep-dive rejected V2 target formation without changing decisions.")
    ap.add_argument("--input", required=True, help="Four-song target-formation ZIP or directory")
    ap.add_argument("--out", required=True, help="Output directory")
    args=ap.parse_args()
    source=Path(args.input); outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    data=load_inputs(source)
    all_runs=[]; all_ap=[]; all_dur=[]; summaries=[]
    for song,df in data.items():
        runs,aperture,duration,summary=analyze_song(song,df)
        runs.to_csv(outdir/f"{song}_rejected_runs.csv",index=False)
        aperture.to_csv(outdir/f"{song}_aperture_probe.csv",index=False)
        duration.to_csv(outdir/f"{song}_duration_probe.csv",index=False)
        if len(runs): all_runs.append(runs)
        all_ap.append(aperture); all_dur.append(duration); summaries.append(summary)
    runs_all=pd.concat(all_runs,ignore_index=True) if all_runs else pd.DataFrame()
    ap_all=pd.concat(all_ap,ignore_index=True); dur_all=pd.concat(all_dur,ignore_index=True)
    sum_df=pd.DataFrame(summaries)
    sum_df.to_csv(outdir/"summary_by_song.csv",index=False)
    runs_all.to_csv(outdir/"all_rejected_runs.csv",index=False)
    ap_all.to_csv(outdir/"aperture_probe_all.csv",index=False)
    dur_all.to_csv(outdir/"duration_probe_all.csv",index=False)
    if len(runs_all):
        priority=runs_all[runs_all["triage"].isin(["HIGH_GT_LIKE","MEDIUM_GT_LIKE","SHORT_COHERENT"])].copy()
        priority=priority.sort_values(["triage","duration_ms","dominant_note_fraction"],ascending=[True,False,False])
        priority.to_csv(outdir/"priority_gt_like_runs.csv",index=False)
    else:
        pd.DataFrame().to_csv(outdir/"priority_gt_like_runs.csv",index=False)
    manifest=pd.DataFrame([{
        "input":str(source),"current_stability_window_ms":CURRENT_WINDOW_MS,"current_stable_range_st":CURRENT_RANGE_ST,
        "current_min_stable_ms":CURRENT_MIN_STABLE_MS,"probe_windows_ms":";".join(map(str,PROBE_WINDOWS_MS)),
        "probe_min_durations_ms":";".join(map(str,PROBE_MIN_DURATIONS_MS)),
        "note":"Read-only audit. Counterfactual probes do not change V2 decisions."
    }])
    manifest.to_csv(outdir/"manifest.csv",index=False)
    print(sum_df.to_string(index=False))
    if len(runs_all):
        print("\nPriority GT-like rejected runs:",len(pd.read_csv(outdir/"priority_gt_like_runs.csv")))
    print("Reports:",outdir)

if __name__=="__main__":
    main()
