#!/usr/bin/env python3
"""
Gain-invariance audit for python_eckf V2.

Compares two complete CLI output bundles produced from the same audio content at
(two ideally linearly gain-scaled) levels.  It does NOT alter tracker decisions.

Expected bundle for --reference / --test:
  PREFIX
  PREFIX.frames.csv
  PREFIX.trajectory.csv
  PREFIX.target_formation.csv

The audit aligns by frame_index for frame evidence and by sample for 100 Hz
tracker output, then reports the earliest pipeline layer at which each aligned
observation differs.

Standard-library only.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

EPS = 1e-12


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _s(row: dict, key: str) -> str:
    v = row.get(key, "")
    return "" if v is None else str(v)


def _b(row: dict, key: str) -> Optional[bool]:
    v = _s(row, key).strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    return None


def cents(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return 1200.0 * math.log2(b / a)


def nearest_midi(freq: Optional[float]) -> Optional[int]:
    if freq is None or freq <= 0.0:
        return None
    return int(math.floor(69.0 + 12.0 * math.log2(freq / 440.0) + 0.5))


def midi_name(m: Optional[int]) -> str:
    if m is None:
        return ""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


def _neq_text(a: dict, b: dict, key: str) -> bool:
    return _s(a, key) != _s(b, key)


def _neq_bool(a: dict, b: dict, key: str) -> bool:
    return _b(a, key) != _b(b, key)


def _neq_num(a: dict, b: dict, key: str, abs_tol: float = 0.0) -> bool:
    x, y = _f(a, key), _f(b, key)
    if x is None or y is None:
        return x != y
    return abs(x - y) > abs_tol


def _index(rows: List[dict], key: str) -> Dict[int, dict]:
    out = {}
    for r in rows:
        try:
            k = int(round(float(r[key])))
        except Exception:
            continue
        out[k] = r
    return out


def _write(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def classify_frame_stage(r: dict, t: dict, cent_threshold: float) -> str:
    # Ordered deliberately: the label means earliest *observable* divergence.
    for k in ("energy_silence", "flatness_silence", "original_silence"):
        if _neq_bool(r, t, k):
            return "01_SILENCE_CLASSIFICATION"
    for k in ("periodicity_voiced", "periodicity_reason"):
        if (_neq_bool(r, t, k) if k == "periodicity_voiced" else _neq_text(r, t, k)):
            return "02_PERIODICITY_DECISION"
    for k in ("periodicity_acf_frequency_hz", "periodicity_cmndf_frequency_hz"):
        d = cents(_f(r, k), _f(t, k))
        if d is not None and abs(d) >= cent_threshold:
            return "03_PERIODICITY_FREQUENCY"
    for k in ("initialization_source", "lookahead_frames", "lookahead_reason"):
        if _neq_text(r, t, k):
            return "04_INITIALIZATION_PATH"
    d = cents(_f(r, "initialization_frequency_hz"), _f(t, "initialization_frequency_hz"))
    if d is not None and abs(d) >= cent_threshold:
        return "05_INITIALIZATION_FREQUENCY"
    if _neq_text(r, t, "frame_acoustic_state_name") or _neq_text(r, t, "frame_decision"):
        return "06_FRAME_STATE"
    return "00_NO_MATERIAL_FRAME_DIVERGENCE"


def classify_sample_stage(r: dict, t: dict, cent_threshold: float) -> str:
    # Same ordering principle for the dense tracker output.
    for k in ("energy_silence", "flatness_silence", "original_silence"):
        if _neq_bool(r, t, k):
            return "01_SILENCE_CLASSIFICATION"
    for k in ("periodicity_voiced", "periodicity_reason"):
        if (_neq_bool(r, t, k) if k == "periodicity_voiced" else _neq_text(r, t, k)):
            return "02_PERIODICITY_DECISION"
    for k in ("periodicity_acf_frequency_hz", "periodicity_cmndf_frequency_hz"):
        d = cents(_f(r, k), _f(t, k))
        if d is not None and abs(d) >= cent_threshold:
            return "03_PERIODICITY_FREQUENCY"
    if _neq_text(r, t, "initialization_source") or _neq_text(r, t, "lookahead_reason"):
        return "04_INITIALIZATION_PATH"
    d = cents(_f(r, "initialization_frequency_hz"), _f(t, "initialization_frequency_hz"))
    if d is not None and abs(d) >= cent_threshold:
        return "05_INITIALIZATION_FREQUENCY"
    d = cents(_f(r, "f0_hz"), _f(t, "f0_hz"))
    if d is not None and abs(d) >= cent_threshold:
        return "06_RAW_ECKF_F0"
    for k in ("first_pass_valid", "corrected_valid", "first_pass_reason", "validity_reason", "correction_reason"):
        if (_neq_bool(r, t, k) if k in {"first_pass_valid", "corrected_valid"} else _neq_text(r, t, k)):
            return "07_VALIDITY_OR_RESCUE"
    d = cents(_f(r, "clean_f0_hz"), _f(t, "clean_f0_hz"))
    if d is not None and abs(d) >= cent_threshold:
        return "08_CLEAN_F0"
    for k in ("trajectory_region_kind", "trajectory_stable"):
        if _neq_text(r, t, k):
            return "09_TRAJECTORY"
    d = None
    x, y = _f(r, "structural_pitch_st"), _f(t, "structural_pitch_st")
    if x is not None and y is not None:
        d = 100.0 * (y - x)
    if d is not None and abs(d) >= cent_threshold:
        return "10_STRUCTURAL_PITCH"
    return "00_NO_MATERIAL_SAMPLE_DIVERGENCE"


def percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    q = (len(s) - 1) * p
    lo = int(math.floor(q)); hi = int(math.ceil(q))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - q) + s[hi] * (q - lo)


def fmt(x: Optional[float], n: int = 4) -> str:
    return "" if x is None else f"{x:.{n}f}"


def compare_frame_bundles(ref_prefix: Path, test_prefix: Path, out: Path, cent_threshold: float) -> Counter:
    rr = _read_csv(Path(str(ref_prefix) + ".frames.csv"))
    tt = _read_csv(Path(str(test_prefix) + ".frames.csv"))
    R, T = _index(rr, "frame_index"), _index(tt, "frame_index")
    common = sorted(set(R) & set(T))
    rows = []
    counts = Counter()
    for k in common:
        r, t = R[k], T[k]
        stage = classify_frame_stage(r, t, cent_threshold)
        counts[stage] += 1
        acf_c = cents(_f(r,"periodicity_acf_frequency_hz"), _f(t,"periodicity_acf_frequency_hz"))
        cm_c = cents(_f(r,"periodicity_cmndf_frequency_hz"), _f(t,"periodicity_cmndf_frequency_hz"))
        init_c = cents(_f(r,"initialization_frequency_hz"), _f(t,"initialization_frequency_hz"))
        rows.append({
            "frame_index": k,
            "time_s": _s(r,"start_time_s"),
            "earliest_divergence_stage": stage,
            "ref_energy": _s(r,"energy_on_matlab_scale"), "test_energy": _s(t,"energy_on_matlab_scale"),
            "ref_energy_silence": _s(r,"energy_silence"), "test_energy_silence": _s(t,"energy_silence"),
            "ref_periodicity_voiced": _s(r,"periodicity_voiced"), "test_periodicity_voiced": _s(t,"periodicity_voiced"),
            "ref_periodicity_reason": _s(r,"periodicity_reason"), "test_periodicity_reason": _s(t,"periodicity_reason"),
            "acf_delta_cents": fmt(acf_c), "cmndf_delta_cents": fmt(cm_c),
            "ref_init_source": _s(r,"initialization_source"), "test_init_source": _s(t,"initialization_source"),
            "init_delta_cents": fmt(init_c),
            "ref_frame_state": _s(r,"frame_acoustic_state_name"), "test_frame_state": _s(t,"frame_acoustic_state_name"),
            "ref_frame_decision": _s(r,"frame_decision"), "test_frame_decision": _s(t,"frame_decision"),
        })
    fields = list(rows[0].keys()) if rows else []
    if rows:
        _write(out / "frame_stage_trace.csv", rows, fields)
    return counts


def compare_sample_bundles(ref_prefix: Path, test_prefix: Path, out: Path, cent_threshold: float) -> Tuple[Counter, dict]:
    rr = _read_csv(ref_prefix)
    tt = _read_csv(test_prefix)
    R, T = _index(rr, "sample"), _index(tt, "sample")
    common = sorted(set(R) & set(T))
    rows = []
    counts = Counter()
    raw_diffs, clean_diffs = [], []
    raw_note_flips = clean_note_flips = 0
    corrected_valid_flips = 0
    materially_different = 0

    for k in common:
        r, t = R[k], T[k]
        stage = classify_sample_stage(r, t, cent_threshold)
        counts[stage] += 1
        if stage != "00_NO_MATERIAL_SAMPLE_DIVERGENCE":
            materially_different += 1
        raw_c = cents(_f(r,"f0_hz"), _f(t,"f0_hz"))
        clean_c = cents(_f(r,"clean_f0_hz"), _f(t,"clean_f0_hz"))
        if raw_c is not None: raw_diffs.append(abs(raw_c))
        if clean_c is not None: clean_diffs.append(abs(clean_c))
        rm, tm = nearest_midi(_f(r,"f0_hz")), nearest_midi(_f(t,"f0_hz"))
        rcm, tcm = nearest_midi(_f(r,"clean_f0_hz")), nearest_midi(_f(t,"clean_f0_hz"))
        raw_flip = rm is not None and tm is not None and rm != tm
        clean_flip = rcm is not None and tcm is not None and rcm != tcm
        raw_note_flips += int(raw_flip)
        clean_note_flips += int(clean_flip)
        cv_flip = _b(r,"corrected_valid") != _b(t,"corrected_valid")
        corrected_valid_flips += int(cv_flip)
        rows.append({
            "sample": k,
            "time_s": _s(r,"time_s"),
            "earliest_divergence_stage": stage,
            "raw_f0_ref_hz": _s(r,"f0_hz"), "raw_f0_test_hz": _s(t,"f0_hz"),
            "raw_delta_cents": fmt(raw_c),
            "raw_note_ref": midi_name(rm), "raw_note_test": midi_name(tm), "raw_nearest_note_flip": int(raw_flip),
            "corrected_valid_ref": _s(r,"corrected_valid"), "corrected_valid_test": _s(t,"corrected_valid"),
            "corrected_valid_flip": int(cv_flip),
            "clean_f0_ref_hz": _s(r,"clean_f0_hz"), "clean_f0_test_hz": _s(t,"clean_f0_hz"),
            "clean_delta_cents": fmt(clean_c),
            "clean_note_ref": midi_name(rcm), "clean_note_test": midi_name(tcm), "clean_nearest_note_flip": int(clean_flip),
            "periodicity_acf_ref_hz": _s(r,"periodicity_acf_frequency_hz"), "periodicity_acf_test_hz": _s(t,"periodicity_acf_frequency_hz"),
            "periodicity_cmndf_ref_hz": _s(r,"periodicity_cmndf_frequency_hz"), "periodicity_cmndf_test_hz": _s(t,"periodicity_cmndf_frequency_hz"),
            "init_ref_hz": _s(r,"initialization_frequency_hz"), "init_test_hz": _s(t,"initialization_frequency_hz"),
            "init_ref_source": _s(r,"initialization_source"), "init_test_source": _s(t,"initialization_source"),
            "validity_ref": _s(r,"validity_reason"), "validity_test": _s(t,"validity_reason"),
            "correction_ref": _s(r,"correction_reason"), "correction_test": _s(t,"correction_reason"),
            "trajectory_ref": _s(r,"trajectory_region_kind"), "trajectory_test": _s(t,"trajectory_region_kind"),
        })
    if rows:
        _write(out / "sample_stage_trace.csv", rows, list(rows[0].keys()))
        risky = [x for x in rows if x["clean_nearest_note_flip"] or x["raw_nearest_note_flip"] or x["corrected_valid_flip"]]
        _write(out / "note_identity_and_validity_flips.csv", risky, list(rows[0].keys()))
        diverged = [x for x in rows if x["earliest_divergence_stage"] != "00_NO_MATERIAL_SAMPLE_DIVERGENCE"]
        _write(out / "material_divergences.csv", diverged, list(rows[0].keys()))

    stats = {
        "aligned_samples": len(common),
        "materially_different_samples": materially_different,
        "corrected_valid_flips": corrected_valid_flips,
        "raw_nearest_note_flips": raw_note_flips,
        "clean_nearest_note_flips": clean_note_flips,
        "raw_cents_median": percentile(raw_diffs, .5),
        "raw_cents_p95": percentile(raw_diffs, .95),
        "raw_cents_max": max(raw_diffs) if raw_diffs else None,
        "clean_cents_median": percentile(clean_diffs, .5),
        "clean_cents_p95": percentile(clean_diffs, .95),
        "clean_cents_max": max(clean_diffs) if clean_diffs else None,
    }
    return counts, stats


def compare_target_formation(ref_prefix: Path, test_prefix: Path, out: Path) -> Counter:
    rp = Path(str(ref_prefix) + ".target_formation.csv")
    tp = Path(str(test_prefix) + ".target_formation.csv")
    if not rp.exists() or not tp.exists():
        return Counter()
    R, T = _index(_read_csv(rp), "sample"), _index(_read_csv(tp), "sample")
    rows = []
    counts = Counter()
    for k in sorted(set(R)&set(T)):
        r,t = R[k],T[k]
        changed = []
        for fld in ("local_stability_pass","va_first_passing_aperture_ms","va_largest_passing_aperture_ms","min_duration_pass","final_stable","target_formation_reason"):
            if _s(r,fld) != _s(t,fld): changed.append(fld)
        if changed:
            counts["changed"] += 1
            rows.append({"sample":k,"time_s":_s(r,"time_s"),"changed_fields":";".join(changed),
                         "ref_final_stable":_s(r,"final_stable"),"test_final_stable":_s(t,"final_stable"),
                         "ref_reason":_s(r,"target_formation_reason"),"test_reason":_s(t,"target_formation_reason"),
                         "ref_first_aperture":_s(r,"va_first_passing_aperture_ms"),"test_first_aperture":_s(t,"va_first_passing_aperture_ms"),
                         "ref_largest_aperture":_s(r,"va_largest_passing_aperture_ms"),"test_largest_aperture":_s(t,"va_largest_passing_aperture_ms")})
        else:
            counts["unchanged"] += 1
    if rows: _write(out/"target_formation_changes.csv", rows, list(rows[0].keys()))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit gain invariance between two python_eckf V2 runs")
    ap.add_argument("--reference", required=True, help="Reference main CSV prefix/path")
    ap.add_argument("--test", required=True, help="Gain-scaled test main CSV prefix/path")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--cent-threshold", type=float, default=1.0,
                    help="Minimum F0 difference called material for stage attribution (default: 1 cent)")
    args = ap.parse_args()

    ref, test, out = Path(args.reference), Path(args.test), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in (ref, test, Path(str(ref)+".frames.csv"), Path(str(test)+".frames.csv")):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    frame_counts = compare_frame_bundles(ref,test,out,args.cent_threshold)
    sample_counts, stats = compare_sample_bundles(ref,test,out,args.cent_threshold)
    target_counts = compare_target_formation(ref,test,out)

    lines = []
    lines.append("GAIN INVARIANCE AUDIT V1")
    lines.append(f"reference: {ref}")
    lines.append(f"test:      {test}")
    lines.append(f"material F0 threshold for stage attribution: {args.cent_threshold:g} cent(s)")
    lines.append("")
    lines.append("SAMPLE SUMMARY")
    for k,v in stats.items():
        lines.append(f"  {k}: {fmt(v) if isinstance(v,float) else v}")
    lines.append("")
    lines.append("EARLIEST OBSERVABLE FRAME DIVERGENCE")
    for k,v in sorted(frame_counts.items()): lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("EARLIEST OBSERVABLE 100-HZ SAMPLE DIVERGENCE")
    for k,v in sorted(sample_counts.items()): lines.append(f"  {k}: {v}")
    if target_counts:
        lines.append("")
        lines.append("TARGET FORMATION")
        for k,v in sorted(target_counts.items()): lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Important: 'earliest' means earliest stage observable in the exported evidence, not proof of the internal causal instruction.")
    text = "\n".join(lines) + "\n"
    (out/"summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Reports: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
