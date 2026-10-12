#!/usr/bin/env python3
"""
V22 adaptive-only shadow adjudicator.

Purpose
-------
Conservatively adjudicate integer-related candidate families using:
  1) acoustic admissibility first,
  2) distinctive spectral support as the strongest discriminator,
  3) temporal persistence / onset-decay evidence,
  4) unchanged interval penalty when an adjacent trusted F0 exists,
  5) optional singer-range prior only when independently established,
  6) abstention whenever evidence is insufficiently separated.

This script is diagnostic/shadow only. It does NOT modify production, MIDI,
or any previously accepted F0 sequence.

Expected inputs from prior audits:
  tests/temporal_integer_families_v21/
  tests/adaptive_waveform_v19/
  tests/full_timeline_v13/

It also reads the original WAVs in tests/ if present.

The scoring weights are intentionally CLI-configurable so the same evidence
can be replayed under different what-if combinations without remeasuring audio.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

try:
    from python_eckf.trajectory_resolver import vocal_transition_penalty
    CURVE_IMPORTED = True
except Exception:
    vocal_transition_penalty = None
    CURVE_IMPORTED = False

CASES = ["Ochiitai", "PREDESTINATI", "RATATA", "Trandafiri"]


def ffloat(v, default=None):
    try:
        if v is None or v == "":
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def parse_json_cell(v, default):
    if not v:
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def cents(a, b):
    if not a or not b or a <= 0 or b <= 0:
        return None
    return 1200.0 * math.log2(a / b)


def norm01(x, lo, hi):
    if x is None:
        return 0.0
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def robust_center(vals):
    vals = sorted(v for v in vals if v is not None and v > 0)
    if not vals:
        return None
    n = len(vals)
    if n % 2:
        return vals[n//2]
    return 0.5*(vals[n//2-1] + vals[n//2])


def percentile(vals, q):
    vals = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    p = (len(vals)-1)*q
    i = int(math.floor(p)); j = int(math.ceil(p))
    if i == j:
        return vals[i]
    a = p-i
    return vals[i]*(1-a)+vals[j]*a


def discover_case_files(root, case):
    candidates = {}
    patterns = {
        "v21": [f"{case}_*.csv"],
        "v19": [f"{case}_*.csv"],
        "v13": [f"{case}_*.csv"],
    }
    dirs = {
        "v21": root / "temporal_integer_families_v21",
        "v19": root / "adaptive_waveform_v19",
        "v13": root / "full_timeline_v13",
    }
    for key, d in dirs.items():
        files = []
        if d.exists():
            files = sorted(d.glob(f"{case}_*.csv"))
        candidates[key] = files
    return candidates


def load_v21_rows(files):
    rows = []
    for p in files:
        r = load_csv(p)
        # pair files should contain low/high candidate information.
        if r and any(k in r[0] for k in ["low_hz", "high_hz", "lower_hz", "higher_hz"]):
            rows.extend(r)
    return rows


def load_v13_timeline(files):
    best = []
    for p in files:
        r = load_csv(p)
        if len(r) > len(best):
            best = r
    return best


def load_v19_controls(files):
    rows = []
    for p in files:
        rows.extend(load_csv(p))
    return rows


def extract_pair(row):
    low = ffloat(row.get("low_hz") or row.get("lower_hz") or row.get("candidate_low_hz"))
    high = ffloat(row.get("high_hz") or row.get("higher_hz") or row.get("candidate_high_hz"))
    return low, high


def extract_time(row):
    for k in ["time_s", "time", "t_s", "center_time_s"]:
        x = ffloat(row.get(k))
        if x is not None:
            return x
    return None


def shifted_acf(row, which):
    # Supports either explicit columns or JSON measurements from V21.
    vals = []
    prefixes = [which, "lower" if which == "low" else "higher"]
    for pre in prefixes:
        for ms in [24,40,64]:
            for shift in ["m8", "0", "p8", "minus8", "plus8", "center"]:
                for k in [f"{pre}_acf_{ms}_{shift}", f"acf_{pre}_{ms}_{shift}"]:
                    v = ffloat(row.get(k))
                    if v is not None:
                        vals.append(v)
    blob = parse_json_cell(row.get("measurements_json"), [])
    if isinstance(blob, list):
        target = "low" if which == "low" else "high"
        for m in blob:
            if not isinstance(m, dict):
                continue
            label = str(m.get("candidate") or m.get("which") or "").lower()
            if target in label:
                v = ffloat(m.get("acf") or m.get("autocorrelation"))
                if v is not None:
                    vals.append(v)
    return vals


def exclusive_energy(row, which="low"):
    keys = []
    if which == "low":
        keys = ["low_exclusive_energy", "lower_exclusive_energy", "exclusive_energy_low",
                "low_exclusive_fraction", "lower_exclusive_fraction"]
    else:
        keys = ["high_exclusive_energy", "higher_exclusive_energy", "exclusive_energy_high",
                "high_exclusive_fraction", "higher_exclusive_fraction"]
    vals = [ffloat(row.get(k)) for k in keys]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def temporal_pattern(row):
    return str(row.get("temporal_pattern") or row.get("pattern") or "").strip()


def temporal_score(row, candidate):
    patt = temporal_pattern(row).lower()
    # Conservative support map. Positive means evidence favors candidate.
    if candidate == "high":
        if "higher_only" in patt or "high_only" in patt:
            return 1.0
        if "lower_only" in patt or "low_only" in patt:
            return -1.0
        if "both_integer_periods_shift_persistent" in patt or "both_persistent" in patt:
            return 0.0
    else:
        if "lower_only" in patt or "low_only" in patt:
            return 1.0
        if "higher_only" in patt or "high_only" in patt:
            return -1.0
        if "both_integer_periods_shift_persistent" in patt or "both_persistent" in patt:
            return 0.0
    return 0.0


def spectral_score(row, candidate, cfg):
    low_acf = robust_center(shifted_acf(row, "low"))
    high_acf = robust_center(shifted_acf(row, "high"))
    low_excl = exclusive_energy(row, "low")
    high_excl = exclusive_energy(row, "high")

    # Strongest criterion: distinctive support. For the lower candidate, exclusive
    # low-frequency energy matters heavily. For the higher candidate, a clear ACF
    # advantage combined with weak lower-exclusive energy is favorable.
    if candidate == "low":
        excl = norm01(low_excl, cfg.excl_low_floor, cfg.excl_low_good)
        acf_adv = 0.0
        if low_acf is not None and high_acf is not None:
            acf_adv = max(-1.0, min(1.0, (low_acf - high_acf) / cfg.acf_adv_scale))
        return cfg.spectral_exclusive_mix*excl + (1-cfg.spectral_exclusive_mix)*acf_adv
    else:
        acf_adv = 0.0
        if low_acf is not None and high_acf is not None:
            acf_adv = max(-1.0, min(1.0, (high_acf - low_acf) / cfg.acf_adv_scale))
        weak_low_excl_bonus = 0.0
        if low_excl is not None:
            weak_low_excl_bonus = 1.0 - norm01(low_excl, cfg.excl_low_floor, cfg.excl_low_good)
        # If explicit high-exclusive energy exists, include it; otherwise don't invent it.
        high_excl_term = norm01(high_excl, cfg.excl_high_floor, cfg.excl_high_good) if high_excl is not None else 0.0
        return (0.55*acf_adv + 0.35*weak_low_excl_bonus + 0.10*high_excl_term)


def admissibility(row, candidate, cfg):
    vals = shifted_acf(row, candidate)
    if not vals:
        return False, "no_shifted_acf"
    supported = sum(v >= cfg.min_acf for v in vals)
    if supported < cfg.min_supported_measurements:
        return False, f"acf_support_{supported}"
    return True, f"acf_support_{supported}"


def build_trusted_timeline(v13_rows, cfg):
    # Only use rows already marked supported and with a unique frequency if present.
    # This does not create new F0s; it only supplies adjacent trusted references.
    out = {}
    for r in v13_rows:
        t = extract_time(r)
        if t is None:
            continue
        status = str(r.get("diagnostic_settlement") or r.get("status") or r.get("classification") or "").lower()
        hz = None
        for k in ["selected_hz", "measured_hz", "supported_hz", "adaptive_hz", "f0_hz"]:
            hz = ffloat(r.get(k))
            if hz is not None:
                break
        if hz is None:
            continue
        if ("supported" in status or "settled" in status) and "unresolved" not in status:
            out[round(t, 6)] = hz
    return out


def adjacent_reference(trusted, t, max_gap_s):
    if not trusted:
        return None, None
    keys = sorted(trusted.keys())
    # nearest strictly earlier trusted observation
    prev = None
    for k in keys:
        if k < t:
            prev = k
        else:
            break
    if prev is None or t-prev > max_gap_s:
        return None, None
    return prev, trusted[prev]


def interval_component(prev_hz, cand_hz, dt_ms, cfg):
    if prev_hz is None or cand_hz is None or vocal_transition_penalty is None:
        return None
    try:
        p = float(vocal_transition_penalty(prev_hz, cand_hz, dt_ms))
        # Convert penalty into a bounded negative contribution.
        return -min(1.0, max(0.0, p / cfg.penalty_scale))
    except Exception:
        return None


def derive_range(v19_rows, v13_rows, cfg):
    # Diagnostic only. We require many independently settled/supported anchors.
    anchors = []
    for r in v13_rows:
        status = str(r.get("diagnostic_settlement") or r.get("status") or r.get("classification") or "").lower()
        if "supported" not in status and "settled" not in status:
            continue
        if "unresolved" in status:
            continue
        hz = None
        for k in ["selected_hz", "measured_hz", "supported_hz", "adaptive_hz", "f0_hz"]:
            hz = ffloat(r.get(k))
            if hz:
                break
        if hz:
            anchors.append(hz)
    if len(anchors) < cfg.min_range_anchors:
        return {"available": False, "anchors": len(anchors)}
    lo = percentile(anchors, cfg.range_lo_q)
    hi = percentile(anchors, cfg.range_hi_q)
    return {"available": True, "anchors": len(anchors), "low_hz": lo, "high_hz": hi}


def range_component(cand_hz, rinfo, cfg):
    if not rinfo.get("available") or cand_hz is None:
        return None
    lo = rinfo["low_hz"]; hi = rinfo["high_hz"]
    if lo <= cand_hz <= hi:
        return 0.0
    # soft penalty based on cents outside provisional range
    if cand_hz < lo:
        d = abs(cents(cand_hz, lo) or 0.0)
    else:
        d = abs(cents(cand_hz, hi) or 0.0)
    return -min(1.0, d / cfg.range_penalty_full_cents)


def adjudicate(row, trusted, rinfo, cfg):
    t = extract_time(row)
    low, high = extract_pair(row)
    if t is None or low is None or high is None:
        return None

    low_ok, low_reason = admissibility(row, "low", cfg)
    high_ok, high_reason = admissibility(row, "high", cfg)
    if not low_ok and not high_ok:
        outcome = "abstain_no_admissible_candidate"
    elif low_ok and not high_ok:
        outcome = "provisional_low_only"
    elif high_ok and not low_ok:
        outcome = "provisional_high_only"
    else:
        outcome = "compare"

    prev_t, prev_hz = adjacent_reference(trusted, t, cfg.max_adjacent_gap_s)
    dt_ms = (t-prev_t)*1000.0 if prev_t is not None else None

    scores = {}
    details = {}
    for cand_name, hz, ok in [("low", low, low_ok), ("high", high, high_ok)]:
        if not ok:
            scores[cand_name] = None
            details[cand_name] = {"admissible": False}
            continue
        spec = spectral_score(row, cand_name, cfg)
        temp = temporal_score(row, cand_name)
        inter = interval_component(prev_hz, hz, dt_ms, cfg) if prev_hz is not None else None
        rng = range_component(hz, rinfo, cfg)
        score = cfg.w_spectral*spec + cfg.w_temporal*temp
        if inter is not None:
            score += cfg.w_interval*inter
        if rng is not None:
            score += cfg.w_range*rng
        scores[cand_name] = score
        details[cand_name] = {
            "admissible": True,
            "spectral_component": spec,
            "temporal_component": temp,
            "interval_component": inter,
            "range_component": rng,
            "score": score,
        }

    if outcome == "compare":
        diff = scores["high"] - scores["low"]
        if abs(diff) < cfg.min_score_margin:
            outcome = "abstain_insufficient_separation"
        elif diff > 0:
            outcome = "provisional_high"
        else:
            outcome = "provisional_low"
    elif outcome == "provisional_low_only":
        if scores["low"] is None or scores["low"] < cfg.min_absolute_score:
            outcome = "abstain_low_only_weak"
    elif outcome == "provisional_high_only":
        if scores["high"] is None or scores["high"] < cfg.min_absolute_score:
            outcome = "abstain_high_only_weak"

    low_acf = robust_center(shifted_acf(row, "low"))
    high_acf = robust_center(shifted_acf(row, "high"))

    return {
        "time_s": t,
        "low_hz": low,
        "high_hz": high,
        "ratio": high/low if low > 0 else None,
        "low_admissible": low_ok,
        "high_admissible": high_ok,
        "low_admissibility_reason": low_reason,
        "high_admissibility_reason": high_reason,
        "low_acf_median": low_acf,
        "high_acf_median": high_acf,
        "low_exclusive_energy": exclusive_energy(row, "low"),
        "temporal_pattern": temporal_pattern(row),
        "prev_trusted_time_s": prev_t,
        "prev_trusted_hz": prev_hz,
        "dt_ms": dt_ms,
        "low_score": scores.get("low"),
        "high_score": scores.get("high"),
        "outcome": outcome,
        "details_json": json.dumps(details, sort_keys=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all"] + CASES)
    ap.add_argument("--tests-root", default="tests")
    ap.add_argument("--out", default="tests/shadow_adjudicator_v22")

    # Tunable what-if weights. Defaults intentionally conservative.
    ap.add_argument("--w-spectral", type=float, default=0.60)
    ap.add_argument("--w-temporal", type=float, default=0.20)
    ap.add_argument("--w-interval", type=float, default=0.15)
    ap.add_argument("--w-range", type=float, default=0.05)
    ap.add_argument("--min-score-margin", type=float, default=0.20)
    ap.add_argument("--min-absolute-score", type=float, default=-0.10)

    # Acoustic screening.
    ap.add_argument("--min-acf", type=float, default=0.72)
    ap.add_argument("--min-supported-measurements", type=int, default=2)
    ap.add_argument("--acf-adv-scale", type=float, default=0.12)
    ap.add_argument("--spectral-exclusive-mix", type=float, default=0.70)
    ap.add_argument("--excl-low-floor", type=float, default=0.002)
    ap.add_argument("--excl-low-good", type=float, default=0.040)
    ap.add_argument("--excl-high-floor", type=float, default=0.002)
    ap.add_argument("--excl-high-good", type=float, default=0.040)

    # Interval / range contextual evidence.
    ap.add_argument("--penalty-scale", type=float, default=2.5)
    ap.add_argument("--max-adjacent-gap-s", type=float, default=0.015)
    ap.add_argument("--min-range-anchors", type=int, default=40)
    ap.add_argument("--range-lo-q", type=float, default=0.02)
    ap.add_argument("--range-hi-q", type=float, default=0.98)
    ap.add_argument("--range-penalty-full-cents", type=float, default=700.0)

    args = ap.parse_args()
    root = Path(args.tests_root)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case == "all" else [args.case]

    manifest = {
        "schema": "v22_adaptive_shadow_adjudicator",
        "production_used": False,
        "production_modified": False,
        "midi_used": False,
        "curve_imported_unchanged": CURVE_IMPORTED,
        "spectral_is_strongest_weight": True,
        "range_prior_is_soft_and_optional": True,
        "abstention_supported": True,
        "weights": {k:getattr(args,k) for k in ["w_spectral","w_temporal","w_interval","w_range"]},
        "thresholds": {
            "min_score_margin": args.min_score_margin,
            "min_absolute_score": args.min_absolute_score,
            "min_acf": args.min_acf,
            "min_supported_measurements": args.min_supported_measurements,
        },
        "cases": {},
    }

    for case in cases:
        files = discover_case_files(root, case)
        v21 = load_v21_rows(files["v21"])
        v13 = load_v13_timeline(files["v13"])
        v19 = load_v19_controls(files["v19"])
        trusted = build_trusted_timeline(v13, args)
        rinfo = derive_range(v19, v13, args)

        results = []
        counts = defaultdict(int)
        seen_obs = defaultdict(list)
        for row in v21:
            res = adjudicate(row, trusted, rinfo, args)
            if res is None:
                continue
            res["case"] = case
            results.append(res)
            counts[res["outcome"]] += 1
            seen_obs[round(res["time_s"], 6)].append(res)

        # Conservative observation-level consolidation: only call a provisional
        # winner if all decisive pair records at that time agree on direction.
        obs_rows = []
        obs_counts = defaultdict(int)
        for t, rr in sorted(seen_obs.items()):
            dirs = []
            for r in rr:
                if r["outcome"] in ("provisional_high", "provisional_high_only"):
                    dirs.append("high")
                elif r["outcome"] in ("provisional_low", "provisional_low_only"):
                    dirs.append("low")
            if dirs and all(d == dirs[0] for d in dirs):
                status = f"provisional_{dirs[0]}_consistent_across_pairs"
            elif dirs:
                status = "abstain_pair_disagreement"
            else:
                status = "abstain_no_decisive_pair"
            obs_counts[status] += 1
            obs_rows.append({
                "case": case,
                "time_s": t,
                "pair_records": len(rr),
                "decisive_pair_records": len(dirs),
                "observation_outcome": status,
            })

        write_csv(outdir / f"{case}_pair_decisions.csv", results)
        write_csv(outdir / f"{case}_observation_summary.csv", obs_rows)
        manifest["cases"][case] = {
            "v21_pair_records": len(v21),
            "pair_outcomes": dict(counts),
            "observations_with_pairs": len(seen_obs),
            "observation_outcomes": dict(obs_counts),
            "trusted_adjacent_reference_count": len(trusted),
            "provisional_range": rinfo,
        }

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
