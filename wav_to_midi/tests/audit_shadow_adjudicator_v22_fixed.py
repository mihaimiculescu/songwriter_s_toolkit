#!/usr/bin/env python3
"""
V22 corrected adaptive-only shadow adjudicator.

Repairs the cross-version contracts against the actual V13 and V21 outputs.
Diagnostic/shadow only: production, MIDI, accepted F0s, and WAVs are not modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

CURVE_IMPORT_ERROR = None
try:
    from python_eckf.trajectory_resolver import vocal_transition_penalty
    CURVE_IMPORTED = True
except Exception as exc:  # diagnostics must remain runnable
    vocal_transition_penalty = None
    CURVE_IMPORTED = False
    CURVE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

CASES = ["Ochiitai", "PREDESTINATI", "RATATA", "Trandafiri"]


def ffloat(v, default=None):
    try:
        if v is None or v == "":
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_json_cell(v, default):
    if not v:
        return default
    try:
        return json.loads(v)
    except (TypeError, json.JSONDecodeError):
        return default


def percentile(values, q):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    p = (len(vals) - 1) * q
    i, j = int(math.floor(p)), int(math.ceil(p))
    if i == j:
        return vals[i]
    a = p - i
    return vals[i] * (1.0 - a) + vals[j] * a


def median(values):
    return percentile(values, 0.5)


def cents(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return 1200.0 * math.log2(a / b)


def norm01(x, lo, hi):
    if x is None or hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def discover_case_files(root, case):
    dirs = {
        "v21": root / "temporal_integer_families_v21",
        "v19": root / "adaptive_waveform_v19",
        "v13": root / "full_timeline_v13",
    }
    return {key: sorted(d.glob(f"{case}_*.csv")) if d.exists() else [] for key, d in dirs.items()}


def load_pair_rows(files):
    out = []
    for path in files:
        rows = load_csv(path)
        if rows and "low_hz" in rows[0] and "high_hz" in rows[0]:
            out.extend(rows)
    return out


def load_timeline(files):
    candidates = []
    for path in files:
        rows = load_csv(path)
        if rows and {"time_s", "evidence_status", "selected_measured_hz"}.issubset(rows[0]):
            candidates.append((path, rows))
    if not candidates:
        return []
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0][1]


def extract_time(row):
    for key in ("time_s", "time", "t_s", "center_time_s"):
        value = ffloat(row.get(key))
        if value is not None:
            return value
    return None


def extract_pair(row):
    return ffloat(row.get("low_hz")), ffloat(row.get("high_hz"))


def v21_evidence(row):
    blob = parse_json_cell(row.get("evidence_json"), {})
    return blob if isinstance(blob, dict) else {}


def candidate_measurements(row, which):
    """Return only V21 measurements explicitly marked testable."""
    evidence = v21_evidence(row)
    windows = evidence.get("shift_windows", {})
    out = []
    if isinstance(windows, dict):
        for window_name, payload in windows.items():
            if not isinstance(payload, dict):
                continue
            candidate = payload.get(which, {})
            if not isinstance(candidate, dict) or not as_bool(candidate.get("testable")):
                continue
            acf = ffloat(candidate.get("acf"))
            if acf is not None:
                out.append({"window": window_name, "acf": acf,
                            "measured_hz": ffloat(candidate.get("measured_hz"))})
    return out


def low_exclusive_measurements(row):
    evidence = v21_evidence(row)
    windows = evidence.get("shift_windows", {})
    vals = []
    if isinstance(windows, dict):
        for payload in windows.values():
            if not isinstance(payload, dict):
                continue
            spec = payload.get("spectral", {})
            if not isinstance(spec, dict) or not as_bool(spec.get("testable")):
                continue
            value = ffloat(spec.get("low_exclusive_power_fraction"))
            if value is not None:
                vals.append(value)
    return vals


def temporal_score(row, candidate):
    pattern = str(row.get("temporal_pattern") or "").strip().lower()
    if pattern == "higher_period_persistent_lower_not":
        return 1.0 if candidate == "high" else -1.0
    if pattern == "lower_period_persistent_higher_not":
        return 1.0 if candidate == "low" else -1.0
    # Both-persistent, mixed/weak, and insufficient remain neutral.
    return 0.0


def acoustic_admissibility(row, candidate, cfg):
    measurements = candidate_measurements(row, candidate)
    supported = sum(m["acf"] >= cfg.min_acf for m in measurements)
    if not measurements:
        return False, "no_testable_shifted_acf", 0, 0
    if supported < cfg.min_supported_measurements:
        return False, f"acf_support_{supported}_of_{len(measurements)}", len(measurements), supported
    return True, f"acf_support_{supported}_of_{len(measurements)}", len(measurements), supported


def spectral_components(row, cfg):
    low_acfs = [m["acf"] for m in candidate_measurements(row, "low")]
    high_acfs = [m["acf"] for m in candidate_measurements(row, "high")]
    low_med, high_med = median(low_acfs), median(high_acfs)
    advantage = 0.0
    if low_med is not None and high_med is not None:
        advantage = max(-1.0, min(1.0, (high_med - low_med) / cfg.acf_adv_scale))

    exclusive = low_exclusive_measurements(row)
    excl_center = median(exclusive)
    low_excl_support = norm01(excl_center, cfg.excl_low_floor, cfg.excl_low_good)
    weak_low_excl = 1.0 - low_excl_support if excl_center is not None else 0.0

    # Positive advantage means high has stronger ACF. Low-exclusive energy is
    # evidence unique to a genuine lower fundamental.
    low_score = cfg.spectral_exclusive_mix * low_excl_support + (1.0 - cfg.spectral_exclusive_mix) * (-advantage)
    high_score = cfg.spectral_exclusive_mix * weak_low_excl + (1.0 - cfg.spectral_exclusive_mix) * advantage
    return {
        "low": max(-1.0, min(1.0, low_score)),
        "high": max(-1.0, min(1.0, high_score)),
        "low_acf_median": low_med,
        "high_acf_median": high_med,
        "low_exclusive_median": excl_center,
        "low_exclusive_min": min(exclusive) if exclusive else None,
        "low_exclusive_max": max(exclusive) if exclusive else None,
        "exclusive_measurement_count": len(exclusive),
    }


def build_supported_timeline(v13_rows):
    """Use the V13 contract exactly; these are supported, source-unverified references."""
    refs = []
    for row in v13_rows:
        t = extract_time(row)
        hz = ffloat(row.get("selected_measured_hz"))
        if t is None or hz is None:
            continue
        if row.get("evidence_status") != "acoustic_period_supported_source_unverified":
            continue
        if not as_bool(row.get("diagnostic_settled")) or not as_bool(row.get("usable_as_next_reference")):
            continue
        refs.append((t, hz))
    refs.sort()
    return refs


def adjacent_references(refs, t, max_gap_s):
    if not refs:
        return None, None
    times = [x[0] for x in refs]
    i = bisect_left(times, t)
    prev = refs[i - 1] if i > 0 and 0 < t - refs[i - 1][0] <= max_gap_s else None
    nxt = refs[i] if i < len(refs) and 0 < refs[i][0] - t <= max_gap_s else None
    # Exact-time references are deliberately excluded to avoid circular reuse.
    return prev, nxt


def transition_component(candidate_hz, reference, t, cfg):
    if reference is None or vocal_transition_penalty is None:
        return None
    rt, rhz = reference
    interval_st = abs(cents(candidate_hz, rhz) or 0.0) / 100.0
    available_ms = abs(t - rt) * 1000.0
    try:
        penalty = float(vocal_transition_penalty(interval_st, available_ms))
        return -min(1.0, max(0.0, penalty / cfg.penalty_scale))
    except Exception:
        return None


def derive_range(refs, cfg):
    """Soft observed-performance region from independently supported V13 rows."""
    hz = [value for _, value in refs]
    if len(hz) < cfg.min_range_anchors:
        return {"available": False, "anchors": len(hz), "kind": "supported_source_unverified"}
    lo, hi = percentile(hz, cfg.range_lo_q), percentile(hz, cfg.range_hi_q)
    return {"available": True, "anchors": len(hz), "low_hz": lo, "high_hz": hi,
            "kind": "supported_source_unverified_observed_region"}


def range_component(candidate_hz, rinfo, cfg):
    if not rinfo.get("available"):
        return None
    lo, hi = rinfo["low_hz"], rinfo["high_hz"]
    if lo <= candidate_hz <= hi:
        return 0.0
    boundary = lo if candidate_hz < lo else hi
    distance = abs(cents(candidate_hz, boundary) or 0.0)
    return -min(1.0, distance / cfg.range_penalty_full_cents)


def adjudicate(row, refs, rinfo, cfg):
    t = extract_time(row)
    low, high = extract_pair(row)
    if t is None or low is None or high is None or low <= 0 or high <= 0:
        return None

    low_ok, low_reason, low_n, low_supported = acoustic_admissibility(row, "low", cfg)
    high_ok, high_reason, high_n, high_supported = acoustic_admissibility(row, "high", cfg)
    spectral = spectral_components(row, cfg)
    prev_ref, next_ref = adjacent_references(refs, t, cfg.max_adjacent_gap_s)

    scores, details = {}, {}
    for name, hz, ok in (("low", low, low_ok), ("high", high, high_ok)):
        if not ok:
            scores[name] = None
            details[name] = {"admissible": False}
            continue
        temp = temporal_score(row, name)
        prev_inter = transition_component(hz, prev_ref, t, cfg)
        next_inter = transition_component(hz, next_ref, t, cfg)
        inter_values = [v for v in (prev_inter, next_inter) if v is not None]
        inter = sum(inter_values) / len(inter_values) if inter_values else None
        rng = range_component(hz, rinfo, cfg)
        score = cfg.w_spectral * spectral[name] + cfg.w_temporal * temp
        if inter is not None:
            score += cfg.w_interval * inter
        if rng is not None:
            score += cfg.w_range * rng
        scores[name] = score
        details[name] = {
            "admissible": True,
            "spectral_component": spectral[name],
            "temporal_component": temp,
            "interval_component": inter,
            "previous_interval_component": prev_inter,
            "following_interval_component": next_inter,
            "range_component": rng,
            "score": score,
        }

    if not low_ok and not high_ok:
        outcome = "abstain_no_admissible_candidate"
    elif low_ok and not high_ok:
        outcome = "provisional_low_only" if scores["low"] >= cfg.min_absolute_score else "abstain_low_only_weak"
    elif high_ok and not low_ok:
        outcome = "provisional_high_only" if scores["high"] >= cfg.min_absolute_score else "abstain_high_only_weak"
    else:
        diff = scores["high"] - scores["low"]
        if abs(diff) < cfg.min_score_margin:
            outcome = "abstain_insufficient_separation"
        else:
            winner = "high" if diff > 0 else "low"
            if scores[winner] < cfg.min_absolute_score:
                outcome = f"abstain_{winner}_winner_weak"
            else:
                outcome = f"provisional_{winner}"

    winner_hz = None
    if outcome in {"provisional_low", "provisional_low_only"}:
        winner_hz = low
    elif outcome in {"provisional_high", "provisional_high_only"}:
        winner_hz = high

    return {
        "time_s": t, "low_hz": low, "high_hz": high, "ratio": high / low,
        "low_admissible": low_ok, "high_admissible": high_ok,
        "low_admissibility_reason": low_reason, "high_admissibility_reason": high_reason,
        "low_testable_measurements": low_n, "high_testable_measurements": high_n,
        "low_supported_measurements": low_supported, "high_supported_measurements": high_supported,
        "low_acf_median": spectral["low_acf_median"], "high_acf_median": spectral["high_acf_median"],
        "low_exclusive_energy_median": spectral["low_exclusive_median"],
        "low_exclusive_energy_min": spectral["low_exclusive_min"],
        "low_exclusive_energy_max": spectral["low_exclusive_max"],
        "exclusive_measurement_count": spectral["exclusive_measurement_count"],
        "temporal_pattern": row.get("temporal_pattern", ""),
        "previous_supported_time_s": prev_ref[0] if prev_ref else None,
        "previous_supported_hz": prev_ref[1] if prev_ref else None,
        "following_supported_time_s": next_ref[0] if next_ref else None,
        "following_supported_hz": next_ref[1] if next_ref else None,
        "low_score": scores["low"], "high_score": scores["high"],
        "winner_hz": winner_hz, "outcome": outcome,
        "details_json": json.dumps(details, sort_keys=True, separators=(",", ":")),
    }


def cluster_frequency(values, tolerance_cents):
    clusters = []
    for value in sorted(values):
        placed = False
        for cluster in clusters:
            center = median(cluster)
            if abs(cents(value, center) or float("inf")) <= tolerance_cents:
                cluster.append(value)
                placed = True
                break
        if not placed:
            clusters.append([value])
    return clusters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="all", choices=["all"] + CASES)
    ap.add_argument("--tests-root", type=Path, default=Path("tests"))
    ap.add_argument("--out", type=Path, default=Path("tests/shadow_adjudicator_v22"))
    ap.add_argument("--w-spectral", type=float, default=0.60)
    ap.add_argument("--w-temporal", type=float, default=0.20)
    ap.add_argument("--w-interval", type=float, default=0.15)
    ap.add_argument("--w-range", type=float, default=0.05)
    ap.add_argument("--min-score-margin", type=float, default=0.20)
    ap.add_argument("--min-absolute-score", type=float, default=-0.10)
    ap.add_argument("--min-acf", type=float, default=0.72)
    ap.add_argument("--min-supported-measurements", type=int, default=2)
    ap.add_argument("--acf-adv-scale", type=float, default=0.12)
    ap.add_argument("--spectral-exclusive-mix", type=float, default=0.70)
    ap.add_argument("--excl-low-floor", type=float, default=0.002)
    ap.add_argument("--excl-low-good", type=float, default=0.040)
    ap.add_argument("--penalty-scale", type=float, default=2.5)
    ap.add_argument("--max-adjacent-gap-s", type=float, default=0.050)
    ap.add_argument("--min-range-anchors", type=int, default=40)
    ap.add_argument("--range-lo-q", type=float, default=0.02)
    ap.add_argument("--range-hi-q", type=float, default=0.98)
    ap.add_argument("--range-penalty-full-cents", type=float, default=700.0)
    ap.add_argument("--observation-cluster-cents", type=float, default=35.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case == "all" else [args.case]
    manifest = {
        "schema": "v22_corrected_adaptive_shadow_adjudicator",
        "production_used": False, "production_modified": False, "midi_used": False,
        "curve_imported_unchanged": CURVE_IMPORTED,
        "curve_import_error": CURVE_IMPORT_ERROR,
        "v21_evidence_json_shift_windows_used": True,
        "v13_supported_source_unverified_references_used": True,
        "range_prior_is_soft_and_optional": True, "abstention_supported": True,
        "weights": {k: getattr(args, k) for k in ("w_spectral", "w_temporal", "w_interval", "w_range")},
        "thresholds": {k: getattr(args, k) for k in ("min_score_margin", "min_absolute_score", "min_acf", "min_supported_measurements")},
        "cases": {},
    }

    for case in cases:
        files = discover_case_files(args.tests_root, case)
        v21 = load_pair_rows(files["v21"])
        v13 = load_timeline(files["v13"])
        refs = build_supported_timeline(v13)
        rinfo = derive_range(refs, args)
        results, pair_counts = [], defaultdict(int)
        observations = defaultdict(list)
        for row in v21:
            result = adjudicate(row, refs, rinfo, args)
            if result is None:
                continue
            result["case"] = case
            results.append(result)
            pair_counts[result["outcome"]] += 1
            observations[round(result["time_s"], 6)].append(result)

        obs_rows, obs_counts = [], defaultdict(int)
        for t, records in sorted(observations.items()):
            winners = [r["winner_hz"] for r in records if r["winner_hz"] is not None]
            clusters = cluster_frequency(winners, args.observation_cluster_cents) if winners else []
            if not winners:
                status, selected = "abstain_no_decisive_pair", None
            elif len(clusters) == 1:
                status, selected = "provisional_frequency_consistent_across_pairs", median(clusters[0])
            else:
                status, selected = "abstain_pair_frequency_disagreement", None
            obs_counts[status] += 1
            obs_rows.append({
                "case": case, "time_s": t, "pair_records": len(records),
                "decisive_pair_records": len(winners), "winner_frequency_clusters": len(clusters),
                "selected_provisional_hz": selected, "observation_outcome": status,
            })

        write_csv(args.out / f"{case}_pair_decisions.csv", results)
        write_csv(args.out / f"{case}_observation_summary.csv", obs_rows)
        manifest["cases"][case] = {
            "v21_pair_records": len(v21), "pair_outcomes": dict(pair_counts),
            "observations_with_pairs": len(observations), "observation_outcomes": dict(obs_counts),
            "supported_adjacent_reference_count": len(refs), "provisional_range": rinfo,
            "input_files": {k: [str(p) for p in paths] for k, paths in files.items()},
        }

    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
