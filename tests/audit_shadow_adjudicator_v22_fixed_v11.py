#!/usr/bin/env python3
"""
V22 V11: V9 tournament with equal-tempered close-frequency ACF referee.

Repairs the cross-version contracts against the actual V13 and V21 outputs.
Diagnostic/shadow only: production, MIDI, accepted F0s, and WAVs are not modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
import sys

# Running `python tests/script.py` puts tests/, not the repository root, on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def exclusive_half_cosine(value, floor, good):
    """Bounded, smooth response: 0 at/below floor, 1 at/above good.

    An absent measurement is not interpreted as zero acoustic evidence.
    The sign/direction of the existing V5 scoring remains unchanged.
    """
    if value is None:
        return None
    if not (math.isfinite(value) and math.isfinite(floor) and math.isfinite(good)):
        raise ValueError("exclusive-energy measurement and endpoints must be finite")
    if floor < 0 or good <= floor:
        raise ValueError("exclusive-energy endpoints must satisfy 0 <= floor < good")
    u = max(0.0, min(1.0, (value - floor) / (good - floor)))
    return 0.5 - 0.5 * math.cos(math.pi * u)


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
    low_excl_support = exclusive_half_cosine(excl_center, cfg.excl_low_floor, cfg.excl_low_good)
    # A zero/weak exclusive fraction is NOT standalone proof of the high F0.
    # Positive exclusive support favors low and opposes high; absence of that
    # support stays neutral. ACF advantage supplies independent direction.
    # The scores are intentionally antisymmetric to avoid an implicit high
    # intercept (V4 scored high +0.70 even when both ACFs tied and exclusive=0).
    exclusive_signed = low_excl_support if excl_center is not None else 0.0
    low_score = cfg.spectral_exclusive_mix * exclusive_signed - (1.0 - cfg.spectral_exclusive_mix) * advantage
    high_score = -cfg.spectral_exclusive_mix * exclusive_signed + (1.0 - cfg.spectral_exclusive_mix) * advantage
    return {
        "low": max(-1.0, min(1.0, low_score)),
        "high": max(-1.0, min(1.0, high_score)),
        "low_acf_median": low_med,
        "high_acf_median": high_med,
        "low_exclusive_median": excl_center,
        "low_exclusive_min": min(exclusive) if exclusive else None,
        "low_exclusive_max": max(exclusive) if exclusive else None,
        "exclusive_measurement_count": len(exclusive),
        "low_exclusive_normalized": low_excl_support if excl_center is not None else None,
        "acf_advantage_high_minus_low": advantage,
        "spectral_scoring_model": "symmetric_half_cosine_exclusive_v7",
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


def interruption_times(v13_rows):
    """Explicit V13 low-energy/decaying flags only; unresolved F0 alone is not silence."""
    times = []
    for row in v13_rows:
        t = extract_time(row)
        if t is None:
            continue
        if as_bool(row.get("low_energy_flag")) or as_bool(row.get("decaying_flag")):
            times.append(t)
    return sorted(set(times))


def adjacent_references(refs, t, max_gap_s, interruptions=()):
    """Independent past/future lookup; do not cross a flagged interruption."""
    if not refs:
        return None, None
    times = [x[0] for x in refs]
    i = bisect_left(times, t)
    j = bisect_right(times, t)
    prev = refs[i-1] if i > 0 and 0 < t-refs[i-1][0] <= max_gap_s else None
    nxt = refs[j] if j < len(refs) and 0 < refs[j][0]-t <= max_gap_s else None
    if prev is not None:
        k = bisect_right(interruptions, prev[0])
        if k < len(interruptions) and interruptions[k] <= t:
            prev = None
    if nxt is not None:
        k = bisect_left(interruptions, t)
        if k < len(interruptions) and interruptions[k] < nxt[0]:
            nxt = None
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


def nearest_midi(hz):
    """Equal temperament A4=440 Hz; ties round upwards deterministically."""
    return math.floor(69.0 + 12.0*math.log2(hz/440.0) + 0.5)


def range_limits(rinfo, cfg):
    if not rinfo.get("available"):
        return None
    lower_note = nearest_midi(rinfo["low_hz"])
    upper_note = nearest_midi(rinfo["high_hz"])
    lo = lower_note - cfg.range_plateau_extension_st
    hi = upper_note + cfg.range_plateau_extension_st
    return lo, hi


def range_confidence(candidate_hz, rinfo, cfg):
    limits = range_limits(rinfo, cfg)
    if limits is None or candidate_hz is None or candidate_hz <= 0:
        return None
    lo, hi = limits
    note = 69.0 + 12.0*math.log2(candidate_hz/440.0)
    distance = lo-note if note < lo else note-hi if note > hi else 0.0
    if distance <= 0:
        return 1.0
    if distance >= cfg.range_shoulder_st:
        return 0.0
    return 0.5*(1.0+math.cos(math.pi*distance/cfg.range_shoulder_st))


def range_component(candidate_hz, rinfo, cfg):
    confidence = range_confidence(candidate_hz, rinfo, cfg)
    return None if confidence is None else -(1.0-confidence)


def adjudicate(row, refs, rinfo, cfg, interruptions=()):
    t = extract_time(row)
    low, high = extract_pair(row)
    if t is None or low is None or high is None or low <= 0 or high <= 0:
        return None

    # Range admission precedes acoustic admissibility; fail closed if range is unavailable.
    low_conf = range_confidence(low, rinfo, cfg)
    high_conf = range_confidence(high, rinfo, cfg)
    low_admitted = low_conf is not None and low_conf > 0.0
    high_admitted = high_conf is not None and high_conf > 0.0
    low_ok, low_reason, low_n, low_supported = (
        acoustic_admissibility(row, "low", cfg) if low_admitted
        else (False, "range_unavailable" if low_conf is None else "range_zero_confidence", 0, 0)
    )
    high_ok, high_reason, high_n, high_supported = (
        acoustic_admissibility(row, "high", cfg) if high_admitted
        else (False, "range_unavailable" if high_conf is None else "range_zero_confidence", 0, 0)
    )
    spectral = spectral_components(row, cfg)
    # Spectral scores are pair-relative. If range rejects one hypothesis, its
    # measurements may remain in diagnostic CSV, but cannot influence the
    # survivor's adjudication score.
    single_range_survivor = low_admitted != high_admitted
    prev_ref, next_ref = adjacent_references(refs, t, cfg.max_adjacent_gap_s, interruptions)

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
        spec_for_score = 0.0 if single_range_survivor else spectral[name]
        score = cfg.w_spectral * spec_for_score + cfg.w_temporal * temp
        if inter is not None:
            score += cfg.w_interval * inter
        if rng is not None:
            score += cfg.w_range * rng
        scores[name] = score
        details[name] = {
            "admissible": True,
            "spectral_component": spec_for_score,
            "pair_relative_spectral_excluded_due_to_range_gate": single_range_survivor,
            "temporal_component": temp,
            "interval_component": inter,
            "previous_interval_component": prev_inter,
            "following_interval_component": next_inter,
            "range_component": rng,
            "range_confidence": range_confidence(hz, rinfo, cfg),
            "previous_reference_time_s": prev_ref[0] if prev_ref else None,
            "previous_reference_hz": prev_ref[1] if prev_ref else None,
            "following_reference_time_s": next_ref[0] if next_ref else None,
            "following_reference_hz": next_ref[1] if next_ref else None,
            "previous_gap_ms": (t-prev_ref[0])*1000 if prev_ref else None,
            "following_gap_ms": (next_ref[0]-t)*1000 if next_ref else None,
            "interval_directions_used": len(inter_values),
            "score": score,
        }

    if not low_admitted and not high_admitted:
        outcome = "abstain_all_candidates_out_of_range" if low_conf is not None and high_conf is not None else "abstain_range_unavailable"
    elif not low_ok and not high_ok:
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
        "low_range_admitted": low_admitted, "high_range_admitted": high_admitted,
        "low_range_confidence": low_conf, "high_range_confidence": high_conf,
        "pair_relative_spectral_disabled": single_range_survivor,
        "low_range_rejection_reason": "" if low_admitted else ("range_unavailable" if low_conf is None else "zero_range_confidence"),
        "high_range_rejection_reason": "" if high_admitted else ("range_unavailable" if high_conf is None else "zero_range_confidence"),
        "low_admissible": low_ok, "high_admissible": high_ok,
        "low_admissibility_reason": low_reason, "high_admissibility_reason": high_reason,
        "low_testable_measurements": low_n, "high_testable_measurements": high_n,
        "low_supported_measurements": low_supported, "high_supported_measurements": high_supported,
        "low_acf_median": spectral["low_acf_median"], "high_acf_median": spectral["high_acf_median"],
        "low_exclusive_energy_median": spectral["low_exclusive_median"],
        "low_exclusive_energy_min": spectral["low_exclusive_min"],
        "low_exclusive_energy_max": spectral["low_exclusive_max"],
        "exclusive_measurement_count": spectral["exclusive_measurement_count"],
        "low_exclusive_normalized": spectral["low_exclusive_normalized"],
        "acf_advantage_high_minus_low": spectral["acf_advantage_high_minus_low"],
        "temporal_pattern": row.get("temporal_pattern", ""),
        "previous_supported_time_s": prev_ref[0] if prev_ref else None,
        "previous_gap_ms": (t-prev_ref[0])*1000 if prev_ref else None,
        "following_gap_ms": (next_ref[0]-t)*1000 if next_ref else None,
        "interval_directions_available": int(prev_ref is not None)+int(next_ref is not None),
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
            distance = cents(value, center)
            if distance is not None and abs(distance) <= tolerance_cents:
                cluster.append(value)
                placed = True
                break
        if not placed:
            clusters.append([value])
    return clusters



def tournament_adjudicate(records, tolerance_cents):
    """Resolve an observation's observed comparisons, never treating a local win as final.

    Edges point winner -> loser. A champion must dominate every observed hypothesis
    through decisive comparisons, have no recorded loss, and have no unresolved
    admissible direct comparison. Missing links, cycles and contradictions abstain.
    """
    # Out-of-range hypotheses must not become nodes, edges, or undefeated rivals.
    frequencies = [hz for r in records for hz, admitted in (
        (r["low_hz"], r["low_range_admitted"]),
        (r["high_hz"], r["high_range_admitted"])
    ) if admitted]
    if not frequencies:
        return "abstain_all_candidates_out_of_range", None, {
            "tournament_candidate_frequencies_hz": "[]",
            "tournament_decisive_edges": 0,
            "tournament_undecided_admissible_pairs": 0,
            "tournament_unmapped_records": 0,
            "tournament_champion_frequency_hz": None,
        }
    clusters = cluster_frequency(frequencies, tolerance_cents)
    centers = [median(c) for c in clusters]

    def node(hz):
        matches = [i for i, center in enumerate(centers)
                   if (d := cents(hz, center)) is not None and abs(d) <= tolerance_cents]
        if len(matches) != 1:
            return None
        return matches[0]

    edges = set()
    undecided = set()
    unassigned = 0
    for r in records:
        lo = node(r["low_hz"]) if r["low_range_admitted"] else None
        hi = node(r["high_hz"]) if r["high_range_admitted"] else None
        # Missing node from a range rejection is intentional, not a mapping error.
        if lo is None and hi is None:
            continue
        if (lo is None) != (hi is None):
            continue
        if lo is None or hi is None:
            unassigned += 1
            continue
        if lo == hi:
            continue
        winner = node(r["winner_hz"]) if r["winner_hz"] is not None else None
        if winner is not None and winner in (lo, hi):
            loser = hi if winner == lo else lo
            edges.add((winner, loser))
        elif r["low_admissible"] and r["high_admissible"]:
            undecided.add(frozenset((lo, hi)))

    detail = {
        "tournament_candidate_frequencies_hz": json.dumps(centers),
        "tournament_decisive_edges": len(edges),
        "tournament_undecided_admissible_pairs": len(undecided),
        "tournament_unmapped_records": unassigned,
    }
    if unassigned:
        return "abstain_tournament_unmapped", None, detail
    if len(centers) == 1:
        # One range survivor still requires at least one acoustically supported,
        # sufficiently strong pair-level winner. Never crown it by elimination alone.
        decisive = [r for r in records if r["winner_hz"] is not None
                    and (d := cents(r["winner_hz"], centers[0])) is not None
                    and abs(d) <= tolerance_cents]
        if not decisive:
            return "abstain_single_candidate_no_acoustic_support", None, detail
        detail["tournament_champion_frequency_hz"] = centers[0]
        return "provisional_tournament_champion", centers[0], detail
    if not edges:
        return "abstain_no_decisive_pair", None, detail

    n = len(centers)
    graph = {i: set() for i in range(n)}
    for winner, loser in edges:
        graph[winner].add(loser)

    # Contradictory direct comparisons and directed cycles cannot establish
    # a transitive ordering, even if one vertex seems to dominate overall.
    visited, active = set(), set()
    def cycle(v):
        if v in active:
            return True
        if v in visited:
            return False
        active.add(v)
        if any(cycle(w) for w in graph[v]):
            return True
        active.remove(v)
        visited.add(v)
        return False
    if any(cycle(i) for i in range(n)):
        return "abstain_tournament_cycle", None, detail

    incoming = {v for _, v in edges}
    candidates = [i for i in range(n) if i not in incoming]
    if len(candidates) != 1:
        return "abstain_tournament_no_unique_champion", None, detail
    champion = candidates[0]
    reached = {champion}
    stack = [champion]
    while stack:
        v = stack.pop()
        for w in graph[v] - reached:
            reached.add(w)
            stack.append(w)
    if len(reached) != n:
        return "abstain_tournament_incomplete_comparisons", None, detail
    if any(champion in pair for pair in undecided):
        return "abstain_tournament_champion_ambiguous", None, detail
    detail["tournament_champion_frequency_hz"] = centers[champion]
    return "provisional_tournament_champion", centers[champion], detail



def abstention_diagnostic(records, status, min_score_margin):
    """Descriptive causes only; never changes acoustic decisions or tournament."""
    outcomes = [r["outcome"] for r in records]
    admitted = [r for r in records if r["low_admissible"] or r["high_admissible"]]
    both = [r for r in records if r["low_admissible"] and r["high_admissible"]]
    decisive = [r for r in records if r["winner_hz"] is not None]
    return {
        "pairs_both_inadmissible": sum(not r["low_admissible"] and not r["high_admissible"] for r in records),
        "pairs_one_admissible": len(admitted) - len(both),
        "pairs_both_admissible": len(both),
        "pairs_insufficient_separation": outcomes.count("abstain_insufficient_separation"),
        "pairs_weak_winner": sum(o.startswith("abstain_") and o.endswith("_weak") for o in outcomes),
        "pairs_decisive": len(decisive),
        "pairs_without_exclusive_measurement": sum(r["exclusive_measurement_count"] == 0 for r in records),
        "pairs_without_any_interval_component": sum(all(
            json.loads(r["details_json"]).get(c, {}).get("interval_component") is None
            for c in ("low", "high")) for r in records),
        "pairs_score_margin_under_threshold": sum(
            r["low_score"] is not None and r["high_score"] is not None
            and abs(r["high_score"] - r["low_score"]) < min_score_margin
            for r in records),
        "abstention_category": (
            "no_decisive_acoustic_pair" if status == "abstain_no_decisive_pair"
            else "tournament_cycle" if status == "abstain_tournament_cycle"
            else "incomplete_comparison_graph" if status == "abstain_tournament_incomplete_comparisons"
            else "multiple_undefeated_candidates" if status == "abstain_tournament_no_unique_champion"
            else "unresolved_champion_direct_comparison" if status == "abstain_tournament_champion_ambiguous"
            else "all_candidates_out_of_range" if status == "abstain_all_candidates_out_of_range"
            else "single_candidate_without_acoustic_support" if status == "abstain_single_candidate_no_acoustic_support"
            else "unmapped_frequency" if status == "abstain_tournament_unmapped"
            else "provisional_champion" if status == "provisional_tournament_champion"
            else "other_abstention"
        ),
    }



def same_note_acf_referee(result, low, high, tolerance=49.0):
    """Judge only newly measured close-frequency matches, after V8 admission.

    Both frequencies must round to the same A4=440 equal-tempered semitone,
    and both must be STRICTLY within 49 cents of that note. Do not infer
    a victory from one inadmissible hypothesis or from missing/tied ACF.
    This intentionally bypasses V9's pair-relative spectral score and its
    decision margin ONLY for these novel close-frequency head-to-heads.
    """
    result["close_referee"] = "equal_tempered_median_acf"
    result["close_referee_note_midi"] = None
    result["close_referee_low_cents"] = None
    result["close_referee_high_cents"] = None
    result["close_referee_acf_difference"] = None
    result["close_referee_reason"] = ""
    result["winner_hz"] = None
    if not (result["low_range_admitted"] and result["high_range_admitted"]):
        result["outcome"] = "abstain_close_range_gate"
        result["close_referee_reason"] = "range_gate"
        return result
    if not (result["low_admissible"] and result["high_admissible"]):
        result["outcome"] = "abstain_close_acoustic_gate"
        result["close_referee_reason"] = "acoustic_gate"
        return result
    if low <= 0 or high <= 0:
        result["outcome"] = "abstain_close_invalid_frequency"
        result["close_referee_reason"] = "invalid_frequency"
        return result
    midi_low = round(69 + 12*math.log2(low/440.0))
    midi_high = round(69 + 12*math.log2(high/440.0))
    result["close_referee_note_midi"] = midi_low if midi_low == midi_high else None
    if midi_low != midi_high:
        result["outcome"] = "abstain_close_different_notes"
        result["close_referee_reason"] = "different_nearest_notes"
        return result
    center = 440.0 * 2**((midi_low-69)/12.0)
    d_low = 1200*math.log2(low/center)
    d_high = 1200*math.log2(high/center)
    result["close_referee_low_cents"] = d_low
    result["close_referee_high_cents"] = d_high
    if abs(d_low) >= tolerance or abs(d_high) >= tolerance:
        result["outcome"] = "abstain_close_outside_49_cents"
        result["close_referee_reason"] = "outside_note_tolerance"
        return result
    a = result.get("low_acf_median")
    b = result.get("high_acf_median")
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        result["outcome"] = "abstain_close_missing_acf"
        result["close_referee_reason"] = "missing_median_acf"
        return result
    result["close_referee_acf_difference"] = b-a
    if b == a:
        result["outcome"] = "abstain_close_acf_tie"
        result["close_referee_reason"] = "equal_median_acf"
        return result
    chosen = "high" if b > a else "low"
    result["winner_hz"] = high if chosen == "high" else low
    result["outcome"] = f"provisional_{chosen}"
    result["close_referee_reason"] = "higher_median_acf"
    return result


def complete_head_to_head(case, t, records, waveform, cfg, refs, rinfo, interruptions, v21_measure):
    """Measure every missing edge between DISTINCT admitted frequency clusters.

    An edge is considered measured even when its result abstains. Existing pairs
    are never remeasured. All novel comparisons use V21's original WAV probes,
    followed by precisely the same V8 range/acoustic/scoring rules.
    """
    freqs = [hz for r in records for hz, ok in
             ((r["low_hz"], r["low_range_admitted"]),
              (r["high_hz"], r["high_range_admitted"])) if ok]
    if len(freqs) < 2:
        return [], 0
    clusters = cluster_frequency(freqs, cfg.observation_cluster_cents)
    centers = [median(c) for c in clusters]
    if len(centers) < 2:
        return [], 0

    def node(hz):
        matching = [i for i, center in enumerate(centers)
                    if (d := cents(hz, center)) is not None
                    and abs(d) <= cfg.observation_cluster_cents]
        # No invented assignment if clustering is ambiguous.
        return matching[0] if len(matching) == 1 else None

    measured = set()
    for r in records:
        if not r["low_range_admitted"] or not r["high_range_admitted"]:
            continue
        a, b = node(r["low_hz"]), node(r["high_hz"])
        if a is not None and b is not None and a != b:
            measured.add(tuple(sorted((a, b))))

    new_rows = []
    missing = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if (i, j) in measured:
                continue
            missing += 1
            low, high = sorted((centers[i], centers[j]))
            evidence = v21_measure.audit_pair(waveform[0], waveform[1], t, low, high)
            row = {"time_s": t, "low_hz": low, "high_hz": high,
                   "temporal_pattern": evidence["temporal_pattern"],
                   "evidence_json": json.dumps(evidence, separators=(",", ":"))}
            result = adjudicate(row, refs, rinfo, cfg, interruptions)
            if result is None:
                raise RuntimeError(f"Failed novel comparison {case} {t} {low} {high}")
            result["case"] = case
            # V21 validated integer-family hypotheses, not close-pitch resolution.
            # Flag its extrapolation transparently without changing the decision.
            result["close_pair_below_64ms_fft_bin"] = abs(high - low) < (waveform[1] / round(0.064 * waveform[1]))
            if result["close_pair_below_64ms_fft_bin"]:
                same_note_acf_referee(result, low, high)
            else:
                result["close_referee"] = "not_applicable_integer_family_v9"
                result["close_referee_reason"] = "ordinary_v9_scoring"
            result["comparison_origin"] = "new_direct_wav_v21_measurement"
            result["synthetic_comparison"] = False  # Measured directly from audio.
            new_rows.append(result)
    return new_rows, missing


def load_v21_measurement_module(tests_root):
    import importlib.util
    source = tests_root / "audit_temporal_integer_families_v21.py"
    if not source.is_file():
        raise SystemExit(f"Missing V21 acoustic measurement source: {source}")
    spec = importlib.util.spec_from_file_location("v21_measurement_for_v9", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_original_wav(case, tests_root, wav_root):
    for candidate in (wav_root / f"{case}.wav", tests_root / f"{case}.wav",
                      tests_root.parent / f"{case}.wav"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Missing ORIGINAL WAV for {case}. Supply --wav-root with the WAV directory.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="all", choices=["all"] + CASES)
    ap.add_argument("--tests-root", type=Path, default=Path("tests"))
    ap.add_argument("--wav-root", type=Path, default=Path("tests"), help="Directory containing original case WAVs")
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
    ap.add_argument("--range-plateau-extension-st", type=float, default=2.0)
    ap.add_argument("--range-shoulder-st", type=float, default=2.0)
    ap.add_argument("--observation-cluster-cents", type=float, default=35.0)
    ap.add_argument("--ratata-active-end-s", type=float, default=87.081,
                    help="Externally annotated active-performance end; RATATA tail is excluded from active-F0 adjudication.")
    args = ap.parse_args()
    if not (0 <= args.excl_low_floor < args.excl_low_good):
        ap.error("exclusive-energy limits must satisfy 0 <= --excl-low-floor < --excl-low-good")

    if args.range_plateau_extension_st < 0 or args.range_shoulder_st <= 0 or args.max_adjacent_gap_s <= 0:
        ap.error("range plateau extension must be >=0; shoulder and adjacent gap must be >0")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case == "all" else [args.case]
    manifest = {
        "schema": "v22_v11_equal_tempered_close_acf_shadow",
        "close_referee_policy": "new_close_pairs_same_nearest_note_strict_49_cents_then_higher_median_acf_else_abstain",
        "missing_head_to_head_policy": "measure_all_missing_admitted_cluster_pairs_from_original_wav_with_v21_probes",
        "spectral_model": "symmetric_half_cosine_exclusive_v7",
        "exclusive_energy_curve": {"shape": "half_cosine_rising", "floor": args.excl_low_floor, "good": args.excl_low_good, "missing": "neutral"},
        "abstention_audit_only_no_decision_override": True,
        "production_used": False, "production_modified": False, "midi_used": False,
        "curve_imported_unchanged": CURVE_IMPORTED,
        "curve_import_error": CURVE_IMPORT_ERROR,
        "v21_evidence_json_shift_windows_used": True,
        "v13_supported_source_unverified_references_used": True,
        "range_prior_is_soft_and_optional": False, "abstention_supported": True,
        "range_gate": "upstream_strict_positive_confidence_only_fail_closed_if_unavailable",
        "shoulder_confidence_retained_as_soft_range_score": True,
        "range_model": "nearest_equal_tempered_note_plus_minus_2st_plateau_half_cosine_2st_shoulders",
        "interval_model": "original_transition_curve_past_future_mean_available_directions",
        "interruption_policy": "V13_explicit_low_energy_or_decaying_flags",
        "weights": {k: getattr(args, k) for k in ("w_spectral", "w_temporal", "w_interval", "w_range")},
        "thresholds": {k: getattr(args, k) for k in ("min_score_margin", "min_absolute_score", "min_acf", "min_supported_measurements")},
        "cases": {},
    }

    for case in cases:
        files = discover_case_files(args.tests_root, case)
        if not files["v13"]:
            raise SystemExit(
                f"Missing V13 input for {case}: expected {args.tests_root / 'full_timeline_v13' / (case + '_full_timeline.csv')}. "
                "Extract full_timeline_v13.zip into repository root before running. "
                "No partial adjudication was performed."
            )
        if not CURVE_IMPORTED:
            raise SystemExit(f"Transition-curve import failed: {CURVE_IMPORT_ERROR}. No partial adjudication was performed.")
        v21 = load_pair_rows(files["v21"])
        v13 = load_timeline(files["v13"])
        if not v13:
            raise SystemExit(f"V13 timeline schema invalid for {case}; no partial adjudication was performed.")
        active_end = args.ratata_active_end_s if case == "RATATA" else None
        refs = build_supported_timeline(v13)
        if active_end is not None:
            refs = [(t, hz) for t, hz in refs if t < active_end]
        rinfo = derive_range(refs, args)
        if not rinfo.get("available"):
            raise SystemExit(f"Range unavailable for {case}; V8 admission gate requires an established range. No partial adjudication performed.")
        interruptions = interruption_times(v13)
        limits = range_limits(rinfo, args)
        if limits is not None:
            rinfo["plateau_low_midi"], rinfo["plateau_high_midi"] = limits
            rinfo["zero_low_midi"] = limits[0]-args.range_shoulder_st
            rinfo["zero_high_midi"] = limits[1]+args.range_shoulder_st
        results, pair_counts = [], defaultdict(int)
        inactive_tail_rows = []
        observations = defaultdict(list)
        for row in v21:
            if active_end is not None and (extract_time(row) is not None and extract_time(row) >= active_end):
                inactive_tail_rows.append(row)
                continue
            result = adjudicate(row, refs, rinfo, args, interruptions)
            if result is None:
                continue
            result["case"] = case
            result["comparison_origin"] = "original_v21_pair"
            results.append(result)
            pair_counts[result["outcome"]] += 1
            observations[round(result["time_s"], 6)].append(result)

        # Complete ALL unmeasured eligible head-to-head matches with original WAV
        # measurements, not assumed victories or recycled pair-relative scores.
        import soundfile as sf
        measure = load_v21_measurement_module(args.tests_root)
        wav_path = locate_original_wav(case, args.tests_root, args.wav_root)
        audio, sample_rate = sf.read(str(wav_path), dtype="float64", always_2d=False)
        if getattr(audio, "ndim", 1) != 1:
            raise SystemExit(f"Original WAV must be mono for V21 parity: {wav_path}")
        waveform = (audio, sample_rate)
        extra_rows = []
        previously_missing = 0
        new_outcomes = defaultdict(int)
        for t, records in sorted(observations.items()):
            measured, missing = complete_head_to_head(
                case, t, records, waveform, args, refs, rinfo, interruptions, measure)
            extra_rows.extend(measured)
            previously_missing += missing
            for result in measured:
                new_outcomes[result["outcome"]] += 1
                records.append(result)
        results.extend(extra_rows)
        for outcome, count in new_outcomes.items():
            pair_counts[outcome] += count
        obs_rows, obs_counts = [], defaultdict(int)
        abstention_counts = defaultdict(int)
        for t, records in sorted(observations.items()):
            status, selected, tournament_details = tournament_adjudicate(
                records, args.observation_cluster_cents
            )
            winners = [r["winner_hz"] for r in records if r["winner_hz"] is not None]
            clusters = cluster_frequency(winners, args.observation_cluster_cents) if winners else []
            obs_counts[status] += 1
            diagnostic = abstention_diagnostic(records, status, args.min_score_margin)
            abstention_counts[diagnostic["abstention_category"]] += 1
            obs_rows.append({
                "case": case, "time_s": t, "pair_records": len(records),
                "decisive_pair_records": len(winners), "winner_frequency_clusters": len(clusters),
                "selected_provisional_hz": selected, "observation_outcome": status,
                **tournament_details,
                **diagnostic,
            })

        write_csv(args.out / f"{case}_pair_decisions.csv", results)
        write_csv(args.out / f"{case}_new_direct_comparisons.csv", extra_rows)
        if inactive_tail_rows:
            write_csv(args.out / f"{case}_excluded_inactive_tail.csv", inactive_tail_rows)
        write_csv(args.out / f"{case}_observation_summary.csv", obs_rows)
        manifest["cases"][case] = {
            "v21_pair_records": len(v21), "new_direct_wav_comparisons": len(extra_rows),
            "previously_unmeasured_edges": previously_missing,
            "new_direct_pair_outcomes": dict(new_outcomes),
            "novel_pairs_below_64ms_fft_resolution": sum(bool(r["close_pair_below_64ms_fft_bin"]) for r in extra_rows),
            "original_wav": str(wav_path), "pair_outcomes": dict(pair_counts),
            "observations_with_pairs": len(observations), "observation_outcomes": dict(obs_counts),
            "abstention_cause_summary": dict(abstention_counts),
            "supported_adjacent_reference_count": len(refs), "provisional_range": rinfo,
            "explicit_interruption_timestamps": len(interruptions),
            "active_end_s_exclusive": active_end,
            "excluded_inactive_tail_pair_records": len(inactive_tail_rows),
            "analyzed_pair_records": len(results),
            "range_rejected_candidate_records": sum(int(not r["low_range_admitted"]) + int(not r["high_range_admitted"]) for r in results),
            "pairs_both_range_rejected": sum(not r["low_range_admitted"] and not r["high_range_admitted"] for r in results),
            "pairs_one_range_rejected": sum(r["low_range_admitted"] != r["high_range_admitted"] for r in results),
            "input_files": {k: [str(p) for p in paths] for k, paths in files.items()},
        }

    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
