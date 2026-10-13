from __future__ import annotations

"""MIDI Renderer V18.3 -- unsupported-start retraction with revealed micro-remnant topology cooperation.

V18 builds on V17.1 and does not call the legacy midi_event_builder/midi_articulation
pipeline.  It assembles musical events once from parallel evidence:

  * frozen V3R9 corrected-valid / structural pitch (normal pitch authority)
  * frozen V3R9 resets and target stability (supporting structural evidence)
  * frozen V3R9 primary frame periodicity (bounded orphan-attack rescue only)
  * high-resolution WAV attack evidence (timing only)
  * frozen V3R9 expressive pitch/amplitude lanes (PB and CC11)

The MIDI wire contract remains compatible with the project: Type 1, PPQ 960,
GM #65 Soprano Sax, channel 1 by default, fixed velocity 100, CC11 dynamics,
and explicit per-note pitch-bend range/reset.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import mido
import numpy as np

from .midi_event_assembler_v18_3 import (
    AssemblyResult, MusicalEvent, assemble_events,
    REVEALED_MICRO_MAX_S, REVEALED_MICRO_MAX_GAP_NEXT_S,
)
from .midi_event_assembler_v18_2 import (
    UNSUPPORTED_BACKDATE_MIN_LAG_S, UNSUPPORTED_BACKDATE_START_DBFS,
    UNSUPPORTED_BACKDATE_MAX_MEDIAN_DBFS, UNSUPPORTED_BACKDATE_MIN_REMAINING_S, UNSUPPORTED_PRIMARY_MIN_ACF,
    UNSUPPORTED_PRIMARY_MAX_CMNDF, UNSUPPORTED_PRIMARY_MAX_ACF_CMNDF_CENTS,
    UNSUPPORTED_PRIMARY_MAX_FAMILY_GAP_ST,
)

from .midi_event_assembler_v18 import (
    BREATH_BACKDATE_MIN_LAG_S, BREATH_REPLACEMENT_MAX_PREANCHOR_S,
    ULTRASHORT_MAX_S, TRANSIENT_HEAD_MAX_S,
    ACOUSTIC_RESCUE_MIN_ACF, ACOUSTIC_RESCUE_MAX_CMNDF,
    SUBHARMONIC_MIN_OCTAVE_DOMINANCE_DB,
)

from .midi_event_assembler_v18_1 import (
    WEAK_BACKDATE_START_DBFS, WEAK_BACKDATE_MIN_LAG_S,
    WEAK_ISLAND_MAX_S, WEAK_ISLAND_START_DBFS, WEAK_ISLAND_MIN_NEXT_GAIN_DB,
    TERMINAL_DECAY_MAX_S, TERMINAL_DECAY_MIN_LEFT_DEPTH_DB, TERMINAL_DECAY_MAX_RIGHT_DEPTH_DB,
    REPEAT_MIN_ATTACK_SCORE, REPEAT_MIN_LEFT_DEPTH_DB, REPEAT_MIN_RIGHT_DEPTH_DB,
    DECAY_SUBHARMONIC_HEAD1_DB, DECAY_SUBHARMONIC_HEAD2_DB, DECAY_SUBHARMONIC_MIN_RMS_FALL_DB,
    FAST_PB_TREND_HALF_WINDOW_S, FAST_PB_MIN_HALF_CYCLE_S, FAST_PB_MAX_HALF_CYCLE_S,
    FAST_PB_COMPRESSION,
)

from .midi_event_assembler_v17_1 import (
    RESIDUAL_VALID_MIN_STABLE_FACTOR, RESIDUAL_ATTACK_ANCHOR_TOLERANCE_S,
    RESIDUAL_JOIN_MAX_GAP_S, RESIDUAL_JOIN_MAX_TAIL_GAP_ST,
    RESIDUAL_REATTACK_PROOF_WINDOW_S,
)
from .midi_event_assembler_v16_1 import (
    RESET_POST_ATTACK_GRACE_S, PRIMARY_QUERY_PREROLL_S, PRIMARY_BRIDGE_POSTROLL_S,
    CORRIDOR_LONG_UNPITCHED_S, CORRIDOR_EARLY_PRIMARY_SUPPORT_S,
    CORRIDOR_PRIMARY_DELAY_FRACTION, CORRIDOR_ATTACK_FLOOR_MARGIN_DB,
    CORRIDOR_OWNER_LEVEL_GAIN_DB,
    MOTION_WINDOW_S, MOTION_ACTIVE_SPAN_ST, MOTION_TRANSITION_RADIUS_S,
    MOTION_LARGE_SPAN_ST, MOTION_STRICT_MONOTONIC_FRACTION, MOTION_LARGE_MONOTONIC_FRACTION,
    REATTACK_MIN_BILATERAL_DEPTH_DB,
    DIRECTED_CROSSING_HALF_WINDOW_S, DIRECTED_CROSSING_MIN_SLOPE_ST_PER_S, DIRECTED_CROSSING_MAX_SLOPE_RATIO,
    POST_STRUCTURAL_DUPLICATE_WINDOW_S, SHORT_ISLAND_SMOOTHING_GUARD_S, SHORT_ISLAND_LATE_ATTACK_S,
    HANDOFF_MAX_GAP_S, HANDOFF_MIN_ACF, HANDOFF_MAX_CMNDF,
    HANDOFF_MAX_ACF_CMNDF_DISAGREEMENT_CENTS, HANDOFF_ENDPOINT_MARGIN_ST,
    HANDOFF_MIN_AUDIBLE_MARGIN_DB, PITCH_ONLY_FUTURE_MATCH_ST, PITCH_ONLY_MAX_FUTURE_LAG_S,
)


PPQ = 960
MELODY_CHANNEL = 0
GM_PROGRAM_NUMBER = 65
MIDO_PROGRAM = GM_PROGRAM_NUMBER - 1
DEFAULT_VELOCITY = 100
CC7_VOLUME = 7
CC7_NEUTRAL = 127
CC11_EXPRESSION = 11
CC11_NEUTRAL = 127
CC11_DB_COMPRESSION = 0.3
PB_CENTER = 0
PB_MIN = -8192
PB_MAX = 8191


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _sidecar(prefix: Path, suffix: str) -> Path:
    p = Path(str(prefix) + suffix)
    if not p.exists():
        raise FileNotFoundError(f"required renderer input not found: {p}")
    return p


def _seconds_to_tick(seconds: float, bpm: float) -> int:
    return int(round(float(seconds) * float(bpm) / 60.0 * PPQ))


def _parse_time_signature(text: str) -> tuple[int, int]:
    try:
        a, b = text.split("/", 1)
        num, den = int(a), int(b)
    except Exception as exc:
        raise argparse.ArgumentTypeError("time signature must look like 4/4") from exc
    if num <= 0 or den <= 0 or den & (den - 1):
        raise argparse.ArgumentTypeError("time signature denominator must be a power of two")
    return num, den


def _rpn_pitch_bend_range(channel: int, semitones: int) -> list[mido.Message]:
    semitones = int(np.clip(semitones, 0, 127))
    return [
        mido.Message("control_change", channel=channel, control=101, value=0, time=0),
        mido.Message("control_change", channel=channel, control=100, value=0, time=0),
        mido.Message("control_change", channel=channel, control=6, value=semitones, time=0),
        mido.Message("control_change", channel=channel, control=38, value=0, time=0),
        mido.Message("control_change", channel=channel, control=101, value=127, time=0),
        mido.Message("control_change", channel=channel, control=100, value=127, time=0),
    ]


def _bend_value(pitch_st: float, base_midi: int, range_st: int) -> int:
    if range_st <= 0:
        return 0
    frac = (float(pitch_st) - float(base_midi)) / float(range_st)
    return int(np.clip(int(round(frac * 8192.0)), PB_MIN, PB_MAX))


def _load_amplitude(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(_sidecar(prefix, ".amplitude_expression.csv"))
    t = np.asarray([_f(r, "time_s") for r in rows], dtype=float)
    shape = np.asarray([_f(r, "shape_rms_dbfs") for r in rows], dtype=float)
    residual = np.asarray([_f(r, "amplitude_residual_db") for r in rows], dtype=float)
    valid = np.asarray([bool(_i(r, "corrected_valid")) for r in rows], dtype=bool)
    valid &= np.isfinite(t) & np.isfinite(shape) & np.isfinite(residual)
    return t, shape, residual, valid


def _assign_expression_baselines(
    prefix: Path,
    assembly: AssemblyResult,
    velocity_override: int | None,
) -> tuple[dict[int, int], dict[int, list[tuple[float, float]]]]:
    """Assign CC11 from the voiced owner, not from an early consonant/breath.

    Note-On timing may legitimately precede trustworthy periodic F0.  V10/V11
    accidentally measured the sustained loudness baseline from that early MIDI
    start, so a quiet consonant or breath could pin an otherwise loud note to a
    very low CC11 value.  V18 retains V17.1/V16.1/V13/V14's separation of the two authorities:

      * event.start_s          -> articulation timing
      * event.expression_anchor_s -> first trustworthy voiced/pitch ownership

    The modulation lane is likewise referenced from the first valid amplitude
    point at/after the voiced anchor.
    """
    events = assembly.events
    t, shape, residual, valid = _load_amplitude(prefix)
    attack_window = assembly.pitch.min_stable_s
    fixed_velocity = DEFAULT_VELOCITY if velocity_override is None else int(np.clip(velocity_override, 1, 127))

    levels: list[float] = []
    for event in events:
        event.velocity = fixed_velocity
        # Keep the physical attack loudness for audit only.  It does not define
        # the sustained expression baseline.
        event.attack_dbfs = assembly.attack_lane.rms_at(event.start_s)

        anchor = event.expression_anchor_s
        if not math.isfinite(anchor):
            anchor = event.start_s
        anchor = float(np.clip(anchor, event.start_s, event.end_s))
        right = min(event.end_s, anchor + attack_window)
        mask = valid & (t >= anchor - 1e-9) & (t <= right + 1e-9)
        if np.any(mask):
            level = float(np.max(shape[mask]))
        else:
            # Primary-rescue notes can lack corrected-valid amplitude sidecars.
            # Fall back at the *pitch/expression anchor*, never at an earlier
            # consonant-aligned Note-On.
            level = assembly.attack_lane.rms_at(anchor)
        event.expression_dbfs = level
        levels.append(level)

    finite = [x for x in levels if math.isfinite(x)]
    peak = max(finite) if finite else math.nan
    baselines: dict[int, int] = {}
    modulation: dict[int, list[tuple[float, float]]] = {}

    for idx, event in enumerate(events):
        if math.isfinite(event.expression_dbfs) and math.isfinite(peak):
            ratio = 10.0 ** ((CC11_DB_COMPRESSION * (event.expression_dbfs - peak)) / 20.0)
            baselines[idx] = int(np.clip(round(CC11_NEUTRAL * ratio), 1, 127))
        else:
            baselines[idx] = CC11_NEUTRAL

        anchor = event.expression_anchor_s if math.isfinite(event.expression_anchor_s) else event.start_s
        mask = valid & (t >= anchor - 1e-9) & (t <= event.end_s + 1e-9)
        tt = t[mask]
        rr = residual[mask]
        if len(tt) < 1:
            modulation[idx] = []
            continue
        ref = float(rr[0])
        modulation[idx] = [(float(a), float(b - ref)) for a, b in zip(tt, rr)]

    return baselines, modulation


def _write_audits(prefix: Path, output_mid: Path, assembly: AssemblyResult, cc11: dict[int, int]) -> None:
    events_path = output_mid.with_suffix(output_mid.suffix + ".events.csv")
    with events_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "event_index", "start_s", "end_s", "duration_s", "base_midi", "base_st",
            "velocity", "attack_dbfs", "expression_anchor_s", "expression_dbfs",
            "cc11_baseline", "source_kind", "source_id",
            "pitch_source", "attack_score", "bend_range_semitones", "max_abs_bend_semitones",
            "pitch_point_count", "interpretation_dt_s", "interpretation_hz", "renderer_version",
        ])
        for i, e in enumerate(assembly.events):
            w.writerow([
                i, e.start_s, e.end_s, e.end_s - e.start_s, e.base_midi, e.base_st,
                e.velocity, e.attack_dbfs, e.expression_anchor_s, e.expression_dbfs,
                cc11.get(i, CC11_NEUTRAL), e.source_kind, e.source_id,
                e.pitch_source, e.attack_score, e.bend_range_st, e.max_abs_bend_st,
                len(e.pitch_times_s), assembly.pitch.dt, assembly.pitch.interpretation_hz, "V18.3_revealed_micro_remnant_topology_cooperation",
            ])

    attacks_path = output_mid.with_suffix(output_mid.suffix + ".attacks.csv")
    with attacks_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s", "accepted", "source", "reason", "acoustic_score", "acoustic_prominence",
            "pitch_source", "pitch_anchor_s", "pitch_anchor_st",
            "continuity_status", "barrier_time_s", "barrier_valley_dbfs",
            "barrier_depth_left_db", "barrier_depth_right_db",
            "barrier_low_duration_s",
            "owner_corridor_status", "owner_corridor_conflict_duration_s",
            "owner_corridor_primary_support_delay_s",
            "motion_status", "motion_span_st", "motion_max_step_st", "motion_monotonic_fraction",
            "reattack_status", "reattack_depth_left_db", "reattack_depth_right_db",
            "interpretation_dt_s", "interpretation_hz",
        ])
        for d in sorted(assembly.attack_decisions, key=lambda x: (x.time_s, not x.accepted)):
            w.writerow([
                d.time_s, int(d.accepted), d.source, d.reason,
                d.acoustic_score, d.acoustic_prominence, d.pitch_source,
                d.pitch_anchor_s, d.pitch_anchor_st,
                d.continuity_status, d.barrier_time_s, d.barrier_valley_dbfs,
                d.barrier_depth_left_db, d.barrier_depth_right_db,
                d.barrier_low_duration_s,
                d.owner_corridor_status, d.owner_corridor_conflict_duration_s,
                d.owner_corridor_primary_support_delay_s,
                d.motion_status, d.motion_span_st, d.motion_max_step_st, d.motion_monotonic_fraction,
                d.reattack_status, d.reattack_depth_left_db, d.reattack_depth_right_db,
                assembly.pitch.dt, assembly.pitch.interpretation_hz,
            ])

    # Corrected-valid coverage remains a hard audit invariant, but unlike V9 a
    # short island may be intentionally admitted by independent attack evidence.
    coverage_path = output_mid.with_suffix(output_mid.suffix + ".coverage.csv")
    with coverage_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "pitch_st", "status", "event_index"])
        uncovered = 0
        for t, st, valid in zip(assembly.pitch.t, assembly.pitch.shape, assembly.pitch.valid):
            if not valid:
                continue
            owner = next((i for i, e in enumerate(assembly.events) if e.start_s - 1e-9 <= t < e.end_s + 1e-9), None)
            if owner is None:
                uncovered += 1
                w.writerow([t, st, "UNOWNED_VALID_F0", ""])
            else:
                w.writerow([t, st, "covered", owner])
    if uncovered:
        print(f"WARNING: {uncovered} corrected-valid interpretation points are not rendered; see {coverage_path}")


def _write_rate_contract(output_mid: Path, assembly: AssemblyResult) -> Path:
    """Write the renderer's explicit timing/rate contract for regression audits."""
    path = output_mid.with_suffix(output_mid.suffix + ".rate_contract.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "value", "units", "policy"])
        rows = [
            ("renderer_version", "V18.3_revealed_micro_remnant_topology_cooperation", "", "renderer-only; frozen V3R9 input"),
            ("v17_1_residual_min_stable_factor", RESIDUAL_VALID_MIN_STABLE_FACTOR, "x upstream min_stable", "V17.1 cannot promote micro residuals below the existing target-former stability scale"),
            ("v17_1_existing_attack_anchor_tolerance", RESIDUAL_ATTACK_ANCHOR_TOLERANCE_S, "s", "accepted attack may predate first trustworthy residual F0 only when its frozen pitch anchor lies in that run"),
            ("v17_1_same_family_join_max_gap", RESIDUAL_JOIN_MAX_GAP_S, "s", "maximum gap for residual ownership transfer into immediately preceding same-family event"),
            ("v17_1_same_family_tail_gap", RESIDUAL_JOIN_MAX_TAIL_GAP_ST, "semitones", "preceding sounding tail must agree with residual family"),
            ("v17_1_reattack_proof_window", RESIDUAL_REATTACK_PROOF_WINDOW_S, "s", "accepted physical reattack near residual start preserves a separate Note-On"),
            ("v18_breath_backdate_min_lag", BREATH_BACKDATE_MIN_LAG_S, "s", "minimum early-attack lag before breath/noise backdating adjudication"),
            ("v18_breath_replacement_max_preanchor", BREATH_REPLACEMENT_MAX_PREANCHOR_S, "s", "later accepted trusted/reset attack must lie near the same pitch anchor"),
            ("v18_ultrashort_max", ULTRASHORT_MAX_S, "s", "ultra-short same-owner WAV rearticulation cleanup ceiling"),
            ("v18_transient_head_max", TRANSIENT_HEAD_MAX_S, "s", "short backdated transient head eligible for PB folding"),
            ("v18_acoustic_rescue_min_acf", ACOUSTIC_RESCUE_MIN_ACF, "ratio", "moderate primary rescue requires independent strong WAV attack"),
            ("v18_acoustic_rescue_max_cmndf", ACOUSTIC_RESCUE_MAX_CMNDF, "ratio", "moderate primary rescue requires ACF/CMNDF pitch agreement"),
            ("v18_subharmonic_min_octave_dominance", SUBHARMONIC_MIN_OCTAVE_DOMINANCE_DB, "dB", "extreme octave-up spectral certificate plus local-context improvement"),
            ("v18_1_weak_backdate_start", WEAK_BACKDATE_START_DBFS, "dBFS", "weak borrowed starts may use near-anchor primary periodicity"),
            ("v18_1_weak_backdate_min_lag", WEAK_BACKDATE_MIN_LAG_S, "s", "minimum lag for calibrated weak-start fallback"),
            ("v18_1_weak_island_max", WEAK_ISLAND_MAX_S, "s", "short weak trusted island eligible for PB folding"),
            ("v18_1_weak_island_start", WEAK_ISLAND_START_DBFS, "dBFS", "absolute weak-start threshold"),
            ("v18_1_weak_island_next_gain", WEAK_ISLAND_MIN_NEXT_GAIN_DB, "dB", "following owner must be materially stronger"),
            ("v18_1_terminal_decay_max", TERMINAL_DECAY_MAX_S, "s", "terminal decay island ceiling"),
            ("v18_1_terminal_decay_left_depth", TERMINAL_DECAY_MIN_LEFT_DEPTH_DB, "dB", "decay morphology: substantial left collapse"),
            ("v18_1_terminal_decay_right_depth", TERMINAL_DECAY_MAX_RIGHT_DEPTH_DB, "dB", "decay morphology: no right restart"),
            ("v18_1_repeat_min_score", REPEAT_MIN_ATTACK_SCORE, "z-score composite", "asymmetric repetition attack strength"),
            ("v18_1_repeat_left_depth", REPEAT_MIN_LEFT_DEPTH_DB, "dB", "minimum left dip for isolated asymmetric repeat"),
            ("v18_1_repeat_right_depth", REPEAT_MIN_RIGHT_DEPTH_DB, "dB", "minimum right restart for repetition"),
            ("v18_1_decay_subharmonic_head1", DECAY_SUBHARMONIC_HEAD1_DB, "dB", "octave-up dominance at decaying event head"),
            ("v18_1_decay_subharmonic_head2", DECAY_SUBHARMONIC_HEAD2_DB, "dB", "second early octave-up support sample"),
            ("v18_1_decay_subharmonic_rms_fall", DECAY_SUBHARMONIC_MIN_RMS_FALL_DB, "dB", "required decay across event"),
            ("v18_1_fast_pb_trend_half_window", FAST_PB_TREND_HALF_WINDOW_S, "s", "local least-squares trend half-window"),
            ("v18_1_fast_pb_half_cycle_min", FAST_PB_MIN_HALF_CYCLE_S, "s", "fast alternating residual minimum half-cycle"),
            ("v18_1_fast_pb_half_cycle_max", FAST_PB_MAX_HALF_CYCLE_S, "s", "fast alternating residual maximum half-cycle"),
            ("v18_1_fast_pb_residual_factor", FAST_PB_COMPRESSION, "ratio", "only alternating residual is compressed; local trend is unchanged"),
            ("v18_2_unsupported_backdate_min_lag", UNSUPPORTED_BACKDATE_MIN_LAG_S, "s", "minimum unsupported borrowed lead"),
            ("v18_2_unsupported_backdate_start", UNSUPPORTED_BACKDATE_START_DBFS, "dBFS", "maximum weak start level"),
            ("v18_2_unsupported_backdate_median", UNSUPPORTED_BACKDATE_MAX_MEDIAN_DBFS, "dBFS", "maximum median level of borrowed corridor"),
            ("v18_2_unsupported_backdate_min_remaining", UNSUPPORTED_BACKDATE_MIN_REMAINING_S, "s", "inherited revealed-micro threshold; V18.3 adjudicates continuous sub-threshold remnants"),
            ("v18_3_revealed_micro_max", REVEALED_MICRO_MAX_S, "s", "sub-threshold unsupported-start remnant routed through existing topology"),
            ("v18_3_revealed_micro_next_gap", REVEALED_MICRO_MAX_GAP_NEXT_S, "s", "existing V18 transient adjacency aperture reused for micro-remnant folding"),
            ("v18_2_unsupported_primary_min_acf", UNSUPPORTED_PRIMARY_MIN_ACF, "ratio", "moderate primary support veto"),
            ("v18_2_unsupported_primary_max_cmndf", UNSUPPORTED_PRIMARY_MAX_CMNDF, "ratio", "moderate primary support veto"),
            ("v18_2_unsupported_primary_max_disagreement", UNSUPPORTED_PRIMARY_MAX_ACF_CMNDF_CENTS, "cents", "ACF/CMNDF agreement veto"),
            ("v18_2_unsupported_primary_family_gap", UNSUPPORTED_PRIMARY_MAX_FAMILY_GAP_ST, "st", "target-family periodicity veto"),
            ("interpretation_dt", assembly.pitch.dt, "s", "measured from input; grid-dependent logic derives from this"),
            ("interpretation_hz", assembly.pitch.interpretation_hz, "Hz", "diagnostic only; no 100-Hz assumption"),
            ("attack_lane_hop", assembly.attack_lane.hop_s, "s", "physical WAV-analysis hop"),
            ("reset_post_attack_grace", RESET_POST_ATTACK_GRACE_S, "s", "physical acoustic/reset association tolerance"),
            ("primary_query_preroll", PRIMARY_QUERY_PREROLL_S, "s", "physical primary-frame query tolerance"),
            ("primary_bridge_postroll", PRIMARY_BRIDGE_POSTROLL_S, "s", "physical primary-frame bridge tolerance"),
            ("corridor_long_unpitched", CORRIDOR_LONG_UNPITCHED_S, "s", "physical inherited V15/V16 owner-corridor threshold"),
            ("corridor_early_primary_support", CORRIDOR_EARLY_PRIMARY_SUPPORT_S, "s", "physical inherited V15/V16 owner-corridor threshold"),
            ("corridor_primary_delay_fraction", CORRIDOR_PRIMARY_DELAY_FRACTION, "ratio", "fraction of attack->owner lag"),
            ("corridor_attack_floor_margin", CORRIDOR_ATTACK_FLOOR_MARGIN_DB, "dB", "recording-adaptive near-floor gate"),
            ("corridor_owner_level_gain", CORRIDOR_OWNER_LEVEL_GAIN_DB, "dB", "later trusted owner must be materially stronger"),
            ("motion_window", MOTION_WINDOW_S, "s", "physical trusted-curve continuity aperture"),
            ("motion_active_span", MOTION_ACTIVE_SPAN_ST, "semitones", "minimum local excursion for strict directed motion"),
            ("motion_large_span", MOTION_LARGE_SPAN_ST, "semitones", "large-gliss directed-motion threshold"),
            ("motion_strict_monotonic_fraction", MOTION_STRICT_MONOTONIC_FRACTION, "ratio", "strict small/medium gliss directionality"),
            ("motion_large_monotonic_fraction", MOTION_LARGE_MONOTONIC_FRACTION, "ratio", "large-gliss directionality"),
            ("motion_transition_radius", MOTION_TRANSITION_RADIUS_S, "s", "physical proximity to tracker transition for false-rearticulation protection"),
            ("reattack_min_bilateral_depth", REATTACK_MIN_BILATERAL_DEPTH_DB, "dB", "independent physical reattack requires envelope collapse and restart"),
            ("directed_crossing_half_window", DIRECTED_CROSSING_HALF_WINDOW_S, "s", "trusted pitch must move on both sides of a suppressed structural boundary"),
            ("directed_crossing_min_slope", DIRECTED_CROSSING_MIN_SLOPE_ST_PER_S, "st/s", "minimum same-direction slope on each side"),
            ("directed_crossing_max_slope_ratio", DIRECTED_CROSSING_MAX_SLOPE_RATIO, "ratio", "pre/post slope magnitudes must describe one coherent ramp"),
            ("post_structural_duplicate_window", POST_STRUCTURAL_DUPLICATE_WINDOW_S, "s", "suppresses detector echo immediately after an already-owned stable-target boundary"),
            ("short_island_smoothing_guard", SHORT_ISLAND_SMOOTHING_GUARD_S, "s", "do not smooth away late articulation in short trusted islands"),
            ("short_island_late_attack", SHORT_ISLAND_LATE_ATTACK_S, "s", "late-attack threshold for short-island coverage guard"),
            ("handoff_max_gap", HANDOFF_MAX_GAP_S, "s", "V16.1 only: maximum V16-created corridor gap eligible for continuity handoff"),
            ("handoff_min_acf", HANDOFF_MIN_ACF, "ratio", "V16.1 transition evidence; cannot create Note-On or pitch family"),
            ("handoff_max_cmndf", HANDOFF_MAX_CMNDF, "ratio", "V16.1 transition evidence; cannot create Note-On or pitch family"),
            ("handoff_max_acf_cmndf_disagreement", HANDOFF_MAX_ACF_CMNDF_DISAGREEMENT_CENTS, "cents", "V16.1 transition ACF/CMNDF family agreement"),
            ("handoff_endpoint_margin", HANDOFF_ENDPOINT_MARGIN_ST, "semitones", "V16.1 transition candidate must remain bracketed by adjacent trusted owners"),
            ("handoff_min_audible_margin", HANDOFF_MIN_AUDIBLE_MARGIN_DB, "dB", "V16.1 transition gap must remain above recording-adaptive floor"),
            ("pitch_only_future_match", PITCH_ONLY_FUTURE_MATCH_ST, "semitones", "suppressed articulation may hand PB authority only to a corroborated future trusted family"),
            ("pitch_only_max_future_lag", PITCH_ONLY_MAX_FUTURE_LAG_S, "s", "maximum pitch-only handoff lookahead; no Note-On is created"),
        ]
        w.writerows(rows)
    return path


def render_midi(
    prefix: Path,
    wav_path: Path,
    output_mid: Path,
    *,
    bpm: float,
    time_signature: tuple[int, int] = (4, 4),
    velocity: int | None = None,
    channel: int = MELODY_CHANNEL,
    write_audits: bool = True,
) -> list[MusicalEvent]:
    prefix = Path(prefix)
    wav_path = Path(wav_path)
    output_mid = Path(output_mid)
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    if not 0 <= channel <= 15 or channel == 9:
        raise ValueError("melody channel must be 0..15 and must not be percussion channel 9")

    assembly = assemble_events(prefix, wav_path)
    events = assembly.events
    cc11_baselines, modulation = _assign_expression_baselines(prefix, assembly, velocity)

    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    meta = mido.MidiTrack(); midi.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0))
    num, den = time_signature
    meta.append(mido.MetaMessage(
        "time_signature", numerator=num, denominator=den,
        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
    ))
    meta.append(mido.MetaMessage("end_of_track", time=0))

    melody = mido.MidiTrack(); midi.tracks.append(melody)
    melody.append(mido.MetaMessage("track_name", name="ECKF Melody - Renderer V18.3 Revealed Micro-Remnant Topology Cooperation", time=0))
    melody.append(mido.Message("program_change", channel=channel, program=MIDO_PROGRAM, time=0))
    melody.append(mido.Message("pitchwheel", channel=channel, pitch=PB_CENTER, time=0))
    melody.append(mido.Message("control_change", channel=channel, control=CC7_VOLUME, value=CC7_NEUTRAL, time=0))
    melody.append(mido.Message("control_change", channel=channel, control=CC11_EXPRESSION, value=CC11_NEUTRAL, time=0))

    wire: list[tuple[int, int, mido.Message]] = []
    for idx, event in enumerate(events):
        start_tick = _seconds_to_tick(event.start_s, bpm)
        end_tick = max(start_tick + 1, _seconds_to_tick(event.end_s, bpm))
        if event.bend_range_st > 0:
            priority = 20
            for msg in _rpn_pitch_bend_range(channel, event.bend_range_st):
                wire.append((start_tick, priority, msg)); priority += 1
            wire.append((start_tick, 30, mido.Message("pitchwheel", channel=channel, pitch=0, time=0)))

        baseline = int(cc11_baselines.get(idx, CC11_NEUTRAL))
        wire.append((start_tick, 35, mido.Message(
            "control_change", channel=channel, control=CC11_EXPRESSION, value=baseline, time=0,
        )))
        wire.append((start_tick, 40, mido.Message(
            "note_on", channel=channel, note=event.base_midi, velocity=event.velocity, time=0,
        )))

        last_cc11 = baseline
        for t_s, delta_db in modulation.get(idx, []):
            tick = int(np.clip(_seconds_to_tick(t_s, bpm), start_tick, end_tick - 1))
            ratio = 10.0 ** ((CC11_DB_COMPRESSION * delta_db) / 20.0)
            value = int(np.clip(round(baseline * ratio), 1, 127))
            if value == last_cc11:
                continue
            wire.append((tick, 55, mido.Message(
                "control_change", channel=channel, control=CC11_EXPRESSION, value=value, time=0,
            )))
            last_cc11 = value

        if event.bend_range_st > 0:
            last_pb: int | None = None
            for t_s, p_st in zip(event.pitch_times_s, event.pitch_st):
                tick = int(np.clip(_seconds_to_tick(float(t_s), bpm), start_tick, end_tick - 1))
                value = _bend_value(float(p_st), event.base_midi, event.bend_range_st)
                if last_pb == value:
                    continue
                wire.append((tick, 50, mido.Message("pitchwheel", channel=channel, pitch=value, time=0)))
                last_pb = value

        wire.append((end_tick, 0, mido.Message(
            "note_off", channel=channel, note=event.base_midi, velocity=0, time=0,
        )))
        # Monophonic wire invariant: every Note Off returns PB to center.
        # At a same-tick legato boundary, Note Off and this reset are emitted
        # before the next note's RPN/PB setup and Note On.
        wire.append((end_tick, 10, mido.Message("pitchwheel", channel=channel, pitch=0, time=0)))

    wire.sort(key=lambda x: (x[0], x[1]))
    last_tick = 0
    for tick, _, msg in wire:
        msg.time = max(0, tick - last_tick)
        melody.append(msg)
        last_tick = tick
    melody.append(mido.MetaMessage("end_of_track", time=0))

    output_mid.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_mid)
    if write_audits:
        _write_audits(prefix, output_mid, assembly, cc11_baselines)
        _write_rate_contract(output_mid, assembly)
    return events


def _infer_wav(prefix: Path) -> Path | None:
    name = prefix.name
    # tests/Song_v3R5.csv -> tests/Song.wav
    stem = name[:-4] if name.lower().endswith(".csv") else name
    marker = stem.lower().rfind("_v")
    if marker > 0:
        candidate = prefix.with_name(stem[:marker] + ".wav")
        if candidate.exists():
            return candidate
    return None


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="MIDI Renderer V18: frozen V3R9 + accepted V17.1 ownership + acoustic adjudication/transient cleanup.",
    )
    ap.add_argument("prefix", type=Path, help="frozen V3R9 main CSV prefix (older compatible prefixes remain supported)")
    ap.add_argument("--wav", type=Path, default=None,
                    help="source WAV for acoustic attack timing; normally required (auto-inferred when possible)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--time-signature", default="4/4", type=_parse_time_signature)
    ap.add_argument("--velocity", type=int, default=None,
                    help="fixed Note-On velocity override; default 100; dynamics live on CC11")
    ap.add_argument("--channel", type=int, default=MELODY_CHANNEL,
                    help="zero-based MIDI channel; default 0 (DAW channel 1); channel 9 forbidden")
    args = ap.parse_args(list(argv) if argv is not None else None)

    wav = args.wav or _infer_wav(args.prefix)
    if wav is None:
        ap.error("--wav is required when the source WAV cannot be inferred from the CSV prefix")

    notes = render_midi(
        args.prefix, wav, args.out,
        bpm=args.bpm, time_signature=args.time_signature,
        velocity=args.velocity, channel=args.channel,
    )
    print(f"MIDI: {args.out}")
    print(f"Events: {args.out.with_suffix(args.out.suffix + '.events.csv')}")
    print(f"Attacks: {args.out.with_suffix(args.out.suffix + '.attacks.csv')}")
    print(f"Coverage: {args.out.with_suffix(args.out.suffix + '.coverage.csv')}")
    print(f"Rate contract: {args.out.with_suffix(args.out.suffix + '.rate_contract.csv')}")
    print(f"Notes: {len(notes)}")
    print(f"Renderer: V18.3 -- V18.2 unsupported-start retraction + revealed micro-remnant topology cooperation")
    print(f"Format: MIDI 1.0 SMF Type 1, PPQ={PPQ}")
    print(f"GM program: #{GM_PROGRAM_NUMBER} Soprano Saxophone")


if __name__ == "__main__":
    main()
