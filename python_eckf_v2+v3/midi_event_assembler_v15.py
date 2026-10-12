from __future__ import annotations

"""Clean-sheet musical-event assembler for MIDI Renderer V15.

The assembler consumes parallel evidence lanes once.  It does not replay the
legacy renderer's chain of rescues/reinterpretations.
"""

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

from .midi_attack_evidence_v13 import AcousticAttack, AttackLane, OwnershipContinuity, analyze_wav_attacks
from .midi_pitch_evidence_v10 import PitchEvidence, PitchIsland, PrimaryPitchPoint


# ---------------------------------------------------------------------------
# Renderer timing contract
# ---------------------------------------------------------------------------
# Interpretation-grid quantities MUST derive from PitchEvidence.dt.  Constants
# below are deliberately expressed in physical seconds because they describe
# acoustic/musical apertures rather than a number of tracker samples.
RESET_PRE_ATTACK_WINDOW_S = 0.080
RESET_POST_ATTACK_GRACE_S = 0.010
PRIMARY_QUERY_PREROLL_S = 0.010
PRIMARY_BRIDGE_POSTROLL_S = 0.020

# V15 attack->owner corridor compatibility.
CORRIDOR_ENTRY_PITCH_WINDOW_S = 0.040
CORRIDOR_CONTRADICTORY_GAP_ST = 2.50
CORRIDOR_CONTRADICTORY_MIN_S = 0.025
CORRIDOR_LONG_UNPITCHED_S = 0.140
CORRIDOR_EARLY_PRIMARY_SUPPORT_S = 0.100
CORRIDOR_PRIMARY_DELAY_FRACTION = 0.90
CORRIDOR_ATTACK_FLOOR_MARGIN_DB = 18.0
CORRIDOR_OWNER_LEVEL_GAIN_DB = 5.0
CORRIDOR_PRIMARY_MATCH_ST = 2.50
CORRIDOR_BORROW_STRENGTH_PERCENTILE = 70.0
CORRIDOR_BORROW_STRENGTH_FLOOR = 12.0


@dataclass(frozen=True)
class OwnerCorridorCompatibility:
    compatible: bool
    status: str
    conflict_duration_s: float = math.nan
    primary_support_delay_s: float = math.nan


@dataclass
class AttackDecision:
    time_s: float
    accepted: bool
    source: str
    reason: str
    acoustic_score: float = math.nan
    acoustic_prominence: float = math.nan
    pitch_source: str = ""
    pitch_anchor_s: float = math.nan
    pitch_anchor_st: float = math.nan
    continuity_status: str = ""
    barrier_time_s: float = math.nan
    barrier_valley_dbfs: float = math.nan
    barrier_depth_left_db: float = math.nan
    barrier_depth_right_db: float = math.nan
    barrier_low_duration_s: float = math.nan
    owner_corridor_status: str = ""
    owner_corridor_conflict_duration_s: float = math.nan
    owner_corridor_primary_support_delay_s: float = math.nan


@dataclass
class MusicalEvent:
    start_s: float
    end_s: float
    base_st: float
    base_midi: int
    pitch_times_s: np.ndarray
    pitch_st: np.ndarray
    source_kind: str
    source_id: str
    attack_score: float = math.nan
    pitch_source: str = "v3r5_corrected_valid"
    attack_dbfs: float = math.nan
    expression_anchor_s: float = math.nan
    expression_dbfs: float = math.nan
    velocity: int = 100
    bend_range_st: int = 0
    max_abs_bend_st: float = 0.0


@dataclass
class AssemblyResult:
    events: list[MusicalEvent]
    attack_decisions: list[AttackDecision]
    attack_lane: AttackLane
    pitch: PitchEvidence


def _nearest_midi(st: float) -> int:
    return int(np.clip(int(round(float(st))), 0, 127))


def _dedupe_times(items: list[tuple[float, str, float]], radius_s: float) -> list[tuple[float, str, float]]:
    chosen: list[tuple[float, str, float]] = []
    for t, source, strength in sorted(items, key=lambda x: (-x[2], x[0])):
        if all(abs(t - q[0]) > radius_s for q in chosen):
            chosen.append((float(t), source, float(strength)))
    return sorted(chosen, key=lambda x: x[0])


def _attack_strength(p: AcousticAttack | None) -> float:
    if p is None:
        return -math.inf
    return float(p.novelty_z + 0.5 * p.prominence_z)


def _eligible_short_island(island: PitchIsland, attack: AcousticAttack | None, has_reset: bool) -> bool:
    if island.duration_s + 1e-9 >= island.min_stable_s:
        return True
    # Duration is evidence, not a veto.  A short trusted F0 island survives when
    # an independent articulation lane or ECKF reset supports it.
    if has_reset:
        return True
    return attack is not None and _attack_strength(attack) >= 2.0


def _event_curve_from_island(pitch: PitchEvidence, island: PitchIsland, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    tt, pp = pitch.pitch_curve(max(start_s, island.start_s), min(end_s, island.end_s), island.median_st)
    # Acoustic Note-On may precede trustworthy F0.  Hold the first trusted pitch
    # backward to the attack; pitch *timing* and pitch *authority* remain
    # separate questions.
    if start_s < tt[0] - 1e-9:
        tt = np.insert(tt, 0, start_s)
        pp = np.insert(pp, 0, pp[0])
    if end_s > tt[-1] + 1e-9:
        tt = np.append(tt, end_s)
        pp = np.append(pp, pp[-1])
    return tt, pp


def _event_curve_from_primary(points: list[PrimaryPitchPoint], start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    tt = np.asarray([p.time_s for p in points], dtype=float)
    pp = np.asarray([p.pitch_st for p in points], dtype=float)
    if len(tt) == 0:
        raise ValueError("primary curve needs points")
    if start_s < tt[0] - 1e-9:
        tt = np.insert(tt, 0, start_s); pp = np.insert(pp, 0, pp[0])
    if end_s > tt[-1] + 1e-9:
        tt = np.append(tt, end_s); pp = np.append(pp, pp[-1])
    return tt, pp


def _entry_pitch_st(island: PitchIsland, pitch_dt: float) -> float:
    """Pitch family actually entering the island, not its whole-island median."""
    window_s = max(CORRIDOR_ENTRY_PITCH_WINDOW_S, 4.0 * float(pitch_dt))
    mask = island.times_s <= island.start_s + window_s + 1e-9
    vals = island.pitch_st[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals):
        return float(np.median(vals))
    return float(island.median_st)


def _longest_true_run_duration(times_s: np.ndarray, mask: np.ndarray, dt: float) -> float:
    """Physical duration of the longest contiguous True run on an input grid."""
    times_s = np.asarray(times_s, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(times_s) == 0 or not np.any(mask):
        return 0.0
    best = 0.0
    run_start: float | None = None
    prev: float | None = None
    for t, yes in zip(times_s, mask):
        if not yes:
            if run_start is not None and prev is not None:
                best = max(best, prev - run_start + dt)
            run_start = None; prev = None
            continue
        if run_start is None or prev is None or t - prev > 1.6 * dt:
            if run_start is not None and prev is not None:
                best = max(best, prev - run_start + dt)
            run_start = float(t)
        prev = float(t)
    if run_start is not None and prev is not None:
        best = max(best, prev - run_start + dt)
    return float(best)


def _owner_corridor_compatibility(
    lane: AttackLane,
    pitch: PitchEvidence,
    attack: AcousticAttack,
    island: PitchIsland,
    borrow_strength_gate: float,
) -> OwnerCorridorCompatibility:
    """V15 fail-safe for renderer-created backward attack ownership.

    A future trusted owner may still borrow a consonant/plosive attack before
    corrected-valid F0.  V15 refuses only two much stronger contradictions:

      1. sustained *trusted* corrected-valid pitch of a different family already
         occupies the attack->owner corridor; or
      2. the renderer is borrowing a weak/ordinary attack across a long pitchless
         lead, with no owner-compatible primary periodicity arriving early.

    All durations are evaluated in physical time.  Input-grid continuity uses
    pitch.dt, so this logic is not tied to the old 100-Hz interpretation rate.
    """
    lag = float(island.start_s - attack.time_s)
    if not math.isfinite(lag) or lag <= max(0.0, pitch.dt):
        return OwnerCorridorCompatibility(True, "short_or_nonbackdated")

    entry_st = _entry_pitch_st(island, pitch.dt)

    # Certificate A: another trusted pitch family already occupies the corridor.
    m = (
        pitch.valid
        & (pitch.t >= attack.time_s - 0.5 * pitch.dt)
        & (pitch.t < island.start_s - 0.5 * pitch.dt)
    )
    if np.any(m):
        tt = pitch.t[m]
        pp = pitch.shape[m]
        conflict = np.abs(pp - entry_st) >= CORRIDOR_CONTRADICTORY_GAP_ST
        conflict_duration = _longest_true_run_duration(tt, conflict, pitch.dt)
        if conflict_duration + 1e-9 >= CORRIDOR_CONTRADICTORY_MIN_S:
            # MRV15 records this geometry conceptually but does not revoke
            # ownership on trusted-pitch contradiction alone.  Corpus preflight
            # showed that doing so before continuous-motion/rearticulation
            # cleanup exposed too many inherited splits.  The V14 nearer-stable
            # certificate remains the authoritative contradiction guard.
            pass

    # Certificate B: long pitchless lead.  This is deliberately NOT a simple
    # maximum attack->F0 lag: an unusually strong plosive/consonant may still own
    # a later vowel.  Ordinary attacks need owner-compatible primary periodicity
    # to appear reasonably soon.
    if lag + 1e-9 >= CORRIDOR_LONG_UNPITCHED_S:
        primary = pitch.primary_support(
            attack.time_s - PRIMARY_QUERY_PREROLL_S,
            island.start_s + min(pitch.dt, RESET_POST_ATTACK_GRACE_S),
            future_st=entry_st,
        )
        matching = [
            q for q in primary
            if abs(q.pitch_st - entry_st) <= CORRIDOR_PRIMARY_MATCH_ST
        ]
        support_delay = (
            float(matching[0].time_s - attack.time_s) if matching else math.inf
        )
        delayed_support = support_delay > max(
            CORRIDOR_EARLY_PRIMARY_SUPPORT_S,
            CORRIDOR_PRIMARY_DELAY_FRACTION * lag,
        ) + 1e-9
        near_floor_attack = (
            math.isfinite(attack.rms_dbfs)
            and attack.rms_dbfs <= lane.active_rms_floor_db + CORRIDOR_ATTACK_FLOOR_MARGIN_DB
        )
        owner_rms = lane.rms_at(island.start_s)
        owner_level_gain = (
            owner_rms - attack.rms_dbfs
            if math.isfinite(owner_rms) and math.isfinite(attack.rms_dbfs)
            else -math.inf
        )
        if (
            delayed_support
            and near_floor_attack
            and owner_level_gain >= CORRIDOR_OWNER_LEVEL_GAIN_DB
            and _attack_strength(attack) < borrow_strength_gate
        ):
            return OwnerCorridorCompatibility(
                False,
                "long_unpitched_weak_lead_v15",
                primary_support_delay_s=support_delay,
            )
        return OwnerCorridorCompatibility(
            True,
            "long_lead_supported",
            primary_support_delay_s=support_delay,
        )

    return OwnerCorridorCompatibility(True, "compatible")


def _select_attack_for_island(
    lane: AttackLane, pitch: PitchEvidence, island: PitchIsland,
    association_s: float, reset_radius_s: float,
    earliest_attack_s: float = 0.0,
) -> tuple[AcousticAttack | None, OwnershipContinuity, list[tuple[AcousticAttack, OwnershipContinuity]]]:
    # Pitch-ownership topology is a hard boundary.  A future trusted island may
    # look backward into the tail of the immediately preceding owner, because a
    # real rearticulation can precede the first trustworthy frame.  It may not
    # look backward *across the start* of that preceding trusted island and steal
    # an attack belonging to an earlier ownership object.
    start = max(float(earliest_attack_s), 0.0, island.start_s - association_s)
    # The interpreted pitch point is window-centered evidence, not a
    # sample-accurate attack timestamp.  Permit a small post-anchor grace so a
    # physical restart a few milliseconds after the first trusted frame can own
    # that frame.
    post_anchor_grace = float(np.clip(1.5 * pitch.dt, 0.012, 0.020))
    end = island.start_s + min(RESET_POST_ATTACK_GRACE_S, pitch.dt)
    candidates = [p for p in lane.peaks if start <= p.time_s <= end and lane.is_audible(p)]

    # V13 object-corridor invariant: an attack can own later pitch only if the acoustic
    # path remains connected.  A breath/noise object that dies before a later
    # restart is rejected even if both lie inside the same association aperture.
    evaluated: list[tuple[AcousticAttack, OwnershipContinuity]] = []
    for candidate in candidates:
        ev = lane.ownership_continuity(
            candidate.time_s, island.start_s, post_anchor_grace_s=post_anchor_grace,
        )
        evaluated.append((candidate, ev))
    connected = [(p, ev) for p, ev in evaluated if ev.connected]
    rejected = [(p, ev) for p, ev in evaluated if not ev.connected]

    if not connected:
        fallback = lane.best_peak(start, end)
        if fallback is None:
            return None, OwnershipContinuity(True, "no_attack_candidate"), rejected
        ev = lane.ownership_continuity(
            fallback.time_s, island.start_s, post_anchor_grace_s=post_anchor_grace,
        )
        if not ev.connected:
            if all(abs(fallback.time_s - p.time_s) > lane.hop_s for p, _ in rejected):
                rejected.append((fallback, ev))
            return None, ev, rejected
        connected = [(fallback, ev)]

    # First prefer the latest clear amplitude restart before trustworthy F0.
    # This finds consonant/noisy attacks without confusing a later vowel
    # periodicity peak for the musical Note-On.
    valley = [(p, ev) for p, ev in connected
              if p.time_s <= island.start_s - 0.012
              and lane.preceding_valley_db(p.time_s, max(0.06, pitch.min_stable_s)) >= 3.0]
    if valley:
        return (*max(valley, key=lambda x: x[0].time_s), rejected)

    # If the CLI emitted a reset in the same ownership aperture, use the
    # strongest connected audible peak immediately preceding that reset.
    resets = [r for r in pitch.reset_times if start <= r <= island.start_s + reset_radius_s]
    reset_candidates: list[tuple[AcousticAttack, OwnershipContinuity]] = []
    for r in resets:
        xs = [(p, ev) for p, ev in connected if r - RESET_PRE_ATTACK_WINDOW_S <= p.time_s <= r + RESET_POST_ATTACK_GRACE_S]
        if xs:
            reset_candidates.append(max(xs, key=lambda x: _attack_strength(x[0])))
    if reset_candidates:
        p, ev = max(reset_candidates, key=lambda x: _attack_strength(x[0]))
        return p, ev, rejected

    # Fallback: strongest connected acoustic event in the association aperture.
    p, ev = max(connected, key=lambda x: _attack_strength(x[0]))
    return p, ev, rejected


def assemble_events(prefix: Path, wav_path: Path) -> AssemblyResult:
    prefix = Path(prefix)
    pitch = PitchEvidence(prefix)
    lane = analyze_wav_attacks(Path(wav_path))
    decisions: list[AttackDecision] = []

    # Existing project timing scale remains useful, but it no longer acts as a
    # minimum-note-duration veto.  It determines only how far evidence may be
    # associated around a CLI ownership boundary.
    association_s = float(np.clip(4.0 * pitch.min_stable_s, 0.22, 0.34))
    local_align_s = float(np.clip(0.75 * pitch.min_stable_s, 0.04, 0.08))
    reset_radius_s = max(0.06, pitch.min_stable_s)
    dedupe_s = max(0.018, 2.0 * lane.hop_s)
    attack_strengths = np.asarray([
        _attack_strength(p) for p in lane.peaks if math.isfinite(_attack_strength(p))
    ], dtype=float)
    borrow_strength_gate = (
        max(CORRIDOR_BORROW_STRENGTH_FLOOR,
            float(np.percentile(attack_strengths, CORRIDOR_BORROW_STRENGTH_PERCENTILE)))
        if len(attack_strengths) else CORRIDOR_BORROW_STRENGTH_FLOOR
    )

    boundaries: list[tuple[float, str, float]] = []
    island_info: list[tuple[PitchIsland, float, AcousticAttack | None]] = []
    reserved_attack_to_pitch: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # 1) Every trusted corrected-valid island gets independent ownership
    #    consideration, irrespective of whether target formation promoted it.
    # ------------------------------------------------------------------
    for island_index, island in enumerate(pitch.islands):
        # A physical attack may occur in the decaying tail of the immediately
        # preceding trusted owner, so we do not force it to wait for prev.end_s.
        # But it may never cross the *start* of that previous owner.  This keeps
        # a future island from backdating through an intervening trusted island
        # and later erasing that island during monophonic overlap cleanup.
        previous_island = pitch.islands[island_index - 1] if island_index > 0 else None
        topology_floor = (previous_island.start_s + dedupe_s) if previous_island is not None else 0.0
        attack, continuity, rejected_attacks = _select_attack_for_island(
            lane, pitch, island, association_s, reset_radius_s, topology_floor
        )
        for rejected_attack, rejected_continuity in rejected_attacks:
            decisions.append(AttackDecision(
                rejected_attack.time_s, False, "wav_attack",
                "ownership_interruption_barrier",
                _attack_strength(rejected_attack), rejected_attack.prominence_z,
                island.source, island.start_s, island.median_st,
                rejected_continuity.status, rejected_continuity.barrier_time_s,
                rejected_continuity.valley_dbfs, rejected_continuity.depth_left_db,
                rejected_continuity.depth_right_db, rejected_continuity.low_duration_s,
            ))
        reset = pitch.reset_near(island.start_s, reset_radius_s)
        eligible = _eligible_short_island(island, attack, reset is not None)
        if not eligible:
            decisions.append(AttackDecision(
                island.start_s, False, "trusted_island_start",
                "short_island_without_independent_attack_or_reset",
                _attack_strength(attack) if attack else math.nan,
                attack.prominence_z if attack else math.nan,
                island.source, island.start_s, island.median_st,
            ))
            continue

        start = island.start_s
        source = "trusted_island_start"
        score = 50.0
        if attack is not None:
            lag = max(0.0, island.start_s - attack.time_s)
            # Long backward ownership is allowed for consonant/half-spoken
            # attacks, but it needs proportionally stronger acoustic evidence.
            # A weak local wiggle 300 ms before stable pitch must not steal the
            # Note-On; a huge plosive/transient may legitimately do so.
            required = 2.0 + 18.0 * (lag / max(association_s, 1e-9)) ** 2
            if _attack_strength(attack) >= required:
                # V14 owner-arbitration fail-safe.  V10-V13 correctly allowed a
                # trusted island to borrow an earlier physical attack when F0
                # became trustworthy late.  The dangerous edge case is a later
                # *different-pitch* island borrowing an attack that already has
                # a CLI-frozen stable-target owner right on top of it.  Generic
                # monophonic overlap cleanup would then let the future owner
                # erase the nearer one.
                #
                # Do not weaken or remove any V13 protection.  We only revoke
                # this renderer-created backdating when all three ownership
                # facts are explicit: (1) stable target is physically local to
                # the attack, (2) future trusted island is materially later,
                # and (3) the two pitch owners are materially different.
                stable_owner_radius_s = max(0.030, 3.0 * pitch.dt)
                future_owner_gap_s = max(0.080, 6.0 * pitch.dt)
                owner_pitch_gap_st = 4.0
                competing = [
                    b for b in pitch.structural_boundaries
                    if abs(b.time_s - attack.time_s) <= stable_owner_radius_s
                    and island.start_s - b.time_s >= future_owner_gap_s
                    and abs(b.pitch_st - island.median_st) >= owner_pitch_gap_st
                    and abs(b.time_s - attack.time_s) < abs(island.start_s - attack.time_s)
                ]
                if competing:
                    owner = min(competing, key=lambda b: abs(b.time_s - attack.time_s))
                    decisions.append(AttackDecision(
                        attack.time_s, False, "wav_attack",
                        "nearer_stable_target_owns_physical_attack_v14",
                        _attack_strength(attack), attack.prominence_z,
                        "v3r5_structural_pitch", owner.time_s, owner.pitch_st,
                        continuity.status, continuity.barrier_time_s,
                        continuity.valley_dbfs, continuity.depth_left_db,
                        continuity.depth_right_db, continuity.low_duration_s,
                        "v14_nearer_stable_owner", math.nan, math.nan,
                    ))
                    attack = None
                else:
                    corridor = _owner_corridor_compatibility(
                        lane, pitch, attack, island, borrow_strength_gate
                    )
                    if not corridor.compatible:
                        decisions.append(AttackDecision(
                            attack.time_s, False, "wav_attack", corridor.status,
                            _attack_strength(attack), attack.prominence_z,
                            island.source, island.start_s, _entry_pitch_st(island, pitch.dt),
                            continuity.status, continuity.barrier_time_s,
                            continuity.valley_dbfs, continuity.depth_left_db,
                            continuity.depth_right_db, continuity.low_duration_s,
                            corridor.status, corridor.conflict_duration_s,
                            corridor.primary_support_delay_s,
                        ))
                        attack = None
                    else:
                        start = min(island.start_s, attack.time_s)
                        source += "+wav_attack"
                        score += min(40.0, max(0.0, _attack_strength(attack)))
            else:
                attack = None
        if reset is not None:
            # A reset is supporting CLI evidence, but it must not move Note-On
            # later than an already-observed acoustic attack.
            if attack is None:
                start = min(start, reset)
            source += "+eckf_reset"
            score += 20.0
        boundaries.append((start, source, score))
        island_info.append((island, start, attack))
        if start < island.start_s - dedupe_s:
            reserved_attack_to_pitch.append((start, island.start_s))
        decisions.append(AttackDecision(
            start, True, source, "trusted_pitch_owner",
            _attack_strength(attack) if attack else math.nan,
            attack.prominence_z if attack else math.nan,
            island.source, island.start_s, island.median_st,
            continuity.status if attack is not None else "no_wav_attack",
            continuity.barrier_time_s if attack is not None else math.nan,
            continuity.valley_dbfs if attack is not None else math.nan,
            continuity.depth_left_db if attack is not None else math.nan,
            continuity.depth_right_db if attack is not None else math.nan,
            continuity.low_duration_s if attack is not None else math.nan,
        ))

    # ------------------------------------------------------------------
    # 2) Persistent stable-center changes from the CLI are legato Note-Ons.
    #    Nearby WAV novelty may sharpen their timing but cannot invent pitch.
    # ------------------------------------------------------------------
    for b in pitch.structural_boundaries:
        attack = lane.best_peak(b.time_s - local_align_s, b.time_s + local_align_s)
        reset = pitch.reset_near(b.time_s, reset_radius_s)
        interval = abs(b.pitch_st - b.previous_pitch_st)
        # A genuinely new stable center can articulate legato without a
        # waveform transient.  Near-same-pitch target restarts need independent
        # attack/reset evidence.
        supported = interval > pitch.stable_range_st or reset is not None or (attack is not None and _attack_strength(attack) >= 4.0)
        if not supported:
            continue
        t = b.time_s
        src = b.source
        strength = 80.0
        structural_continuity = OwnershipContinuity(True, "no_wav_attack")
        if attack is not None and _attack_strength(attack) >= 4.0 and lane.is_audible(attack):
            structural_continuity = lane.ownership_continuity(
                attack.time_s, b.time_s,
                post_anchor_grace_s=float(np.clip(1.5 * pitch.dt, 0.012, 0.020)),
            )
            if structural_continuity.connected:
                t = attack.time_s
                src += "+wav_aligned"
                strength += min(30.0, _attack_strength(attack))
            else:
                decisions.append(AttackDecision(
                    attack.time_s, False, "wav_attack",
                    "structural_alignment_interruption_barrier",
                    _attack_strength(attack), attack.prominence_z,
                    "v3r5_structural_pitch", b.time_s, b.pitch_st,
                    structural_continuity.status, structural_continuity.barrier_time_s,
                    structural_continuity.valley_dbfs, structural_continuity.depth_left_db,
                    structural_continuity.depth_right_db, structural_continuity.low_duration_s,
                ))
                attack = None
        boundaries.append((t, src, strength))
        decisions.append(AttackDecision(
            t, True, src, "cli_stable_target_start",
            _attack_strength(attack) if attack else math.nan,
            attack.prominence_z if attack else math.nan,
            "v3r5_structural_pitch", b.time_s, b.pitch_st,
            structural_continuity.status if attack is not None else "no_wav_alignment",
            structural_continuity.barrier_time_s if attack is not None else math.nan,
            structural_continuity.valley_dbfs if attack is not None else math.nan,
            structural_continuity.depth_left_db if attack is not None else math.nan,
            structural_continuity.depth_right_db if attack is not None else math.nan,
            structural_continuity.low_duration_s if attack is not None else math.nan,
        ))

    # ------------------------------------------------------------------
    # 3) Free acoustic attacks: accept only with pitch ownership evidence.
    #    This lane is what can recover same-pitch rearticulation and an attack
    #    that occurs before pitch becomes trustworthy.
    # ------------------------------------------------------------------
    accepted_free: list[tuple[float, str, float]] = []
    all_attack_strengths = np.asarray([_attack_strength(p) for p in lane.peaks if math.isfinite(_attack_strength(p))], dtype=float)
    reartic_gate = float(np.percentile(all_attack_strengths, 80.0)) if len(all_attack_strengths) else 16.0
    reset_attack_gate = float(np.percentile(all_attack_strengths, 40.0)) if len(all_attack_strengths) else 8.0
    reartic_gate = max(12.0, reartic_gate)
    reset_attack_gate = max(4.0, reset_attack_gate)
    for attack in lane.peaks:
        strength = _attack_strength(attack)
        if not lane.is_audible(attack):
            continue
        # When a trusted pitch island has already been pulled backward to a
        # preceding acoustic attack, intermediate spectral peaks belong to that
        # same attack-to-vowel settling corridor.  They must not become a stack
        # of extra MIDI notes before the trusted pitch anchor.
        corridor = next(((a, b) for a, b in reserved_attack_to_pitch
                         if a + dedupe_s < attack.time_s < b - dedupe_s), None)
        if corridor is not None:
            a, _b = corridor
            # Intermediate settling peaks are suppressed, but a clearly distinct
            # later articulation may still exist before F0 becomes trustworthy.
            # It needs both temporal separation and independent restart evidence.
            valley_here = lane.preceding_valley_db(attack.time_s, max(0.06, pitch.min_stable_s))
            reset_here = pitch.reset_near(attack.time_s, reset_radius_s)
            if attack.time_s - a < 0.10 or not (strength >= reartic_gate and (valley_here >= 2.5 or reset_here is not None)):
                continue
        # Ignore a free proposal already represented by a strong CLI-derived
        # boundary.  Fine timing alignment was handled above.
        if any(abs(attack.time_s - t) <= dedupe_s for t, _, _ in boundaries):
            continue

        current = pitch.island_containing(attack.time_s, tolerance_s=pitch.dt)
        future = pitch.next_island(attack.time_s, association_s)
        reset = pitch.reset_near(attack.time_s, reset_radius_s)

        # A) Attack before trustworthy pitch is handled by the trusted-island
        # alignment pass above.  Do not add a second boundary here: doing so
        # would turn one consonant-to-vowel onset into two MIDI notes.

        # B) Same-pitch / continuous-voicing rearticulation.  Strong waveform
        # novelty alone is not enough: require either an ECKF reset, an
        # amplitude restart, or unusually strong novelty.
        if current is not None:
            valley = lane.preceding_valley_db(attack.time_s, max(0.06, pitch.min_stable_s))
            reset_supported = reset is not None and strength >= reset_attack_gate
            envelope_supported = strength >= reartic_gate and valley >= 3.0
            if reset_supported or envelope_supported:
                accepted_free.append((attack.time_s, "wav_rearticulation", 45.0 + min(45.0, strength)))
                decisions.append(AttackDecision(
                    attack.time_s, True, "wav_rearticulation",
                    "continuous_pitch_with_independent_attack", strength, attack.prominence_z,
                    current.source, attack.time_s, current.median_st,
                ))
                continue

        # C) No corrected-valid owner: bounded primary-periodicity rescue.
        future_st = future.median_st if future is not None else None
        horizon_end = attack.time_s + association_s
        # another strong attack closes the ownership aperture
        later = [p for p in lane.peaks if p.time_s > attack.time_s + dedupe_s and p.time_s < horizon_end and _attack_strength(p) >= max(4.0, 0.8 * strength)]
        if later:
            horizon_end = min(p.time_s for p in later)
        primary = pitch.primary_support(attack.time_s - PRIMARY_QUERY_PREROLL_S, horizon_end, future_st=future_st)
        if primary:
            # one extremely clean frame can service a very short spoken/pitched
            # attack; otherwise require persistence across at least two frames.
            persistent = len(primary) >= 2 or (primary[0].acf >= 0.93 and primary[0].cmndf <= 0.08)
            primary_continuity = lane.ownership_continuity(
                attack.time_s, primary[0].time_s,
                post_anchor_grace_s=float(np.clip(1.5 * pitch.dt, 0.012, 0.020)),
            )
            if not primary_continuity.connected:
                decisions.append(AttackDecision(
                    attack.time_s, False, "wav_attack+primary_periodicity",
                    "primary_rescue_interruption_barrier", strength, attack.prominence_z,
                    primary[0].source, primary[0].time_s, primary[0].pitch_st,
                    primary_continuity.status, primary_continuity.barrier_time_s,
                    primary_continuity.valley_dbfs, primary_continuity.depth_left_db,
                    primary_continuity.depth_right_db, primary_continuity.low_duration_s,
                ))
                continue
            if persistent:
                if current is not None:
                    # Inside an already-trusted V3R5 owner, primary periodicity
                    # has NO pitch authority.  It may only corroborate a new
                    # articulation when the WAV attack is itself among the
                    # recording's strongest attacks and the primary lane must
                    # independently confirm coherent periodicity.  Pitch change
                    # is deliberately *not* required: same-pitch rearticulation
                    # is a real Note-On when the attack itself is proven.
                    k = int(np.argmin(np.abs(current.times_s - attack.time_s)))
                    local_trusted_st = float(current.pitch_st[k])
                    if strength >= reartic_gate:
                        accepted_free.append((
                            attack.time_s,
                            "wav_attack+primary_periodicity_rearticulation",
                            55.0 + min(40.0, strength),
                        ))
                        decisions.append(AttackDecision(
                            attack.time_s, True, "wav_attack+primary_periodicity_rearticulation",
                            "trusted_pitch_split_by_independent_attack", strength, attack.prominence_z,
                            current.source, attack.time_s, local_trusted_st,
                            primary_continuity.status, primary_continuity.barrier_time_s,
                            primary_continuity.valley_dbfs, primary_continuity.depth_left_db,
                            primary_continuity.depth_right_db, primary_continuity.low_duration_s,
                        ))
                        continue
                elif strength >= 12.0:
                    accepted_free.append((attack.time_s, "wav_attack+primary_periodicity", 55.0 + min(40.0, strength)))
                    decisions.append(AttackDecision(
                        attack.time_s, True, "wav_attack+primary_periodicity",
                        "orphan_attack_primary_pitch_rescue", strength, attack.prominence_z,
                        primary[0].source, primary[0].time_s, primary[0].pitch_st,
                        primary_continuity.status, primary_continuity.barrier_time_s,
                        primary_continuity.valley_dbfs, primary_continuity.depth_left_db,
                        primary_continuity.depth_right_db, primary_continuity.low_duration_s,
                    ))
                    continue

        # The free acoustic lane is intentionally allowed to abstain.
        decisions.append(AttackDecision(
            attack.time_s, False, "wav_attack", "no_defensible_pitch_owner",
            strength, attack.prominence_z,
        ))

    boundaries.extend(accepted_free)
    boundaries = _dedupe_times(boundaries, dedupe_s)

    # A primary-periodicity rescue may form a continuous bridge from a strong
    # acoustic attack into a later trusted island.  In that case the trusted
    # island owns the attack from the beginning; do not render a tiny rescue
    # note followed by a duplicate trusted Note-On.
    reconciled_info: list[tuple[PitchIsland, float, AcousticAttack | None]] = []
    primary_free = [(t, src, st) for t, src, st in accepted_free
                    if src.startswith("wav_attack+primary_periodicity")]
    island_index_by_start = {round(x.start_s, 9): i for i, x in enumerate(pitch.islands)}
    for island, aligned_start, aligned_attack in island_info:
        bridges: list[tuple[float, float]] = []
        ii = island_index_by_start.get(round(island.start_s, 9), 0)
        prev = pitch.islands[ii - 1] if ii > 0 else None
        topology_floor = (prev.start_s + dedupe_s) if prev is not None else 0.0
        for t0, _src, strength0 in primary_free:
            if not (max(topology_floor, island.start_s - association_s) <= t0 < island.start_s - dedupe_s):
                continue
            pts = pitch.primary_support(t0 - PRIMARY_QUERY_PREROLL_S, island.start_s + PRIMARY_BRIDGE_POSTROLL_S, future_st=island.median_st)
            if not pts:
                continue
            med = float(np.median([q.pitch_st for q in pts]))
            if abs(med - island.median_st) > 3.0:
                continue
            if pts[-1].time_s < island.start_s - 0.12:
                continue
            bridge_continuity = lane.ownership_continuity(
                t0, island.start_s,
                post_anchor_grace_s=float(np.clip(1.5 * pitch.dt, 0.012, 0.020)),
            )
            if not bridge_continuity.connected:
                decisions.append(AttackDecision(
                    t0, False, "wav_attack+primary_periodicity",
                    "primary_bridge_object_barrier", strength0, math.nan,
                    island.source, island.start_s, island.median_st,
                    bridge_continuity.status, bridge_continuity.barrier_time_s,
                    bridge_continuity.valley_dbfs, bridge_continuity.depth_left_db,
                    bridge_continuity.depth_right_db, bridge_continuity.low_duration_s,
                ))
                continue
            bridges.append((t0, strength0))
        if bridges:
            # Prefer the strongest defensible bridge; tie-break toward the later
            # attack because it is less likely to belong to the previous note.
            t_bridge, _ = max(bridges, key=lambda x: (x[1], x[0]))
            aligned_start = min(aligned_start, t_bridge)
        reconciled_info.append((island, aligned_start, aligned_attack))
    island_info = reconciled_info

    # ------------------------------------------------------------------
    # 4) Assemble trusted events *per corrected-valid island*.  This guarantees
    #    that the CLI pitch authority is never accidentally orphaned by an
    #    unrelated global boundary.  Structural/acoustic boundaries may split
    #    an island, but they cannot steal its ownership.
    # ------------------------------------------------------------------
    events: list[MusicalEvent] = []

    # Map each eligible island to the independently aligned attack chosen in
    # pass 1.
    eligible_map = {round(island.start_s, 9): (aligned_start, attack)
                    for island, aligned_start, attack in island_info}

    for island in pitch.islands:
        key = round(island.start_s, 9)
        if key not in eligible_map:
            continue
        aligned_start, _attack = eligible_map[key]

        # Only boundaries that actually occur inside this trusted owner may
        # split it.  The aligned physical attack is the first boundary.
        internal = [(t, src, strength) for t, src, strength in boundaries
                    if island.start_s + dedupe_s < t < island.end_s - dedupe_s]
        split_points = [(aligned_start, "trusted_owner", 100.0)] + internal
        split_points = _dedupe_times(split_points, dedupe_s)
        split_points.sort(key=lambda x: x[0])

        # Make sure an internal boundary cannot precede the independently
        # aligned attack.
        split_points = [x for x in split_points if x[0] >= aligned_start - 1e-9]
        if not split_points or abs(split_points[0][0] - aligned_start) > dedupe_s:
            split_points.insert(0, (aligned_start, "trusted_owner", 100.0))

        segment_starts = split_points
        for si, (seg_start, seg_src, seg_strength) in enumerate(segment_starts):
            seg_end = segment_starts[si + 1][0] if si + 1 < len(segment_starts) else island.end_s
            if seg_end <= seg_start + 1e-4:
                continue

            # Slice the trusted expressive curve only from points belonging to
            # this island.  If the physical attack predates trustworthy F0,
            # hold the first trusted pitch backward to Note On.
            mask = (island.times_s >= max(seg_start, island.start_s) - 1e-9) & \
                   (island.times_s <= seg_end + 1e-9)
            tt = island.times_s[mask].copy()
            pp = island.pitch_st[mask].copy()
            if len(tt) == 0:
                # A very early split before the first trusted point still owns
                # the island's first reliable pitch.
                tt = np.asarray([max(seg_start, island.start_s)], dtype=float)
                pp = np.asarray([island.pitch_st[0]], dtype=float)
            if seg_start < tt[0] - 1e-9:
                tt = np.insert(tt, 0, seg_start); pp = np.insert(pp, 0, pp[0])
            if seg_end > tt[-1] + 1e-9:
                tt = np.append(tt, seg_end); pp = np.append(pp, pp[-1])
            base_st = float(np.median(pp[np.isfinite(pp)]))
            events.append(MusicalEvent(
                seg_start, seg_end, base_st, _nearest_midi(base_st), tt, pp,
                seg_src if seg_src != "trusted_owner" else "trusted_island_owner",
                f"trusted@{island.start_s:.6f}", seg_strength,
                "v3r5_corrected_valid",
                expression_anchor_s=max(seg_start, island.start_s),
            ))

    # ------------------------------------------------------------------
    # 5) Primary-periodicity *pitch* lives only in gaps that have no trusted
    #    owner.  The same acoustic attack may still split a trusted owner in
    #    pass 4; in that case V3R5 corrected-valid pitch remains authoritative.
    # ------------------------------------------------------------------
    primary_boundaries = [(t, src, strength) for t, src, strength in boundaries
                          if src.startswith("wav_attack+primary_periodicity")
                          and pitch.island_containing(t, tolerance_s=pitch.dt) is None]
    primary_boundaries.sort(key=lambda x: x[0])
    trusted_spans = [(e.start_s, e.end_s) for e in events if e.pitch_source == "v3r5_corrected_valid"]

    for bi, (start_p, source_p, strength_p) in enumerate(primary_boundaries):
        if any(a - dedupe_s <= start_p < b + dedupe_s for a, b in trusted_spans):
            continue
        next_primary = primary_boundaries[bi + 1][0] if bi + 1 < len(primary_boundaries) else math.inf
        next_trusted = min([a for a, _b in trusted_spans if a > start_p] or [math.inf])
        horizon_end = min(start_p + association_s, next_primary, next_trusted)
        if horizon_end <= start_p + 1e-4:
            continue
        primary = pitch.primary_support(start_p - PRIMARY_QUERY_PREROLL_S, horizon_end)
        if not primary:
            continue
        if len(primary) < 2 and not (primary[0].acf >= 0.93 and primary[0].cmndf <= 0.08):
            continue
        frame_dt = float(np.median(np.diff([q.time_s for q in primary]))) if len(primary) >= 2 else 0.046
        end_p = min(horizon_end, primary[-1].time_s + frame_dt)
        if end_p <= start_p + 1e-4:
            continue
        tt, pp = _event_curve_from_primary(primary, start_p, end_p)
        base_st = float(np.median(pp))
        events.append(MusicalEvent(
            start_p, end_p, base_st, _nearest_midi(base_st), tt, pp,
            source_p, f"primary@{primary[0].time_s:.6f}", strength_p,
            "v3r5_primary_periodicity_rescue",
            expression_anchor_s=float(primary[0].time_s),
        ))

    # ------------------------------------------------------------------
    # 6) Final monophonic ordering and bend metadata.  A defensible new attack
    #    wins and clips the previous decaying owner.  Topology guards above
    #    prevent a future owner from reaching backward across an intervening
    #    trusted island.  Pitch Bend therefore expresses motion *within* an
    #    articulation object; it cannot absorb an independently proven attack.
    # ------------------------------------------------------------------
    events.sort(key=lambda e: (e.start_s, e.end_s, -e.attack_score))
    cleaned: list[MusicalEvent] = []
    for e in events:
        if cleaned and e.start_s < cleaned[-1].end_s - 1e-9:
            cleaned[-1].end_s = e.start_s
            if cleaned[-1].end_s <= cleaned[-1].start_s + 1e-4:
                cleaned.pop()
        if e.end_s <= e.start_s + 1e-4:
            continue
        deviations = e.pitch_st - float(e.base_midi)
        e.max_abs_bend_st = float(np.nanmax(np.abs(deviations))) if len(deviations) else 0.0
        e.bend_range_st = max(1, int(math.ceil(e.max_abs_bend_st - 1e-12))) if e.max_abs_bend_st > 1e-4 else 0
        cleaned.append(e)

    return AssemblyResult(cleaned, decisions, lane, pitch)
