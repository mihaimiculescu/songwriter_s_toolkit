from __future__ import annotations

"""MIDI Renderer V18 -- acoustic adjudication and transient cleanup.

V18 runs *after* the accepted V17.1 renderer.  V17.1 remains the authority for
all corrected-valid V3R9 ownership and guarantees zero unowned valid-F0 points.
V18 is the first explicitly acoustic/adjudicative layer.  It is deliberately
split into independent certificates:

  1. breath/noise backdating trim when a later already-accepted trusted/reset
     attack owns the same pitch anchor;
  2. collapse of ultra-short same-owner WAV rearticulations that have no proven
     bilateral envelope restart;
  3. collapse of a short, weakly backdated transient head into the following
     continuous voiced event as pitch bend rather than a separate Note-On;
  4. bounded WAV/primary rescue for tracker-abstention gaps;
  5. exceptionally strong octave-up correction when the rendered low family is
     spectrally subharmonic and local event context independently favors the
     octave-up family.

No V3R9 tracker state is changed.  Every correction is renderer-side only.
"""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import soundfile as sf

from .midi_event_assembler_v17_1 import (
    AssemblyResult,
    MusicalEvent,
    AttackDecision,
    assemble_events as assemble_events_v17_1,
    _reclean_monophonic,
)
from .midi_event_assembler_v16_1 import _physical_reattack_evidence
from .midi_pitch_evidence_v10 import hz_to_st, st_to_hz, cents_between_hz


# ---------------------------------------------------------------------------
# V18 physical-time / acoustic contracts
# ---------------------------------------------------------------------------

# Breath/backdating: never use this simply because F0 arrives late.  A later
# attack must already be accepted by the inherited renderer and point to the
# same trusted pitch anchor.
BREATH_BACKDATE_MIN_LAG_S = 0.070
BREATH_REPLACEMENT_MAX_PREANCHOR_S = 0.090
BREATH_MIN_LEVEL_GAIN_DB = 8.0
BREATH_EARLY_PRIMARY_GUARD_S = 0.030
BREATH_EARLY_VOICED_MIN_ACF = 0.55
BREATH_EARLY_VOICED_MAX_CMNDF = 0.40
BREATH_ANCHOR_TOLERANCE_S = 0.012

# Ultra-short transient/tail articulation cleanup.
ULTRASHORT_MAX_S = 0.035
ULTRASHORT_JOIN_GAP_S = 0.025
ULTRASHORT_MAX_FAMILY_GAP_ST = 2.25

# A weakly backdated tiny head may be preserved acoustically but represented as
# the PB lead-in of the following event instead of an independent MIDI Note-On.
TRANSIENT_HEAD_MAX_S = 0.120
TRANSIENT_HEAD_MAX_GAP_TO_NEXT_S = 0.030
TRANSIENT_HEAD_MIN_BACKDATE_S = 0.035
TRANSIENT_HEAD_MAX_NEXT_GAP_ST = 5.0
TRANSIENT_HEAD_WEAK_START_DBFS = -42.0

# Tracker-abstention rescue: intentionally weaker than the legacy primary lane,
# but only with an exceptionally strong WAV attack and ACF/CMNDF pitch agreement.
ACOUSTIC_RESCUE_MIN_ATTACK_SCORE = 80.0
ACOUSTIC_RESCUE_MIN_ACF = 0.68
ACOUSTIC_RESCUE_MAX_CMNDF = 0.30
ACOUSTIC_RESCUE_MAX_ACF_CMNDF_CENTS = 100.0
ACOUSTIC_RESCUE_ATTACK_RADIUS_S = 0.030
ACOUSTIC_RESCUE_MAX_EVENT_S = 0.100
ACOUSTIC_RESCUE_MIN_EVENT_S = 0.030
ACOUSTIC_RESCUE_FRAME_TAIL_S = 0.0465
ACOUSTIC_RESCUE_MIN_AUDIBLE_GAIN_DB = 8.0

# Existing accepted structural/trusted attack may pull a later trusted event
# backward across a short tracker-abstention gap only when a direct WAV spectrum
# supports the same pitch family there.
ACOUSTIC_OWNER_BACKFILL_MAX_S = 0.060
ACOUSTIC_OWNER_BACKFILL_MAX_ST = 1.25
ACOUSTIC_OWNER_BACKFILL_MIN_RMS_DBFS = -35.0

# Subharmonic certificate.  18 dB is too permissive for ordinary vocal second
# harmonics; V18 requires a much more extreme spectral inversion *and* local
# contextual improvement on both sides.
SUBHARMONIC_MIN_EVENT_S = 0.080
SUBHARMONIC_MIN_OCTAVE_DOMINANCE_DB = 30.0
SUBHARMONIC_MIN_STRONG_FRAME_FRACTION = 0.60
SUBHARMONIC_CONTEXT_IMPROVEMENT_ST = 8.0
SUBHARMONIC_MAX_CORRECTED_NEIGHBOR_GAP_ST = 5.0
SUBHARMONIC_SPECTRAL_WINDOW_S = 0.046


@dataclass(frozen=True)
class AcousticRescueCandidate:
    attack_s: float
    frame_s: float
    pitch_st: float
    f0_hz: float
    acf: float
    cmndf: float
    score: float


class _WavProbe:
    def __init__(self, wav_path: Path):
        audio, sr = sf.read(Path(wav_path), always_2d=True, dtype="float32")
        self.y = np.mean(audio, axis=1, dtype=np.float64)
        self.sr = int(sr)

    def rms_db(self, time_s: float, window_s: float = 0.023) -> float:
        n = max(64, int(round(window_s * self.sr)))
        c = int(round(float(time_s) * self.sr))
        a = max(0, c - n // 2)
        b = min(len(self.y), a + n)
        a = max(0, b - n)
        if b <= a:
            return -120.0
        seg = self.y[a:b]
        rms = math.sqrt(float(np.mean(seg * seg)) + 1e-15)
        return 20.0 * math.log10(rms + 1e-15)

    def spectral_pair_db(self, time_s: float, f0_hz: float) -> tuple[float, float]:
        """Return local spectral magnitudes (dB) near f0 and 2*f0."""
        n = max(256, int(round(SUBHARMONIC_SPECTRAL_WINDOW_S * self.sr)))
        c = int(round(float(time_s) * self.sr))
        a = max(0, c - n // 2)
        b = min(len(self.y), a + n)
        a = max(0, b - n)
        if b - a < 128:
            return -120.0, -120.0
        seg = self.y[a:b] * np.hanning(b - a)
        nfft = 1
        while nfft < 2 * len(seg):
            nfft <<= 1
        spec = np.abs(np.fft.rfft(seg, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sr)

        def peak_db(f: float) -> float:
            bw = max(4.0, 0.02 * f)
            mask = (freqs >= f - bw) & (freqs <= f + bw)
            if not np.any(mask):
                return -120.0
            mag = float(np.max(spec[mask]))
            return 20.0 * math.log10(mag + 1e-15)

        return peak_db(float(f0_hz)), peak_db(2.0 * float(f0_hz))


# ---------------------------------------------------------------------------
# Common event helpers
# ---------------------------------------------------------------------------

def _finite_curve(event: MusicalEvent) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(event.pitch_times_s, dtype=float)
    p = np.asarray(event.pitch_st, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p)
    return t[mask], p[mask]


def _recompute_bend(event: MusicalEvent) -> None:
    t, p = _finite_curve(event)
    event.pitch_times_s = t
    event.pitch_st = p
    if not len(p):
        event.max_abs_bend_st = 0.0
        event.bend_range_st = 0
        return
    dev = p - float(event.base_midi)
    event.max_abs_bend_st = float(np.max(np.abs(dev)))
    event.bend_range_st = max(1, int(math.ceil(event.max_abs_bend_st - 1e-12))) if event.max_abs_bend_st > 1e-4 else 0


def _trim_event_start(event: MusicalEvent, new_start_s: float) -> None:
    new_start_s = float(np.clip(new_start_s, event.start_s, event.end_s - 1e-5))
    t, p = _finite_curve(event)
    if len(t):
        # Interpolate/hold the pitch at the new start, then keep later points.
        if new_start_s <= t[0]:
            p0 = float(p[0])
        elif new_start_s >= t[-1]:
            p0 = float(p[-1])
        else:
            p0 = float(np.interp(new_start_s, t, p))
        keep = t > new_start_s + 1e-9
        t = np.concatenate([[new_start_s], t[keep]])
        p = np.concatenate([[p0], p[keep]])
    event.start_s = new_start_s
    event.pitch_times_s = t
    event.pitch_st = p
    _recompute_bend(event)


def _merge_curves(first: MusicalEvent, second: MusicalEvent, *, keep_base: str = "first") -> MusicalEvent:
    """Mutate and return the chosen owner with the union of both pitch curves."""
    if keep_base == "second":
        owner, other = second, first
        new_start = first.start_s
        new_end = max(first.end_s, second.end_s)
        base_midi = second.base_midi
        base_st = second.base_st
        expression_anchor = min(
            x for x in (first.expression_anchor_s, second.expression_anchor_s)
            if math.isfinite(x)
        ) if any(math.isfinite(x) for x in (first.expression_anchor_s, second.expression_anchor_s)) else new_start
    else:
        owner, other = first, second
        new_start = min(first.start_s, second.start_s)
        new_end = second.end_s
        base_midi = first.base_midi
        base_st = first.base_st
        expression_anchor = first.expression_anchor_s

    t1, p1 = _finite_curve(first)
    t2, p2 = _finite_curve(second)
    tt = np.concatenate([t1, t2]) if len(t1) or len(t2) else np.zeros(0)
    pp = np.concatenate([p1, p2]) if len(p1) or len(p2) else np.zeros(0)
    if len(tt):
        order = np.argsort(tt, kind="stable")
        tt, pp = tt[order], pp[order]
        keep = np.ones(len(tt), dtype=bool)
        keep[1:] = np.diff(tt) > 1e-9
        tt, pp = tt[keep], pp[keep]
        if tt[0] > new_start + 1e-9:
            tt = np.insert(tt, 0, new_start)
            pp = np.insert(pp, 0, pp[0])
        if tt[-1] < new_end - 1e-9:
            tt = np.append(tt, new_end)
            pp = np.append(pp, pp[-1])

    owner.start_s = float(new_start)
    owner.end_s = float(new_end)
    owner.base_midi = int(base_midi)
    owner.base_st = float(base_st)
    owner.pitch_times_s = tt
    owner.pitch_st = pp
    owner.expression_anchor_s = float(expression_anchor)
    _recompute_bend(owner)
    return owner


def _attack_at(result: AssemblyResult, time_s: float, radius_s: float = 0.006) -> AttackDecision | None:
    xs = [d for d in result.attack_decisions if d.accepted and abs(float(d.time_s) - float(time_s)) <= radius_s]
    return min(xs, key=lambda d: abs(float(d.time_s) - float(time_s))) if xs else None


def _physical_reattack_proven(result: AssemblyResult, time_s: float, radius_s: float = 0.015) -> bool:
    # Prefer already-audited bilateral evidence.
    for d in result.attack_decisions:
        if not d.accepted or abs(float(d.time_s) - float(time_s)) > radius_s:
            continue
        if d.reattack_status == "bilateral_envelope_restart":
            return True
    # Trusted/reset attacks did not always run through the V16 envelope test.
    # Re-evaluate the WAV morphology directly rather than treating the reset as
    # automatic proof of musical rearticulation.
    attack = result.attack_lane.best_peak(float(time_s) - radius_s, float(time_s) + radius_s)
    return bool(_physical_reattack_evidence(result.attack_lane, attack).proven)



# ---------------------------------------------------------------------------
# 1. Breath/noise backdating
# ---------------------------------------------------------------------------

def _early_corridor_has_voiced_primary(result: AssemblyResult, start_s: float, end_s: float) -> bool:
    end_s = float(end_s) - BREATH_EARLY_PRIMARY_GUARD_S
    if end_s <= start_s:
        return False
    for r in result.pitch.frame_rows:
        try:
            t = float(r.get("start_time_s", "nan"))
            acf = float(r.get("periodicity_acf_peak", "nan"))
            cm = float(r.get("periodicity_cmndf_minimum", "nan"))
        except Exception:
            continue
        if start_s <= t <= end_s and math.isfinite(acf) and math.isfinite(cm):
            if acf >= BREATH_EARLY_VOICED_MIN_ACF and cm <= BREATH_EARLY_VOICED_MAX_CMNDF:
                return True
    return False


def _trim_breath_backdating(result: AssemblyResult, probe: _WavProbe) -> int:
    changed = 0
    for e in result.events:
        anchor = float(e.expression_anchor_s)
        if not math.isfinite(anchor):
            continue
        lag = anchor - e.start_s
        if lag < BREATH_BACKDATE_MIN_LAG_S:
            continue
        start_level = probe.rms_db(e.start_s)

        candidates: list[AttackDecision] = []
        for d in result.attack_decisions:
            if not d.accepted:
                continue
            if not (e.start_s + 1e-6 < d.time_s <= anchor + BREATH_ANCHOR_TOLERANCE_S):
                continue
            if not math.isfinite(d.pitch_anchor_s):
                continue
            if abs(float(d.pitch_anchor_s) - anchor) > BREATH_ANCHOR_TOLERANCE_S:
                continue
            if anchor - float(d.time_s) > BREATH_REPLACEMENT_MAX_PREANCHOR_S:
                continue
            if not (d.source.startswith("trusted_island_start") or "eckf_reset" in d.source):
                continue
            candidates.append(d)
        if not candidates:
            continue
        replacement = min(candidates, key=lambda d: d.time_s)

        # If coherent primary periodicity already exists well before the later
        # trusted/reset attack, the early corridor is voiced motion, not breath.
        if _early_corridor_has_voiced_primary(result, e.start_s, replacement.time_s):
            continue

        # Prefer the already-accepted reset attack when it is materially louder.
        # Otherwise, for a long breath corridor, move to the earliest moderate
        # ACF/CMNDF-agreeing primary frame near the trusted anchor.
        replacement_level = probe.rms_db(replacement.time_s)
        chosen_start = None
        if replacement_level >= start_level + BREATH_MIN_LEVEL_GAIN_DB:
            chosen_start = float(replacement.time_s)

        near_primary = []
        for r in result.pitch.frame_rows:
            try:
                t = float(r.get("start_time_s", "nan"))
                acf = float(r.get("periodicity_acf_peak", "nan"))
                cm = float(r.get("periodicity_cmndf_minimum", "nan"))
                fa = float(r.get("periodicity_acf_frequency_hz", "nan"))
                fc = float(r.get("periodicity_cmndf_frequency_hz", "nan"))
            except Exception:
                continue
            if not (max(e.start_s, replacement.time_s - 0.035) <= t <= anchor + BREATH_ANCHOR_TOLERANCE_S):
                continue
            if not (math.isfinite(acf) and math.isfinite(cm) and acf >= BREATH_EARLY_VOICED_MIN_ACF and cm <= BREATH_EARLY_VOICED_MAX_CMNDF):
                continue
            if not (math.isfinite(fa) and fa > 0 and math.isfinite(fc) and fc > 0 and cents_between_hz(fa, fc) <= 120.0):
                continue
            if probe.rms_db(t) < start_level + BREATH_MIN_LEVEL_GAIN_DB:
                continue
            near_primary.append(t)
        if near_primary:
            pstart = min(near_primary)
            chosen_start = pstart if chosen_start is None else min(chosen_start, pstart)

        if chosen_start is None:
            continue
        old = e.start_s
        _trim_event_start(e, chosen_start)
        e.source_kind = f"{e.source_kind}+breath_trim_v18"
        result.attack_decisions.append(AttackDecision(
            old, False, "acoustic_breath_backdating_v18",
            "later_trusted_or_primary_voiced_onset_replaces_breath_v18",
            pitch_source=e.pitch_source, pitch_anchor_s=anchor, pitch_anchor_st=e.base_st,
            continuity_status=f"trimmed_to={chosen_start:.9f}",
        ))
        changed += 1
    return changed



# ---------------------------------------------------------------------------
# 2. Ultra-short tail/rearticulation cleanup
# ---------------------------------------------------------------------------

def _merge_ultrashort_rearticulations(result: AssemblyResult) -> int:
    original = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    kept: list[MusicalEvent] = []
    changed = 0
    for i, e in enumerate(original):
        dur = e.end_s - e.start_s
        nxt = original[i + 1] if i + 1 < len(original) else None
        prev = kept[-1] if kept else None
        eligible = (
            prev is not None and nxt is not None
            and dur <= ULTRASHORT_MAX_S + 1e-9
            and "wav_rearticulation" in e.source_kind
            and e.source_id == prev.source_id
            and e.start_s - prev.end_s <= ULTRASHORT_JOIN_GAP_S
            and nxt.start_s - e.end_s <= ULTRASHORT_JOIN_GAP_S
        )
        if eligible:
            attack = _attack_at(result, e.start_s)
            bilateral = attack is not None and attack.reattack_status == "bilateral_envelope_restart"
            sandwich = (
                abs(float(prev.base_st) - float(nxt.base_st)) <= 1.50
                and abs(float(e.base_st) - float(prev.base_st)) <= 3.50
            )
            tail_before_new_owner = (
                nxt.source_id != e.source_id
                and abs(float(e.base_st) - float(prev.base_st)) <= ULTRASHORT_MAX_FAMILY_GAP_ST
            )
            if not bilateral and (sandwich or tail_before_new_owner):
                _merge_curves(prev, e, keep_base="first")
                result.attack_decisions.append(AttackDecision(
                    e.start_s, False, "ultrashort_rearticulation_v18",
                    "ultrashort_tail_or_detour_without_bilateral_restart_v18",
                    pitch_source=e.pitch_source, pitch_anchor_s=e.expression_anchor_s,
                    pitch_anchor_st=e.base_st,
                ))
                changed += 1
                continue
        kept.append(e)

    result.events = _reclean_monophonic(kept)
    return changed


# ---------------------------------------------------------------------------
# 3. Short transient head -> PB lead-in of next real note
# ---------------------------------------------------------------------------

def _collapse_transient_heads(result: AssemblyResult, probe: _WavProbe) -> int:
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    removed: set[int] = set()
    changed = 0
    for i, e in enumerate(events[:-1]):
        if i in removed:
            continue
        nxt = events[i + 1]
        dur = e.end_s - e.start_s
        if dur > TRANSIENT_HEAD_MAX_S:
            continue
        anchor = float(e.expression_anchor_s)
        if not math.isfinite(anchor) or anchor - e.start_s < TRANSIENT_HEAD_MIN_BACKDATE_S:
            continue
        # Only a very short stretch of trustworthy voiced pitch survives after
        # the weak/noisy borrowed start.  This is onset ornament/transient
        # morphology, not enough evidence for an independent MIDI note.
        if e.end_s - anchor > 0.050:
            continue
        if nxt.start_s - e.end_s > TRANSIENT_HEAD_MAX_GAP_TO_NEXT_S:
            continue
        if abs(float(e.base_st) - float(nxt.base_st)) > TRANSIENT_HEAD_MAX_NEXT_GAP_ST:
            continue
        if probe.rms_db(e.start_s) > TRANSIENT_HEAD_WEAK_START_DBFS:
            continue

        next_anchor = float(nxt.expression_anchor_s) if math.isfinite(nxt.expression_anchor_s) else float(nxt.start_s)
        continuity = result.attack_lane.ownership_continuity(
            anchor, next_anchor, post_anchor_grace_s=float(np.clip(1.5 * result.pitch.dt, 0.012, 0.020))
        )
        if not continuity.connected:
            # A proven acoustic object break means the tiny transient is not
            # allowed to smear pitch into the next event.  It is simply dropped
            # as a non-musical onset fragment.
            removed.add(i)
            result.attack_decisions.append(AttackDecision(
                anchor, False, "transient_head_v18",
                "short_backdated_transient_dropped_before_independent_object_v18",
                pitch_source=e.pitch_source, pitch_anchor_s=anchor, pitch_anchor_st=e.base_st,
                continuity_status=continuity.status,
            ))
            changed += 1
            continue

        # Connected voiced transition: keep its actual pitch as PB lead-in of
        # the following event, but do not issue its own Note-On.
        _trim_event_start(e, anchor)
        merged = _merge_curves(e, nxt, keep_base="second")
        merged.source_kind = f"{nxt.source_kind}+transient_head_pb_v18"
        merged.pitch_source = nxt.pitch_source
        removed.add(i)
        result.attack_decisions.append(AttackDecision(
            anchor, False, "transient_head_v18",
            "short_backdated_head_folded_into_following_pitch_curve_v18",
            pitch_source=merged.pitch_source,
            pitch_anchor_s=merged.expression_anchor_s, pitch_anchor_st=merged.base_st,
            continuity_status=continuity.status,
        ))
        changed += 1

    result.events = [e for i, e in enumerate(events) if i not in removed]
    result.events = _reclean_monophonic(result.events)
    return changed


# ---------------------------------------------------------------------------
# 4. WAV/primary acoustic rescue in tracker-invalid gaps
# ---------------------------------------------------------------------------

def _moderate_primary_candidates(result: AssemblyResult, start_s: float, end_s: float) -> list[tuple[float, float, float, float, float]]:
    out = []
    for r in result.pitch.frame_rows:
        try:
            t = float(r.get("start_time_s", "nan"))
            acf = float(r.get("periodicity_acf_peak", "nan"))
            cm = float(r.get("periodicity_cmndf_minimum", "nan"))
            fa = float(r.get("periodicity_acf_frequency_hz", "nan"))
            fc = float(r.get("periodicity_cmndf_frequency_hz", "nan"))
        except Exception:
            continue
        if not (start_s - 1e-9 <= t <= end_s + 1e-9):
            continue
        if not (math.isfinite(acf) and math.isfinite(cm) and acf >= ACOUSTIC_RESCUE_MIN_ACF and cm <= ACOUSTIC_RESCUE_MAX_CMNDF):
            continue
        if not (math.isfinite(fa) and fa > 0 and math.isfinite(fc) and fc > 0):
            continue
        if cents_between_hz(fa, fc) > ACOUSTIC_RESCUE_MAX_ACF_CMNDF_CENTS:
            continue
        hz = math.sqrt(fa * fc)
        st = hz_to_st(hz)
        if st < result.pitch.zero_low_st or st > result.pitch.zero_high_st:
            continue
        out.append((t, st, hz, acf, cm))
    return out


def _make_acoustic_rescue_event(result: AssemblyResult, c: AcousticRescueCandidate, end_s: float, serial: int) -> MusicalEvent:
    end_s = max(c.attack_s + ACOUSTIC_RESCUE_MIN_EVENT_S, min(end_s, c.attack_s + ACOUSTIC_RESCUE_MAX_EVENT_S))
    times = np.asarray([c.attack_s, c.frame_s, end_s], dtype=float)
    pitch = np.asarray([c.pitch_st, c.pitch_st, c.pitch_st], dtype=float)
    midi = int(np.clip(round(c.pitch_st), 0, 127))
    e = MusicalEvent(
        start_s=float(c.attack_s), end_s=float(end_s),
        base_st=float(c.pitch_st), base_midi=midi,
        pitch_times_s=times, pitch_st=pitch,
        source_kind="wav_primary_acoustic_rescue_v18",
        source_id=f"acoustic_rescue_v18_{serial}",
        attack_score=float(c.score),
        pitch_source="wav_primary_agreement_v18",
        expression_anchor_s=float(c.frame_s),
    )
    _recompute_bend(e)
    return e


def _acoustic_rescue(result: AssemblyResult, probe: _WavProbe) -> tuple[int, int]:
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    added: list[MusicalEvent] = []
    shifted = 0
    serial = 0

    # A) Standalone moderate-primary rescue in gaps.
    gaps: list[tuple[float, float, MusicalEvent | None, MusicalEvent | None]] = []
    prev = None
    for e in events:
        if prev is None:
            if e.start_s > 0.03:
                gaps.append((0.0, e.start_s, None, e))
        elif e.start_s - prev.end_s > ACOUSTIC_RESCUE_MIN_EVENT_S:
            gaps.append((prev.end_s, e.start_s, prev, e))
        prev = e

    for g0, g1, before, after in gaps:
        # Never use this mechanism where ordinary corrected-valid V3R9 pitch is
        # present; V17.1 already guarantees ownership there.
        mask = result.pitch.valid & (result.pitch.t >= g0 - 1e-9) & (result.pitch.t < g1 + 1e-9)
        if np.any(mask):
            continue
        prim = _moderate_primary_candidates(result, g0, g1)
        if not prim:
            continue
        attacks = [
            p for p in result.attack_lane.peaks
            if g0 <= p.time_s <= g1
            and (p.novelty_z + 0.5 * p.prominence_z) >= ACOUSTIC_RESCUE_MIN_ATTACK_SCORE
        ]
        if not attacks:
            continue
        best = None
        for p in attacks:
            near = [q for q in prim if abs(q[0] - p.time_s) <= ACOUSTIC_RESCUE_ATTACK_RADIUS_S]
            if not near:
                continue
            q = min(near, key=lambda x: abs(x[0] - p.time_s))
            if probe.rms_db(p.time_s) < result.attack_lane.active_rms_floor_db + ACOUSTIC_RESCUE_MIN_AUDIBLE_GAIN_DB:
                continue
            score = float(p.novelty_z + 0.5 * p.prominence_z)
            cand = AcousticRescueCandidate(float(p.time_s), q[0], q[1], q[2], q[3], q[4], score)
            if best is None or cand.score > best.score:
                best = cand
        if best is None:
            continue

        # End when the supporting primary frame expires, or at the next event.
        rescue_end = min(g1, best.frame_s + ACOUSTIC_RESCUE_FRAME_TAIL_S)
        if rescue_end - best.attack_s < ACOUSTIC_RESCUE_MIN_EVENT_S:
            continue
        e = _make_acoustic_rescue_event(result, best, rescue_end, serial)
        serial += 1
        added.append(e)
        result.attack_decisions.append(AttackDecision(
            e.start_s, True, "wav_primary_acoustic_rescue_v18",
            "strong_wav_attack_plus_moderate_acf_cmndf_agreement_v18",
            e.attack_score, math.nan, e.pitch_source, best.frame_s, best.pitch_st,
            continuity_status="tracker_abstention_acoustic_rescue",
        ))

    if added:
        events = _reclean_monophonic(events + added)

    # B) Backfill a short gap to the next already-trusted owner only when an
    # accepted structural/trusted attack occurs inside the gap and the local WAV
    # spectrum's strongest low-order periodic family agrees with that owner.
    events = sorted(events, key=lambda e: (e.start_s, e.end_s))
    for i in range(1, len(events)):
        prev = events[i - 1]
        e = events[i]
        gap = e.start_s - prev.end_s
        if not (1e-4 < gap <= ACOUSTIC_OWNER_BACKFILL_MAX_S):
            continue
        accepted = [
            d for d in result.attack_decisions
            if d.accepted and prev.end_s <= d.time_s < e.start_s
            and (d.source.startswith("v3r5_stable_target_start") or d.source.startswith("trusted_island_start"))
            and math.isfinite(d.pitch_anchor_st)
            and abs(float(d.pitch_anchor_st) - float(e.base_st)) <= ACOUSTIC_OWNER_BACKFILL_MAX_ST
        ]
        if not accepted:
            continue
        d = max(accepted, key=lambda x: x.time_s)
        if probe.rms_db(d.time_s) < ACOUSTIC_OWNER_BACKFILL_MIN_RMS_DBFS:
            continue
        # Primary evidence at/near the accepted attack must agree with next owner.
        prim = _moderate_primary_candidates(result, d.time_s - 0.030, e.start_s + 0.010)
        if not prim:
            continue
        q = min(prim, key=lambda x: abs(x[0] - d.time_s))
        if abs(q[1] - float(e.base_st)) > ACOUSTIC_OWNER_BACKFILL_MAX_ST:
            continue
        old = e.start_s
        _trim_event_start(e, d.time_s)
        e.source_kind = f"{e.source_kind}+acoustic_backfill_v18"
        result.attack_decisions.append(AttackDecision(
            old, False, "acoustic_owner_backfill_v18",
            "accepted_structural_attack_plus_primary_family_support_v18",
            pitch_source=e.pitch_source, pitch_anchor_s=q[0], pitch_anchor_st=q[1],
            continuity_status=f"shifted_to={d.time_s:.9f}",
        ))
        shifted += 1

    result.events = _reclean_monophonic(events)
    return len(added), shifted


# ---------------------------------------------------------------------------
# 5. Strong subharmonic octave-up adjudication
# ---------------------------------------------------------------------------

def _subharmonic_octave_corrections(result: AssemblyResult, probe: _WavProbe) -> int:
    events = sorted(result.events, key=lambda e: (e.start_s, e.end_s))
    changed = 0
    for i, e in enumerate(events):
        dur = e.end_s - e.start_s
        if dur < SUBHARMONIC_MIN_EVENT_S:
            continue
        prev = events[i - 1] if i > 0 else None
        nxt = events[i + 1] if i + 1 < len(events) else None
        if prev is None or nxt is None:
            continue

        corrected_st = float(e.base_st) + 12.0
        current_context = abs(float(e.base_st) - float(prev.base_st)) + abs(float(e.base_st) - float(nxt.base_st))
        corrected_context = abs(corrected_st - float(prev.base_st)) + abs(corrected_st - float(nxt.base_st))
        if current_context - corrected_context < SUBHARMONIC_CONTEXT_IMPROVEMENT_ST:
            continue
        if max(abs(corrected_st - float(prev.base_st)), abs(corrected_st - float(nxt.base_st))) > SUBHARMONIC_MAX_CORRECTED_NEIGHBOR_GAP_ST:
            continue

        t, p = _finite_curve(e)
        base_pitch = float(np.median(p)) if len(p) else float(e.base_st)
        f0 = st_to_hz(base_pitch)
        if 2.0 * f0 > 1200.0:
            continue
        sample_times = np.linspace(
            e.start_s + min(0.025, 0.15 * dur),
            e.end_s - min(0.025, 0.15 * dur),
            5,
        )
        dominance = []
        for ts in sample_times:
            lo, hi = probe.spectral_pair_db(float(ts), f0)
            dominance.append(hi - lo)
        med = float(np.median(dominance))
        frac = float(np.mean(np.asarray(dominance) >= 18.0))
        if med < SUBHARMONIC_MIN_OCTAVE_DOMINANCE_DB or frac < SUBHARMONIC_MIN_STRONG_FRAME_FRACTION:
            continue

        e.base_st += 12.0
        e.base_midi = int(np.clip(e.base_midi + 12, 0, 127))
        e.pitch_st = np.asarray(e.pitch_st, dtype=float) + 12.0
        e.pitch_source = f"{e.pitch_source}+wav_subharmonic_octave_up_v18"
        e.source_kind = f"{e.source_kind}+subharmonic_corrected_v18"
        _recompute_bend(e)
        result.attack_decisions.append(AttackDecision(
            e.start_s, False, "subharmonic_adjudication_v18",
            "wav_octave_up_dominates_missing_subharmonic_plus_context_v18",
            pitch_source=e.pitch_source, pitch_anchor_s=e.expression_anchor_s,
            pitch_anchor_st=e.base_st,
            continuity_status=f"median_octave_dominance_db={med:.3f};strong_fraction={frac:.3f}",
        ))
        changed += 1
    return changed


# ---------------------------------------------------------------------------
# Public assembly
# ---------------------------------------------------------------------------

def assemble_events(prefix: Path, wav_path: Path) -> AssemblyResult:
    result = assemble_events_v17_1(prefix, wav_path)
    probe = _WavProbe(wav_path)

    breath_trimmed = _trim_breath_backdating(result, probe)
    ultrashort_merged = _merge_ultrashort_rearticulations(result)
    transient_heads = _collapse_transient_heads(result, probe)
    rescues_added, rescue_backfills = _acoustic_rescue(result, probe)
    octave_corrected = _subharmonic_octave_corrections(result, probe)

    result.events = _reclean_monophonic(result.events)
    result.attack_decisions.append(AttackDecision(
        -1.0, False, "v18_summary",
        (
            f"breath_trimmed={breath_trimmed};"
            f"ultrashort_merged={ultrashort_merged};"
            f"transient_heads_collapsed={transient_heads};"
            f"acoustic_rescues_added={rescues_added};"
            f"acoustic_owner_backfills={rescue_backfills};"
            f"subharmonic_octave_corrected={octave_corrected}"
        ),
        pitch_source="renderer_acoustic_adjudication_v18",
    ))
    return result
