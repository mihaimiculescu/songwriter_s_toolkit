from __future__ import annotations

"""MIDI Renderer V18.2 -- unsupported borrowed-start retraction.

V18.2 is intentionally a one-mechanism child of V18.1.  Every V18.1
calibration remains unchanged.  The only new authority retracts a weak Note-On
that was borrowed far backward from a later frozen trusted pitch owner when the
borrowed corridor contains neither corrected-valid V3R9 pitch nor even moderate
ACF/CMNDF-agreeing primary periodicity of the event's target family.

This is deliberately stricter than V18.1's near-anchor weak-start trim.  V18.2
exists for the residual case where there is *no* defensible earlier target-family
onset to choose.  In that situation it starts the event at the first frozen
corrected-valid target-family point, preserving the V17.1 zero-unowned-valid
contract exactly.
"""

from pathlib import Path
import math
import numpy as np

from .midi_event_assembler_v18_1 import (
    AssemblyResult, MusicalEvent, AttackDecision,
    assemble_events as assemble_events_v18_1,
)
from .midi_event_assembler_v18 import _WavProbe, _trim_event_start
from .midi_event_assembler_v17_1 import _reclean_monophonic
from .midi_pitch_evidence_v10 import cents_between_hz, hz_to_st


# ---------------------------------------------------------------------------
# V18.2 -- one narrow fallback only
# ---------------------------------------------------------------------------

UNSUPPORTED_BACKDATE_MIN_LAG_S = 0.090
UNSUPPORTED_BACKDATE_START_DBFS = -48.0
UNSUPPORTED_BACKDATE_MAX_MEDIAN_DBFS = -40.0
UNSUPPORTED_BACKDATE_RMS_SAMPLES = 7
UNSUPPORTED_BACKDATE_MIN_REMAINING_S = 0.055

# "No target-family periodicity" is intentionally a permissive veto: even
# moderate primary evidence blocks the trim.  This makes the fallback
# conservative rather than a replacement for V18.1's normal onset logic.
UNSUPPORTED_PRIMARY_MIN_ACF = 0.45
UNSUPPORTED_PRIMARY_MAX_CMNDF = 0.55
UNSUPPORTED_PRIMARY_MAX_ACF_CMNDF_CENTS = 150.0
UNSUPPORTED_PRIMARY_MAX_FAMILY_GAP_ST = 1.75
UNSUPPORTED_PRIMARY_ANCHOR_GUARD_S = 0.008

# The first frozen corrected-valid point used as replacement must still belong
# to the event's own family.  This protects against trimming into a neighboring
# pitch object merely because it happens to be valid.
UNSUPPORTED_VALID_MAX_FAMILY_GAP_ST = 2.0


def _target_family_primary_exists(
    result: AssemblyResult,
    start_s: float,
    end_s: float,
    target_st: float,
) -> bool:
    """Return True if even moderate target-family primary evidence exists.

    The last 8 ms before the frozen trusted anchor are excluded.  That tiny
    neighborhood is the owner itself, not evidence that the earlier borrowed
    corridor was already voiced.
    """
    hi = float(end_s) - UNSUPPORTED_PRIMARY_ANCHOR_GUARD_S
    if hi <= start_s:
        return False
    for r in result.pitch.frame_rows:
        try:
            t = float(r.get("start_time_s", "nan"))
            acf = float(r.get("periodicity_acf_peak", "nan"))
            cm = float(r.get("periodicity_cmndf_minimum", "nan"))
            fa = float(r.get("periodicity_acf_frequency_hz", "nan"))
            fc = float(r.get("periodicity_cmndf_frequency_hz", "nan"))
        except Exception:
            continue
        if not (start_s <= t < hi):
            continue
        if not (
            math.isfinite(acf) and acf >= UNSUPPORTED_PRIMARY_MIN_ACF
            and math.isfinite(cm) and cm <= UNSUPPORTED_PRIMARY_MAX_CMNDF
            and math.isfinite(fa) and fa > 0.0
            and math.isfinite(fc) and fc > 0.0
        ):
            continue
        if cents_between_hz(fa, fc) > UNSUPPORTED_PRIMARY_MAX_ACF_CMNDF_CENTS:
            continue
        primary_st = hz_to_st(math.sqrt(fa * fc))
        if abs(primary_st - float(target_st)) <= UNSUPPORTED_PRIMARY_MAX_FAMILY_GAP_ST:
            return True
    return False


def _first_target_valid_time(result: AssemblyResult, e: MusicalEvent) -> float | None:
    mask = (
        result.pitch.valid
        & (result.pitch.t >= float(e.start_s) - 1e-9)
        & (result.pitch.t <= float(e.end_s) + 1e-9)
        & np.isfinite(result.pitch.shape)
        & (np.abs(result.pitch.shape - float(e.base_st)) <= UNSUPPORTED_VALID_MAX_FAMILY_GAP_ST)
    )
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    return float(result.pitch.t[int(idx[0])])


def _borrowed_corridor_median_dbfs(
    probe: _WavProbe,
    start_s: float,
    end_s: float,
) -> float:
    if end_s <= start_s:
        return -120.0
    # endpoint=False keeps the trusted owner itself from making a silent/noisy
    # borrowed corridor appear loud merely because the real note starts there.
    ts = np.linspace(
        float(start_s), float(end_s),
        UNSUPPORTED_BACKDATE_RMS_SAMPLES,
        endpoint=False,
    )
    return float(np.median([probe.rms_db(float(t)) for t in ts]))


def _retract_unsupported_borrowed_starts(
    result: AssemblyResult,
    probe: _WavProbe,
) -> int:
    changed = 0
    for e in result.events:
        # Residual-valid and acoustic-rescue owners have different semantics.
        # This fallback is only for the inherited trusted-island backdating
        # class that holds later frozen pitch backward to an earlier attack.
        if "trusted_island_owner" not in e.source_kind:
            continue

        anchor = float(e.expression_anchor_s)
        if not math.isfinite(anchor):
            continue
        old_start = float(e.start_s)
        if anchor - old_start < UNSUPPORTED_BACKDATE_MIN_LAG_S:
            continue

        start_level = probe.rms_db(old_start)
        if start_level > UNSUPPORTED_BACKDATE_START_DBFS:
            continue

        # Hard authority guard: if *any* frozen corrected-valid pitch already
        # exists before the trusted anchor, this is not the unsupported class.
        prevalid = (
            result.pitch.valid
            & (result.pitch.t >= old_start - 1e-9)
            & (result.pitch.t < anchor - 1e-9)
        )
        if np.any(prevalid):
            continue

        # A modest target-family primary estimate anywhere in the corridor is
        # sufficient evidence to leave V18.1 untouched.
        if _target_family_primary_exists(result, old_start, anchor, float(e.base_st)):
            continue

        median_level = _borrowed_corridor_median_dbfs(probe, old_start, anchor)
        if median_level > UNSUPPORTED_BACKDATE_MAX_MEDIAN_DBFS:
            continue

        first_valid = _first_target_valid_time(result, e)
        if first_valid is None:
            continue
        # The event's frozen owner should begin at/near its expression anchor;
        # never manufacture a later start than the first valid target point.
        new_start = float(first_valid)
        if new_start <= old_start + 1e-9:
            continue
        # Do not turn a long bad backdate into a brand-new micro-note.  V18.2
        # is a timing repair only; sub-55 ms residual topology remains under
        # the already-frozen transient/articulation policy.
        if float(e.end_s) - new_start < UNSUPPORTED_BACKDATE_MIN_REMAINING_S:
            continue

        _trim_event_start(e, new_start)
        e.source_kind = f"{e.source_kind}+unsupported_borrowed_start_trim_v18_2"
        result.attack_decisions.append(AttackDecision(
            old_start,
            False,
            "unsupported_borrowed_start_v18_2",
            "weak_preanchor_corridor_has_no_valid_or_primary_target_family_v18_2",
            pitch_source=e.pitch_source,
            pitch_anchor_s=new_start,
            pitch_anchor_st=float(e.base_st),
            continuity_status=(
                f"old_start_dbfs={start_level:.3f};"
                f"borrowed_median_dbfs={median_level:.3f};"
                f"lag_s={anchor-old_start:.6f};"
                f"trimmed_to={new_start:.9f}"
            ),
        ))
        changed += 1

    result.events = _reclean_monophonic(result.events)
    return changed


def assemble_events(prefix: Path, wav_path: Path) -> AssemblyResult:
    result = assemble_events_v18_1(prefix, wav_path)
    probe = _WavProbe(wav_path)
    retracted = _retract_unsupported_borrowed_starts(result, probe)
    result.events = _reclean_monophonic(result.events)
    result.attack_decisions.append(AttackDecision(
        -1.0,
        False,
        "v18_2_summary",
        f"unsupported_borrowed_starts_retracted={retracted}",
        pitch_source="renderer_calibration_v18_2",
    ))
    return result
