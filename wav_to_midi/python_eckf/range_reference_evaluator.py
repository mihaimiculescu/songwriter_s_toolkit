from __future__ import annotations

"""Historical V13-style range-reference evaluator, adapted to the live V2 pipeline.

This restores the *reference population* contract that made the old V22 range
calibration robust.  It does not alter the V22 range geometry itself.

Historical contract, preserved here:
  * choose a locally measured acoustic candidate close to the current
    observational trajectory (<= 35 cents);
  * reject the observation as ambiguous when a rival > 70 cents away has
    comparable periodic support (ACF within 0.07 and >= 0.5 of the selected
    component strength);
  * optionally reject the selected period as a subsequent range reference when
    the unchanged vocal-transition curve challenges it *and* the local waveform
    is both weak relative to its own neighbourhood and rapidly decaying;
  * only unique locally supported periods become range anchors.

The old V13 used ``fundamental_fraction`` for the 0.5 comparison.  V2 exposes
measured component amplitude instead.  Because the comparison is within the
same frame, the common frame scale cancels; the >= 0.5 ratio has the same
within-frame role while remaining invariant to global WAV gain.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .trajectory_resolver import vocal_transition_penalty

SELECT_MATCH_CENTS = 35.0
COMPETITOR_SEPARATION_CENTS = 70.0
COMPETITOR_ACF_DELTA = 0.07
COMPETITOR_COMPONENT_RATIO = 0.50
LOW_RELATIVE_ENERGY = 0.25
DECAY_DB = -4.0
LOCAL_ENERGY_CONTEXT_S = 0.30
LOCAL_RMS_PROBE_S = 0.012
LOCAL_ENERGY_HOP_S = 0.010


def _cents(a: float, b: float) -> float:
    return abs(1200.0 * math.log2(float(a) / float(b)))


def _finite_positive(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0.0 else None


@dataclass(frozen=True)
class RangeReferenceEvidence:
    frame_index: int
    frame_start_sample: int
    time_s: float
    observational_hz: float | None
    selected_measured_hz: float | None
    selected_acf: float | None
    selected_component_amplitude: float | None
    local_candidate_count: int
    competing_candidate_count: int
    acoustic_measurement: str
    relative_energy: float | None
    energy_slope_db: float | None
    low_energy_flag: bool
    decaying_flag: bool
    prior_reference_hz: float | None
    penalty_elapsed_ms: float | None
    transition_penalty: float | None
    interval_energy_joint_challenge: bool
    evidence_status: str
    diagnostic_settled: bool
    usable_as_next_reference: bool


@dataclass(frozen=True)
class RangeReferenceResult:
    references_hz: tuple[float, ...]
    rows: tuple[RangeReferenceEvidence, ...]


def _rms_window(integral: np.ndarray, center: int, n: int) -> float | None:
    lo = int(center) - int(n) // 2
    hi = lo + int(n)
    if lo < 0 or hi >= len(integral):
        return None
    return float(math.sqrt(max(0.0, (integral[hi] - integral[lo]) / float(n))))


def _energy_at(audio: np.ndarray, sample_rate: float, integral: np.ndarray, sample: int):
    fs = float(sample_rate)
    probe = max(32, int(round(LOCAL_RMS_PROBE_S * fs)))
    hop = max(1, int(round(LOCAL_ENERGY_HOP_S * fs)))
    now = _rms_window(integral, sample, probe)
    left = _rms_window(integral, sample - probe, probe)
    right = _rms_window(integral, sample + probe, probe)

    lo = max(probe, int(sample - round(LOCAL_ENERGY_CONTEXT_S * fs)))
    hi = min(len(audio) - probe, int(sample + round(LOCAL_ENERGY_CONTEXT_S * fs)))
    if hi >= lo:
        centers = np.arange(lo, hi + 1, hop, dtype=np.int64)
        vals = [
            _rms_window(integral, int(c), probe)
            for c in centers
        ]
        vals = [v for v in vals if v is not None]
    else:
        vals = []
    ref = float(np.quantile(np.asarray(vals, dtype=np.float64), 0.80)) if vals else None
    ratio = now / ref if now is not None and ref is not None and ref > 1e-12 else None
    slope = (
        20.0 * math.log10((right + 1e-12) / (left + 1e-12))
        if left is not None and right is not None else None
    )
    return ratio, slope


def _frame_observational_hz(
    *,
    sample_frame_index: np.ndarray,
    clean_f0_hz: np.ndarray,
    valid: np.ndarray,
) -> dict[int, float]:
    by_frame: dict[int, list[float]] = defaultdict(list)
    for fi, hz, ok in zip(sample_frame_index, clean_f0_hz, valid):
        if not bool(ok):
            continue
        v = _finite_positive(hz)
        if v is not None:
            by_frame[int(fi)].append(v)
    return {
        fi: float(np.median(np.asarray(vals, dtype=np.float64)))
        for fi, vals in by_frame.items() if vals
    }


def build_v13_style_range_references(
    *,
    pitch_candidates: Iterable,
    audio: np.ndarray,
    sample_rate: float,
    sample_frame_index: np.ndarray,
    clean_f0_hz: np.ndarray,
    valid: np.ndarray,
) -> RangeReferenceResult:
    """Build the trusted per-file range-reference population.

    This stage is deliberately independent of the range gate itself, so there
    is no circular ``range says candidate is valid, therefore candidate defines
    range`` feedback loop.
    """
    candidates = list(pitch_candidates)
    by_frame: dict[int, list] = defaultdict(list)
    for row in candidates:
        by_frame[int(row.frame_index)].append(row)

    observational = _frame_observational_hz(
        sample_frame_index=np.asarray(sample_frame_index),
        clean_f0_hz=np.asarray(clean_f0_hz, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
    )

    y = np.asarray(audio, dtype=np.float64)
    sq = np.r_[0.0, np.cumsum(np.square(y, dtype=np.float64))]

    rows: list[RangeReferenceEvidence] = []
    refs: list[float] = []
    previous_ref_hz: float | None = None
    previous_ref_sample: int | None = None

    for fi in sorted(by_frame):
        group = by_frame[fi]
        start = int(group[0].frame_start_sample)
        time_s = float(group[0].time_s)
        obs = observational.get(fi)

        selected = None
        if obs is not None:
            near = [r for r in group if _cents(float(r.candidate_hz), obs) <= SELECT_MATCH_CENTS]
            if near:
                selected = min(near, key=lambda r: _cents(float(r.candidate_hz), obs))

        competing_rows = []
        if selected is not None:
            s_hz = float(selected.candidate_hz)
            s_acf = float(selected.measured_period_acf)
            s_amp = max(0.0, float(selected.measured_component_amplitude))
            for other in group:
                o_hz = float(other.candidate_hz)
                if _cents(o_hz, s_hz) <= COMPETITOR_SEPARATION_CENTS:
                    continue
                o_acf = float(other.measured_period_acf)
                o_amp = max(0.0, float(other.measured_component_amplitude))
                if (
                    o_acf >= s_acf - COMPETITOR_ACF_DELTA
                    and o_amp >= s_amp * COMPETITOR_COMPONENT_RATIO
                ):
                    competing_rows.append(other)

        ratio, slope = _energy_at(y, sample_rate, sq, start)
        low = bool(ratio is not None and ratio < LOW_RELATIVE_ENERGY)
        decay = bool(slope is not None and slope < DECAY_DB)

        penalty = None
        elapsed_ms = None
        prior_ref_hz = previous_ref_hz
        prior_ref_sample = previous_ref_sample
        if selected is not None and prior_ref_hz is not None and prior_ref_sample is not None:
            elapsed_ms = (start - prior_ref_sample) / float(sample_rate) * 1000.0
            if elapsed_ms > 0.0:
                penalty = float(vocal_transition_penalty(
                    12.0 * math.log2(float(selected.candidate_hz) / prior_ref_hz),
                    elapsed_ms,
                ))

        if selected is None:
            acoustic = "no_unique_locally_supported_period"
        elif competing_rows:
            acoustic = "competing_locally_supported_periods"
        else:
            acoustic = "unique_locally_supported_period"

        joint = bool(
            acoustic == "unique_locally_supported_period"
            and penalty is not None and penalty > 0.0
            and low and decay
        )

        if joint:
            status = "interval_energy_joint_challenge_unresolved"
        elif acoustic == "unique_locally_supported_period":
            status = "acoustic_period_supported_source_unverified"
        else:
            status = "acoustic_ambiguity_unresolved"

        reference_ok = status == "acoustic_period_supported_source_unverified"
        selected_hz = float(selected.candidate_hz) if selected is not None else None
        if reference_ok and selected_hz is not None:
            refs.append(selected_hz)
            previous_ref_hz = selected_hz
            previous_ref_sample = start
        else:
            # Historical V13 rule: never carry a frequency across an unresolved
            # observation.  A later frame must establish itself afresh.
            previous_ref_hz = None
            previous_ref_sample = None

        rows.append(RangeReferenceEvidence(
            frame_index=int(fi),
            frame_start_sample=start,
            time_s=time_s,
            observational_hz=obs,
            selected_measured_hz=selected_hz,
            selected_acf=(float(selected.measured_period_acf) if selected is not None else None),
            selected_component_amplitude=(float(selected.measured_component_amplitude) if selected is not None else None),
            local_candidate_count=len(group),
            competing_candidate_count=len(competing_rows),
            acoustic_measurement=acoustic,
            relative_energy=ratio,
            energy_slope_db=slope,
            low_energy_flag=low,
            decaying_flag=decay,
            prior_reference_hz=(None if elapsed_ms is None else prior_ref_hz),
            penalty_elapsed_ms=elapsed_ms,
            transition_penalty=penalty,
            interval_energy_joint_challenge=joint,
            evidence_status=status,
            diagnostic_settled=(status == "acoustic_period_supported_source_unverified"),
            usable_as_next_reference=reference_ok,
        ))

    return RangeReferenceResult(tuple(refs), tuple(rows))
