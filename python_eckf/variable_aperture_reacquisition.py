from __future__ import annotations

"""V3R3 variable-aperture adjudication for ECKF reacquisition seeds.

This module remains deliberately upstream of the Kalman forth/back filtering
pass.  It does not inspect the already-selected tracker trajectory.  When the
current initialization is absent or acoustically ambiguous among integer-
related measured-period families, it examines the raw waveform over the
existing 24/40/64/96-ms aperture ladder and asks which of those physically
measured families persists.

V3R3 generalises the previous octave-only contest to integer-related families
with ratios x2..x5.  No higher/lower direction is preferred.  No continuity is
forced across silence, unvoiced material, a harmonic-change boundary, or end
of-audio.  If the aperture ladder remains genuinely ambiguous, the caller
keeps the pre-V3R3 choice and falls back to the historical offline lookahead.
"""

from dataclasses import dataclass
import math
import numpy as np

from .initialization_candidates import (
    InitializationChoice,
    _harmonic_support,
    _measured_amplitude_phase,
    _measured_period_candidates,
)
from .periodicity import assess_periodicity
from .silence import is_silent
from .trajectory_resolver import vocal_transition_penalty


APERTURES_MS: tuple[float, ...] = (24.0, 40.0, 64.0, 96.0)
MAX_DISAGREEMENT_CENTS = 50.0


@dataclass(frozen=True)
class ApertureCandidate:
    frequency_hz: float
    midi_group: int
    acf_peak: float
    cmndf_minimum: float
    amplitude: float
    harmonics: tuple[int, ...]
    transition_penalty: float | None


@dataclass(frozen=True)
class VariableApertureDecision:
    choice: InitializationChoice
    state: str
    reason: str
    aperture_ms: float | None
    original_hz: float | None
    selected_hz: float | None
    competing_groups: tuple[int, ...]


def _midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def _midi_group(hz: float) -> int:
    return int(math.floor(_midi(hz) + 0.5))


def _integer_related(higher_hz: float, lower_hz: float) -> bool:
    if higher_hz <= lower_hz * 1.4:
        return False
    ratio = higher_hz / lower_hz
    nearest = round(ratio)
    if not (2 <= nearest <= 5):
        return False
    return abs(1200.0 * math.log2(ratio / nearest)) <= MAX_DISAGREEMENT_CENTS


def _dominates(higher: ApertureCandidate, lower: ApertureCandidate) -> bool:
    """Reuse the existing initialization-candidate dominance semantics.

    This is intentionally asymmetric only in the physically meaningful sense:
    a shorter measured period may expose a submultiple illusion.  It is not an
    octave-up preference; the stronger family still has to satisfy the same
    measured-period and fundamental-amplitude evidence used by V18.
    """
    if not _integer_related(higher.frequency_hz, lower.frequency_hz):
        return False
    clear_period_advantage = (
        higher.acf_peak >= lower.acf_peak + 0.04
        and higher.cmndf_minimum <= lower.cmndf_minimum - 0.03
    )
    overwhelming_fundamental_advantage = (
        higher.amplitude >= 8.0 * lower.amplitude
        and higher.acf_peak >= lower.acf_peak - 0.02
        and higher.cmndf_minimum <= lower.cmndf_minimum + 0.02
    )
    return (
        higher.amplitude >= 3.0 * lower.amplitude and clear_period_advantage
    ) or overwhelming_fundamental_advantage


def _candidate_records(
    detector,
    window: np.ndarray,
    fs: float,
    start_sample: int,
    previous_hz: float | None,
    elapsed_ms: float | None,
) -> tuple[ApertureCandidate, ...]:
    measured = _measured_period_candidates(window, fs)
    if not measured:
        return ()

    records: list[ApertureCandidate] = []
    for hz, _lag, acf_peak, cmndf_minimum in measured:
        if not (math.isfinite(hz) and hz > 0.0 and hz < fs / 2.0):
            continue
        harmonics = _harmonic_support(detector, window, fs, hz)
        if len(harmonics) < 2:
            continue
        amplitude, _phase = _measured_amplitude_phase(window, fs, hz, start_sample)
        if not (math.isfinite(amplitude) and amplitude > 0.0):
            continue
        # Same high-only-family admission rule used by choose_initialization.
        # We first collect all measured families, then apply the relative floor.
        penalty = None
        if (
            previous_hz is not None
            and math.isfinite(previous_hz)
            and previous_hz > 0.0
            and elapsed_ms is not None
        ):
            penalty = float(vocal_transition_penalty(
                12.0 * math.log2(hz / previous_hz), elapsed_ms
            ))
        records.append(ApertureCandidate(
            frequency_hz=float(hz),
            midi_group=_midi_group(hz),
            acf_peak=float(acf_peak),
            cmndf_minimum=float(cmndf_minimum),
            amplitude=float(amplitude),
            harmonics=tuple(int(h) for h in harmonics),
            transition_penalty=penalty,
        ))

    if not records:
        return ()

    # Remove measured submultiple illusions using exactly the evidence shape
    # already present in V18 initialization_candidates.choose_initialization.
    records = [
        r for r in records
        if not any(
            other is not r
            and other.frequency_hz > r.frequency_hz
            and _dominates(other, r)
            for other in records
        )
    ]
    if not records:
        return ()

    floor = max(r.amplitude for r in records)
    records = [
        r for r in records
        if any(h <= 3 for h in r.harmonics)
        or (
            r.amplitude >= 0.25 * floor
            and r.acf_peak >= 0.80
            and r.cmndf_minimum <= 0.20
        )
    ]
    if not records:
        return ()

    # Deduplicate same MIDI family by the existing acoustic ordering.
    by_group: dict[int, ApertureCandidate] = {}
    for r in records:
        incumbent = by_group.get(r.midi_group)
        key = (
            -r.acf_peak,
            r.cmndf_minimum,
            -r.amplitude,
            0.0 if r.transition_penalty is None else r.transition_penalty,
            -len(r.harmonics),
        )
        if incumbent is None:
            by_group[r.midi_group] = r
            continue
        old_key = (
            -incumbent.acf_peak,
            incumbent.cmndf_minimum,
            -incumbent.amplitude,
            0.0 if incumbent.transition_penalty is None else incumbent.transition_penalty,
            -len(incumbent.harmonics),
        )
        if key < old_key:
            by_group[r.midi_group] = r
    return tuple(by_group[g] for g in sorted(by_group))


def _integer_related_records(a: ApertureCandidate, b: ApertureCandidate) -> bool:
    high, low = (a, b) if a.frequency_hz >= b.frequency_hz else (b, a)
    return _integer_related(high.frequency_hz, low.frequency_hz)


def _same_integer_family_as_hz(record: ApertureCandidate, hz: float) -> bool:
    if not (math.isfinite(hz) and hz > 0.0):
        return False
    high = max(record.frequency_hz, hz)
    low = min(record.frequency_hz, hz)
    return _integer_related(high, low)


def _boundary_allows_aperture(
    *, y: np.ndarray, raw_y: np.ndarray, start: int, end: int, block: int,
    original_length: int, fs: float, detector, silence_threshold: float,
    silence_flatness_threshold: float,
) -> tuple[bool, str]:
    """Fail closed if widening would cross an existing tracker boundary."""
    if end > original_length:
        return False, "END_OF_REAL_AUDIO"
    # 24/40 ms remain inside the current frame. Wider apertures can cross one
    # or more block boundaries; vet each newly entered standard frame using
    # the same silence/periodicity/harmonic-change concepts as V18 lookahead.
    current = y[start:min(start + block, original_length)]
    if current.size < 3:
        return False, "ANCHOR_TOO_SHORT"
    boundary = start + block
    prior = current
    while boundary < end:
        if boundary + block > original_length:
            # The aperture may legitimately end inside the final real frame.
            future = y[boundary:original_length]
            future_raw = raw_y[boundary:original_length]
        else:
            future = y[boundary:boundary + block]
            future_raw = raw_y[boundary:boundary + block]
        if future.size < 3:
            return False, "FUTURE_TOO_SHORT"
        silent, _flatness, energy = is_silent(
            future_raw,
            flatness_threshold=silence_flatness_threshold,
            energy_db_threshold=silence_threshold,
        )
        if silent or energy < silence_threshold:
            return False, "FUTURE_SILENCE_BOUNDARY"
        try:
            periodicity = assess_periodicity(future, fs)
        except ValueError:
            return False, "FUTURE_PERIODICITY_UNAVAILABLE"
        if not periodicity.voiced:
            return False, "FUTURE_UNVOICED_BOUNDARY"
        try:
            change = detector.analyze(prior, future, fs)
        except Exception:
            change = None
        if change is not None and bool(getattr(change, "flag", False)):
            return False, "FUTURE_HARMONIC_BOUNDARY"
        prior = future
        boundary += block
    return True, "OK"


REGISTER_FAMILY_FREE_ZONE_ST = 4


def _record_acoustic_key(r: ApertureCandidate):
    return (
        -r.acf_peak,
        r.cmndf_minimum,
        -r.amplitude,
        0.0 if r.transition_penalty is None else r.transition_penalty,
        -len(r.harmonics),
    )


def _family_records(records, centers):
    """Map measured candidates onto octave-register families.

    Family identity is allowed to drift within the interval juror's historical
    <=4-semitone free zone. This lets a real melody move 66->65 across wider
    apertures without being misread as disappearance of the upper register.
    """
    out={}
    for r in records:
        d=[(abs(int(r.midi_group)-int(c)), int(c)) for c in centers]
        if not d:
            continue
        d.sort()
        dist,center=d[0]
        if dist > REGISTER_FAMILY_FREE_ZONE_ST:
            continue
        # Equidistant between two register centers is genuinely ambiguous.
        if len(d)>1 and d[1][0] == dist:
            continue
        incumbent=out.get(center)
        if incumbent is None or _record_acoustic_key(r) < _record_acoustic_key(incumbent):
            out[center]=r
    return out


def _status_vector_for_group(group: int, family_rows):
    """Per-aperture status: 2=unique family, 1=surviving family, 0=absent."""
    out=[]
    for _ms, fmap, unique_group in family_rows:
        if unique_group == group:
            out.append(2)
        elif group in fmap:
            out.append(1)
        else:
            out.append(0)
    return tuple(out)


def _lexicographic_choice(groups, vectors, *, reverse=False):
    if not groups:
        return None
    n=len(next(iter(vectors.values())))
    order=range(n-1,-1,-1) if reverse else range(n)
    seqs={g:tuple(vectors[g][i] for i in order) for g in groups}
    best=max(seqs.values())
    winners=[g for g in groups if seqs[g]==best]
    return winners[0] if len(winners)==1 else None


def _breadth_choice(groups, vectors):
    if not groups:
        return None
    scores={g:sum(vectors[g]) for g in groups}
    best=max(scores.values())
    winners=[g for g in groups if scores[g]==best]
    return winners[0] if len(winners)==1 else None


def _best_record_for_group(group: int, family_rows):
    candidates=[]
    for ms,fmap,_unique_group in family_rows:
        r=fmap.get(group)
        if r is None:
            continue
        key=(-float(ms),)+_record_acoustic_key(r)
        candidates.append((key,r))
    if not candidates:
        return None
    candidates.sort(key=lambda x:x[0])
    return candidates[0][1]


def adjudicate_variable_aperture_reacquisition(
    *, y: np.ndarray, raw_y: np.ndarray, start: int, block: int,
    original_length: int, fs: float, detector, silence_threshold: float,
    silence_flatness_threshold: float,
    original_choice: InitializationChoice, previous_hz: float | None,
    elapsed_ms: float | None,
    apertures_ms: tuple[float, ...] = APERTURES_MS,
) -> VariableApertureDecision:
    """Bidirectional variable-aperture adjudication for an ECKF reacquisition seed.

    Unlike the prototype V19, this function never stops at the first decisive
    aperture.  It evaluates every aperture that can be reached without crossing
    an existing tracker boundary, builds an integer-family support profile, then
    interprets that profile both short->long and long->short.  Overall support
    breadth is a third independent reason.  A family must win a strict majority
    of those reasons to replace an existing seed; ties keep the V18 choice.
    """
    original_hz=original_choice.frequency_hz
    aperture_rows=[]
    reached=[]
    boundary_reason=None

    for aperture_ms in apertures_ms:
        width=max(3, int(round(float(aperture_ms)*fs/1000.0)))
        end=start+width
        allowed, why=_boundary_allows_aperture(
            y=y, raw_y=raw_y, start=start, end=end, block=block,
            original_length=original_length, fs=fs, detector=detector,
            silence_threshold=silence_threshold,
            silence_flatness_threshold=silence_flatness_threshold,
        )
        if not allowed:
            boundary_reason=why
            break
        records=_candidate_records(detector, y[start:end], fs, start, previous_hz, elapsed_ms)
        reached.append(float(aperture_ms))
        aperture_rows.append([float(aperture_ms), records, None])

    if not aperture_rows:
        return VariableApertureDecision(
            original_choice, 'keep', boundary_reason or 'no_aperture_available',
            None, original_hz, original_hz, (),
        )

    # Build one integer-related contest across all reached apertures.  Family
    # membership is based on measured frequencies (x2..x5 within the existing
    # cents tolerance), not on a preferred octave direction.
    contest=set()
    all_records=[r for _ms, records, _ in aperture_rows for r in records]
    if original_hz is not None and math.isfinite(original_hz) and original_hz>0:
        og=_midi_group(original_hz)
        contest.add(og)
        for r in all_records:
            if _same_integer_family_as_hz(r, float(original_hz)):
                contest.add(r.midi_group)
    else:
        for i,a in enumerate(all_records):
            for b in all_records[i+1:]:
                if _integer_related_records(a,b):
                    contest.add(a.midi_group); contest.add(b.midi_group)

    # If no integer-related contest exists and the current frame has no seed,
    # retain the useful V19 behaviour but adjudicate over the reached profile.
    if len(contest)<2:
        if original_hz is None:
            all_groups=sorted({r.midi_group for _ms, records, _ in aperture_rows for r in records})
            if len(all_groups)==1:
                g=all_groups[0]
                r=next((r for _ms,records,_ in reversed(aperture_rows) for r in records if r.midi_group==g), None)
                if r is not None:
                    amp,phase=_measured_amplitude_phase(y[start:start+block],fs,r.frequency_hz,start)
                    if math.isfinite(amp) and amp>0 and math.isfinite(phase):
                        choice=InitializationChoice(r.frequency_hz,amp,phase,
                            'variable_aperture_reacquisition',
                            'bidirectional_unique_family',len(r.harmonics),r.transition_penalty)
                        return VariableApertureDecision(choice,'replace','bidirectional_unique_future_family',
                            max(reached),original_hz,r.frequency_hz,(g,))
        return VariableApertureDecision(original_choice,'keep',boundary_reason or 'no_integer_family_contest',
            None,original_hz,original_hz,tuple(sorted(contest)))

    contest=tuple(sorted(contest))
    # Build integer-family evidence at every aperture. Candidate notes may move
    # within the historical <=4-semitone free zone while remaining the same
    # local family center.
    family_rows=[]
    for ms,records,_ in aperture_rows:
        fmap=_family_records(records,contest)
        unique=None
        if len(fmap)==1:
            unique=next(iter(fmap))
        elif fmap:
            undominated=[]
            for g,r in fmap.items():
                if not any(og!=g and other.frequency_hz>r.frequency_hz and _dominates(other,r)
                           for og,other in fmap.items()):
                    undominated.append(g)
            if len(undominated)==1:
                unique=undominated[0]
        family_rows.append([ms,fmap,unique])

    vectors={g:_status_vector_for_group(g,family_rows) for g in contest}
    short_choice=_lexicographic_choice(contest,vectors,reverse=False)
    long_choice=_lexicographic_choice(contest,vectors,reverse=True)
    breadth_choice=_breadth_choice(contest,vectors)
    reasons=[x for x in (short_choice,long_choice,breadth_choice) if x is not None]
    counts={g:reasons.count(g) for g in contest}
    best_count=max(counts.values()) if counts else 0
    winners=[g for g in contest if counts.get(g,0)==best_count and best_count>0]
    winner_group=winners[0] if len(winners)==1 and best_count>=2 else None

    if winner_group is None:
        return VariableApertureDecision(original_choice,'ambiguous','bidirectional_profile_no_majority',
            None,original_hz,original_hz,contest)

    # When an existing seed is already in the winning family there is nothing
    # to change; record the profile agreement but preserve V18 exactly.
    if original_hz is not None and math.isfinite(original_hz) and original_hz>0:
        if _midi_group(original_hz)==winner_group:
            return VariableApertureDecision(original_choice,'keep','bidirectional_profile_confirms_original',
                max(reached),original_hz,original_hz,contest)

    winner=_best_record_for_group(winner_group,family_rows)
    if winner is None:
        return VariableApertureDecision(original_choice,'ambiguous','bidirectional_winner_missing_record',
            None,original_hz,original_hz,contest)
    amp,phase=_measured_amplitude_phase(y[start:start+block],fs,winner.frequency_hz,start)
    if not (math.isfinite(amp) and amp>0 and math.isfinite(phase)):
        return VariableApertureDecision(original_choice,'ambiguous','bidirectional_winner_unusable_current_frame',
            None,original_hz,original_hz,contest)

    choice=InitializationChoice(
        winner.frequency_hz,amp,phase,'variable_aperture_reacquisition',
        'bidirectional_profile_majority',len(winner.harmonics),winner.transition_penalty,
    )
    return VariableApertureDecision(choice,'replace','bidirectional_profile_majority',
        max(reached),original_hz,winner.frequency_hz,contest)
