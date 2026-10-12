from __future__ import annotations

"""V3R5 variable-aperture adjudication for ECKF reacquisition seeds.

This module remains deliberately upstream of the Kalman forth/back filtering
pass.  It does not inspect the already-selected tracker trajectory.  When the
current initialization is absent or acoustically ambiguous among integer-
related measured-period families, it examines the raw waveform over the
existing 24/40/64/96-ms aperture ladder and asks which of those physically
measured families persists.

V3R5 retains the V3R4 integer-family / subharmonic-borrowing machinery and
adds a narrowly scoped moving-onset rescue for a different failure mode:
after a true tracker reset, the current-frame seed can disappear completely
while one coherent measured family emerges at both 64 and 96 ms.  In that
specific case the persistent family may replace the vanished seed even when
pitch motion has moved it outside the frozen-seed integer-family tolerance.

V3R4 retains the V3R3 integer-family contest and adds a narrowly scoped
subharmonic-borrowing certificate at the two widest apertures.  The certificate
does not prefer higher notes globally: it applies only when a lower measured
integer-related family has almost no spectral evidence exclusive to its own
harmonic grid while a credibly periodic higher family overwhelmingly occupies
the shared grid.

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

# V3R4: conservative subharmonic-borrowing certificate.  These thresholds are
# deliberately applied only inside the variable-aperture integer-family
# contest and only when the same evidence repeats at BOTH wide apertures.
# Nothing here changes choose_initialization(), the global periodicity test,
# the ECKF, range logic, jurors, or MIDI rendering.
BORROWING_REQUIRED_APERTURES_MS: tuple[float, ...] = (64.0, 96.0)
BORROWING_MIN_HIGH_ACF = 0.77
BORROWING_MAX_HIGH_CMNDF = 0.20
BORROWING_MIN_HIGH_TO_LOW_FUNDAMENTAL = 8.0
BORROWING_MAX_LOW_EXCLUSIVE_TO_SHARED_RMS = 0.12
BORROWING_MAX_HARMONIC = 12

# V3R5: moving reacquisition-onset rescue.  This does NOT loosen the existing
# integer-family tolerance.  It is available only after a real state reset
# (previous_hz is unavailable), only when the original seed has no measured
# support anywhere on the 24/40/64/96-ms ladder, and only when exactly one
# independently credible family persists at both wide apertures.  The
# 4-semitone coherence bound reuses the interval juror's historical free zone;
# it is not a newly learned song-specific tolerance.
MOVING_ONSET_REQUIRED_APERTURES_MS: tuple[float, ...] = (64.0, 96.0)
MOVING_ONSET_MIN_ACF = 0.80
MOVING_ONSET_MAX_CMNDF = 0.20
MOVING_ONSET_MAX_DRIFT_ST = 4.0
MOVING_ONSET_SEED_EXCLUSION_ST = 4.0


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


def _low_exclusive_to_shared_harmonic_rms(
    window: np.ndarray, fs: float, start_sample: int,
    lower_hz: float, integer_ratio: int,
) -> float | None:
    """Measure whether a low family has harmonic evidence of its own.

    For an integer-related pair high ~= N * low, every Nth harmonic on the
    low grid is shared with the high family.  Harmonics whose indices are NOT
    divisible by N are exclusive to the longer low period.

    A genuine low F0 may have a weak fundamental, so the fundamental alone is
    not enough to reject it.  Instead compare the RMS sinusoid amplitude of
    the low-exclusive grid (1F, 3F, 5F... for an octave pair) with the RMS of
    the shared grid (2F, 4F, 6F...).  The same least-squares sinusoid
    measurement already used by the initializer is used here.
    """
    if not (math.isfinite(lower_hz) and lower_hz > 0.0):
        return None
    if integer_ratio < 2:
        return None

    exclusive: list[float] = []
    shared: list[float] = []
    for harmonic in range(1, BORROWING_MAX_HARMONIC + 1):
        target = float(lower_hz) * harmonic
        if target >= fs / 2.0:
            break
        amplitude, _phase = _measured_amplitude_phase(
            window, fs, target, start_sample
        )
        if not (math.isfinite(amplitude) and amplitude >= 0.0):
            continue
        if harmonic % integer_ratio == 0:
            shared.append(float(amplitude))
        else:
            exclusive.append(float(amplitude))

    # Fail closed when there is not enough grid evidence on both sides.
    if len(exclusive) < 2 or len(shared) < 2:
        return None
    shared_rms = math.sqrt(sum(a * a for a in shared) / len(shared))
    if not (math.isfinite(shared_rms) and shared_rms > 0.0):
        return None
    exclusive_rms = math.sqrt(sum(a * a for a in exclusive) / len(exclusive))
    if not math.isfinite(exclusive_rms):
        return None
    return float(exclusive_rms / shared_rms)


def _borrowing_pair_passes(
    *, window: np.ndarray, fs: float, start_sample: int,
    higher: ApertureCandidate, lower: ApertureCandidate,
) -> tuple[bool, int | None, float | None]:
    """Return a conservative certificate for one aperture and one pair.

    The higher family must be an independently credible measured period, its
    physical fundamental component must overwhelm the lower candidate, and
    the lower family's *exclusive* harmonic grid must be weak relative to the
    harmonics it shares with the higher family.
    """
    if higher.frequency_hz <= lower.frequency_hz:
        return False, None, None
    if not _integer_related(higher.frequency_hz, lower.frequency_hz):
        return False, None, None
    ratio = int(round(higher.frequency_hz / lower.frequency_hz))
    if not (2 <= ratio <= 5):
        return False, None, None
    if higher.acf_peak < BORROWING_MIN_HIGH_ACF:
        return False, ratio, None
    if higher.cmndf_minimum > BORROWING_MAX_HIGH_CMNDF:
        return False, ratio, None
    if higher.amplitude < BORROWING_MIN_HIGH_TO_LOW_FUNDAMENTAL * lower.amplitude:
        return False, ratio, None

    exclusive_to_shared = _low_exclusive_to_shared_harmonic_rms(
        window, fs, start_sample, lower.frequency_hz, ratio
    )
    if exclusive_to_shared is None:
        return False, ratio, None
    return (
        exclusive_to_shared <= BORROWING_MAX_LOW_EXCLUSIVE_TO_SHARED_RMS,
        ratio,
        exclusive_to_shared,
    )


def _subharmonic_borrowing_suppressions(
    *, aperture_rows, contest, y: np.ndarray, start: int, fs: float,
):
    """Find low families certified as borrowed at BOTH 64 and 96 ms.

    Returns ``(suppressions, certified_pairs)``.  ``suppressions`` maps each
    required aperture to the family centers that should be treated as absent
    for the support-profile vote.  Short apertures are never altered.

    Requiring the same pair and the same integer relation at both wide
    apertures is intentional: a one-window spectral accident cannot trigger
    the certificate.
    """
    raw_family_maps = {
        float(ms): _family_records(records, contest)
        for ms, records, _ in aperture_rows
    }

    required_rows = {}
    for required_ms in BORROWING_REQUIRED_APERTURES_MS:
        match = next(
            (ms for ms in raw_family_maps if math.isclose(
                ms, required_ms, rel_tol=0.0, abs_tol=1e-9
            )),
            None,
        )
        if match is None:
            return {}, ()
        required_rows[required_ms] = raw_family_maps[match]

    suppressions = {float(ms): set() for ms in BORROWING_REQUIRED_APERTURES_MS}
    certified_pairs = []
    groups = tuple(sorted(int(g) for g in contest))

    for i, group_a in enumerate(groups):
        for group_b in groups[i + 1:]:
            orientation = None
            ratio_ref = None
            passed = True

            for required_ms in BORROWING_REQUIRED_APERTURES_MS:
                fmap = required_rows[required_ms]
                a = fmap.get(group_a)
                b = fmap.get(group_b)
                if a is None or b is None:
                    passed = False
                    break

                if a.frequency_hz < b.frequency_hz:
                    low_group, lower, high_group, higher = group_a, a, group_b, b
                else:
                    low_group, lower, high_group, higher = group_b, b, group_a, a

                current_orientation = (low_group, high_group)
                if orientation is None:
                    orientation = current_orientation
                elif current_orientation != orientation:
                    passed = False
                    break

                width = max(3, int(round(float(required_ms) * fs / 1000.0)))
                window = y[start:start + width]
                ok, ratio, _exclusive_to_shared = _borrowing_pair_passes(
                    window=window, fs=fs, start_sample=start,
                    higher=higher, lower=lower,
                )
                if not ok or ratio is None:
                    passed = False
                    break
                if ratio_ref is None:
                    ratio_ref = ratio
                elif ratio != ratio_ref:
                    passed = False
                    break

            if not passed or orientation is None or ratio_ref is None:
                continue

            low_group, high_group = orientation
            for required_ms in BORROWING_REQUIRED_APERTURES_MS:
                suppressions[float(required_ms)].add(int(low_group))
            certified_pairs.append((int(low_group), int(high_group), int(ratio_ref)))

    # Drop empty aperture entries so a false/unused certificate is easy to test.
    suppressions = {ms: groups for ms, groups in suppressions.items() if groups}
    return suppressions, tuple(certified_pairs)


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


def _semitone_distance(a_hz: float, b_hz: float) -> float:
    if not (
        math.isfinite(a_hz) and a_hz > 0.0
        and math.isfinite(b_hz) and b_hz > 0.0
    ):
        return math.inf
    return abs(12.0 * math.log2(float(a_hz) / float(b_hz)))


def _moving_onset_rescue_candidate(
    *, aperture_rows, original_hz: float | None, previous_hz: float | None,
) -> tuple[ApertureCandidate | None, str]:
    """Certify one persistent moving family after a true reacquisition reset.

    This is intentionally stricter than the integer-family path.  It exists
    for the case where a short/current-frame seed is a transient periodicity
    illusion and therefore cannot even form an integer contest once the true
    F0 begins moving during the 64/96-ms future aperture.

    Safety conditions:

    * there must be no immediately preceding valid F0 (true reset/gap only);
    * an original seed must exist, otherwise the historical seedless V19 path
      remains responsible;
    * the original seed's local register family must be absent at EVERY
      reached aperture (24/40/64/96 ms);
    * both 64 and 96 ms must be available without crossing an existing tracker
      boundary;
    * each wide aperture must contain exactly ONE surviving measured family;
    * both wide measurements must independently pass conservative periodicity
      bounds; and
    * their pitch motion must stay inside the existing 4-semitone local-family
      free zone.

    Once 96 ms confirms persistence, return the 64-ms record rather than the
    96-ms record because it is temporally closer to the reacquisition instant.
    """
    if previous_hz is not None:
        return None, "moving_onset_requires_true_reset"
    if original_hz is None or not math.isfinite(original_hz) or original_hz <= 0.0:
        return None, "moving_onset_requires_original_seed"

    # A seed that remains acoustically visible on any aperture is not a
    # vanished onset illusion.  Fail closed rather than reinterpret a possible
    # grace note, ornament, or genuinely short first pitch.
    for _ms, records, _ in aperture_rows:
        for record in records:
            if _semitone_distance(record.frequency_hz, original_hz) <= MOVING_ONSET_SEED_EXCLUSION_ST:
                return None, "moving_onset_seed_still_supported"

    wide: dict[float, ApertureCandidate] = {}
    for required_ms in MOVING_ONSET_REQUIRED_APERTURES_MS:
        row = next(
            (
                (ms, records)
                for ms, records, _ in aperture_rows
                if math.isclose(float(ms), float(required_ms), rel_tol=0.0, abs_tol=1e-9)
            ),
            None,
        )
        if row is None:
            return None, "moving_onset_wide_aperture_unavailable"
        _ms, records = row
        if len(records) != 1:
            return None, "moving_onset_wide_family_not_unique"
        record = records[0]
        if record.acf_peak < MOVING_ONSET_MIN_ACF:
            return None, "moving_onset_wide_acf_too_weak"
        if record.cmndf_minimum > MOVING_ONSET_MAX_CMNDF:
            return None, "moving_onset_wide_cmndf_too_weak"
        wide[float(required_ms)] = record

    first = wide[float(MOVING_ONSET_REQUIRED_APERTURES_MS[0])]
    last = wide[float(MOVING_ONSET_REQUIRED_APERTURES_MS[-1])]
    if _semitone_distance(first.frequency_hz, last.frequency_hz) > MOVING_ONSET_MAX_DRIFT_ST:
        return None, "moving_onset_family_motion_incoherent"

    return first, "persistent_moving_future_family"


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

    # V3R5: if no integer-related contest can be formed, first test the much
    # narrower moving-onset failure mode.  Do NOT widen the 50-cent integer
    # family gate: the whole point is to recognize a vanished seed followed by
    # one coherent moving family after a true reset.
    if len(contest)<2:
        moving_record, moving_reason = _moving_onset_rescue_candidate(
            aperture_rows=aperture_rows,
            original_hz=original_hz,
            previous_hz=previous_hz,
        )
        if moving_record is not None:
            amp, phase = _measured_amplitude_phase(
                y[start:start + block], fs, moving_record.frequency_hz, start
            )
            if math.isfinite(amp) and amp > 0.0 and math.isfinite(phase):
                choice = InitializationChoice(
                    moving_record.frequency_hz,
                    amp,
                    phase,
                    'variable_aperture_reacquisition',
                    moving_reason,
                    len(moving_record.harmonics),
                    moving_record.transition_penalty,
                )
                return VariableApertureDecision(
                    choice,
                    'replace',
                    'moving_reacquisition_onset_persistent_family',
                    max(MOVING_ONSET_REQUIRED_APERTURES_MS),
                    original_hz,
                    moving_record.frequency_hz,
                    (moving_record.midi_group,),
                )

        # If the current frame has no seed, retain the useful V19 behaviour
        # and adjudicate over the reached profile.
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

    # V3R4: before interpreting the support profile, ask whether a lower
    # integer-related family is merely borrowing the harmonic lattice of a
    # much stronger higher family.  This is deliberately NOT folded into the
    # global _dominates() rule: it must repeat at both 64 and 96 ms and only
    # affects those two wide-aperture rows.
    borrowing_suppressions, borrowing_pairs = _subharmonic_borrowing_suppressions(
        aperture_rows=aperture_rows, contest=contest, y=y, start=start, fs=fs
    )

    # Build integer-family evidence at every aperture. Candidate notes may move
    # within the historical <=4-semitone free zone while remaining the same
    # local family center.
    family_rows=[]
    for ms,records,_ in aperture_rows:
        fmap=_family_records(records,contest)
        suppressed = next(
            (groups for required_ms, groups in borrowing_suppressions.items()
             if math.isclose(float(ms), float(required_ms), rel_tol=0.0, abs_tol=1e-9)),
            (),
        )
        if suppressed:
            fmap={g:r for g,r in fmap.items() if g not in suppressed}
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
        return VariableApertureDecision(
            original_choice, 'ambiguous',
            ('bidirectional_profile_no_majority_subharmonic_borrowing_certificate'
             if borrowing_pairs else 'bidirectional_profile_no_majority'),
            None, original_hz, original_hz, contest,
        )

    # When an existing seed is already in the winning family there is nothing
    # to change; record the profile agreement but preserve V18 exactly.
    if original_hz is not None and math.isfinite(original_hz) and original_hz>0:
        if _midi_group(original_hz)==winner_group:
            return VariableApertureDecision(
                original_choice, 'keep',
                ('bidirectional_profile_confirms_original_subharmonic_borrowing_certificate'
                 if borrowing_pairs else 'bidirectional_profile_confirms_original'),
                max(reached), original_hz, original_hz, contest,
            )

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
    return VariableApertureDecision(
        choice, 'replace',
        ('bidirectional_profile_majority_subharmonic_borrowing_certificate'
         if borrowing_pairs else 'bidirectional_profile_majority'),
        max(reached), original_hz, winner.frequency_hz, contest,
    )
