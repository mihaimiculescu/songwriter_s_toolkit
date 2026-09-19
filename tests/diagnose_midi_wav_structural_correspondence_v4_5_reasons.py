#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, math, sys
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# V1 is deliberately reused ONLY for independent alignment.
from diagnose_midi_wav_event_correspondence import (
    Config as V1Config,
    load_eckf_csv,
    load_midi_notes,
    extract_wav_events,
    estimate_global_offset,
    section_offsets,
)

from python_eckf.expressive_smoothing import (
    ExpressiveSmoothingConfig,
    smooth_expressive_pitch,
)
from python_eckf.gesture_features import (
    GestureFeatureConfig,
    extract_gesture_features,
)
from python_eckf.gesture_objects import (
    GestureObjectConfig,
    construct_gesture_objects,
)
from python_eckf.gesture_object_splitter import (
    GestureObjectSplitterConfig,
    analyse_gesture_object_splits,
)
from python_eckf.gesture_structural_splitter import (
    StructuralSplitterConfig,
    apply_structural_splitting,
)
from python_eckf.trajectory_interpreter import (
    RegionKind,
    interpret_pitch_trajectory,
)


TOLS = (50.0, 100.0, 150.0, 250.0)
REFERENCE_BPM = 120.0
TIMING_ALPHA = 0.20

VICINITY_EYEBROW_MULT = 2.0
MIN_VICINITY_MS = 350.0
MAX_VICINITY_MS = 900.0

# Diagnostic matching tolerances only. They do NOT affect the frozen pipeline.
PITCH_EXACT_ST = 1.25
PITCH_LOOSE_ST = 2.25
INTERVAL_EXACT_ST = 1.50
INTERVAL_LOOSE_ST = 3.00


def timing_eyebrow_threshold_ms(
    bpm,
    reference_bpm=REFERENCE_BPM,
    alpha=TIMING_ALPHA,
):
    bpm = float(bpm)
    if not np.isfinite(bpm) or bpm <= 0:
        return float("nan")
    return (
        (30000.0 / bpm)
        * (bpm / reference_bpm) ** alpha
    )


def timing_flag(abs_delta_ms, threshold_ms):
    if (
        not np.isfinite(abs_delta_ms)
        or not np.isfinite(threshold_ms)
    ):
        return "-"
    ratio = abs_delta_ms / threshold_ms
    if ratio < 1.0:
        return "ordinary"
    if ratio < 2.0:
        return "EYEBROW"
    if ratio < 4.0:
        return "LARGE"
    return "EXTREME"


class MidiMap:
    def __init__(self, path):
        mid = mido.MidiFile(path)
        self.tpb = mid.ticks_per_beat

        msgs = []
        serial = 0
        for tr in mid.tracks:
            tick = 0
            for msg in tr:
                tick += msg.time
                msgs.append((tick, serial, msg))
                serial += 1

        # Defaults are inserted first. Stable tick-only collapse below lets a
        # real tick-0 event encountered later replace the default.
        tempos = [(0, 500000)]
        meters = [(0, 4, 4)]

        for tick, _, msg in msgs:
            if msg.type == "set_tempo":
                tempos.append((tick, int(msg.tempo)))
            elif msg.type == "time_signature":
                meters.append(
                    (
                        tick,
                        int(msg.numerator),
                        int(msg.denominator),
                    )
                )

        self.tempos = self._collapse(tempos)
        self.meters = self._collapse(meters)

    @staticmethod
    def _collapse(xs):
        out = []
        # IMPORTANT: stable sort by tick only. Do not tuple-sort same-tick
        # values, otherwise the synthetic default can incorrectly win.
        for x in sorted(xs, key=lambda item: item[0]):
            if out and out[-1][0] == x[0]:
                out[-1] = x
            else:
                out.append(x)
        return out

    def sec2tick(self, sec):
        elapsed = 0.0
        prev = 0.0
        tempo = self.tempos[0][1]

        for tick, newtempo in self.tempos[1:]:
            span = mido.tick2second(
                tick - prev,
                self.tpb,
                tempo,
            )
            if elapsed + span >= sec:
                return (
                    prev
                    + (sec - elapsed)
                    / (tempo / 1e6)
                    * self.tpb
                )
            elapsed += span
            prev = tick
            tempo = newtempo

        return (
            prev
            + (sec - elapsed)
            / (tempo / 1e6)
            * self.tpb
        )

    def bpm_at(self, sec):
        tick = self.sec2tick(sec)
        tempo = self.tempos[0][1]
        for t, newtempo in self.tempos[1:]:
            if t > tick:
                break
            tempo = newtempo
        return 60_000_000.0 / tempo

    def timing_metrics(self, delta_ms, sec):
        bpm = self.bpm_at(sec)
        beat_ms = 60000.0 / bpm
        eighth_ms = beat_ms / 2.0
        eyebrow = timing_eyebrow_threshold_ms(bpm)
        return (
            bpm,
            delta_ms / beat_ms,
            delta_ms / eighth_ms,
            eyebrow,
            timing_flag(abs(delta_ms), eyebrow),
        )

    def pos(self, sec):
        tick = self.sec2tick(sec)
        bar = 1
        section = 0.0
        active = self.meters[0]

        for i, m in enumerate(self.meters):
            if m[0] > tick:
                break
            if i:
                prevm = self.meters[i - 1]
                bt = self.tpb * 4 / prevm[2]
                bart = bt * prevm[1]
                span = m[0] - section
                whole = int(
                    math.floor(span / bart + 1e-9)
                )
                bar += whole
                if span - whole * bart > 1e-6:
                    bar += 1
                section = float(m[0])
            active = m

        bt = self.tpb * 4 / active[2]
        bart = bt * active[1]
        local = max(0.0, tick - section)
        bars = int(
            math.floor(local / bart + 1e-12)
        )
        bar += bars
        wb = local - bars * bart
        beat = int(
            math.floor(wb / bt + 1e-12)
        ) + 1
        within = wb - (beat - 1) * bt
        six = self.tpb / 4.0
        sf = within / six
        slot = int(
            math.floor(sf + 1e-12)
        ) + 1
        frac = sf - (slot - 1)

        if abs(frac) < 0.01:
            return f"{bar}|{beat}|{slot}"
        return (
            f"{bar}|{beat}|{slot}"
            f"+{frac:.2f}"
        )


@dataclass(frozen=True)
class Anchor:
    obj: int
    link: int
    decision: str
    region: int
    start: float
    mid: float
    end: float
    pitch: float
    dur: float
    connectors: str
    boundaries: str
    nc: int
    nb: int


def flags(e):
    c = []
    b = []

    if e.excursion_connector:
        c.append("EXCURSION")
    if e.direct_through_connector:
        c.append("DIRECT")
    if e.transition_similarity_connector:
        c.append("SIMILAR")

    if e.duration_discontinuity:
        b.append("DURATION")
    if e.path_discontinuity:
        b.append("PATH")
    if e.center_shape_discontinuity:
        b.append("CENTER/SHAPE")
    if e.residual_discontinuity:
        b.append("RESIDUAL")
    if e.topology_discontinuity:
        b.append("TOPOLOGY")

    return (
        ",".join(c) or "-",
        ",".join(b) or "-",
    )


def frozen(data):
    hz = 1.0 / float(
        np.median(np.diff(data.time_s))
    )

    interp = interpret_pitch_trajectory(
        clean_f0_hz=data.clean_f0_hz,
        valid=data.corrected_valid,
        analysis_hz=hz,
    )

    sm = smooth_expressive_pitch(
        pitch_st=interp.pitch_st,
        valid=data.corrected_valid,
        analysis_hz=hz,
        config=ExpressiveSmoothingConfig(
            analysis_hz=hz,
            window_ms=50.0,
            polyorder=2,
            center_window_ms=250.0,
        ),
    )

    feat = extract_gesture_features(
        interp,
        sm,
        GestureFeatureConfig(),
    )

    objs = construct_gesture_objects(
        interp,
        feat,
        GestureObjectConfig(
            short_target_max_ms=250.0,
            max_chain_transitions=8,
        ),
    )

    ev = analyse_gesture_object_splits(
        objs,
        feat,
        GestureObjectSplitterConfig(
            near_flat_st=0.05
        ),
    )

    st = apply_structural_splitting(
        ev,
        StructuralSplitterConfig(),
    )

    rmap = {
        int(r.index): r
        for r in interp.regions
    }

    anchors = []

    for o in st.objects:
        for d in o.link_decisions:
            r = rmap[
                int(d.stable_region_index)
            ]

            if (
                r.kind
                is not RegionKind.STABLE_TARGET
            ):
                raise RuntimeError(
                    f"S{r.index} is not STABLE_TARGET"
                )

            c, b = flags(d.evidence)
            a = float(r.start_time_s)
            z = float(r.end_time_s)

            anchors.append(
                Anchor(
                    int(d.object_index),
                    int(d.link_index),
                    d.decision.name,
                    int(r.index),
                    a,
                    (a + z) / 2.0,
                    z,
                    float(r.median_pitch_st),
                    float(r.duration_ms),
                    c,
                    b,
                    int(
                        d.evidence
                        .connector_family_count
                    ),
                    int(
                        d.evidence
                        .boundary_family_count
                    ),
                )
            )

    if len(anchors) != st.total_links:
        raise RuntimeError(
            "structural-link coverage failed"
        )

    return interp, st, tuple(anchors), hz


def note_name(n):
    names = (
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B",
    )
    return f"{names[n % 12]}{n // 12 - 1}"


def bins(label, vals):
    a = np.asarray(
        [
            x
            for x in vals
            if np.isfinite(x)
        ]
    )

    if not len(a):
        print(f"{label:<30} n=0")
        return

    print(
        f"{label:<30} "
        f"n={len(a):4d} "
        f"median={np.median(a):7.1f}ms "
        f"p90={np.percentile(a,90):7.1f}ms"
    )

    for t in TOLS:
        n = int(np.sum(a <= t))
        print(
            f"    within {t:3.0f} ms: "
            f"{n:4d}/{len(a):4d} "
            f"({100*n/len(a):5.1f}%)"
        )


def sgn(x, eps=0.35):
    if not np.isfinite(x):
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def finite_or_nan(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isfinite(value):
        return value
    return float("nan")


def harmonic_adjusted_error(
    observed_pitch,
    expected_pitch,
):
    raw = abs(
        float(observed_pitch)
        - float(expected_pitch)
    )
    best = raw
    label = "-"
    d = (
        float(observed_pitch)
        - float(expected_pitch)
    )

    for target, lab in (
        (12.0, "2x"),
        (-12.0, "1/2x"),
        (19.01955, "3x"),
        (-19.01955, "1/3x"),
    ):
        e = abs(d - target)
        if e < best:
            best = e
            label = lab

    return best, label


# =============================================================================
# V4.2 — THE FROZEN TRANSITION REGION IS THE STRUCTURAL CHANGE CANDIDATE
# =============================================================================

@dataclass(frozen=True)
class WavTransition:
    index: int
    region_index: int

    start_s: float
    mid_s: float
    end_s: float
    duration_ms: float

    source_pitch: float
    destination_pitch: float

    has_source: bool
    has_destination: bool
    context_kind: str

    interval_st: float
    direction: int


@dataclass(frozen=True)
class TransitionCandidate:
    transition: WavTransition

    # Distance from aligned GT boundary to the transition interval.
    # 0 means GT falls inside [start,end].
    interval_delta_ms: float

    # Descriptive distances to the frozen region's three time references.
    start_delta_ms: float
    mid_delta_ms: float
    end_delta_ms: float

    source_err: float
    dest_err: float
    interval_err: float
    direction_match: bool

    harmonic_source: str
    harmonic_dest: str

    pitch_score: float
    correspondence_score: float

    credible: bool
    reason: str


@dataclass(frozen=True)
class GTMatch:
    gt_i: int
    status: str
    candidate: TransitionCandidate | None
    runner_up: TransitionCandidate | None

    vicinity_ms: float
    eyebrow_ms: float

    candidates_seen: int
    credible_candidates: int


def transition_candidates(interp):
    """
    Extract exactly the frozen TRANSITION regions.

    No new segmentation is performed.

    Critically:
      - previous_target_st / next_target_st are consumed exactly as frozen by
        trajectory_interpreter.py;
      - missing target context is NOT reconstructed from neighboring regions;
      - STABLE_TARGET and UNRESOLVED regions are not promoted to pitch changes;
      - region start/end are preserved as the transition's temporal support.

    TWO_SIDED:
        both previous_target_st and next_target_st are finite.

    SOURCE_ONLY / DESTINATION_ONLY:
        useful observational context, but insufficient by themselves for an
        A->B GT pitch-geometry match.

    UNANCHORED:
        frozen transition with neither target context.
    """
    out = []

    for r in interp.regions:
        if (
            r.kind
            is not RegionKind.TRANSITION
        ):
            continue

        src = finite_or_nan(
            getattr(
                r,
                "previous_target_st",
                np.nan,
            )
        )
        dst = finite_or_nan(
            getattr(
                r,
                "next_target_st",
                np.nan,
            )
        )

        hs = bool(np.isfinite(src))
        hd = bool(np.isfinite(dst))

        if hs and hd:
            context_kind = "TWO_SIDED"
            interval = float(dst - src)
            direction = sgn(interval)
        elif hs:
            context_kind = "SOURCE_ONLY"
            interval = float("nan")
            direction = 0
        elif hd:
            context_kind = "DESTINATION_ONLY"
            interval = float("nan")
            direction = 0
        else:
            context_kind = "UNANCHORED"
            interval = float("nan")
            direction = 0

        start = float(r.start_time_s)
        end = float(r.end_time_s)

        out.append(
            WavTransition(
                index=len(out) + 1,
                region_index=int(r.index),
                start_s=start,
                mid_s=(start + end) / 2.0,
                end_s=end,
                duration_ms=float(
                    getattr(
                        r,
                        "duration_ms",
                        max(
                            0.0,
                            1000.0 * (end - start),
                        ),
                    )
                ),
                source_pitch=src,
                destination_pitch=dst,
                has_source=hs,
                has_destination=hd,
                context_kind=context_kind,
                interval_st=interval,
                direction=direction,
            )
        )

    return tuple(out)


def transition_interval_delta_ms(
    gt_time,
    transition,
):
    """
    Signed distance from GT time to transition temporal support.

    Negative:
        transition lies before GT.

    Zero:
        GT lies inside the transition.

    Positive:
        transition lies after GT.
    """
    t = float(gt_time)

    if t < transition.start_s:
        return 1000.0 * (
            transition.start_s - t
        )

    if t > transition.end_s:
        return 1000.0 * (
            transition.end_s - t
        )

    return 0.0


def vicinity_ms(mm, midi_sec):
    bpm = mm.bpm_at(midi_sec)
    eyebrow = timing_eyebrow_threshold_ms(
        bpm
    )
    v = (
        VICINITY_EYEBROW_MULT
        * eyebrow
    )

    return (
        float(
            np.clip(
                v,
                MIN_VICINITY_MS,
                MAX_VICINITY_MS,
            )
        ),
        float(eyebrow),
    )


def candidate_for_gt(
    transition,
    gt_time,
    prev_pitch,
    cur_pitch,
    vicinity,
):
    """
    Only TWO_SIDED frozen transitions can establish A->B correspondence.

    One-sided/unanchored transitions remain in the structural inventory and
    reverse-coverage report; their missing endpoint is never manufactured.
    """
    interval_dms = (
        transition_interval_delta_ms(
            gt_time,
            transition,
        )
    )

    if abs(interval_dms) > vicinity:
        return None

    # Retrieval sees all transitions in temporal vicinity. Geometry matching
    # requires the interpreter to have supplied both endpoint targets.
    if (
        not transition.has_source
        or not transition.has_destination
    ):
        return TransitionCandidate(
            transition=transition,
            interval_delta_ms=float(
                interval_dms
            ),
            start_delta_ms=1000.0 * (
                transition.start_s - gt_time
            ),
            mid_delta_ms=1000.0 * (
                transition.mid_s - gt_time
            ),
            end_delta_ms=1000.0 * (
                transition.end_s - gt_time
            ),
            source_err=float("nan"),
            dest_err=float("nan"),
            interval_err=float("nan"),
            direction_match=False,
            harmonic_source="-",
            harmonic_dest="-",
            pitch_score=float("inf"),
            correspondence_score=float(
                "inf"
            ),
            credible=False,
            reason=(
                "INSUFFICIENT_FROZEN_TARGET_CONTEXT_"
                + transition.context_kind
            ),
        )

    src_err, src_h = (
        harmonic_adjusted_error(
            transition.source_pitch,
            prev_pitch,
        )
    )
    dst_err, dst_h = (
        harmonic_adjusted_error(
            transition.destination_pitch,
            cur_pitch,
        )
    )

    gt_iv = float(
        cur_pitch - prev_pitch
    )

    iv_err = abs(
        abs(transition.interval_st)
        - abs(gt_iv)
    )

    gt_dir = sgn(gt_iv)

    dir_match = (
        gt_dir == transition.direction
        if gt_dir != 0
        else True
    )

    pitch_score = (
        0.45
        * np.clip(
            src_err / PITCH_LOOSE_ST,
            0,
            2,
        )
        + 0.45
        * np.clip(
            dst_err / PITCH_LOOSE_ST,
            0,
            2,
        )
        + 0.10
        * np.clip(
            iv_err / INTERVAL_LOOSE_ST,
            0,
            2,
        )
    )

    direction_penalty = (
        0.0
        if dir_match
        else 0.55
    )

    # Timing remains only a weak tie-breaker inside the retrieval vicinity.
    timing_tiebreak = (
        0.10
        * min(
            1.0,
            abs(interval_dms)
            / max(vicinity, 1.0),
        )
    )

    score = float(
        pitch_score
        + direction_penalty
        + timing_tiebreak
    )

    exact = bool(
        src_err <= PITCH_EXACT_ST
        and dst_err <= PITCH_EXACT_ST
        and iv_err <= INTERVAL_EXACT_ST
        and dir_match
    )

    loose = bool(
        src_err <= PITCH_LOOSE_ST
        and dst_err <= PITCH_LOOSE_ST
        and iv_err <= INTERVAL_LOOSE_ST
        and dir_match
    )

    harmonic_pair = bool(
        src_h != "-"
        and src_h == dst_h
        and src_err <= PITCH_EXACT_ST
        and dst_err <= PITCH_EXACT_ST
        and iv_err <= INTERVAL_LOOSE_ST
        and dir_match
    )

    credible = bool(
        exact
        or loose
        or harmonic_pair
    )

    reason = (
        "EXACT_GEOMETRY"
        if exact
        else (
            "LOOSE_GEOMETRY"
            if loose
            else (
                f"HARMONIC_PAIR_{src_h}"
                if harmonic_pair
                else "GEOMETRY_MISMATCH"
            )
        )
    )

    return TransitionCandidate(
        transition=transition,
        interval_delta_ms=float(
            interval_dms
        ),
        start_delta_ms=1000.0 * (
            transition.start_s - gt_time
        ),
        mid_delta_ms=1000.0 * (
            transition.mid_s - gt_time
        ),
        end_delta_ms=1000.0 * (
            transition.end_s - gt_time
        ),
        source_err=float(src_err),
        dest_err=float(dst_err),
        interval_err=float(iv_err),
        direction_match=bool(dir_match),
        harmonic_source=src_h,
        harmonic_dest=dst_h,
        pitch_score=float(pitch_score),
        correspondence_score=score,
        credible=credible,
        reason=reason,
    )


def match_gt_change(
    i,
    midi,
    gt,
    mm,
    transitions,
):
    prev = midi.notes[i - 1]
    cur = midi.notes[i]

    gt_time = float(gt[i])

    vic, eyebrow = vicinity_ms(
        mm,
        cur.onset_s,
    )

    candidates = []

    for t in transitions:
        c = candidate_for_gt(
            t,
            gt_time,
            prev.pitch,
            cur.pitch,
            vic,
        )

        if c is not None:
            candidates.append(c)

    candidates.sort(
        key=lambda c: (
            not c.credible,
            c.correspondence_score,
            abs(c.interval_delta_ms),
            abs(c.mid_delta_ms),
            c.transition.index,
        )
    )

    credible = [
        c
        for c in candidates
        if c.credible
    ]

    if not credible:
        return GTMatch(
            i,
            "NO_CORRESPONDENCE",
            None,
            (
                candidates[0]
                if candidates
                else None
            ),
            vic,
            eyebrow,
            len(candidates),
            0,
        )

    best = credible[0]

    runner = (
        credible[1]
        if len(credible) > 1
        else None
    )

    ambiguous = False

    if runner is not None:
        score_gap = (
            runner.correspondence_score
            - best.correspondence_score
        )
        pitch_gap = (
            runner.pitch_score
            - best.pitch_score
        )

        if (
            score_gap <= 0.18
            and pitch_gap <= 0.15
        ):
            ambiguous = True

    return GTMatch(
        i,
        (
            "AMBIGUOUS"
            if ambiguous
            else "MATCHED"
        ),
        best,
        runner,
        vic,
        eyebrow,
        len(candidates),
        len(credible),
    )



# =============================================================================
# V4.4 — OBSERVATIONAL ONE-SIDED RAW-CONTEXT PROBE
# =============================================================================
# This probe NEVER creates a target, changes a PitchRegion, or upgrades a V4.2
# correspondence.  It asks only whether corrected-valid F0 immediately beyond
# the missing side of a SOURCE_ONLY / DESTINATION_ONLY frozen TRANSITION
# contains local evidence compatible with the missing GT endpoint.
#
# The 30 ms witness run is diagnostic evidence only.  It is deliberately much
# shorter than a stable-target declaration and MUST NOT be fed back into the
# interpreter.
RAW_WITNESS_MIN_MS = 30.0


@dataclass(frozen=True)
class RawContextProbe:
    gt_i: int
    transition_index: int
    region_index: int
    context_kind: str
    side: str
    expected_pitch: float
    known_pitch: float
    known_error: float
    known_harmonic: str
    window_start_s: float
    window_end_s: float
    valid_rows: int
    best_error: float
    best_harmonic: str
    best_time_s: float
    exact_rows: int
    loose_rows: int
    longest_loose_ms: float
    raw_support: bool
    # V4.4 classification-only direct-pitch metrics.  The existing V4.3
    # harmonic-aware metrics above are preserved unchanged.
    direct_best_error: float = float("inf")
    direct_exact_rows: int = 0
    direct_loose_rows: int = 0
    direct_longest_loose_ms: float = 0.0


def _pitch_st_from_hz(hz):
    hz = np.asarray(hz, dtype=float)
    out = np.full(hz.shape, np.nan, dtype=float)
    ok = np.isfinite(hz) & (hz > 0.0)
    out[ok] = 69.0 + 12.0 * np.log2(hz[ok] / 440.0)
    return out


def _edge_connected_indices(data, edge_s, direction, lo_s, hi_s):
    """Return corrected-valid rows connected to the transition edge.

    We walk away from the frozen transition and stop at the first real
    corrected-valid wall.  Thus the probe cannot jump across rejected audio.
    """
    times = np.asarray(data.time_s, dtype=float)
    valid = np.asarray(data.corrected_valid, dtype=bool)
    if not len(times):
        return np.asarray([], dtype=int)

    edge = int(np.argmin(np.abs(times - float(edge_s))))
    step = 1 if direction > 0 else -1
    out = []
    i = edge

    # The frozen transition edge itself should normally be valid.  If rounding
    # lands one row outside it, permit a one-row move toward the probe side.
    if not valid[i]:
        j = i + step
        if j < 0 or j >= len(times) or not valid[j]:
            return np.asarray([], dtype=int)
        i = j

    while 0 <= i < len(times) and valid[i]:
        t = times[i]
        if lo_s <= t <= hi_s:
            out.append(i)
        if direction > 0 and t > hi_s:
            break
        if direction < 0 and t < lo_s:
            break
        i += step

    if direction < 0:
        out.reverse()
    return np.asarray(out, dtype=int)


def _longest_true_run_ms(mask, analysis_hz):
    best = cur = 0
    for x in np.asarray(mask, dtype=bool):
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    if best == 0 or not np.isfinite(analysis_hz) or analysis_hz <= 0:
        return 0.0
    return 1000.0 * best / float(analysis_hz)


def probe_one_sided_candidate(m, c, midi, gt, data, analysis_hz):
    """Observe the missing side of one one-sided frozen transition.

    The known frozen endpoint must first be compatible with the corresponding
    GT endpoint under the SAME V4.2 diagnostic pitch tolerance.  Only then do
    we inspect raw corrected-valid F0 on the missing side.
    """
    t = c.transition
    prev = midi.notes[m.gt_i - 1]
    cur = midi.notes[m.gt_i]
    gt_time = float(gt[m.gt_i])
    vic_s = float(m.vicinity_ms) / 1000.0

    if t.context_kind == "SOURCE_ONLY":
        known_pitch = t.source_pitch
        known_expected = float(prev.pitch)
        expected = float(cur.pitch)
        side = "AFTER"
        direction = +1
        edge_s = t.end_s
        lo_s = max(t.end_s, gt_time - vic_s)
        hi_s = gt_time + vic_s
    elif t.context_kind == "DESTINATION_ONLY":
        known_pitch = t.destination_pitch
        known_expected = float(cur.pitch)
        expected = float(prev.pitch)
        side = "BEFORE"
        direction = -1
        edge_s = t.start_s
        lo_s = gt_time - vic_s
        hi_s = min(t.start_s, gt_time + vic_s)
    else:
        return None

    known_err, known_h = harmonic_adjusted_error(known_pitch, known_expected)
    if known_err > PITCH_LOOSE_ST:
        return None
    if hi_s < lo_s:
        return None

    idx = _edge_connected_indices(data, edge_s, direction, lo_s, hi_s)
    if not len(idx):
        return RawContextProbe(
            m.gt_i, t.index, t.region_index, t.context_kind, side, expected,
            known_pitch, float(known_err), known_h, lo_s, hi_s, 0,
            float("inf"), "-", float("nan"), 0, 0, 0.0, False
        )

    pitches = _pitch_st_from_hz(np.asarray(data.clean_f0_hz, dtype=float)[idx])
    times = np.asarray(data.time_s, dtype=float)[idx]
    errs = np.full(len(idx), np.inf, dtype=float)
    harms = ["-"] * len(idx)
    for k, pitch in enumerate(pitches):
        if np.isfinite(pitch):
            errs[k], harms[k] = harmonic_adjusted_error(pitch, expected)

    finite = np.isfinite(errs)
    if not np.any(finite):
        return RawContextProbe(
            m.gt_i, t.index, t.region_index, t.context_kind, side, expected,
            known_pitch, float(known_err), known_h, lo_s, hi_s, len(idx),
            float("inf"), "-", float("nan"), 0, 0, 0.0, False
        )

    best_k = int(np.argmin(errs))
    exact = errs <= PITCH_EXACT_ST
    loose = errs <= PITCH_LOOSE_ST
    longest = _longest_true_run_ms(loose, analysis_hz)
    support = bool(longest + 1e-9 >= RAW_WITNESS_MIN_MS)

    # V4.4 observes direct fundamental agreement separately from the existing
    # harmonic-aware witness.  This is classification only; it does not alter
    # raw_support or candidate selection.
    direct_errs = np.abs(pitches - expected)
    direct_finite = np.isfinite(direct_errs)
    direct_best = (float(np.min(direct_errs[direct_finite]))
                   if np.any(direct_finite) else float("inf"))
    direct_exact = direct_errs <= PITCH_EXACT_ST
    direct_loose = direct_errs <= PITCH_LOOSE_ST
    direct_longest = _longest_true_run_ms(direct_loose, analysis_hz)

    return RawContextProbe(
        gt_i=m.gt_i, transition_index=t.index, region_index=t.region_index,
        context_kind=t.context_kind, side=side, expected_pitch=expected,
        known_pitch=known_pitch, known_error=float(known_err),
        known_harmonic=known_h, window_start_s=float(lo_s),
        window_end_s=float(hi_s), valid_rows=int(len(idx)),
        best_error=float(errs[best_k]), best_harmonic=harms[best_k],
        best_time_s=float(times[best_k]), exact_rows=int(np.sum(exact)),
        loose_rows=int(np.sum(loose)), longest_loose_ms=float(longest),
        raw_support=support, direct_best_error=direct_best,
        direct_exact_rows=int(np.sum(direct_exact)),
        direct_loose_rows=int(np.sum(direct_loose)),
        direct_longest_loose_ms=float(direct_longest),
    )


def classify_one_sided_probe(p):
    """V4.4 observational classification; never changes correspondence."""
    if p.valid_rows <= 0:
        return "NO_VALID_F0_ON_MISSING_SIDE"

    direct_support = (
        p.direct_longest_loose_ms + 1e-9 >= RAW_WITNESS_MIN_MS
    )
    harmonic_aware_support = (
        p.longest_loose_ms + 1e-9 >= RAW_WITNESS_MIN_MS
    )

    if direct_support:
        return "DIRECT_RAW_SUPPORT"
    if harmonic_aware_support:
        return "HARMONIC_RAW_SUPPORT"
    if p.loose_rows > 0:
        return "VALID_F0_NEAR_EXPECTED_BUT_INSUFFICIENT_DURATION"
    # A valid trajectory exists, but neither direct nor harmonic-adjusted pitch
    # enters the existing loose diagnostic tolerance.
    return "VALID_F0_WRONG_PITCH"



def one_sided_probe_for_match(m, midi, gt, data, analysis_hz, transitions):
    """"Choose the strongest observational one-sided probe in vicinity."""
    if m.status != "NO_CORRESPONDENCE":
        return None
    gt_time = float(gt[m.gt_i])
    probes = []
    for t in transitions:
        if t.context_kind not in ("SOURCE_ONLY", "DESTINATION_ONLY"):
            continue
        dms = transition_interval_delta_ms(gt_time, t)
        if abs(dms) > m.vicinity_ms:
            continue
        # Reuse a lightweight candidate shell only for the transition reference.
        c = candidate_for_gt(
            t, gt_time, midi.notes[m.gt_i - 1].pitch,
            midi.notes[m.gt_i].pitch, m.vicinity_ms
        )
        if c is None:
            continue
        p = probe_one_sided_candidate(m, c, midi, gt, data, analysis_hz)
        if p is not None:
            probes.append((p, abs(dms)))
    if not probes:
        return None
    probes.sort(key=lambda x: (not x[0].raw_support, x[0].best_error,
                              -x[0].longest_loose_ms, x[1],
                              x[0].transition_index))
    return probes[0][0]


# =============================================================================
# V4.5 — RAW-ECKF AUDIT BENEATH V4.4 VALIDITY-WALL CASES
# =============================================================================
RAW_AUDIT_MIN_HZ = 1.0
RAW_AUDIT_VOCAL_FLOOR_HZ = 120.0  # observational marker only
RAW_AUDIT_MAX_HZ = 1500.0         # observational marker only
RAW_AUDIT_MAX_STEP_ST = 4.0       # observational marker only

@dataclass(frozen=True)
class RawRejectedAudit:
    gt_i:int; transition_index:int; region_index:int; context_kind:str; side:str
    expected_pitch:float; window_start_s:float; window_end_s:float
    rows:int; raw_positive_rows:int; raw_in_range_rows:int
    direct_best_error:float; direct_loose_rows:int; direct_longest_loose_ms:float
    harmonic_best_error:float; harmonic_best_relation:str; harmonic_loose_rows:int; harmonic_longest_loose_ms:float
    adjacent_pairs:int; continuous_pairs:int; continuity_fraction:float
    median_step_st:float; p90_step_st:float; max_step_st:float; median_pitch_st:float; pitch_span_st:float
    raw_below_floor_rows:int; raw_above_ceiling_rows:int; raw_nonpositive_rows:int
    first_pass_reason_counts:tuple; validity_reason_counts:tuple; correction_reason_counts:tuple
    classification:str

def load_reason_audit(path, data):
    """Read CSV reason fields separately; never modify the frozen V1 loader."""
    fields = ('first_pass_reason', 'validity_reason', 'correction_reason')
    reasons = {key: [] for key in fields}
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = set(fields) | {'time_s', 'f0_hz', 'corrected_valid'}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'Reason audit: missing CSV columns: {sorted(missing)}')
        for i, row in enumerate(reader):
            if i >= len(data.time_s):
                raise ValueError(f'Reason audit: extra CSV row {i}')
            t = float(row['time_s'])
            if not np.isclose(t, data.time_s[i], rtol=0, atol=1e-8):
                raise ValueError(f'Reason audit: time mismatch at row {i}: {t} vs {data.time_s[i]}')
            f0 = float(row['f0_hz'])
            if not np.isclose(f0, data.raw_f0_hz[i], rtol=1e-9, atol=1e-7, equal_nan=True):
                raise ValueError(f'Reason audit: raw F0 mismatch at row {i}')
            valid = row['corrected_valid'].strip().lower() in ('1', 'true', 'yes', 'y', 't')
            if valid != bool(data.corrected_valid[i]):
                raise ValueError(f'Reason audit: validity mismatch at row {i}')
            for key in fields:
                reasons[key].append(row[key] if row[key] is not None else '<NULL>')
    if len(reasons[fields[0]]) != len(data.time_s):
        raise ValueError(f'Reason audit: {len(reasons[fields[0]])} rows vs {len(data.time_s)} ECKF rows')
    return SimpleNamespace(**{**vars(data), **{k: np.asarray(v, dtype=object) for k, v in reasons.items()}})

def _reason_counts(data, field, idx):
    if not hasattr(data, field): raise ValueError(f"Reason audit field unavailable: {field}")
    arr=np.asarray(getattr(data,field),dtype=object)
    from collections import Counter
    vals=[str(arr[i]).strip() or "<BLANK>" for i in idx]
    c=Counter(vals)
    return tuple(sorted(c.items(),key=lambda kv:(-kv[1],kv[0])))

def _format_reason_counts(items, limit=4):
    if not items: return '-'
    text=','.join(f'{k}:{v}' for k,v in items[:limit])
    if len(items)>limit: text += f',+{len(items)-limit}more'
    return text

def _raw_window_indices(data,p):
    times=np.asarray(data.time_s,dtype=float)
    return np.flatnonzero((times>=p.window_start_s-1e-12)&(times<=p.window_end_s+1e-12))

def audit_rejected_raw_f0(m,p,data,analysis_hz):
    idx=_raw_window_indices(data,p)
    raw_hz=np.asarray(data.raw_f0_hz,dtype=float)[idx] if len(idx) else np.asarray([],dtype=float)
    raw_st=_pitch_st_from_hz(raw_hz)
    positive=np.isfinite(raw_hz)&(raw_hz>=RAW_AUDIT_MIN_HZ)
    in_range=positive&(raw_hz>=RAW_AUDIT_VOCAL_FLOOR_HZ)&(raw_hz<=RAW_AUDIT_MAX_HZ)
    below=positive&(raw_hz<RAW_AUDIT_VOCAL_FLOOR_HZ); above=positive&(raw_hz>RAW_AUDIT_MAX_HZ)
    direct=np.full(len(raw_st),np.inf); direct[np.isfinite(raw_st)]=np.abs(raw_st[np.isfinite(raw_st)]-p.expected_pitch)
    dloose=direct<=PITCH_LOOSE_ST; dbest=float(np.min(direct)) if len(direct) else float('inf'); dlong=_longest_true_run_ms(dloose,analysis_hz)
    herr=np.full(len(raw_st),np.inf); hrel=['-']*len(raw_st)
    for k,pitch in enumerate(raw_st):
        if np.isfinite(pitch): herr[k],hrel[k]=harmonic_adjusted_error(pitch,p.expected_pitch)
    hloose=herr<=PITCH_LOOSE_ST; hk=int(np.argmin(herr)) if len(herr) else -1
    hbest=float(herr[hk]) if hk>=0 else float('inf'); hbrel=hrel[hk] if hk>=0 else '-'; hlong=_longest_true_run_ms(hloose,analysis_hz)
    steps=[]
    for k in range(max(0,len(raw_st)-1)):
        if positive[k] and positive[k+1]: steps.append(abs(float(raw_st[k+1]-raw_st[k])))
    pairs=len(steps); cont=sum(x<=RAW_AUDIT_MAX_STEP_ST for x in steps); frac=cont/pairs if pairs else 0.0
    finite=raw_st[np.isfinite(raw_st)]
    if not len(idx) or not np.any(positive): cls='RAW_NO_POSITIVE_F0'
    elif dlong+1e-9>=RAW_WITNESS_MIN_MS: cls='RAW_DIRECT_EXPECTED_SUPPORT_REJECTED'
    elif hlong+1e-9>=RAW_WITNESS_MIN_MS: cls='RAW_HARMONIC_EXPECTED_SUPPORT_REJECTED'
    elif np.sum(in_range)>=3 and pairs>=2 and frac>=0.80: cls='RAW_COHERENT_OTHER_PITCH_REJECTED'
    else: cls='RAW_TRACKING_FAILURE_OR_OUT_OF_RANGE'
    return RawRejectedAudit(m.gt_i,p.transition_index,p.region_index,p.context_kind,p.side,p.expected_pitch,p.window_start_s,p.window_end_s,len(idx),int(np.sum(positive)),int(np.sum(in_range)),dbest,int(np.sum(dloose)),dlong,hbest,hbrel,int(np.sum(hloose)),hlong,pairs,cont,frac,float(np.median(steps)) if steps else float('nan'),float(np.percentile(steps,90)) if steps else float('nan'),float(np.max(steps)) if steps else float('nan'),float(np.median(finite)) if len(finite) else float('nan'),float(np.ptp(finite)) if len(finite) else float('nan'),int(np.sum(below)),int(np.sum(above)),int(np.sum(~positive)),_reason_counts(data,'first_pass_reason',idx),_reason_counts(data,'validity_reason',idx),_reason_counts(data,'correction_reason',idx),cls)

def nearest_structural_link(
    transition,
    anchors,
):
    """
    V3.1 cross-reference only. It never participates in GT matching.
    """
    if not anchors:
        return None, float("nan")

    a = min(
        anchors,
        key=lambda x: abs(
            x.mid - transition.mid_s
        ),
    )

    return (
        a,
        1000.0
        * (a.mid - transition.mid_s),
    )


def print_timing(
    mm,
    delta_ms,
    midi_sec,
    indent="    ",
):
    bpm, beats, eighths, thr, flag = (
        mm.timing_metrics(
            delta_ms,
            midi_sec,
        )
    )

    print(
        f"{indent}timing-to-transition-support: "
        f"Δ={delta_ms:+7.1f}ms  "
        f"beats={beats:+.3f}  "
        f"eighths={eighths:+.3f}  "
        f"BPM={bpm:.2f}  "
        f"eyebrow={thr:.1f}ms  "
        f"flag={flag}"
    )


def print_transition(
    prefix,
    c,
):
    t = c.transition

    src = (
        f"{t.source_pitch:.2f}"
        if t.has_source
        else "--"
    )
    dst = (
        f"{t.destination_pitch:.2f}"
        if t.has_destination
        else "--"
    )

    print(
        f"{prefix}T{t.index} R{t.region_index} "
        f"{src}->{dst}st "
        f"context={t.context_kind} "
        f"support={t.start_s:.3f}"
        f"..{t.end_s:.3f}s "
        f"dur={t.duration_ms:.1f}ms"
    )


def print_match(
    mm,
    midi,
    gt,
    m,
    anchors,
    offset,
):
    i = m.gt_i
    prev = midi.notes[i - 1]
    cur = midi.notes[i]
    gt_time = float(gt[i])

    print(
        f"GT{cur.index:03d} "
        f"{note_name(prev.pitch)}"
        f"->{note_name(cur.pitch)} "
        f"({cur.pitch-prev.pitch:+d}st) "
        f"MIDI={cur.onset_s:8.3f}s "
        f"{mm.pos(cur.onset_s):<14s} "
        f"WAVref={gt_time:8.3f}s"
    )

    print(
        f"    status={m.status:<17s} "
        f"vicinity=±{m.vicinity_ms:.1f}ms  "
        f"eyebrow={m.eyebrow_ms:.1f}ms  "
        f"candidates={m.candidates_seen} "
        f"credible={m.credible_candidates}"
    )

    if m.candidate is None:
        if m.runner_up is not None:
            c = m.runner_up
            print_transition(
                "    best rejected ",
                c,
            )

            if np.isfinite(c.source_err):
                print(
                    "    geometry: "
                    f"src_err={c.source_err:.2f}st "
                    f"dst_err={c.dest_err:.2f}st "
                    f"interval_err={c.interval_err:.2f}st "
                    f"direction="
                    f"{'YES' if c.direction_match else 'NO'} "
                    f"harmonic="
                    f"{c.harmonic_source}/"
                    f"{c.harmonic_dest} "
                    f"reason={c.reason}"
                )
            else:
                print(
                    f"    reason={c.reason}"
                )

            print(
                "    temporal support deltas: "
                f"start={c.start_delta_ms:+.1f}ms "
                f"mid={c.mid_delta_ms:+.1f}ms "
                f"end={c.end_delta_ms:+.1f}ms "
                f"interval-distance="
                f"{c.interval_delta_ms:+.1f}ms"
            )
        return

    c = m.candidate
    t = c.transition

    print_transition(
        "    MATCH ",
        c,
    )

    print(
        "    geometry: "
        f"src_err={c.source_err:.2f}st "
        f"dst_err={c.dest_err:.2f}st "
        f"interval_err={c.interval_err:.2f}st "
        f"direction="
        f"{'YES' if c.direction_match else 'NO'} "
        f"reason={c.reason} "
        f"harmonic="
        f"{c.harmonic_source}/"
        f"{c.harmonic_dest}"
    )

    print(
        "    temporal support deltas: "
        f"start={c.start_delta_ms:+.1f}ms "
        f"mid={c.mid_delta_ms:+.1f}ms "
        f"end={c.end_delta_ms:+.1f}ms "
        f"interval-distance="
        f"{c.interval_delta_ms:+.1f}ms"
    )

    # Position of transition start/mid/end on the MIDI reference timeline.
    print(
        "    transition positions: "
        f"start={mm.pos(t.start_s-offset)}  "
        f"mid={mm.pos(t.mid_s-offset)}  "
        f"end={mm.pos(t.end_s-offset)}"
    )

    # Timing judgment uses distance to the frozen transition SUPPORT, not
    # distance to an invented midpoint onset.
    print_timing(
        mm,
        c.interval_delta_ms,
        cur.onset_s,
    )

    a, ad = nearest_structural_link(
        t,
        anchors,
    )

    if a is None:
        print(
            "    V3.1 cross-ref: none"
        )
    else:
        print(
            "    V3.1 cross-ref only: "
            f"{a.decision} "
            f"O{a.obj:02d} "
            f"L{a.link+1} "
            f"S{a.region} "
            "link-mid minus transition-mid="
            f"{ad:+.1f}ms"
        )

    if m.runner_up is not None:
        r = m.runner_up
        rt = r.transition

        print(
            f"    runner-up "
            f"T{rt.index} R{rt.region_index} "
            f"intervalΔ="
            f"{r.interval_delta_ms:+.1f}ms "
            f"midΔ={r.mid_delta_ms:+.1f}ms "
            f"score="
            f"{r.correspondence_score:.3f} "
            f"geometry="
            f"{r.source_err:.2f}/"
            f"{r.dest_err:.2f}/"
            f"{r.interval_err:.2f}st"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "eckf_csv",
        type=Path,
    )
    ap.add_argument(
        "midi",
        type=Path,
    )
    ap.add_argument(
        "--fixed-offset",
        type=float,
        default=None,
    )
    ap.add_argument(
        "--offset-min",
        type=float,
        default=-15.0,
    )
    ap.add_argument(
        "--offset-max",
        type=float,
        default=15.0,
    )

    args = ap.parse_args()

    midi = load_midi_notes(
        args.midi
    )
    mm = MidiMap(
        args.midi
    )
    data = load_eckf_csv(
        args.eckf_csv
    )
    data = load_reason_audit(args.eckf_csv, data)

    # Independent V1 alignment remains conceptually separate.
    cfg = V1Config(
        global_offset_min_s=args.offset_min,
        global_offset_max_s=args.offset_max,
    )

    primitive = extract_wav_events(
        data,
        cfg,
    )

    if args.fixed_offset is None:
        off, score = estimate_global_offset(
            midi,
            data,
            primitive,
            cfg,
        )
        source = "independent V1 estimate"
    else:
        off = float(
            args.fixed_offset
        )
        score = float("nan")
        source = (
            "FIXED independent V1 offset"
        )

    sections = section_offsets(
        midi,
        data,
        primitive,
        off,
    )

    # Frozen production stack: unchanged.
    interp, st, anchors, hz = frozen(
        data
    )

    gt = np.asarray(
        [
            n.onset_s + off
            for n in midi.notes
        ],
        dtype=float,
    )

    transitions = transition_candidates(
        interp
    )

    frozen_transition_count = sum(
        1
        for r in interp.regions
        if (
            r.kind
            is RegionKind.TRANSITION
        )
    )

    if (
        len(transitions)
        != frozen_transition_count
    ):
        raise RuntimeError(
            "V4.4 transition extraction "
            "does not equal frozen "
            "TRANSITION population"
        )

    if hasattr(
        st,
        "frozen_transition_count",
    ):
        if (
            int(st.frozen_transition_count)
            != len(transitions)
        ):
            raise RuntimeError(
                "V4.4 transition population "
                "does not match downstream "
                "frozen-transition invariant"
            )

    context_counts = {
        "TWO_SIDED": 0,
        "SOURCE_ONLY": 0,
        "DESTINATION_ONLY": 0,
        "UNANCHORED": 0,
    }

    for t in transitions:
        context_counts[
            t.context_kind
        ] += 1

    # GT same-pitch rearticulations remain explicitly separate because they
    # have no A->B pitch-change geometry.
    change_indices = [
        i
        for i in range(
            1,
            len(midi.notes),
        )
        if (
            midi.notes[i].pitch
            != midi.notes[i - 1].pitch
        )
    ]

    rearticulation_indices = [
        i
        for i in range(
            1,
            len(midi.notes),
        )
        if (
            midi.notes[i].pitch
            == midi.notes[i - 1].pitch
        )
    ]

    matches = [
        match_gt_change(
            i,
            midi,
            gt,
            mm,
            transitions,
        )
        for i in change_indices
    ]

    print("=" * 148)
    print(
        "MIDI <-> FROZEN WAV TRANSITION "
        "CORRESPONDENCE V4.4"
    )
    print("=" * 148)

    print(
        f"MIDI: {args.midi}"
    )
    print(
        f"ECKF: {args.eckf_csv}"
    )

    print(
        f"MIDI notes={len(midi.notes)} "
        f"pitch_changes={len(change_indices)} "
        f"same_pitch_rearticulations="
        f"{len(rearticulation_indices)} "
        f"rows={len(data.time_s)} "
        f"analysis={hz:.3f}Hz"
    )

    print(
        f"frozen regions={len(interp.regions)} "
        f"frozen TRANSITION regions="
        f"{len(transitions)}"
    )

    print(
        "transition target context: "
        f"TWO_SIDED="
        f"{context_counts['TWO_SIDED']} "
        f"SOURCE_ONLY="
        f"{context_counts['SOURCE_ONLY']} "
        f"DESTINATION_ONLY="
        f"{context_counts['DESTINATION_ONLY']} "
        f"UNANCHORED="
        f"{context_counts['UNANCHORED']}"
    )

    print(
        f"V3.1 objects={st.total_objects} "
        f"links={st.total_links} "
        f"JOIN={st.join_links} "
        f"SPLIT={st.split_links} "
        f"UNRESOLVED="
        f"{st.unresolved_links}"
    )

    print(
        "alignment="
        f"{source}; "
        "WAV_time=MIDI_time+offset; "
        f"offset={off:+.3f}s",
        end="",
    )

    if np.isfinite(score):
        print(
            f" score={score:.4f}"
        )
    else:
        print()

    for k in (
        "EARLY",
        "MIDDLE",
        "LATE",
    ):
        if k in sections:
            print(
                f"  {k:<6} "
                f"{sections[k][0]:+.3f}s "
                f"score="
                f"{sections[k][1]:.4f}"
            )

    print(
        "position="
        "bar|beat|sixteenth-slot"
        "(+fraction)"
    )

    print(
        "V4.5 principle: "
        "FROZEN TRANSITION RETRIEVES; "
        "FROZEN TARGET CONTEXT MATCHES; "
        "TIMING IS MEASURED AGAINST "
        "TRANSITION SUPPORT."
    )

    print(
        "No stable-to-stable boundary is "
        "manufactured. No missing target "
        "context is reconstructed."
    )

    print(
        f"vicinity = clip("
        f"{VICINITY_EYEBROW_MULT:g} * "
        "adaptive-eyebrow, "
        f"{MIN_VICINITY_MS:.0f}.."
        f"{MAX_VICINITY_MS:.0f}ms)"
    )

    print(
        "All matching thresholds are "
        "diagnostic only. Frozen production "
        "decisions are untouched."
    )

    print(
        "\n"
        + "=" * 148
    )
    print(
        "GT PITCH CHANGES -> "
        "FROZEN WAV TRANSITION REGIONS"
    )
    print("=" * 148)

    for m in matches:
        print_match(
            mm,
            midi,
            gt,
            m,
            anchors,
            off,
        )

    matched = [
        m
        for m in matches
        if m.status == "MATCHED"
    ]
    ambiguous = [
        m
        for m in matches
        if m.status == "AMBIGUOUS"
    ]
    missing = [
        m
        for m in matches
        if (
            m.status
            == "NO_CORRESPONDENCE"
        )
    ]

    # V4.4 observational layer.  Core V4.2 match statuses above are NOT changed.
    one_sided_probes = {
        m.gt_i: one_sided_probe_for_match(
            m, midi, gt, data, hz, transitions
        )
        for m in missing
    }
    one_sided_with_raw = [
        (m, one_sided_probes[m.gt_i]) for m in missing
        if one_sided_probes[m.gt_i] is not None
        and one_sided_probes[m.gt_i].raw_support
    ]
    one_sided_without_raw = [
        (m, one_sided_probes[m.gt_i]) for m in missing
        if one_sided_probes[m.gt_i] is not None
        and not one_sided_probes[m.gt_i].raw_support
    ]
    no_eligible_one_sided = [
        m for m in missing if one_sided_probes[m.gt_i] is None
    ]

    # V4.4 is classification only.  V4.3 probe selection and raw_support are
    # retained exactly; we merely partition the selected probes.
    v44_classes = {}
    for m in missing:
        p = one_sided_probes[m.gt_i]
        if p is not None:
            v44_classes.setdefault(classify_one_sided_probe(p), []).append((m, p))

    # Generic population discovery: no song name or GT index is hard-coded.
    v45_audits=[]
    for m,p in v44_classes.get("NO_VALID_F0_ON_MISSING_SIDE",[]):
        v45_audits.append((m,p,audit_rejected_raw_f0(m,p,data,hz)))
    v45_classes={}
    for item in v45_audits:
        v45_classes.setdefault(item[2].classification,[]).append(item)

    print(
        "\n"
        + "=" * 148
    )
    print("SUMMARY")
    print("=" * 148)

    print(
        f"GT pitch changes:       "
        f"{len(matches)}"
    )
    print(
        f"MATCHED:                "
        f"{len(matched)}"
    )
    print(
        f"AMBIGUOUS:              "
        f"{len(ambiguous)}"
    )
    print(
        f"NO_CORRESPONDENCE:      "
        f"{len(missing)}"
    )
    print(
        "  observational ONE_SIDED_WITH_RAW_SUPPORT:    "
        f"{len(one_sided_with_raw)}"
    )
    print(
        "  observational ONE_SIDED_WITHOUT_RAW_SUPPORT: "
        f"{len(one_sided_without_raw)}"
    )
    print(
        "  no eligible one-sided frozen context:        "
        f"{len(no_eligible_one_sided)}"
    )
    print("  V4.4 classification of eligible one-sided probes:")
    for label in (
        "DIRECT_RAW_SUPPORT",
        "HARMONIC_RAW_SUPPORT",
        "VALID_F0_NEAR_EXPECTED_BUT_INSUFFICIENT_DURATION",
        "VALID_F0_WRONG_PITCH",
        "NO_VALID_F0_ON_MISSING_SIDE",
    ):
        print(f"    {label:<49s} {len(v44_classes.get(label, []))}")

    print("  V4.5 raw-ECKF audit of V4.4 NO_VALID_F0 cases:")
    for label in ("RAW_DIRECT_EXPECTED_SUPPORT_REJECTED","RAW_HARMONIC_EXPECTED_SUPPORT_REJECTED","RAW_COHERENT_OTHER_PITCH_REJECTED","RAW_TRACKING_FAILURE_OR_OUT_OF_RANGE","RAW_NO_POSITIVE_F0"):
        print(f"    {label:<49s} {len(v45_classes.get(label, []))}")

    timed = matched + ambiguous

    abs_support_ms = [
        abs(
            m.candidate
            .interval_delta_ms
        )
        for m in timed
        if m.candidate is not None
    ]

    abs_mid_ms = [
        abs(
            m.candidate
            .mid_delta_ms
        )
        for m in timed
        if m.candidate is not None
    ]

    bins(
        "credible support timing",
        abs_support_ms,
    )
    bins(
        "credible midpoint timing",
        abs_mid_ms,
    )

    ordinary = []
    eyebrow = []
    large = []
    extreme = []

    for m in timed:
        if m.candidate is None:
            continue

        _, _, _, _, flag = (
            mm.timing_metrics(
                m.candidate
                .interval_delta_ms,
                midi.notes[
                    m.gt_i
                ].onset_s,
            )
        )

        {
            "ordinary": ordinary,
            "EYEBROW": eyebrow,
            "LARGE": large,
            "EXTREME": extreme,
        }[flag].append(m)

    print(
        "timing flags on credible "
        "matches only, using distance "
        "to transition support: "
        f"ordinary={len(ordinary)} "
        f"EYEBROW={len(eyebrow)} "
        f"LARGE={len(large)} "
        f"EXTREME={len(extreme)}"
    )

    print(
        "\n"
        + "=" * 148
    )
    print(
        "INSPECTION QUEUE A — "
        "CREDIBLE CORRESPONDENCE WITH "
        "BPM-ADAPTIVE SUPPORT-TIMING FLAG"
    )
    print("=" * 148)

    flagged = (
        eyebrow
        + large
        + extreme
    )

    if not flagged:
        print("none")

    for m in flagged:
        c = m.candidate
        n = midi.notes[m.gt_i]
        p = midi.notes[m.gt_i - 1]

        (
            bpm,
            beats,
            eighths,
            thr,
            flag,
        ) = mm.timing_metrics(
            c.interval_delta_ms,
            n.onset_s,
        )

        print(
            f"{flag:7s} "
            f"GT{n.index:03d} "
            f"{note_name(p.pitch)}"
            f"->{note_name(n.pitch)} "
            f"{mm.pos(n.onset_s):<14s} "
            f"T{c.transition.index} "
            f"R{c.transition.region_index} "
            "supportΔ="
            f"{c.interval_delta_ms:+.1f}ms "
            f"midΔ={c.mid_delta_ms:+.1f}ms "
            f"({beats:+.3f} beats, "
            f"{eighths:+.3f} eighths) "
            f"threshold={thr:.1f}ms "
            f"reason={c.reason}"
        )

    print(
        "\n"
        + "=" * 148
    )
    print(
        "INSPECTION QUEUE B — "
        "AMBIGUOUS STRUCTURAL "
        "CORRESPONDENCE"
    )
    print("=" * 148)

    if not ambiguous:
        print("none")

    for m in ambiguous:
        n = midi.notes[m.gt_i]
        p = midi.notes[m.gt_i - 1]
        a = m.candidate
        r = m.runner_up

        print(
            f"GT{n.index:03d} "
            f"{note_name(p.pitch)}"
            f"->{note_name(n.pitch)} "
            f"{mm.pos(n.onset_s):<14s} "
            f"T{a.transition.index}/"
            f"R{a.transition.region_index} "
            f"supportΔ="
            f"{a.interval_delta_ms:+.1f}ms "
            f"score="
            f"{a.correspondence_score:.3f} "
            "| "
            f"T{r.transition.index}/"
            f"R{r.transition.region_index} "
            f"supportΔ="
            f"{r.interval_delta_ms:+.1f}ms "
            f"score="
            f"{r.correspondence_score:.3f}"
        )

    print(
        "\n"
        + "=" * 148
    )
    print(
        "INSPECTION QUEUE C — "
        "NO CREDIBLE TWO-SIDED FROZEN "
        "TRANSITION IN VICINITY"
    )
    print("=" * 148)

    if not missing:
        print("none")

    for m in missing:
        n = midi.notes[m.gt_i]
        p = midi.notes[m.gt_i - 1]

        print(
            f"GT{n.index:03d} "
            f"{note_name(p.pitch)}"
            f"->{note_name(n.pitch)} "
            f"MIDI={n.onset_s:.3f}s "
            f"{mm.pos(n.onset_s):<14s} "
            f"WAVref={gt[m.gt_i]:.3f}s "
            f"vicinity=±"
            f"{m.vicinity_ms:.1f}ms "
            f"candidates="
            f"{m.candidates_seen}"
        )

        if m.runner_up is not None:
            c = m.runner_up
            t = c.transition

            print(
                f"    best observed "
                f"T{t.index} "
                f"R{t.region_index} "
                f"context={t.context_kind} "
                f"support="
                f"{t.start_s:.3f}"
                f"..{t.end_s:.3f}s "
                f"supportΔ="
                f"{c.interval_delta_ms:+.1f}ms "
                f"reason={c.reason}"
            )

    print(
        "\n" + "=" * 148
    )
    print(
        "INSPECTION QUEUE D — V4.4 CLASSIFICATION OF ONE-SIDED "
        "CORRECTED-VALID F0 PROBES"
    )
    print("=" * 148)
    print(
        "Classification only: no missing target is created; V4.2 MATCHED / "
        "NO_CORRESPONDENCE status is unchanged."
    )
    print(
        f"Existing V4.3 witness remains >= {RAW_WITNESS_MIN_MS:.0f}ms contiguous "
        f"corrected-valid F0 within {PITCH_LOOSE_ST:.2f}st. V4.4 merely "
        "separates direct from harmonic-adjusted support and failure causes."
    )

    order = (
        "DIRECT_RAW_SUPPORT",
        "HARMONIC_RAW_SUPPORT",
        "VALID_F0_NEAR_EXPECTED_BUT_INSUFFICIENT_DURATION",
        "VALID_F0_WRONG_PITCH",
        "NO_VALID_F0_ON_MISSING_SIDE",
    )
    if not v44_classes:
        print("none")
    for label in order:
        rows = v44_classes.get(label, [])
        if not rows:
            continue
        print(f"\n{label} ({len(rows)})")
        for m, p in rows:
            n = midi.notes[m.gt_i]
            prev = midi.notes[m.gt_i - 1]
            print(
                f"GT{n.index:03d} {note_name(prev.pitch)}->{note_name(n.pitch)} "
                f"MIDI={n.onset_s:.3f}s {mm.pos(n.onset_s):<14s} "
                f"T{p.transition_index} R{p.region_index} {p.context_kind} "
                f"probe={p.side} expected={p.expected_pitch:.0f}st "
                f"knownErr={p.known_error:.2f}st({p.known_harmonic}) "
                f"window={p.window_start_s:.3f}..{p.window_end_s:.3f}s "
                f"validRows={p.valid_rows} "
                f"directBest={p.direct_best_error:.2f}st "
                f"directLooseRows={p.direct_loose_rows} "
                f"directLongest={p.direct_longest_loose_ms:.1f}ms "
                f"harmBest={p.best_error:.2f}st({p.best_harmonic}) "
                f"harmLooseRows={p.loose_rows} "
                f"harmLongest={p.longest_loose_ms:.1f}ms "
                f"bestAt={p.best_time_s:.3f}s"
            )

    # Reverse coverage: all frozen transitions remain visible, including
    # one-sided/unanchored ones. No musical label is assigned.
    used = {
        m.candidate
        .transition.index
        for m in timed
        if m.candidate is not None
    }

    unused = [
        t
        for t in transitions
        if t.index not in used
    ]

    print(
        "\n"
        + "=" * 148
    )
    print(
        "REVERSE COVERAGE — "
        "FROZEN TRANSITIONS NOT USED BY "
        "A CREDIBLE GT PITCH-CHANGE MATCH"
    )
    print("=" * 148)

    print(
        f"unused={len(unused)}/"
        f"{len(transitions)}"
    )

    for t in unused:
        if change_indices:
            gi = min(
                change_indices,
                key=lambda i: abs(
                    float(gt[i])
                    - t.mid_s
                ),
            )

            n = midi.notes[gi]
            p = midi.notes[gi - 1]

            interval_dms = (
                transition_interval_delta_ms(
                    float(gt[gi]),
                    t,
                )
            )

            src = (
                f"{t.source_pitch:.2f}"
                if t.has_source
                else "--"
            )
            dst = (
                f"{t.destination_pitch:.2f}"
                if t.has_destination
                else "--"
            )

            print(
                f"T{t.index:03d} "
                f"R{t.region_index} "
                f"{src}->{dst}st "
                f"{t.context_kind:<16s} "
                f"support={t.start_s:.3f}"
                f"..{t.end_s:.3f}s "
                "nearest-context "
                f"GT{n.index:03d} "
                f"{note_name(p.pitch)}"
                f"->{note_name(n.pitch)} "
                "supportΔ="
                f"{interval_dms:+.1f}ms"
            )
        else:
            print(
                f"T{t.index:03d} "
                f"R{t.region_index} "
                f"context={t.context_kind} "
                f"support={t.start_s:.3f}"
                f"..{t.end_s:.3f}s"
            )

    print(
        "\n"
        + "=" * 148
    )
    print(
        "GT SAME-PITCH REARTICULATIONS — "
        "REPORTED SEPARATELY, NOT "
        "PITCH-GEOMETRY MATCHED"
    )
    print("=" * 148)

    print(
        f"count="
        f"{len(rearticulation_indices)}"
    )

    for i in rearticulation_indices:
        n = midi.notes[i]
        print(
            f"GT{n.index:03d} "
            f"{note_name(n.pitch)}"
            f"->{note_name(n.pitch)} "
            f"MIDI={n.onset_s:.3f}s "
            f"{mm.pos(n.onset_s):<14s} "
            f"WAVref={gt[i]:.3f}s"
        )

    print("\n"+"="*148)
    print("INSPECTION QUEUE E — V4.5 RAW ECKF BENEATH V4.4 VALIDITY-WALL CASES")
    print("="*148)
    print("Observational only: immutable f0_hz is read beneath rejected intervals. No row is rescued; validity and interpreter remain unchanged.")
    print("120..1500Hz and <=4st adjacent-step markers are descriptive mirrors of current validity diagnostics, not new production thresholds.\n")
    for label in ("RAW_DIRECT_EXPECTED_SUPPORT_REJECTED","RAW_HARMONIC_EXPECTED_SUPPORT_REJECTED","RAW_COHERENT_OTHER_PITCH_REJECTED","RAW_TRACKING_FAILURE_OR_OUT_OF_RANGE","RAW_NO_POSITIVE_F0"):
        items=v45_classes.get(label,[])
        print(f"{label} ({len(items)})")
        for m,p,a in items:
            prev=midi.notes[m.gt_i-1]; cur=midi.notes[m.gt_i]
            print(f"GT{m.gt_i:03d} {note_name(prev.pitch)}->{note_name(cur.pitch)} MIDI={cur.onset_s:.3f}s T{a.transition_index} R{a.region_index} {a.context_kind} probe={a.side} expected={a.expected_pitch:.0f}st window={a.window_start_s:.3f}..{a.window_end_s:.3f}s rows={a.rows} raw+={a.raw_positive_rows} inRange={a.raw_in_range_rows} below120={a.raw_below_floor_rows} above1500={a.raw_above_ceiling_rows} zero/bad={a.raw_nonpositive_rows}")
            print(f"    directBest={a.direct_best_error:.2f}st directLoose={a.direct_loose_rows} directLongest={a.direct_longest_loose_ms:.1f}ms harmBest={a.harmonic_best_error:.2f}st({a.harmonic_best_relation}) harmLoose={a.harmonic_loose_rows} harmLongest={a.harmonic_longest_loose_ms:.1f}ms")
            print(f"    rawGeometry: continuity={a.continuity_fraction:.3f} ({a.continuous_pairs}/{a.adjacent_pairs}) stepMedian={a.median_step_st:.2f}st stepP90={a.p90_step_st:.2f}st stepMax={a.max_step_st:.2f}st pitchMedian={a.median_pitch_st:.2f}st span={a.pitch_span_st:.2f}st")
            print(f"    reasons: first=[{_format_reason_counts(a.first_pass_reason_counts)}] validity=[{_format_reason_counts(a.validity_reason_counts)}] correction=[{_format_reason_counts(a.correction_reason_counts)}]")
        print()

    print(
        "\n"
        + "=" * 148
    )
    print(
        "INVARIANTS / INTERPRETATION"
    )
    print("=" * 148)

    print(
        "V4.5 extracted frozen "
        f"TRANSITION regions: "
        f"{len(transitions)}/"
        f"{frozen_transition_count}"
    )

    print(
        f"V3.1 structural links "
        f"preserved: "
        f"{len(anchors)}/"
        f"{st.total_links}"
    )

    print(
        "Every V4.4 structural-change "
        "candidate is an existing frozen "
        "TRANSITION PitchRegion."
    )

    print(
        "No stable-to-stable boundary is "
        "manufactured."
    )

    print(
        "Missing previous/next target "
        "context is never reconstructed "
        "from neighboring regions."
    )

    print(
        "Only TWO_SIDED transitions can "
        "independently establish A->B "
        "GT pitch geometry."
    )

    print(
        "SOURCE_ONLY, DESTINATION_ONLY, "
        "and UNANCHORED transitions remain "
        "visible observational evidence."
    )

    print(
        "V4.4 one-sided probes read only existing corrected-valid clean_f0_hz "
        "on the missing side and stop at the first validity wall."
    )
    print(
        "V4.4 classification only partitions those probes; it does not alter "
        "probe selection, raw_support, correspondence, or frozen production."
    )
    print(
        "A raw-context witness is observational only: it never creates a "
        "stable target, changes a PitchRegion, or upgrades core correspondence."
    )

    print(
        "UNRESOLVED PitchRegions are not "
        "promoted to pitch-change "
        "boundaries."
    )

    print(
        "Timing flags are computed only "
        "after credible pitch-geometric "
        "correspondence."
    )

    print(
        "Timing judgment uses distance "
        "to the frozen transition temporal "
        "support; midpoint timing is "
        "descriptive only."
    )

    print(
        "GT same-pitch rearticulations are "
        "reported separately."
    )

    print(
        "No gesture labels. "
        "No frozen threshold changed. "
        "trajectory_interpreter.py "
        "is untouched."
    )


if __name__ == "__main__":
    main()
