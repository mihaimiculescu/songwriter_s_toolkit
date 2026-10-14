"""
Optional, observational ECKF diagnostic.

Enable with:
    ECKF_TRACE_CSV=tests/predestinati_internal_trace.csv

Optional time window:
    ECKF_TRACE_START=57.5
    ECKF_TRACE_END=60.0

No environment variable -> no output.
No changes to tracker decisions or numerical calculations.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path


FIELDS = [
    "event",
    "time_s",
    "sample",
    "frame_start",
    "frame_time_s",
    "harm_prev",
    "harm_cur",
    "silent_prev",
    "silent_cur",
    "flag",
    "count",
    "peak_1_hz",
    "peak_2_hz",
    "peak_3_hz",
    "spacing_1_hz",
    "spacing_2_hz",
    "rounded_spacing_1_hz",
    "rounded_spacing_2_hz",
    "init_f0_hz",
    "init_amplitude",
    "gain_min_abs",
    "reset_threshold",
    "reset_accepted",
    "f0_hz",
    "innovation_abs",
    "innovation_after_abs",
    "q",
    "gain_norm",
    "covariance_norm",
    "state_frequency_hz",
    "sample_value_real",
    "predicted_measurement_real",
    "predicted_measurement_imag",
    "predicted_measurement_abs",
    "innovation_real",
    "innovation_imag",
    "kalman_denominator_real",
    "kalman_denominator_imag",
    "k0_real", "k0_imag", "k0_abs",
    "k1_real", "k1_imag", "k1_abs",
    "k2_real", "k2_imag", "k2_abs",
    "p00_abs", "p11_abs", "p22_abs",
    "x1_pre_real", "x1_pre_imag", "x1_pre_abs", "f0_pre_hz",
    "x2_pre_abs", "x3_pre_abs",
    "x1_post_real", "x1_post_imag", "x1_post_abs",
    "x2_post_abs", "x3_post_abs",
]


class ECKFTrace:
    def __init__(self, sample_rate: float):
        path = os.environ.get("ECKF_TRACE_CSV")

        self.enabled = bool(path)
        self.sample_rate = float(sample_rate)

        self.start = float(
            os.environ.get("ECKF_TRACE_START", "-inf")
        )
        self.end = float(
            os.environ.get("ECKF_TRACE_END", "inf")
        )
        self.stride_ms = float(os.environ.get("ECKF_TRACE_STRIDE_MS", "10.0"))
        if not (self.stride_ms > 0):
            raise ValueError("ECKF_TRACE_STRIDE_MS must be > 0")
        self.stride_samples = max(1, round(self.sample_rate * self.stride_ms / 1000.0))

        self.handle = None
        self.writer = None

        if self.enabled:
            destination = Path(path)
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.handle = destination.open(
                "w",
                newline="",
                encoding="utf-8",
            )

            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=FIELDS,
                extrasaction="ignore",
            )
            self.writer.writeheader()

    def wants_sample(self, sample: int) -> bool:
        if not self.enabled:
            return False
        t = sample / self.sample_rate
        return self.start <= t <= self.end and (sample % self.stride_samples == 0)

    def emit(self, event: str, sample: int, **values):
        if not self.enabled:
            return

        t = sample / self.sample_rate

        if not self.start <= t <= self.end:
            return

        row = {
            key: ""
            for key in FIELDS
        }

        row.update(values)
        row["event"] = event
        row["sample"] = sample
        row["time_s"] = f"{t:.9f}"

        self.writer.writerow(row)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            