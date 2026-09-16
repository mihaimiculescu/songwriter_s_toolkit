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
            