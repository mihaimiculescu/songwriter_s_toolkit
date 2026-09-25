from dataclasses import dataclass
from typing import Literal
import math

Mode = Literal["matlab", "offline"]
SilenceMode = Literal["fixed", "adaptive"]


@dataclass(frozen=True)
class ECKFConfig:
    """
    Configuration for the Python port of orchidas/Pitch-Tracking.

    mode="matlab":
        Preserve the checked-in MATLAB behavior as closely as practical.

    mode="offline":
        Vocal-only offline behavior. The active offline tracker currently
        first attempts current-frame initialization, then inspects up to
        num_buf_to_wait future frames if the current choice fails. The future
        must belong to the same supported note and acoustic event; filtering
        returns to the ORIGINAL current audio. Zero disables lookahead.

    MATLAB mode remains compatibility-oriented.
    """
    block_size: int = 2048
    c: float = 7.0
    num_buf_to_wait: int = 2
    npeaks: int = 3
    nsemitones: float = 2.0
    mode: Mode = "matlab"

    # Vocal-only floor.  60 Hz is deliberately a little below C2 (65.406 Hz),
    # so a very low male singer is not clipped at the boundary.
    vocal_floor_hz: float = 60.0

    # Fixed is the ONLY implemented mode. Adaptive calibration will be added later.
    silence_mode: SilenceMode = "fixed"
    silence_flatness_threshold: float = 0.45
    silence_energy_db_threshold: float = -50.0
    kalman_gain_reset_threshold: float = 0.01

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.c <= 0:
            raise ValueError("c must be > 0")
        if self.num_buf_to_wait < 0:
            raise ValueError("num_buf_to_wait must be >= 0")
        if self.npeaks < 2:
            raise ValueError("npeaks must be >= 2")
        if self.nsemitones <= 0:
            raise ValueError("nsemitones must be > 0")
        if self.mode not in ("matlab", "offline"):
            raise ValueError("mode must be 'matlab' or 'offline'")
        if self.silence_mode not in ("fixed", "adaptive"):
            raise ValueError("silence_mode must be 'fixed' or 'adaptive'")
        if self.silence_mode == "adaptive":
            raise NotImplementedError(
                "Adaptive silence calibration is not implemented. "
                "Use silence_mode='fixed' until its separate validation is complete."
            )
        if not math.isfinite(self.silence_energy_db_threshold):
            raise ValueError("silence_energy_db_threshold must be finite")
        if not math.isfinite(self.silence_flatness_threshold) or not 0 <= self.silence_flatness_threshold <= 1:
            raise ValueError("silence_flatness_threshold must be in [0, 1]")
        if self.vocal_floor_hz <= 0:
            raise ValueError("vocal_floor_hz must be > 0")
