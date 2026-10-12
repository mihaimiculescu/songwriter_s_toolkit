from dataclasses import dataclass
from typing import Literal

Mode = Literal["matlab", "offline"]


@dataclass(frozen=True)
class ECKFConfig:
    """
    Configuration for the Python port of orchidas/Pitch-Tracking.

    mode="matlab":
        Preserve the checked-in MATLAB behavior as closely as practical.

    mode="offline":
        Vocal-only offline behavior.  In v1 this deliberately differs from
        MATLAB in three places:
          1. spectral analysis ignores frequencies below vocal_floor_hz;
          2. the final complete/padded frame is processed;
          3. after attack look-ahead, filtering really resumes on the
             backtracked audio frame rather than reusing the future frame.

    Everything else is intentionally kept close to the repository source.
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
        if self.vocal_floor_hz <= 0:
            raise ValueError("vocal_floor_hz must be > 0")
