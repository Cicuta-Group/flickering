from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Microscope(Protocol):
    camera_name: str
    _z: float

    def get_stage_position(self, update: bool = True) -> np.ndarray:
        """Returns the stage position as [x, y] in microns."""
        ...

    def get_pfs_offset(self, update: bool = True) -> float:
        """Returns the PFS offset."""
        ...

    def get_z(self, update: bool = True) -> float:
        """Returns the Z position in microns."""
        ...

    def get_image(self, trigger: bool = True) -> np.ndarray:
        """Captures a single frame from the camera."""
        ...

    def max_image_size(self) -> tuple[int, int]:
        """Returns the maximum width and height of the camera sensor."""
        ...

    def configure_camera(self, config: dict, delay_send: bool = False) -> None:
        """Configures camera settings (e.g. exposure, gain, offset)."""
        ...

    def restore_camera_defaults(self) -> None:
        """Restores default camera configurations."""
        ...

    def pfs_focus(self, update: bool = False) -> bool:
        """Attempts to lock/re-lock the PFS focus."""
        ...

    def set_pfs(self, enable: bool) -> bool:
        """Enables or disables PFS focus."""
        ...

    def move_stage(self, move_by: np.ndarray, absolute: bool = False, speed: float = 1000.0) -> bool:
        """Moves stage by delta or to absolute coordinates."""
        ...

    def move_z(self, move_by: float, absolute: bool = False, wait: bool = True, speed: float = 50.0) -> bool:
        """Moves focus motor by delta or to absolute coordinate."""
        ...

    def move_pfs(self, move_by: float, absolute: bool = False) -> None:
        """Moves PFS offset."""
        ...

    def update_status(self, retries: int = 5) -> None:
        """Updates internal status parameters of the microscope."""
        ...

    def attempt_recovery(self) -> None:
        """Attempts to recover from microscope or camera communication failure."""
        ...

    def set_illumination(self, channel: int, led: str, value: float, trigger=None) -> None:
        """Sets the illumination intensity for a specific LED channel."""
        ...
