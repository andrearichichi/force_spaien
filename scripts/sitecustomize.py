"""Process-local SAPIEN loader calibration, activated only by explicit env vars."""
from __future__ import annotations

import os


def _install() -> None:
    scale_text = os.environ.get("FORCESAPIEN_CALIBRATION_SCALE")
    density_text = os.environ.get("FORCESAPIEN_CALIBRATION_DENSITY")
    if scale_text is None and density_text is None:
        return
    from sapien.wrapper.urdf_loader import URDFLoader
    original_init = URDFLoader.__init__

    def calibrated_init(self):
        original_init(self)
        if scale_text is not None:
            self.scale = float(scale_text)
        if density_text is not None:
            self.set_density(float(density_text))

    URDFLoader.__init__ = calibrated_init

_install()
