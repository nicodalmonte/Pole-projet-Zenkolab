"""Model classes for glaucoma classification."""

from __future__ import annotations

from src.models.dino_v3_1 import DinoV3_1
from src.models.eva02_large import EVA02Large
from src.models.retfound_dinov2 import RETFoundDinoV2
from src.models.vit_generalist import ViTGeneralist

__all__ = ["DinoV3_1", "EVA02Large", "RETFoundDinoV2", "ViTGeneralist"]
