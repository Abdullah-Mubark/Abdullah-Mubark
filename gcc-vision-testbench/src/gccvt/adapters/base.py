"""The model adapter interface.

To plug in a model, subclass BaseAdapter, implement the methods for the tasks it supports,
and register it (see registry.py). Frames are numpy arrays, HxWx3 uint8, BGR (OpenCV order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

TASKS = {"face_det", "face_rec", "plate_det", "plate_ocr"}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str                      # "face" | "plate" (or whatever the detector config sets)
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. landmarks, class name

    def clipped(self, width: int, height: int) -> "Box | None":
        x1, x2 = sorted((max(0.0, min(self.x1, width)), max(0.0, min(self.x2, width))))
        y1, y2 = sorted((max(0.0, min(self.y1, height)), max(0.0, min(self.y2, height))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return Box(x1, y1, x2, y2, self.score, self.label, self.meta)


@dataclass(frozen=True)
class PlateRead:
    text: str                       # raw model output; normalisation happens in evaluation
    score: float = 1.0
    chars: list[tuple[str, float]] | None = None


class BaseAdapter:
    name: str = "base"
    tasks: set[str] = set()

    def load(self, **params: Any) -> None:
        """Load weights. `params` come from the config file."""

    def detect(self, frame: np.ndarray) -> list[Box]:
        raise NotImplementedError(f"{self.name} does not implement detect()")

    def read_plate(self, crop: np.ndarray) -> PlateRead:
        raise NotImplementedError(f"{self.name} does not implement read_plate()")

    def embed_face(self, crop: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not implement embed_face()")

    def close(self) -> None:
        pass
