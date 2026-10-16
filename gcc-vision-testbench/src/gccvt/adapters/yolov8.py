"""Ultralytics YOLOv8 detector (plate or face weights). Requires requirements-models.txt."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ToolMissingError
from .base import BaseAdapter, Box
from .registry import register


@register("yolov8")
class YoloV8Adapter(BaseAdapter):
    tasks = {"plate_det", "face_det"}

    def load(self, weights: str, conf: float = 0.25, iou: float = 0.5, device: str | None = None,
             imgsz: int | None = None, label: str = "object", keep_classes: list[int] | None = None,
             **_: Any) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ToolMissingError("yolov8 adapter needs ultralytics: pip install -r requirements-models.txt") from e
        self.model = YOLO(weights)
        self.conf, self.iou, self.device, self.imgsz = conf, iou, device, imgsz
        self.label = label
        self.keep = set(keep_classes) if keep_classes else None

    def detect(self, frame: np.ndarray) -> list[Box]:
        kwargs: dict[str, Any] = {"conf": self.conf, "iou": self.iou, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        res = self.model.predict(frame, **kwargs)[0]
        out = []
        for xyxy, score, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            cls = int(cls)
            if self.keep is not None and cls not in self.keep:
                continue
            out.append(Box(*xyxy, score=float(score), label=self.label,
                           meta={"class_id": cls, "class_name": res.names.get(cls, str(cls))}))
        return out
