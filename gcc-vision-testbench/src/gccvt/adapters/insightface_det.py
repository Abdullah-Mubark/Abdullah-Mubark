"""InsightFace face detector (RetinaFace / SCRFD family) + ArcFace embeddings.

Requires requirements-models.txt. InsightFace pretrained packs are non-commercial.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ToolMissingError
from .base import BaseAdapter, Box
from .registry import register


@register("insightface")
class InsightFaceAdapter(BaseAdapter):
    tasks = {"face_det", "face_rec"}

    def load(self, model_pack: str = "buffalo_l", conf: float = 0.5, det_size: int = 640,
             gpu: int = -1, recognition: bool = False, label: str = "face", **_: Any) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ToolMissingError("insightface adapter needs insightface + onnxruntime: "
                                   "pip install -r requirements-models.txt") from e
        modules = ["detection", "recognition"] if recognition else ["detection"]
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu >= 0 else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_pack, allowed_modules=modules, providers=providers)
        self.app.prepare(ctx_id=gpu, det_size=(det_size, det_size), det_thresh=conf)
        self.label = label

    def detect(self, frame: np.ndarray) -> list[Box]:
        return [Box(*map(float, f.bbox), score=float(f.det_score), label=self.label,
                    meta={"kps": f.kps.tolist()} if getattr(f, "kps", None) is not None else {})
                for f in self.app.get(frame)]

    def embed_face(self, crop: np.ndarray) -> np.ndarray:
        faces = self.app.get(crop)
        if not faces or getattr(faces[0], "normed_embedding", None) is None:
            raise ValueError("no face found in crop (or recognition=False in params)")
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(best.normed_embedding, dtype=np.float32)
