"""Plane model kind: the run-018 on-device artifact family.

One ONNX graph maps (input_ids, plane_ids) -> per-position class
logits; decode is a host-side K-pass loop (pass k>1 conditions on
pass k-1 argmax); the haraqat plane renders onto the skeleton.
Byte table: id = byte + 3 (the canonical ByT5 table); MASK id =
len(classes).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

MASK_OFFSET = 0  # plane vocab = classes + MASK at index len(classes)


def render_plane(skeleton: str, char_classes: list[str]) -> str:
    """One class per CHARACTER (multi-byte chars consume one class)."""
    return "".join(ch + cls for ch, cls in zip(skeleton, char_classes))


class PlaneModel:
    def __init__(self, graph: bytes, classes: list[str], k_passes: int = 2):
        self.classes = classes
        self.n_classes = len(classes)
        self.mask_id = self.n_classes
        self.k = k_passes
        self.sess = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])

    def encode(self, text: str) -> list[int]:
        return [b + 3 for b in text.encode("utf-8")]

    def predict_token_classes(self, text: str) -> list[int]:
        ids = np.array([self.encode(text)], dtype=np.int64)
        plane = np.full_like(ids, self.mask_id)
        for _ in range(self.k):
            logits = self.sess.run(None, {"input_ids": ids, "plane_ids": plane})[0]
            plane = logits.argmax(-1)
        return plane[0].tolist()

    def translate(self, text: str) -> str:
        tok_preds = self.predict_token_classes(text)
        # majority vote of a char's byte-token predictions
        pos = 0
        char_classes: list[str] = []
        for ch in text:
            n = len(ch.encode("utf-8"))
            votes = tok_preds[pos : pos + n]
            cid = max(set(votes), key=votes.count)
            char_classes.append(self.classes[cid] if cid < self.n_classes else "")
            pos += n
        return render_plane(text, char_classes)
