"""YOLO-backed object detector used by the search_and_follow mission mode.

Design decisions worth being able to defend:

1. The Ultralytics import is lazy (inside Detector.__init__), not at module
   scope. That means this file can be imported and its pure helper functions
   (`_result_to_detections`, `pick_target`) can be unit-tested on a machine
   that doesn't have torch/ultralytics installed at all -- useful for CI and
   for reviewing the bbox-selection logic in isolation from the model.

2. The detector never hardcodes a class name. `target_classes` is a plain
   list of strings supplied by the caller (which gets it from the mission
   JSON's `target_class` field). We validate the requested class exists in
   the model's label set at construction time and fail loudly if not, rather
   than silently detecting nothing.

3. `infer()` returns *all* matching detections for the frame; the executor
   decides how to pick one (we provide `pick_target` as the default policy:
   largest bounding box = closest/most prominent instance, which is a
   reasonable default for "the thing to follow").
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence


@dataclasses.dataclass
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_w: int
    frame_h: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _result_to_detections(result, target_classes: Optional[Sequence[str]], conf_threshold: float) -> List[Detection]:
    """Convert one Ultralytics `Results` object into our plain Detection list.

    Kept separate from Detector.infer() so it can be unit tested with a fake
    "result" stand-in that has the same minimal shape (boxes with .xyxy,
    .conf, .cls, and a .names dict) without needing a real model loaded.
    """
    detections: List[Detection] = []
    frame_h, frame_w = result.orig_shape  # (height, width), as reported by Ultralytics
    names = result.names

    boxes = result.boxes
    if boxes is None:
        return detections

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        if target_classes and class_name not in target_classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            Detection(
                class_name=class_name,
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                frame_w=frame_w, frame_h=frame_h,
            )
        )
    return detections


def pick_target(detections: Sequence[Detection]) -> Optional[Detection]:
    """Default selection policy when more than one match is in frame.

    We pick the largest bounding box (by area). Apparent size is our proxy
    for "closest / most prominent", which is a sane default for "the thing an
    operator most likely wants us to follow" without needing re-ID or
    tracking-by-ID logic for this challenge's scope.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d.width * d.height)


class Detector:
    def __init__(self, model_path: str, target_classes: Sequence[str], confidence_threshold: float = 0.5):
        if not target_classes:
            raise ValueError("target_classes must be a non-empty list, e.g. ['person']")

        from ultralytics import YOLO  # lazy import, see module docstring

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = list(target_classes)

        valid_names = set(self.model.names.values())
        unknown = [c for c in self.target_classes if c not in valid_names]
        if unknown:
            raise ValueError(
                f"target_classes {unknown} are not in this model's label set. "
                f"Available classes: {sorted(valid_names)}"
            )

    def infer(self, frame) -> List[Detection]:
        """Run detection on a single BGR frame (as produced by OpenCV/cv_bridge).

        Returns every detection matching `target_classes` above the
        confidence threshold. Use `pick_target()` to reduce to one, or apply
        your own selection (e.g. nearest to last-tracked position).
        """
        results = self.model(frame, verbose=False)
        result = results[0]
        return _result_to_detections(result, self.target_classes, self.confidence_threshold)
