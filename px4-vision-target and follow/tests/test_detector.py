"""Unit tests for detector.py's pure logic (class filtering, confidence
threshold, target selection). Deliberately does NOT import ultralytics --
Detector's real model loading is lazy-imported inside __init__ specifically
so these tests can exercise `_result_to_detections` and `pick_target`
without torch/ultralytics installed. A fake "Results"-shaped stub stands in
for what Ultralytics would normally hand back.

Run with:  python3 -m pytest tests/test_detector.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.detector import _result_to_detections, pick_target

NAMES = {0: "person", 1: "car", 2: "dog"}


class FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes, names=NAMES, orig_shape=(480, 640)):
        self.boxes = boxes
        self.names = names
        self.orig_shape = orig_shape  # (height, width), matches Ultralytics


def test_filters_to_requested_target_class():
    result = FakeResult([
        FakeBox(0, 0.9, (10, 10, 50, 50)),   # person
        FakeBox(1, 0.9, (60, 60, 120, 120)),  # car
    ])

    detections = _result_to_detections(result, target_classes=["person"], conf_threshold=0.5)

    assert len(detections) == 1
    assert detections[0].class_name == "person"


def test_below_confidence_threshold_is_dropped():
    result = FakeResult([FakeBox(0, 0.3, (10, 10, 50, 50))])

    detections = _result_to_detections(result, target_classes=["person"], conf_threshold=0.5)

    assert detections == []


def test_no_target_classes_filter_returns_everything_above_threshold():
    result = FakeResult([
        FakeBox(0, 0.9, (10, 10, 50, 50)),
        FakeBox(2, 0.9, (60, 60, 120, 120)),
    ])

    detections = _result_to_detections(result, target_classes=None, conf_threshold=0.5)

    assert {d.class_name for d in detections} == {"person", "dog"}


def test_pick_target_selects_largest_bounding_box():
    result = FakeResult([
        FakeBox(0, 0.9, (10, 10, 40, 40)),     # 30x30 = 900
        FakeBox(0, 0.9, (100, 100, 300, 300)),  # 200x200 = 40000, bigger
    ])
    detections = _result_to_detections(result, target_classes=["person"], conf_threshold=0.5)

    best = pick_target(detections)

    assert best.width == 200 and best.height == 200


def test_pick_target_on_empty_list_returns_none():
    assert pick_target([]) is None
