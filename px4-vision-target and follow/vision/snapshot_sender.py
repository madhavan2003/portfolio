"""Snapshot capture + "send to operator" hook.

Fires exactly once per target *acquisition* (i.e. the transition from "no
target" to "target found"), not on every frame -- otherwise a 10Hz control
loop would flood the operator with a snapshot every 100ms for the entire
follow duration. Re-arms itself if the target is lost and later re-acquired
(new acquisition = new snapshot), which mission_executor / search_and_follow
communicates by calling `reset()` at the start of the step and `notify_lost()`
when the controller reports FollowState.LOST.

The actual "send to operator" transport is intentionally not implemented:
this challenge only asks for a clear hook where it would go. `_notify_operator`
is the single place to wire in a real webhook POST / email / MQTT publish.
Everything else here (annotate + save to disk + append a structured JSON
log line) is real and runnable today with zero external services, which is
what lets you demo "sent to operator" live without needing real
infrastructure.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Optional

from vision.detector import Detection

try:
    import cv2
except ImportError:  # pragma: no cover - only needed when actually saving frames
    cv2 = None


@dataclasses.dataclass
class SnapshotRecord:
    path: str
    timestamp: float
    class_name: str
    confidence: float
    bbox: tuple


class SnapshotSender:
    def __init__(self, output_dir: str, operator_endpoint: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.operator_endpoint = operator_endpoint
        self.events_log_path = self.output_dir / "events.jsonl"
        self._armed = True  # True = "next detection should trigger a snapshot"

    def reset(self) -> None:
        """Call at the start of a new search_and_follow step / new target."""
        self._armed = True

    def notify_lost(self) -> None:
        """Call when the follow controller reports the target as LOST, so the
        *next* re-acquisition is treated as a fresh event and gets its own
        snapshot instead of being silently swallowed."""
        self._armed = True

    def maybe_send(self, frame, detection: Detection, mission_context: Optional[dict] = None) -> Optional[SnapshotRecord]:
        """Call once per frame with the current best detection. Only actually
        does anything the first time it's called after arming (i.e. on
        acquisition); returns None on subsequent frames of the same
        acquisition."""
        if not self._armed:
            return None
        self._armed = False
        return self._send(frame, detection, mission_context or {})

    def _send(self, frame, detection: Detection, mission_context: dict) -> SnapshotRecord:
        timestamp = time.time()
        filename = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(timestamp))}_{detection.class_name}.jpg"
        path = self.output_dir / filename

        annotated = self._annotate(frame, detection)
        if cv2 is not None:
            cv2.imwrite(str(path), annotated)
        else:  # pragma: no cover - degrade gracefully without opencv installed
            path = path.with_suffix(".raw.txt")
            path.write_text("opencv not installed; frame not saved")

        record = SnapshotRecord(
            path=str(path),
            timestamp=timestamp,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
        )
        self._append_event_log(record, mission_context)
        self._notify_operator(record, mission_context)
        return record

    def _annotate(self, frame, detection: Detection):
        if cv2 is None:
            return frame
        annotated = frame.copy()
        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(annotated, p1, p2, (0, 0, 255), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(annotated, label, (p1[0], max(0, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return annotated

    def _append_event_log(self, record: SnapshotRecord, mission_context: dict) -> None:
        event = {
            "event": "target_acquired",
            "snapshot_path": record.path,
            "timestamp": record.timestamp,
            "class_name": record.class_name,
            "confidence": record.confidence,
            "bbox": record.bbox,
            "mission_context": mission_context,
        }
        with self.events_log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def _notify_operator(self, record: SnapshotRecord, mission_context: dict) -> None:
        """Hook for a real operator notification.

        This is the one function to replace for production use, e.g.:

            import requests
            requests.post(self.operator_endpoint, json={
                "message": f"Target acquired: {record.class_name} ({record.confidence:.0%})",
                "snapshot_path": record.path,
            }, timeout=5)

        or an email send via smtplib/ses, or a ROS 2 publish onto an
        operator-facing topic. Left as a log line so the demo has no
        dependency on real network/email infra.
        """
        if self.operator_endpoint is None:
            print(f"[operator-notify:noop] target acquired -> {record.path} "
                  f"(no operator_endpoint configured, see snapshot_sender._notify_operator)")
            return
        # Real implementation would POST/publish here using self.operator_endpoint.
        print(f"[operator-notify] would send {record.path} to {self.operator_endpoint}")
