from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.event_bus import EventBus

logger = logging.getLogger(__name__)


class EventLogger:
    """Structured JSONL event logger that subscribes to the event bus."""

    TOPICS = [
        "detection.complete",
        "track.updated",
        "distress.scored",
        "alert.triggered",
        "deploy.commanded",
    ]

    def __init__(
        self,
        event_bus: EventBus,
        log_dir: str = "logs",
        snapshot_on_alert: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir = self._log_dir / "snapshots"
        self._snapshot_on_alert = snapshot_on_alert
        if snapshot_on_alert:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self._log_dir / f"events_{timestamp_str}.jsonl"
        self._file = open(self._log_path, "a", encoding="utf-8")

        self._event_bus = event_bus
        self._last_frame: np.ndarray | None = None

        # Subscribe to all topics
        for topic in self.TOPICS:
            event_bus.subscribe(topic, lambda data, t=topic: self._log(t, data))

        # Track latest frame for snapshot
        event_bus.subscribe("frame.captured", self._on_frame)

        logger.info("EventLogger writing to %s", self._log_path)

    def _on_frame(self, frame_data: Any) -> None:
        self._last_frame = frame_data.frame if frame_data else None

    def _log(self, topic: str, data: Any) -> None:
        record = {
            "timestamp": time.time(),
            "topic": topic,
            "data": self._serialize(data),
        }
        line = json.dumps(record, default=str)
        self._file.write(line + "\n")
        self._file.flush()

        # Save snapshot on alert
        if topic == "alert.triggered" and self._snapshot_on_alert:
            self._save_snapshot(data)

    def _save_snapshot(self, alert_data: Any) -> None:
        if self._last_frame is None:
            return
        frame_id = getattr(alert_data, "frame_id", "unknown")
        path = self._snapshot_dir / f"alert_{frame_id}.jpg"
        cv2.imwrite(str(path), self._last_frame)
        logger.info("Alert snapshot saved: %s", path)

    @staticmethod
    def _serialize(data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, list):
            return [EventLogger._serialize(item) for item in data]
        if isinstance(data, dict):
            return {k: EventLogger._serialize(v) for k, v in data.items()}
        if hasattr(data, "to_dict"):
            return data.to_dict()
        return str(data)

    def close(self) -> None:
        self._file.close()
        logger.info("EventLogger closed: %s", self._log_path)
