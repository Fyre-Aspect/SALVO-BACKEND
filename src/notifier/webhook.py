from __future__ import annotations

import base64
import logging
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np
import requests

from src.models.schemas import Alert, AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: str = ""
    auth_token: str = ""
    timeout_seconds: float = 5.0
    include_snapshot: bool = True
    snapshot_scale: float = 0.5
    only_critical: bool = True
    source_id: str = "drone-1"


class WebhookNotifier:
    """Posts distress alerts to a remote webhook (the website backend).

    Runs the HTTP POST on a background worker so a slow or unreachable
    site never blocks the detection pipeline.
    """

    def __init__(self, config: WebhookConfig) -> None:
        self._config = config
        self._queue: Queue[tuple[Alert, np.ndarray | None]] = Queue(maxsize=64)
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

        if config.enabled and not config.url:
            logger.warning("Webhook enabled but URL is empty — disabling")
            self._config.enabled = False

        if self._config.enabled:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            logger.info("WebhookNotifier active → %s", config.url)

    def notify(self, alert: Alert, frame: np.ndarray | None = None) -> None:
        if not self._config.enabled:
            return
        if self._config.only_critical and alert.level != AlertLevel.CRITICAL:
            return
        try:
            self._queue.put_nowait((alert, frame))
        except Exception:
            logger.warning("Webhook queue full — dropping alert for track %d",
                           alert.track_id)

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert, frame = self._queue.get(timeout=0.5)
            except Empty:
                continue
            self._post(alert, frame)

    def _post(self, alert: Alert, frame: np.ndarray | None) -> None:
        payload: dict[str, Any] = {
            "source_id": self._config.source_id,
            "event": "distress_detected",
            "alert": alert.to_dict(),
        }
        if self._config.include_snapshot and frame is not None:
            payload["snapshot_jpeg_b64"] = _encode_jpeg_b64(
                frame, self._config.snapshot_scale
            )

        headers = {"Content-Type": "application/json"}
        if self._config.auth_token:
            headers["Authorization"] = f"Bearer {self._config.auth_token}"

        try:
            resp = requests.post(
                self._config.url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
            if 200 <= resp.status_code < 300:
                logger.info(
                    "Distress signal delivered: track=%d status=%d",
                    alert.track_id, resp.status_code,
                )
            else:
                logger.warning(
                    "Webhook non-2xx: track=%d status=%d body=%s",
                    alert.track_id, resp.status_code, resp.text[:200],
                )
        except requests.RequestException as exc:
            logger.error("Webhook delivery failed: %s", exc)


def _encode_jpeg_b64(frame: np.ndarray, scale: float) -> str:
    if scale != 1.0:
        frame = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")
