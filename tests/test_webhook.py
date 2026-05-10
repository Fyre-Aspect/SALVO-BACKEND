from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.models.schemas import Alert, AlertLevel, BoundingBox
from src.notifier.webhook import WebhookConfig, WebhookNotifier


def _alert(level: AlertLevel = AlertLevel.CRITICAL) -> Alert:
    return Alert(
        track_id=7,
        level=level,
        distress_score=0.92,
        timestamp=1234.5,
        frame_id=42,
        bbox=BoundingBox(0, 0, 10, 10),
    )


def test_disabled_notifier_does_nothing():
    notifier = WebhookNotifier(WebhookConfig(enabled=False))
    with patch("src.notifier.webhook.requests.post") as post:
        notifier.notify(_alert())
        time.sleep(0.1)
        post.assert_not_called()
    notifier.close()


def test_enabled_with_empty_url_disables_itself():
    notifier = WebhookNotifier(WebhookConfig(enabled=True, url=""))
    assert notifier._config.enabled is False
    notifier.close()


def test_critical_alert_posts_payload():
    cfg = WebhookConfig(
        enabled=True,
        url="http://test.local/distress",
        auth_token="abc",
        include_snapshot=False,
    )
    fake_resp = MagicMock(status_code=200, text="ok")
    with patch(
        "src.notifier.webhook.requests.post", return_value=fake_resp
    ) as post:
        notifier = WebhookNotifier(cfg)
        notifier.notify(_alert(AlertLevel.CRITICAL))
        # Wait for the worker to drain the queue
        for _ in range(50):
            if post.called:
                break
            time.sleep(0.05)
        notifier.close()

    assert post.called
    kwargs = post.call_args.kwargs
    assert kwargs["json"]["event"] == "distress_detected"
    assert kwargs["json"]["alert"]["track_id"] == 7
    assert kwargs["json"]["alert"]["level"] == "critical"
    assert kwargs["headers"]["Authorization"] == "Bearer abc"


def test_warning_alert_skipped_when_only_critical():
    cfg = WebhookConfig(enabled=True, url="http://x", only_critical=True)
    with patch("src.notifier.webhook.requests.post") as post:
        notifier = WebhookNotifier(cfg)
        notifier.notify(_alert(AlertLevel.WARNING))
        time.sleep(0.2)
        notifier.close()
        post.assert_not_called()


def test_snapshot_included_when_frame_provided():
    cfg = WebhookConfig(enabled=True, url="http://x", include_snapshot=True)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    fake_resp = MagicMock(status_code=200, text="ok")
    with patch(
        "src.notifier.webhook.requests.post", return_value=fake_resp
    ) as post:
        notifier = WebhookNotifier(cfg)
        notifier.notify(_alert(), frame=frame)
        for _ in range(50):
            if post.called:
                break
            time.sleep(0.05)
        notifier.close()
    body = post.call_args.kwargs["json"]
    assert "snapshot_jpeg_b64" in body
    assert len(body["snapshot_jpeg_b64"]) > 0
