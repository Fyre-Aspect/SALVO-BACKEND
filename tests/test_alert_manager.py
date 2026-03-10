"""Tests for AlertManager."""

import time

from src.alerting.alert_manager import AlertConfig, AlertManager
from src.models.schemas import AlertLevel, DistressEvent


def _make_event(track_id=1, score=0.5, frame_id=1):
    return DistressEvent(
        track_id=track_id,
        distress_score=score,
        motion_score=score,
        submersion_score=0.0,
        stationary_score=0.0,
        timestamp=time.time(),
        frame_id=frame_id,
    )


def test_no_alert_below_threshold():
    mgr = AlertManager(AlertConfig(warning_threshold=0.4))
    events = [_make_event(score=0.2)]
    alerts = mgr.evaluate(events)
    assert len(alerts) == 0


def test_warning_after_consecutive_frames():
    config = AlertConfig(
        warning_threshold=0.4,
        warning_frames=3,
    )
    mgr = AlertManager(config)

    alerts = []
    for i in range(5):
        result = mgr.evaluate([_make_event(score=0.5, frame_id=i)])
        alerts.extend(result)

    # Should get exactly 1 warning after 3 consecutive frames
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING


def test_critical_alert():
    config = AlertConfig(
        critical_threshold=0.7,
        critical_frames=3,
    )
    mgr = AlertManager(config)

    alerts = []
    for i in range(10):
        result = mgr.evaluate([_make_event(score=0.8, frame_id=i)])
        alerts.extend(result)

    # Should have both warning and critical at some point
    levels = [a.level for a in alerts]
    assert AlertLevel.CRITICAL in levels


def test_cleanup_stale():
    mgr = AlertManager()
    mgr.evaluate([_make_event(track_id=1, score=0.5)])
    mgr.evaluate([_make_event(track_id=2, score=0.5)])

    # Remove track 1
    mgr.cleanup_stale({2})
    assert 1 not in mgr._person_states
    assert 2 in mgr._person_states
