from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.models.schemas import Alert, AlertLevel, AlertState, DistressEvent

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    warning_threshold: float = 0.4
    critical_threshold: float = 0.7
    warning_frames: int = 5
    critical_frames: int = 10
    cooldown_seconds: float = 30.0


@dataclass
class _PersonAlertState:
    state: AlertState = AlertState.MONITORING
    consecutive_warning: int = 0
    consecutive_critical: int = 0
    last_alert_time: float = 0.0


class AlertManager:
    """Per-person alert state machine with threshold logic."""

    def __init__(self, config: AlertConfig | None = None) -> None:
        self._config = config or AlertConfig()
        self._person_states: dict[int, _PersonAlertState] = {}

    def evaluate(self, events: list[DistressEvent]) -> list[Alert]:
        alerts: list[Alert] = []

        for event in events:
            state = self._person_states.setdefault(
                event.track_id, _PersonAlertState()
            )

            # Handle cooldown
            if state.state == AlertState.COOLDOWN:
                elapsed = time.time() - state.last_alert_time
                if elapsed >= self._config.cooldown_seconds:
                    state.state = AlertState.MONITORING
                    state.consecutive_warning = 0
                    state.consecutive_critical = 0
                else:
                    continue

            score = event.distress_score

            # Update consecutive counters
            if score >= self._config.critical_threshold:
                state.consecutive_critical += 1
                state.consecutive_warning += 1
            elif score >= self._config.warning_threshold:
                state.consecutive_critical = 0
                state.consecutive_warning += 1
            else:
                state.consecutive_critical = 0
                state.consecutive_warning = max(0, state.consecutive_warning - 1)

            # State transitions
            if (
                state.consecutive_critical >= self._config.critical_frames
                and state.state != AlertState.ALERT
            ):
                state.state = AlertState.ALERT
                state.last_alert_time = time.time()
                alert = Alert(
                    track_id=event.track_id,
                    level=AlertLevel.CRITICAL,
                    distress_score=score,
                    timestamp=event.timestamp,
                    frame_id=event.frame_id,
                    bbox=self._stub_bbox(),
                )
                alerts.append(alert)
                logger.warning(
                    "CRITICAL ALERT: track_id=%d score=%.2f",
                    event.track_id,
                    score,
                )
                # Move to cooldown after alert fires
                state.state = AlertState.COOLDOWN

            elif (
                state.consecutive_warning >= self._config.warning_frames
                and state.state == AlertState.MONITORING
            ):
                state.state = AlertState.WARNED
                alert = Alert(
                    track_id=event.track_id,
                    level=AlertLevel.WARNING,
                    distress_score=score,
                    timestamp=event.timestamp,
                    frame_id=event.frame_id,
                    bbox=self._stub_bbox(),
                )
                alerts.append(alert)
                logger.info(
                    "WARNING: track_id=%d score=%.2f",
                    event.track_id,
                    score,
                )

        return alerts

    def update_bbox(self, alert: Alert, track_id: int, bbox) -> Alert:
        """Attach the latest bounding box to an alert after creation."""
        alert.bbox = bbox
        return alert

    def cleanup_stale(self, active_track_ids: set[int]) -> None:
        """Remove state for tracks that no longer exist."""
        stale = [tid for tid in self._person_states if tid not in active_track_ids]
        for tid in stale:
            del self._person_states[tid]

    @staticmethod
    def _stub_bbox():
        from src.models.schemas import BoundingBox
        return BoundingBox(0, 0, 0, 0)
