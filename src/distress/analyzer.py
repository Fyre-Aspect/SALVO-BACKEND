from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.distress.features import (
    compute_motion_irregularity,
    compute_stationary_duration,
    compute_submersion_ratio,
)
from src.distress.pose_classifier import GestureResult
from src.models.schemas import DistressEvent, GestureType, TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class DistressConfig:
    motion_weight: float = 0.4
    submersion_weight: float = 0.4
    stationary_weight: float = 0.2
    min_track_seconds: float = 1.0
    fps: float = 15.0
    # When a gesture is detected, the heuristic score is replaced by this floor
    # so a clear SOS signal fires CRITICAL even from someone standing still.
    gesture_score_floor: float = 0.95


class DistressAnalyzer:
    """Computes per-person distress scores from tracked movement data."""

    def __init__(self, config: DistressConfig | None = None) -> None:
        self._config = config or DistressConfig()

    def analyze(
        self,
        tracked_persons: list[TrackedPerson],
        frame_id: int,
        gestures: dict[int, GestureResult] | None = None,
    ) -> list[DistressEvent]:
        events: list[DistressEvent] = []
        now = time.time()
        gestures = gestures or {}

        for person in tracked_persons:
            if person.frames_since_seen > 0:
                continue

            gesture_result = gestures.get(person.track_id)
            has_gesture = (
                gesture_result is not None
                and gesture_result.gesture != GestureType.NONE
            )

            # Need minimum observation window for HEURISTIC scoring,
            # but a recognized gesture fires immediately.
            duration = person.last_seen - person.first_seen
            if duration < self._config.min_track_seconds and not has_gesture:
                continue

            motion = compute_motion_irregularity(person.positions)
            submersion = compute_submersion_ratio(person.bboxes)
            stationary = compute_stationary_duration(
                person.positions, self._config.fps
            )

            cfg = self._config
            heuristic_score = min(
                cfg.motion_weight * motion
                + cfg.submersion_weight * submersion
                + cfg.stationary_weight * stationary,
                1.0,
            )

            if has_gesture:
                # Gesture is a strong, intentional signal — dominates the score
                gesture_score = max(
                    cfg.gesture_score_floor, gesture_result.confidence
                )
                final_score = max(heuristic_score, gesture_score)
                gesture_type = gesture_result.gesture
                gesture_conf = gesture_result.confidence
            else:
                final_score = heuristic_score
                gesture_type = GestureType.NONE
                gesture_conf = 0.0

            events.append(
                DistressEvent(
                    track_id=person.track_id,
                    distress_score=final_score,
                    motion_score=motion,
                    submersion_score=submersion,
                    stationary_score=stationary,
                    timestamp=now,
                    frame_id=frame_id,
                    gesture=gesture_type,
                    gesture_confidence=gesture_conf,
                )
            )

        return events
