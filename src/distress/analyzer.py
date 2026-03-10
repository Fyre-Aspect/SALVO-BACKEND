from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.distress.features import (
    compute_motion_irregularity,
    compute_stationary_duration,
    compute_submersion_ratio,
)
from src.models.schemas import DistressEvent, TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class DistressConfig:
    motion_weight: float = 0.4
    submersion_weight: float = 0.4
    stationary_weight: float = 0.2
    min_track_seconds: float = 1.0
    fps: float = 15.0


class DistressAnalyzer:
    """Computes per-person distress scores from tracked movement data."""

    def __init__(self, config: DistressConfig | None = None) -> None:
        self._config = config or DistressConfig()

    def analyze(
        self,
        tracked_persons: list[TrackedPerson],
        frame_id: int,
    ) -> list[DistressEvent]:
        events: list[DistressEvent] = []
        now = time.time()

        for person in tracked_persons:
            if person.frames_since_seen > 0:
                continue

            # Need minimum observation window
            duration = person.last_seen - person.first_seen
            if duration < self._config.min_track_seconds:
                continue

            motion = compute_motion_irregularity(person.positions)
            submersion = compute_submersion_ratio(person.bboxes)
            stationary = compute_stationary_duration(
                person.positions, self._config.fps
            )

            cfg = self._config
            score = (
                cfg.motion_weight * motion
                + cfg.submersion_weight * submersion
                + cfg.stationary_weight * stationary
            )
            score = min(score, 1.0)

            events.append(
                DistressEvent(
                    track_id=person.track_id,
                    distress_score=score,
                    motion_score=motion,
                    submersion_score=submersion,
                    stationary_score=stationary,
                    timestamp=now,
                    frame_id=frame_id,
                )
            )

        return events
