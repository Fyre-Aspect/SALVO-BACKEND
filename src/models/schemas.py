from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Bounding Box
# ---------------------------------------------------------------------------
@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


# ---------------------------------------------------------------------------
# Frame Data
# ---------------------------------------------------------------------------
@dataclass
class FrameData:
    frame: np.ndarray
    timestamp: float
    frame_id: int


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    frame_id: int
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Tracked Person
# ---------------------------------------------------------------------------
@dataclass
class TrackedPerson:
    track_id: int
    bbox: BoundingBox
    positions: deque = field(default_factory=lambda: deque(maxlen=90))
    bboxes: deque = field(default_factory=lambda: deque(maxlen=90))
    frames_since_seen: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_dict(),
            "frames_since_seen": self.frames_since_seen,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


# ---------------------------------------------------------------------------
# Gesture
# ---------------------------------------------------------------------------
class GestureType(str, Enum):
    NONE = "none"
    SOS_BOTH_ARMS = "sos_both_arms"
    SOS_SINGLE_ARM = "sos_single_arm"
    CROSSED_ARMS_OVERHEAD = "crossed_arms_overhead"
    DROWNING_POSTURE = "drowning_posture"


# ---------------------------------------------------------------------------
# Distress Event
# ---------------------------------------------------------------------------
@dataclass
class DistressEvent:
    track_id: int
    distress_score: float
    motion_score: float
    submersion_score: float
    stationary_score: float
    timestamp: float
    frame_id: int
    gesture: GestureType = GestureType.NONE
    gesture_confidence: float = 0.0
    face_emotion_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "distress_score": self.distress_score,
            "motion_score": self.motion_score,
            "submersion_score": self.submersion_score,
            "stationary_score": self.stationary_score,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "gesture": self.gesture.value,
            "gesture_confidence": self.gesture_confidence,
            "face_emotion_score": self.face_emotion_score,
        }


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------
class AlertLevel(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(str, Enum):
    MONITORING = "monitoring"
    WARNED = "warned"
    ALERT = "alert"
    COOLDOWN = "cooldown"


@dataclass
class Alert:
    track_id: int
    level: AlertLevel
    distress_score: float
    timestamp: float
    frame_id: int
    bbox: BoundingBox
    gesture: GestureType = GestureType.NONE

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "level": self.level.value,
            "distress_score": self.distress_score,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "bbox": self.bbox.to_dict(),
            "gesture": self.gesture.value,
        }


# ---------------------------------------------------------------------------
# Deploy Command
# ---------------------------------------------------------------------------
class CommandType(str, Enum):
    ARM = "ARM"
    DISARM = "DISARM"
    DEPLOY = "DEPLOY"
    STOP = "STOP"
    STATUS = "STATUS"


class ControllerState(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"


@dataclass
class DeployCommand:
    command: CommandType
    target_track_id: int | None
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "command": self.command.value,
            "target_track_id": self.target_track_id,
            "timestamp": self.timestamp,
        }
