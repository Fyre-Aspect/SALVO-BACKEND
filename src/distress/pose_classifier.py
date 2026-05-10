"""Pose-based gesture classifier for distress signals.

Uses YOLOv8-pose to extract 17 COCO keypoints per person, then a small
rule-based classifier to recognize universal SOS gestures:

    - SOS_BOTH_ARMS:        both wrists held above the head
    - SOS_SINGLE_ARM:       one wrist above head + waving (lateral oscillation)
    - CROSSED_ARMS_OVERHEAD: both wrists above head AND crossed past the midline
    - DROWNING_POSTURE:     head Y >= shoulders Y + erratic wrist motion

A gesture must persist for `min_hold_frames` to count, suppressing one-off
arm raises (scratching head, stretching, etc.) from misfiring as distress.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.models.schemas import (
    BoundingBox,
    FrameData,
    GestureType,
    TrackedPerson,
)

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices used by YOLOv8-pose
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

KEYPOINT_CONF_THRESHOLD = 0.3


@dataclass
class GestureConfig:
    model_path: str = "yolov8n-pose.pt"
    device: str = "cpu"
    detection_confidence: float = 0.4
    min_hold_frames: int = 6           # ~0.4s at 15 FPS
    wave_history_frames: int = 12      # window for detecting oscillation
    wave_min_amplitude_px: float = 25.0
    drowning_motion_px: float = 18.0
    iou_match_threshold: float = 0.20


@dataclass
class GestureResult:
    gesture: GestureType
    confidence: float


class PoseGestureClassifier:
    """Detects distress gestures from body keypoints, per tracked person."""

    def __init__(self, config: GestureConfig | None = None) -> None:
        self._config = config or GestureConfig()
        self._model = YOLO(self._config.model_path)
        # Per-track temporal state
        self._gesture_streak: dict[int, dict[GestureType, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._wrist_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self._config.wave_history_frames)
        )
        logger.info(
            "PoseGestureClassifier loaded model=%s device=%s",
            self._config.model_path, self._config.device,
        )

    def classify(
        self,
        frame_data: FrameData,
        tracked: list[TrackedPerson],
    ) -> dict[int, GestureResult]:
        """Return {track_id: GestureResult} for visible tracked persons."""
        results: dict[int, GestureResult] = {}
        if not tracked:
            return results

        active_ids = {t.track_id for t in tracked if t.frames_since_seen == 0}

        # Run pose model on the full frame once
        try:
            yolo_out = self._model.predict(
                frame_data.frame,
                conf=self._config.detection_confidence,
                device=self._config.device,
                verbose=False,
            )
        except Exception:
            logger.exception("Pose inference failed")
            return results

        # Extract pose detections (bbox + keypoints array)
        pose_dets: list[tuple[BoundingBox, np.ndarray]] = []
        for r in yolo_out:
            if r.keypoints is None or r.boxes is None:
                continue
            kps_xy = r.keypoints.xy.cpu().numpy()       # (N, 17, 2)
            kps_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            boxes = r.boxes.xyxy.cpu().numpy()           # (N, 4)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                kp = kps_xy[i]
                conf = kps_conf[i] if kps_conf is not None else np.ones(17)
                kp_with_conf = np.concatenate([kp, conf[:, None]], axis=1)  # (17, 3)
                pose_dets.append((BoundingBox(x1, y1, x2, y2), kp_with_conf))

        # Match pose detections to tracked persons by IoU
        for person in tracked:
            if person.frames_since_seen != 0:
                continue
            best_kp = self._best_match(person.bbox, pose_dets)
            if best_kp is None:
                continue
            results[person.track_id] = self._classify_keypoints(
                person.track_id, best_kp
            )

        # Clean up state for tracks that no longer exist
        stale = [tid for tid in self._gesture_streak if tid not in active_ids]
        for tid in stale:
            self._gesture_streak.pop(tid, None)
            self._wrist_history.pop(tid, None)

        return results

    def _best_match(
        self,
        bbox: BoundingBox,
        pose_dets: list[tuple[BoundingBox, np.ndarray]],
    ) -> np.ndarray | None:
        best_iou = 0.0
        best_kp = None
        for pbox, kp in pose_dets:
            iou = _iou(bbox, pbox)
            if iou > best_iou:
                best_iou = iou
                best_kp = kp
        if best_iou < self._config.iou_match_threshold:
            return None
        return best_kp

    def _classify_keypoints(
        self, track_id: int, kp: np.ndarray
    ) -> GestureResult:
        """kp shape (17, 3): x, y, confidence per keypoint."""
        cfg = self._config

        def get(idx: int) -> tuple[float, float] | None:
            x, y, c = kp[idx]
            if c < KEYPOINT_CONF_THRESHOLD:
                return None
            return float(x), float(y)

        nose = get(KP_NOSE)
        ls = get(KP_LEFT_SHOULDER)
        rs = get(KP_RIGHT_SHOULDER)
        lw = get(KP_LEFT_WRIST)
        rw = get(KP_RIGHT_WRIST)

        # Need at least shoulders to do anything meaningful
        if ls is None or rs is None:
            return self._update_streak(track_id, GestureType.NONE, 0.0)

        shoulder_y = (ls[1] + rs[1]) / 2
        # "Head normal position" = nose visible AND clearly above shoulders.
        # If the head has dropped to/below shoulder line, the subject is in
        # a drowning-candidate posture — don't run SOS checks on them.
        head_normal = nose is not None and nose[1] < shoulder_y - 10
        # Strict "above head" reference: clearly above both nose and shoulders.
        head_y = min(nose[1], shoulder_y) if nose is not None else shoulder_y

        # Track wrist X over time for oscillation detection
        wrist_sample = (
            lw[0] if lw else None,
            rw[0] if rw else None,
        )
        self._wrist_history[track_id].append(wrist_sample)

        left_above = head_normal and lw is not None and lw[1] < head_y
        right_above = head_normal and rw is not None and rw[1] < head_y

        # 1) Crossed arms overhead — both wrists above head, each on the
        #    OPPOSITE side of the midline from its own shoulder (forming an X).
        if left_above and right_above and lw is not None and rw is not None:
            midline = (ls[0] + rs[0]) / 2
            l_crossed = (lw[0] - midline) * (ls[0] - midline) < 0
            r_crossed = (rw[0] - midline) * (rs[0] - midline) < 0
            if l_crossed and r_crossed:
                return self._update_streak(
                    track_id, GestureType.CROSSED_ARMS_OVERHEAD, 0.95
                )

        # 2) Both arms overhead = SOS wave
        if left_above and right_above:
            return self._update_streak(
                track_id, GestureType.SOS_BOTH_ARMS, 0.9
            )

        # 3) Single-arm overhead + waving (lateral oscillation)
        if left_above ^ right_above:
            wrist_idx = 0 if left_above else 1
            amp = self._wave_amplitude(track_id, wrist_idx)
            if amp >= cfg.wave_min_amplitude_px:
                return self._update_streak(
                    track_id, GestureType.SOS_SINGLE_ARM, 0.8
                )

        # 4) Drowning posture: head AT or below shoulder line + erratic wrists
        if not head_normal:
            motion = max(
                self._wave_amplitude(track_id, 0),
                self._wave_amplitude(track_id, 1),
            )
            if motion >= cfg.drowning_motion_px:
                return self._update_streak(
                    track_id, GestureType.DROWNING_POSTURE, 0.75
                )

        return self._update_streak(track_id, GestureType.NONE, 0.0)

    def _wave_amplitude(self, track_id: int, wrist_idx: int) -> float:
        """Peak-to-peak X amplitude of a wrist over the recent window."""
        history = self._wrist_history[track_id]
        xs = [s[wrist_idx] for s in history if s[wrist_idx] is not None]
        if len(xs) < 4:
            return 0.0
        return float(max(xs) - min(xs))

    def _update_streak(
        self, track_id: int, gesture: GestureType, confidence: float
    ) -> GestureResult:
        """Hold-frame filter: only emit a gesture once it persists."""
        streaks = self._gesture_streak[track_id]
        for g in list(streaks.keys()):
            if g != gesture:
                streaks[g] = 0

        if gesture == GestureType.NONE:
            return GestureResult(GestureType.NONE, 0.0)

        streaks[gesture] += 1
        if streaks[gesture] >= self._config.min_hold_frames:
            return GestureResult(gesture, confidence)
        return GestureResult(GestureType.NONE, 0.0)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
