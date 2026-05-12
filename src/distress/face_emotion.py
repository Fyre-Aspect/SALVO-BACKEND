"""Facial expression distress analyzer.

Uses DeepFace to detect fear/surprise from face crops of tracked persons.
Gracefully disabled when deepface is not installed.

Install the optional dependency with:
    pip install deepface tf-keras
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace  # type: ignore[import]
    _DEEPFACE_AVAILABLE = True
    logger.info("DeepFace loaded — face emotion scoring enabled")
except ImportError:
    _DEEPFACE_AVAILABLE = False
    logger.info("DeepFace not installed — face emotion scoring disabled (pip install deepface tf-keras)")

# Maps DeepFace emotion labels to distress probability
EMOTION_DISTRESS_WEIGHTS: dict[str, float] = {
    "fear":     0.90,
    "surprise": 0.45,
    "angry":    0.30,
    "disgust":  0.25,
    "sad":      0.20,
    "happy":    0.00,
    "neutral":  0.00,
}


@dataclass
class FaceEmotionConfig:
    enabled: bool = True
    # Run analysis every N frames per track (emotion changes slowly; saves CPU)
    analyze_every_n_frames: int = 5
    # Top fraction of the person bbox assumed to contain the face
    face_crop_top_ratio: float = 0.28
    # Skip crops smaller than this in pixels (noisy for distant persons)
    min_face_dimension_px: int = 32
    # Don't raise if no face found in the crop
    enforce_detection: bool = False


class FaceEmotionAnalyzer:
    """Per-track facial emotion scorer using DeepFace."""

    def __init__(self, config: FaceEmotionConfig | None = None) -> None:
        self._cfg = config or FaceEmotionConfig()
        self._available = _DEEPFACE_AVAILABLE and self._cfg.enabled
        self._last_score: dict[int, float] = {}
        self._frame_count: dict[int, int] = defaultdict(int)
        if not self._available:
            reason = "config disabled" if not self._cfg.enabled else "deepface not installed"
            logger.debug("FaceEmotionAnalyzer inactive (%s)", reason)

    @property
    def available(self) -> bool:
        return self._available

    def score(
        self,
        frame: np.ndarray,
        track_id: int,
        x1: float, y1: float, x2: float, y2: float,
    ) -> float:
        """Return emotion-based distress score [0.0, 1.0] for one tracked person.

        Returns the cached value on frames that are skipped for performance.
        """
        if not self._available:
            return 0.0

        self._frame_count[track_id] += 1
        if self._frame_count[track_id] % self._cfg.analyze_every_n_frames != 1:
            return self._last_score.get(track_id, 0.0)

        crop = self._crop_face(frame, x1, y1, x2, y2)
        if crop is None:
            return self._last_score.get(track_id, 0.0)

        try:
            result = DeepFace.analyze(
                crop,
                actions=["emotion"],
                enforce_detection=self._cfg.enforce_detection,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotions: dict[str, float] = result.get("emotion", {})
            new_score = _emotions_to_distress(emotions)
        except Exception as exc:
            logger.debug("Face emotion failed track=%d: %s", track_id, exc)
            new_score = self._last_score.get(track_id, 0.0)

        self._last_score[track_id] = new_score
        if new_score > 0.3:
            logger.debug("Face distress track=%d score=%.2f", track_id, new_score)
        return new_score

    def cleanup_track(self, track_id: int) -> None:
        self._last_score.pop(track_id, None)
        self._frame_count.pop(track_id, None)

    def _crop_face(
        self,
        frame: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
    ) -> np.ndarray | None:
        fh, fw = frame.shape[:2]
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(fw, int(x2))
        face_h = int((y2 - y1) * self._cfg.face_crop_top_ratio)
        cy2 = min(fh, cy1 + face_h)
        if (cx2 - cx1) < self._cfg.min_face_dimension_px:
            return None
        if (cy2 - cy1) < self._cfg.min_face_dimension_px:
            return None
        return frame[cy1:cy2, cx1:cx2]


def _emotions_to_distress(emotions: dict[str, float]) -> float:
    """Weighted average of emotion confidences mapped to distress probability."""
    total = sum(emotions.values())
    if total <= 0:
        return 0.0
    score = sum(
        (v / total) * EMOTION_DISTRESS_WEIGHTS.get(k.lower(), 0.0)
        for k, v in emotions.items()
    )
    return min(float(score), 1.0)
