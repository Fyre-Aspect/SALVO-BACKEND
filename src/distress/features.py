from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.models.schemas import BoundingBox


def compute_motion_irregularity(positions: deque) -> float:
    """Score erratic movement patterns (0.0 = smooth, 1.0 = very erratic).

    Measures variance in direction changes between consecutive positions.
    Drowning victims exhibit rapid, irregular arm/body movements.
    """
    if len(positions) < 4:
        return 0.0

    pts = list(positions)
    # Compute direction angles between consecutive positions
    angles: list[float] = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angles.append(math.atan2(dy, dx))

    if len(angles) < 2:
        return 0.0

    # Compute angular changes
    changes: list[float] = []
    for i in range(1, len(angles)):
        diff = abs(angles[i] - angles[i - 1])
        # Normalize to [0, pi]
        diff = min(diff, 2 * math.pi - diff)
        changes.append(diff)

    if not changes:
        return 0.0

    # High variance in direction change = erratic movement
    mean_change = sum(changes) / len(changes)
    # Normalize: pi would be a complete reversal every frame
    score = min(mean_change / math.pi, 1.0)
    return score


def compute_submersion_ratio(bboxes: deque) -> float:
    """Score how often the bounding box height shrinks significantly (0.0-1.0).

    A person going under water shows a shrinking visible bbox (head dipping).
    """
    if len(bboxes) < 5:
        return 0.0

    boxes: list[BoundingBox] = list(bboxes)
    heights = [b.height for b in boxes]
    max_height = max(heights)

    if max_height < 1.0:
        return 0.0

    # Count frames where height is significantly reduced (< 60% of max)
    submersion_threshold = 0.6
    submerged_count = sum(1 for h in heights if h < max_height * submersion_threshold)
    ratio = submerged_count / len(heights)
    return min(ratio, 1.0)


def compute_stationary_duration(positions: deque, fps: float = 15.0) -> float:
    """Score how long the person has been motionless (0.0-1.0).

    A motionless person floating in water is a critical signal.
    """
    if len(positions) < 5:
        return 0.0

    pts = list(positions)
    # Compute total displacement over recent window
    displacements: list[float] = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        displacements.append(math.sqrt(dx * dx + dy * dy))

    # Average displacement per frame
    avg_displacement = sum(displacements) / len(displacements)

    # If average movement is < 2 pixels per frame, considered stationary
    stationary_threshold = 2.0
    if avg_displacement >= stationary_threshold:
        return 0.0

    # Score based on how many consecutive recent frames are stationary
    consecutive = 0
    for d in reversed(displacements):
        if d < stationary_threshold:
            consecutive += 1
        else:
            break

    # Normalize: 3 seconds of stillness = 1.0
    max_still_frames = fps * 3.0
    score = min(consecutive / max_still_frames, 1.0)
    return score
