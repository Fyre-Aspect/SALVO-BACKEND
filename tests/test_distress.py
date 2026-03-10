"""Tests for distress feature extraction and analyzer."""

import math
from collections import deque

from src.distress.features import (
    compute_motion_irregularity,
    compute_stationary_duration,
    compute_submersion_ratio,
)
from src.models.schemas import BoundingBox, TrackedPerson


def test_motion_irregularity_smooth():
    """Smooth linear motion should score low."""
    positions = deque(maxlen=30)
    for i in range(20):
        positions.append([100 + i * 5, 200])  # Straight line
    score = compute_motion_irregularity(positions)
    assert score < 0.2


def test_motion_irregularity_erratic():
    """Zigzag motion should score higher."""
    positions = deque(maxlen=30)
    for i in range(20):
        y = 200 + (30 if i % 2 == 0 else -30)
        positions.append([100 + i * 2, y])
    score = compute_motion_irregularity(positions)
    assert score > 0.3


def test_motion_irregularity_insufficient_data():
    positions = deque([[100, 200], [110, 210]], maxlen=30)
    score = compute_motion_irregularity(positions)
    assert score == 0.0


def test_submersion_ratio_no_submersion():
    bboxes = deque(maxlen=30)
    for _ in range(10):
        bboxes.append(BoundingBox(0, 0, 50, 200))  # Constant height
    score = compute_submersion_ratio(bboxes)
    assert score == 0.0


def test_submersion_ratio_frequent_dips():
    bboxes = deque(maxlen=30)
    for i in range(10):
        h = 200 if i % 2 == 0 else 80  # Alternates between full and short
        bboxes.append(BoundingBox(0, 0, 50, h))
    score = compute_submersion_ratio(bboxes)
    assert score > 0.3


def test_stationary_duration_moving():
    positions = deque(maxlen=30)
    for i in range(20):
        positions.append([100 + i * 10, 200])  # Moving steadily
    score = compute_stationary_duration(positions, fps=15.0)
    assert score == 0.0


def test_stationary_duration_still():
    positions = deque(maxlen=90)
    for _ in range(60):
        positions.append([100.0, 200.0])  # Completely still
    score = compute_stationary_duration(positions, fps=15.0)
    assert score > 0.5
