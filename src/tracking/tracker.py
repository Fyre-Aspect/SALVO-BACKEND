from __future__ import annotations

import logging
import time
from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import cdist

from src.models.schemas import BoundingBox, Detection, TrackedPerson

logger = logging.getLogger(__name__)


class PersonTracker:
    """Centroid-based multi-object tracker with persistent IDs."""

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 80.0,
    ) -> None:
        self._next_id = 0
        self._tracks: OrderedDict[int, TrackedPerson] = OrderedDict()
        self._max_disappeared = max_disappeared
        self._max_distance = max_distance

    @property
    def tracks(self) -> dict[int, TrackedPerson]:
        return dict(self._tracks)

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        if not detections:
            return self._mark_all_disappeared()

        det_centroids = np.array([d.bbox.center for d in detections])
        det_bboxes = [d.bbox for d in detections]
        now = detections[0].timestamp

        if not self._tracks:
            for i, det in enumerate(detections):
                self._register(det_centroids[i], det_bboxes[i], now)
            return list(self._tracks.values())

        track_ids = list(self._tracks.keys())
        track_centroids = np.array(
            [self._tracks[tid].bbox.center for tid in track_ids]
        )

        # Compute pairwise distances between existing tracks and new detections
        dist_matrix = cdist(track_centroids, det_centroids)

        # Greedy assignment: pick the smallest distance first
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        matched_pairs: list[tuple[int, int]] = []

        flat_indices = dist_matrix.flatten().argsort()
        for flat_idx in flat_indices:
            row = int(flat_idx // dist_matrix.shape[1])
            col = int(flat_idx % dist_matrix.shape[1])
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self._max_distance:
                break
            matched_pairs.append((row, col))
            used_rows.add(row)
            used_cols.add(col)

        # Update matched tracks
        for row, col in matched_pairs:
            tid = track_ids[row]
            track = self._tracks[tid]
            track.bbox = det_bboxes[col]
            track.positions.append(det_centroids[col].tolist())
            track.bboxes.append(det_bboxes[col])
            track.frames_since_seen = 0
            track.last_seen = now

        # Increment disappeared counter for unmatched tracks
        for row in range(len(track_ids)):
            if row not in used_rows:
                tid = track_ids[row]
                self._tracks[tid].frames_since_seen += 1
                if self._tracks[tid].frames_since_seen > self._max_disappeared:
                    del self._tracks[tid]

        # Register new tracks for unmatched detections
        for col in range(len(detections)):
            if col not in used_cols:
                self._register(det_centroids[col], det_bboxes[col], now)

        return list(self._tracks.values())

    def _register(
        self, centroid: np.ndarray, bbox: BoundingBox, timestamp: float
    ) -> None:
        track = TrackedPerson(
            track_id=self._next_id,
            bbox=bbox,
            first_seen=timestamp,
            last_seen=timestamp,
        )
        track.positions.append(centroid.tolist())
        track.bboxes.append(bbox)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _mark_all_disappeared(self) -> list[TrackedPerson]:
        to_remove: list[int] = []
        for tid, track in self._tracks.items():
            track.frames_since_seen += 1
            if track.frames_since_seen > self._max_disappeared:
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]
        return list(self._tracks.values())
