#!/usr/bin/env python3
"""Drowning detection threshold calibrator.

Run a video through the detection pipeline and get suggested values for
config/default.yaml based on the observed feature distributions.

Usage:
    python scripts/calibrate_drowning.py test_videos/your_drowning_video.mp4
    python scripts/calibrate_drowning.py test_videos/your_drowning_video.mp4 --config config/default.yaml
    python scripts/calibrate_drowning.py --list     # show videos in test_videos/
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distress.features import (
    compute_motion_irregularity,
    compute_stationary_duration,
    compute_submersion_ratio,
)
from src.distress.pose_classifier import GestureConfig, PoseGestureClassifier
from src.models.schemas import FrameData, GestureType
from src.tracking.tracker import PersonTracker
from src.vision.detector import PersonDetector


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def list_test_videos() -> None:
    folder = Path("test_videos")
    if not folder.exists():
        print("test_videos/ directory not found.")
        return
    videos = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.avi")) + sorted(folder.glob("*.mov"))
    if not videos:
        print("No video files found in test_videos/")
        return
    print("Videos in test_videos/:")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  {v.name}  ({size_mb:.1f} MB)")


def calibrate(video_path: str, config_path: str) -> None:
    config = load_config(config_path)

    print(f"\nCalibrating from: {video_path}")
    print("=" * 64)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video '{video_path}'")
        print("Tip: drop your video into test_videos/ then pass the path here.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps

    print(f"Resolution : {width}x{height}")
    print(f"Duration   : {duration_s:.1f}s  ({total_frames} frames @ {fps:.1f} FPS)")

    det_cfg = config.get("detector", {})
    detector = PersonDetector(
        det_cfg.get("model_path", "yolov8n.pt"),
        det_cfg.get("confidence_threshold", 0.5),
        det_cfg.get("device", "cpu"),
    )

    trk_cfg = config.get("tracker", {})
    tracker = PersonTracker(
        trk_cfg.get("max_disappeared", 30),
        trk_cfg.get("max_distance", 80.0),
    )

    ges_cfg = config.get("gesture", {})
    gesture_clf = PoseGestureClassifier(
        GestureConfig(
            model_path=ges_cfg.get("model_path", "yolov8n-pose.pt"),
            device=ges_cfg.get("device", "cpu"),
            detection_confidence=ges_cfg.get("detection_confidence", 0.4),
            drowning_head_drop_px=ges_cfg.get("drowning_head_drop_px", 15.0),
            drowning_motion_px=ges_cfg.get("drowning_motion_px", 60.0),
        )
    )

    motion_by_track: dict[int, list[float]] = defaultdict(list)
    sub_by_track: dict[int, list[float]] = defaultdict(list)
    stat_by_track: dict[int, list[float]] = defaultdict(list)
    drowning_frames_by_track: dict[int, int] = defaultdict(int)
    gesture_hits: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    frame_id = 0
    print("\nProcessing", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = FrameData(frame=frame, timestamp=time.time(), frame_id=frame_id)
        detections = detector.detect(frame_data)
        tracked = tracker.update(detections)
        gestures = gesture_clf.classify(frame_data, tracked)

        for person in tracked:
            if person.frames_since_seen > 0:
                continue
            tid = person.track_id

            motion_by_track[tid].append(compute_motion_irregularity(person.positions))
            sub_by_track[tid].append(compute_submersion_ratio(person.bboxes))
            stat_by_track[tid].append(compute_stationary_duration(person.positions, fps))

            gest = gestures.get(tid)
            if gest and gest.gesture != GestureType.NONE:
                gesture_hits[tid][gest.gesture.value] += 1
                if gest.gesture == GestureType.DROWNING_POSTURE:
                    drowning_frames_by_track[tid] += 1

        frame_id += 1
        if frame_id % 30 == 0:
            print(".", end="", flush=True)

    cap.release()
    print(f" done ({frame_id} frames)\n")

    if not motion_by_track:
        print("No persons detected. Check that the video contains visible people.")
        return

    print(f"Detected {len(motion_by_track)} track(s)")
    print("=" * 64)

    all_motion: list[float] = []
    all_sub: list[float] = []
    max_drowning_ratio = 0.0

    for tid in sorted(motion_by_track):
        ms = motion_by_track[tid]
        ss = sub_by_track[tid]
        st = stat_by_track[tid]
        n = len(ms)
        d_frames = drowning_frames_by_track.get(tid, 0)
        d_ratio = d_frames / n if n else 0.0
        max_drowning_ratio = max(max_drowning_ratio, d_ratio)

        print(f"\nTrack #{tid}  ({n} frames, {n/fps:.1f}s):")
        print(f"  motion_irregularity  mean={np.mean(ms):.3f}  p75={np.percentile(ms,75):.3f}  max={np.max(ms):.3f}")
        print(f"  submersion_ratio     mean={np.mean(ss):.3f}  p75={np.percentile(ss,75):.3f}  max={np.max(ss):.3f}")
        print(f"  stationary_duration  mean={np.mean(st):.3f}  p75={np.percentile(st,75):.3f}  max={np.max(st):.3f}")
        if gesture_hits[tid]:
            for g, count in sorted(gesture_hits[tid].items()):
                print(f"  gesture [{g}]: {count} frames ({count/n:.1%})")
        else:
            print("  gesture: none detected")

        all_motion.extend(ms)
        all_sub.extend(ss)

    print("\n" + "=" * 64)
    print("SUGGESTED CONFIG CHANGES")
    print("=" * 64)

    cur_ges = config.get("gesture", {})
    cur_alert = config.get("alerting", {})
    changes: dict[str, tuple[str, float, float]] = {}  # key -> (section.key, old, new)

    if max_drowning_ratio > 0.05:
        old_hdrop = cur_ges.get("drowning_head_drop_px", 15.0)
        new_hdrop = max(8.0, old_hdrop - 3.0)
        if new_hdrop != old_hdrop:
            changes["gesture.drowning_head_drop_px"] = ("gesture.drowning_head_drop_px", old_hdrop, new_hdrop)

        old_motion_px = cur_ges.get("drowning_motion_px", 60.0)
        new_motion_px = max(40.0, old_motion_px - 10.0)
        if new_motion_px != old_motion_px:
            changes["gesture.drowning_motion_px"] = ("gesture.drowning_motion_px", old_motion_px, new_motion_px)

    if all_motion:
        p90_motion = float(np.percentile(all_motion, 90))
        p90_sub = float(np.percentile(all_sub, 90))

        if p90_motion > 0.55 or p90_sub > 0.55:
            old_warn = cur_alert.get("warning_threshold", 0.6)
            old_crit = cur_alert.get("critical_threshold", 0.85)
            new_warn = round(max(0.45, old_warn - 0.1), 2)
            new_crit = round(max(0.70, old_crit - 0.1), 2)
            if new_warn != old_warn:
                changes["alerting.warning_threshold"] = ("alerting.warning_threshold", old_warn, new_warn)
            if new_crit != old_crit:
                changes["alerting.critical_threshold"] = ("alerting.critical_threshold", old_crit, new_crit)

    if changes:
        for key, (label, old, new) in changes.items():
            print(f"  {label}: {new}  (was {old})")
        print("\nApply by editing config/default.yaml with the values above,")
        print("then re-run the pipeline.")
    else:
        print("  Current thresholds look well-calibrated for this video.")
        print("  No changes needed.")

    if max_drowning_ratio == 0.0 and total_frames > 50:
        print("\nNote: no drowning posture detected in this video.")
        print("If this IS a drowning video, consider lowering gesture.drowning_motion_px")
        print("and gesture.drowning_head_drop_px manually to increase sensitivity.")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate drowning detection thresholds from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to video file (e.g. test_videos/drowning.mp4)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Config file to read thresholds from (default: config/default.yaml)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available videos in test_videos/ and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_test_videos()
        return

    if not args.video:
        print("Usage: python scripts/calibrate_drowning.py <video_path>")
        print("       python scripts/calibrate_drowning.py --list")
        sys.exit(1)

    calibrate(args.video, args.config)


if __name__ == "__main__":
    main()
