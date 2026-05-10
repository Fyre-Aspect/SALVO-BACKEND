"""End-to-end demo: run the trained model on a sample video.

Lowers alert thresholds so a CRITICAL distress signal will fire within
the clip — proving the full chain (detector → tracker → distress score
→ alert → webhook → controller) executes against real video input.

Usage:
    python scripts/run_test_video.py
    python scripts/run_test_video.py --video path/to/video.mp4 --webhook-url http://...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline  # noqa: E402

DEFAULT_VIDEO = Path("test_videos/people-detection.mp4")


def build_demo_config(video: Path, webhook_url: str | None) -> dict:
    return {
        "camera": {"source": str(video)},
        "detector": {
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.4,
            "device": "cpu",
        },
        "tracker": {"max_disappeared": 30, "max_distance": 80.0},
        "distress": {
            "motion_weight": 0.4,
            "submersion_weight": 0.4,
            "stationary_weight": 0.2,
            "min_track_seconds": 0.5,
            "fps": 15.0,
        },
        # Production thresholds. Heuristic motion alone should NOT fire these
        # on a walking video — only a recognized SOS gesture should reach
        # the gesture_score_floor (0.95) needed to cross critical (0.85).
        "alerting": {
            "warning_threshold": 0.6,
            "critical_threshold": 0.85,
            "warning_frames": 4,
            "critical_frames": 6,
            "cooldown_seconds": 15.0,
        },
        "gesture": {
            "enabled": True,
            "model_path": "yolov8n-pose.pt",
            "device": "cpu",
            "detection_confidence": 0.4,
            "min_hold_frames": 6,
            "wave_history_frames": 12,
            "wave_min_amplitude_px": 25.0,
            "drowning_head_drop_px": 15.0,
            "drowning_motion_px": 60.0,
        },
        "serial": {"enabled": False},
        "control": {"auto_arm": False},
        "dashboard": {"enabled": False},
        "webhook": {
            "enabled": bool(webhook_url),
            "url": webhook_url or "",
            "include_snapshot": True,
            "only_critical": True,
            "source_id": "demo-drone",
        },
        "logging": {"log_dir": "logs", "snapshot_on_alert": True},
        "pipeline": {"target_fps": 30, "show_preview": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--webhook-url", default=None,
        help="If set, POST distress signals to this URL.",
    )
    parser.add_argument("--show-preview", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("demo")

    if not args.video.exists():
        log.error(
            "Video not found: %s — run scripts/download_test_video.py first",
            args.video,
        )
        return 2

    Path("logs").mkdir(exist_ok=True)
    config = build_demo_config(args.video, args.webhook_url)
    if args.show_preview:
        config["pipeline"]["show_preview"] = True

    log.info("Demo config: video=%s webhook=%s", args.video, args.webhook_url)
    Pipeline(config).run()
    log.info("Demo finished — see logs/events_*.jsonl for the recorded chain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
