"""Live webcam demo — make an SOS gesture and watch it fire.

The pipeline runs on your default webcam. Wave both arms over your head
(or perform any other recognized SOS gesture) and a CRITICAL alert will
fire within ~0.5 seconds.

Usage:
    python scripts/run_live_demo.py
    python scripts/run_live_demo.py --webhook-url https://your-site/api/distress
    python scripts/run_live_demo.py --camera 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline  # noqa: E402


def build_live_config(camera: int, webhook_url: str | None) -> dict:
    return {
        "camera": {"source": camera},
        "detector": {
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "device": "cpu",
        },
        "tracker": {"max_disappeared": 30, "max_distance": 80.0},
        "distress": {
            "motion_weight": 0.4,
            "submersion_weight": 0.4,
            "stationary_weight": 0.2,
            "min_track_seconds": 1.0,
            "fps": 15.0,
            "gesture_score_floor": 0.95,
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
        # Real-world thresholds — walking around does NOT fire these.
        # The gesture classifier bypasses them when a real SOS is seen.
        "alerting": {
            "warning_threshold": 0.6,
            "critical_threshold": 0.85,
            "warning_frames": 4,
            "critical_frames": 6,
            "cooldown_seconds": 15.0,
        },
        "serial": {"enabled": False},
        "control": {"auto_arm": True},
        "dashboard": {"enabled": True, "host": "0.0.0.0", "port": 3000},
        "webhook": {
            "enabled": bool(webhook_url),
            "url": webhook_url or "",
            "include_snapshot": True,
            "only_critical": True,
            "source_id": "live-cam",
        },
        "logging": {"log_dir": "logs", "snapshot_on_alert": True},
        "pipeline": {"target_fps": 15, "show_preview": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default 0)")
    parser.add_argument("--webhook-url", default=None,
                        help="If set, POST distress signals here on CRITICAL.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("live")

    Path("logs").mkdir(exist_ok=True)
    config = build_live_config(args.camera, args.webhook_url)
    log.info("Live demo: camera=%d webhook=%s", args.camera, args.webhook_url)
    log.info("Press 'q' in the preview window to quit.")
    Pipeline(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
