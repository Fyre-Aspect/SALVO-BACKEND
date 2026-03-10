"""Project Doe — AI Drowning Detection System.

Usage:
    python -m src.main
    python -m src.main --source path/to/video.mp4
    python -m src.main --config config/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.pipeline import Pipeline


def setup_logging(level: str = "INFO") -> None:
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/system.log", mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logging.warning("Config file not found: %s — using defaults", config_path)
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Doe — Drowning Detection")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Camera index (int) or video file path",
    )
    parser.add_argument(
        "--serial-port", type=str, default=None,
        help="Serial port for microcontroller (e.g., COM3, /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable local OpenCV preview window",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable web dashboard",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = load_config(args.config)

    # CLI overrides
    if args.source is not None:
        try:
            config.setdefault("camera", {})["source"] = int(args.source)
        except ValueError:
            config.setdefault("camera", {})["source"] = args.source

    if args.serial_port:
        config.setdefault("serial", {})["enabled"] = True
        config["serial"]["port"] = args.serial_port

    if args.no_preview:
        config.setdefault("pipeline", {})["show_preview"] = False

    if args.no_dashboard:
        config.setdefault("dashboard", {})["enabled"] = False

    logger.info("Starting Project Doe")
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
