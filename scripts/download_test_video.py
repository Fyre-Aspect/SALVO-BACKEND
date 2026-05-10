"""Download a public sample video for testing the trained model end-to-end.

Default source: Intel IoT DevKit's MIT-licensed people-detection.mp4.
Run:
    python scripts/download_test_video.py
    python scripts/download_test_video.py --url <other-url> --out path/to.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

DEFAULT_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "people-detection.mp4"
)
DEFAULT_OUT = Path("test_videos/people-detection.mp4")


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"     -> {out}")
    with urlopen(url, timeout=60) as resp, open(out, "wb") as f:
        total = 0
        while chunk := resp.read(64 * 1024):
            f.write(chunk)
            total += len(chunk)
            print(f"  {total / 1024:8.0f} KB", end="\r")
    size_kb = out.stat().st_size / 1024
    print(f"\nDone — {size_kb:.0f} KB at {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        print(f"Already exists: {args.out} (delete to re-download)")
        return 0

    try:
        download(args.url, args.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
