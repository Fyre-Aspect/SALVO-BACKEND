"""Download YOLO model weights for Project Doe."""

from ultralytics import YOLO


def main():
    print("Downloading YOLOv8n (person detection)...")
    YOLO("yolov8n.pt")
    print("Downloading YOLOv8n-pose (gesture / SOS detection)...")
    YOLO("yolov8n-pose.pt")
    print("Ready to use.")


if __name__ == "__main__":
    main()
