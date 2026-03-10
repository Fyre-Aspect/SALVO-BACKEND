"""Download YOLO model weights for Project Doe."""

from ultralytics import YOLO


def main():
    print("Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print(f"Model downloaded: {model.model_name}")
    print("Ready to use.")


if __name__ == "__main__":
    main()
