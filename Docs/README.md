# Project Doe — AI Drowning Detection System

A prototype system that detects persons in distress in water and deploys a flotation device.

## Core Flow

```
Camera → Person Detection (YOLO) → Tracking → Distress Scoring → Alert → Deploy Flotation Device
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download YOLO model
python scripts/download_model.py

# Run with default camera
python -m src.main

# Run with video file
python -m src.main --source path/to/video.mp4

# Run with simulated serial device
python scripts/simulate_serial.py &
python -m src.main --serial-port COM3
```

## Configuration

Edit `config/default.yaml` to adjust thresholds, model paths, and hardware settings.

## Project Structure

```
src/
  vision/       Frame capture + YOLO person detection
  tracking/     Multi-object centroid tracker
  distress/     Distress scoring and feature extraction
  alerting/     Alert thresholds and state management
  control/      Deployment logic and serial communication
  dashboard/    FastAPI web dashboard with WebSocket
  logging_/     Structured JSON event logging
  models/       Shared data structures
```

## Testing

```bash
pytest tests/ -v
```

## Hardware

The system communicates with an ESP32/Arduino via serial (115200 baud).
See `firmware/deploy_controller/` for the microcontroller sketch.
