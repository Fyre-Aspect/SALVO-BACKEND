# Project Doe — Test Plan & Getting Started

## Current Status

**All 31 unit tests pass.** Every core module works independently:

| Module | Tests | Status |
|--------|-------|--------|
| Event Bus | 5 | PASS |
| Person Tracker | 5 | PASS |
| Distress Features | 7 | PASS |
| Alert Manager | 4 | PASS |
| Deploy Controller | 6 | PASS |
| YOLO Detector (mocked) | 2 | PASS |
| Pipeline Integration | 2 | PASS |

---

## How to Run

### Prerequisites

```powershell
cd C:\Users\arunp\OneDrive\Desktop\Project-Doe
.\venv\Scripts\Activate.ps1
```

### 1. Run Unit Tests

```powershell
python -m pytest tests/ -v
```

This verifies all modules work correctly with synthetic data. No camera or hardware needed.

### 2. Run with Webcam (Live Detection)

```powershell
python -m src.main
```

This will:
- Open your default webcam (source 0)
- Run YOLOv8n person detection on each frame
- Track persons across frames
- Compute distress scores
- Show an annotated preview window (bounding boxes, IDs, scores)
- Start the web dashboard at `http://localhost:8000`
- Log events to `logs/`

**Controls in the preview window:**
- `q` — quit
- `a` — arm the deployment system
- `d` — disarm the deployment system

### 3. Run with a Video File

```powershell
python -m src.main --source path\to\video.mp4
```

Use any video of people near/in water. The pipeline processes each frame the same as live camera.

### 4. Run Headless (No Preview Window)

```powershell
python -m src.main --no-preview
```

Monitor via the web dashboard at `http://localhost:8000` instead.

### 5. Run Without Dashboard

```powershell
python -m src.main --no-dashboard
```

### 6. Run with Debug Logging

```powershell
python -m src.main --log-level DEBUG
```

---

## What Each Test Proves

### Test Level 1 — Unit Tests (All Working Now)

| Test | What It Proves |
|------|---------------|
| `test_event_bus.py` | Pub/sub messaging works, bad subscribers don't crash the system |
| `test_tracker.py` | Persons get unique IDs, IDs persist across frames, lost tracks get removed |
| `test_distress.py` | Smooth motion scores low, erratic motion scores high, stillness is detected, submersion dips are caught |
| `test_alert_manager.py` | Alerts only fire after sustained distress, WARNING before CRITICAL, stale tracks get cleaned up |
| `test_controller.py` | Can't deploy unless armed, can't double-deploy same person, emergency stop works |
| `test_detector.py` | YOLO integration produces correct Detection objects (mocked model) |
| `test_pipeline.py` | Detection → Tracking → Distress flows through correctly |

### Test Level 2 — Manual Integration Tests (Do These Now)

These require running the system and observing behavior:

#### Test 2A: Camera Feed Works
```
1. Run: python -m src.main
2. EXPECT: Preview window opens showing your webcam feed
3. EXPECT: Console shows "Pipeline started (target FPS: 15)"
4. VERIFY: Press 'q' to quit cleanly
```

#### Test 2B: Person Detection Works
```
1. Run: python -m src.main
2. Stand in front of the camera
3. EXPECT: Green bounding box drawn around you with "ID:0"
4. EXPECT: Console shows no errors
5. Walk out of frame, walk back in
6. VERIFY: You get a new ID (ID:1) since the old track expired
```

#### Test 2C: Multi-Person Tracking
```
1. Run: python -m src.main
2. Have 2+ people in frame
3. EXPECT: Each person gets a unique ID
4. Have one person leave and return
5. VERIFY: IDs are stable while visible, new ID after disappearing
```

#### Test 2D: Distress Scoring Appears
```
1. Run: python -m src.main --log-level DEBUG
2. Stand in front of camera and move erratically (wave arms)
3. EXPECT: Distress score (D:XX%) appears next to your bounding box
4. Stand still for several seconds
5. EXPECT: Stationary score component increases
6. CHECK: logs/ folder contains events_*.jsonl with scored events
```

#### Test 2E: Web Dashboard Works
```
1. Run: python -m src.main
2. Open browser to http://localhost:8000
3. EXPECT: Dashboard shows live video feed, tracked persons, distress scores
4. EXPECT: Dashboard updates ~10 times/second
5. VERIFY: Controller state shows "DISARMED"
```

#### Test 2F: Alert System Fires
```
1. Edit config/default.yaml temporarily:
   - Set warning_threshold to 0.1 (very low for testing)
   - Set warning_frames to 2
   - Set critical_threshold to 0.2
   - Set critical_frames to 3
2. Run: python -m src.main
3. Move around in front of camera
4. EXPECT: Bounding box turns orange (WARNING) then red (CRITICAL)
5. EXPECT: Console logs "WARNING" and "CRITICAL ALERT" messages
6. CHECK: logs/snapshots/ has alert screenshot images
7. RESET config/default.yaml to original values after testing
```

#### Test 2G: Arm/Disarm via Keyboard
```
1. Run: python -m src.main
2. Press 'a' in the preview window
3. EXPECT: Console logs "System ARMED", overlay text changes
4. Press 'd'
5. EXPECT: Console logs "System DISARMED"
```

#### Test 2H: Mock Deployment Fires
```
1. Set low alert thresholds in config (same as Test 2F)
2. Add to config: control.auto_arm: true
3. Run: python -m src.main
4. Move erratically in front of camera
5. EXPECT: Console shows "DEPLOY command sent for track_id=X"
6. EXPECT: MockSerialComm logs the DEPLOY command
7. CHECK: logs/ events file shows deploy.commanded event
```

#### Test 2I: Video File Processing
```
1. Find or record a short video with people in it
2. Run: python -m src.main --source video.mp4
3. EXPECT: Pipeline processes all frames and shows detections
4. EXPECT: Pipeline exits cleanly with "End of video stream" when done
```

#### Test 2J: Event Log Replay Check
```
1. After any run, open logs/events_*.jsonl
2. EXPECT: Each line is valid JSON with timestamp, topic, data
3. EXPECT: Topics include detection.complete, track.updated, distress.scored
4. VERIFY: Timestamps are monotonically increasing
```

### Test Level 3 — Hardware Tests (When Hardware is Ready)

#### Test 3A: Serial Communication
```
1. Connect ESP32/Arduino with firmware uploaded
2. Run: python -m src.main --serial-port COM3
3. Press 'a' to arm
4. EXPECT: Microcontroller responds with ACK:ARM
5. VERIFY: LED on microcontroller changes blink pattern
```

#### Test 3B: Full End-to-End Deploy
```
1. Connect hardware + servo mechanism
2. Set appropriate thresholds
3. Run with --serial-port and auto_arm: true
4. Simulate distress in front of camera
5. EXPECT: Servo actuates (deployment mechanism fires)
6. VERIFY: System goes to DEPLOYING → DEPLOYED state
```

---

## What's Possible Right Now (Without Hardware)

| Capability | Status | How |
|-----------|--------|-----|
| Live webcam person detection | **READY** | `python -m src.main` |
| Video file person detection | **READY** | `python -m src.main --source video.mp4` |
| Multi-person tracking with IDs | **READY** | Automatic — just run the pipeline |
| Distress scoring | **READY** | Scores appear on bounding boxes |
| Alert system (warning + critical) | **READY** | Lower thresholds in config to test |
| Web dashboard with live feed | **READY** | Open `http://localhost:8000` |
| Structured event logging (JSONL) | **READY** | Check `logs/` after any run |
| Alert snapshots (JPG) | **READY** | Check `logs/snapshots/` after alert fires |
| Mock deployment (no real hardware) | **READY** | Uses MockSerialComm by default |
| Real hardware deployment | **NEEDS HARDWARE** | ESP32 + firmware upload + servo |
| Keyboard arm/disarm | **READY** | Press 'a'/'d' in preview window |

## What to Try First

**Recommended order:**

1. `python -m pytest tests/ -v` — confirm everything passes
2. `python -m src.main` — see live detection on your webcam
3. Open `http://localhost:8000` — check the dashboard
4. `python -m src.main --source some_video.mp4` — test with a video
5. Lower thresholds in config, re-run, trigger alerts on purpose
6. Check `logs/` for structured event data

## Known Limitations (v1 Prototype)

- **Distress scoring is basic** — uses motion patterns only, not pose estimation
- **Single camera only** — no multi-camera support yet
- **No real drowning model** — detects "person" class, scores based on movement heuristics
- **Single-threaded** — FPS depends on YOLO inference speed (~15 FPS on CPU)
- **No authentication** on the dashboard (it's a local prototype)
- **No automated end-to-end tests** — integration testing is manual
