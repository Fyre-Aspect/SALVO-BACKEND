# Project Doe — AI Drowning Detection System

A prototype system that detects persons in distress in water and deploys a
flotation device. This repo (**SALVO-BACKEND**) is one half of the SALVO
project — the other half is the **SALVO-PWA-APP** webapp, a separate repo
that receives distress alerts and displays them to a human responder.

## Core Flow

```
Camera → Person Detection (YOLO) → Tracking → Distress Scoring → Alert → Deploy Flotation Device
                                                        │
                                                        └─▶ Webhook POST ──▶ SALVO-PWA-APP (/api/distress)
```

## How the two repos fit together

| Repo | Stack | Runs on | Role |
|------|-------|---------|------|
| **SALVO-BACKEND** (this repo) | Python, OpenCV, YOLO, FastAPI | `http://localhost:8000` (operator dashboard) | Reads camera/video, detects + tracks people, scores distress, fires alerts, drives the deployment hardware. |
| **SALVO-PWA-APP** | Next.js, React | `http://localhost:3000` | Responder-facing webapp. Receives distress alerts over a webhook and streams them to the UI in real time. |

The two repos talk over one HTTP call: on a `CRITICAL` alert, this backend
`POST`s a JSON payload (see `src/notifier/webhook.py`) to
`SALVO-PWA-APP`'s `/api/distress` route. That route pushes the event onto an
in-memory bus (`src/lib/distress-store.ts` in the webapp) and the dashboard
page subscribes to it via Server-Sent Events at `/api/distress/stream`.

This backend also serves its own lightweight **operator dashboard**
(raw camera feed + live overlay, via FastAPI/WebSocket) — that's a separate,
simpler view meant for whoever is standing next to the hardware, and is
unrelated to the webapp. **It intentionally runs on port 8000, not 3000**,
so it never collides with the webapp's Next.js dev server. Don't move it
back to 3000 without also moving the webapp.

## Prerequisites

- Python 3.10+ (tested on 3.11)
- Node.js 20+ and npm (for the webapp)
- A webcam, or a test video file, for the backend
- Windows/macOS/Linux — commands below show PowerShell first, POSIX second

## Repo layout (as checked out on this machine)

```
Jeevo Projects/
  SALVO BACKEND/SALVO-BACKEND/   ← this repo
  SALVO APP/SALVO-PWA-APP/       ← the webapp repo
```

They don't need to live next to each other — the only coupling is the
webhook URL/port, which defaults to `localhost` for local dev.

## Quick Start — Backend (this repo)

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download YOLO model weights (person detection + pose/gesture)
python scripts/download_model.py

# Run with default camera
python -m src.main

# Run with video file
python -m src.main --source path/to/video.mp4

# Run with simulated serial device (mock hardware)
python scripts/simulate_serial.py &
python -m src.main --serial-port COM3
```

The operator dashboard (if enabled in config) comes up at
`http://localhost:8000`.

## Quick Start — Webapp (SALVO-PWA-APP)

From the webapp repo root:

```powershell
npm install

# Optional — only needed for real Firebase auth/data. Without it, the app
# runs in a mock-auth mode using localStorage, which is fine for local dev.
copy .env.local.example .env.local

npm run dev
```

The webapp comes up at `http://localhost:3000`. `/api/distress` is the
webhook receiver this backend posts to; `/dashboard` is the responder UI
that subscribes to those events live.

## Running both together locally

This is the setup you want for an end-to-end test (backend detects → webapp
shows the alert):

1. **Start the webapp first** so port 3000 is claimed by it:
   ```powershell
   cd path\to\SALVO-PWA-APP
   npm run dev
   ```
2. **Start the backend**, pointing its webhook at the webapp. This is
   already the default in `config/default.yaml`:
   ```yaml
   webhook:
     enabled: true
     url: "http://localhost:3000/api/distress"
   ```
   Then run one of:
   ```powershell
   # Real webcam, waves/SOS gestures trigger alerts
   python scripts/run_live_demo.py

   # Canned test video (downloads a walking-only clip; proves no false positives)
   python scripts/download_test_video.py
   python scripts/run_test_video.py --webhook-url http://localhost:3000/api/distress
   ```
3. Open `http://localhost:3000/dashboard` in a browser (log in — mock auth
   accepts anything) and trigger a `CRITICAL` alert (e.g. raise both arms
   overhead in front of the webcam). It should appear on the webapp within
   about a second.
4. The backend's own operator view is at `http://localhost:8000` — a raw
   camera feed with bounding boxes, independent of the webapp.

If you only want to develop the backend in isolation, set
`webhook.enabled: false` in `config/default.yaml` (or don't run the webapp
at all — a failed webhook POST just logs a warning and doesn't block
detection).

## Configuration

Edit `config/default.yaml` to adjust thresholds, model paths, and hardware
settings. Key sections:

- `detector` / `gesture` — YOLO model paths, confidence thresholds
- `distress` — how motion/submersion/gesture/face-emotion scores are weighted
- `alerting` — score thresholds and consecutive-frame requirements before firing
- `webhook` — where alerts get POSTed (see above)
- `dashboard` — the backend's own operator view (host/port)
- `serial` — microcontroller connection for physical deployment

## Testing

```bash
pytest tests/ -v
```

For manual/scenario testing (test video negative control, live webcam
positive control, webhook delivery), see [`Docs/TESTPLAN.md`](TESTPLAN.md).

## Project Structure

```
src/
  vision/       Frame capture + YOLO person detection
  tracking/     Multi-object centroid tracker
  distress/     Distress scoring and feature extraction
  alerting/     Alert thresholds and state management
  control/      Deployment logic and serial communication
  dashboard/    FastAPI operator dashboard with WebSocket
  notifier/     Webhook delivery to the SALVO-PWA-APP webapp
  logging_/     Structured JSON event logging
  models/       Shared data structures
scripts/        download_model, download_test_video, run_test_video,
                run_live_demo, calibrate_drowning, simulate_serial
```

## Hardware

The system communicates with an ESP32/Arduino via serial (115200 baud).
See `firmware/deploy_controller/` for the microcontroller sketch. Use
`scripts/simulate_serial.py` to develop without physical hardware attached.

## Troubleshooting

- **Webapp dev server won't bind to 3000 / picks 3001 instead**: something
  else is already on 3000. Next.js auto-increments the port on conflict —
  check its terminal output for the actual port, and update
  `webhook.url` / your browser URL to match.
- **Backend dashboard fails to start**: check `dashboard.port` in
  `config/default.yaml` isn't already in use (defaults to 8000, deliberately
  separate from the webapp's 3000).
- **Webhook POSTs never show up in the webapp**: confirm the webapp is
  running and `/api/distress` is reachable at the URL in `webhook.url`;
  `only_critical: true` means `WARNING`-level alerts are never sent, only
  `CRITICAL`.
