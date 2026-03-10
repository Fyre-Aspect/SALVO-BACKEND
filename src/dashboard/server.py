from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.event_bus import EventBus

logger = logging.getLogger(__name__)

app = FastAPI(title="Project Doe Dashboard")

# Shared state updated from pipeline thread
_dashboard_state: dict[str, Any] = {
    "frame_jpeg_b64": None,
    "detections": [],
    "tracks": [],
    "distress_events": [],
    "alerts": [],
    "controller_state": "DISARMED",
    "fps": 0.0,
}
_state_lock = threading.Lock()
_connected_clients: list[WebSocket] = []

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Project Doe — Drowning Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }
        .header { background: #16213e; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.3em; color: #0f9; }
        .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }
        .status-armed { background: #e74c3c; }
        .status-disarmed { background: #555; }
        .container { display: flex; gap: 16px; padding: 16px; height: calc(100vh - 56px); }
        .video-panel { flex: 2; background: #0f3460; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        .video-panel img { max-width: 100%; max-height: 100%; }
        .side-panel { flex: 1; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
        .card { background: #16213e; border-radius: 8px; padding: 16px; }
        .card h3 { color: #0f9; margin-bottom: 8px; font-size: 0.95em; }
        .metric { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1a1a3e; }
        .alert-item { padding: 8px; margin: 4px 0; border-radius: 4px; }
        .alert-warning { background: #f39c1233; border-left: 3px solid #f39c12; }
        .alert-critical { background: #e74c3c33; border-left: 3px solid #e74c3c; }
        .no-data { color: #666; font-style: italic; }
        #fps { color: #0f9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Project Doe — Drowning Detection System</h1>
        <div>
            <span id="fps">-- FPS</span> &nbsp;
            <span id="ctrl-state" class="status-badge status-disarmed">DISARMED</span>
        </div>
    </div>
    <div class="container">
        <div class="video-panel">
            <img id="video-feed" alt="No video feed" />
        </div>
        <div class="side-panel">
            <div class="card">
                <h3>Tracked Persons</h3>
                <div id="tracks-list"><p class="no-data">No persons detected</p></div>
            </div>
            <div class="card">
                <h3>Distress Scores</h3>
                <div id="distress-list"><p class="no-data">No distress events</p></div>
            </div>
            <div class="card">
                <h3>Alerts</h3>
                <div id="alerts-list"><p class="no-data">No alerts</p></div>
            </div>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            // Video feed
            if (data.frame_jpeg_b64) {
                document.getElementById('video-feed').src = 'data:image/jpeg;base64,' + data.frame_jpeg_b64;
            }

            // FPS
            document.getElementById('fps').textContent = (data.fps || 0).toFixed(1) + ' FPS';

            // Controller state
            const ctrlEl = document.getElementById('ctrl-state');
            ctrlEl.textContent = data.controller_state || 'DISARMED';
            ctrlEl.className = 'status-badge ' + (data.controller_state === 'ARMED' ? 'status-armed' : 'status-disarmed');

            // Tracks
            const tracksList = document.getElementById('tracks-list');
            if (data.tracks && data.tracks.length > 0) {
                tracksList.innerHTML = data.tracks.map(t =>
                    `<div class="metric"><span>Person #${t.track_id}</span><span>${t.frames_since_seen === 0 ? 'Visible' : 'Lost'}</span></div>`
                ).join('');
            } else {
                tracksList.innerHTML = '<p class="no-data">No persons detected</p>';
            }

            // Distress
            const distressList = document.getElementById('distress-list');
            if (data.distress_events && data.distress_events.length > 0) {
                distressList.innerHTML = data.distress_events.map(d =>
                    `<div class="metric"><span>Person #${d.track_id}</span><span style="color:${d.distress_score > 0.7 ? '#e74c3c' : d.distress_score > 0.4 ? '#f39c12' : '#0f9'}">${(d.distress_score * 100).toFixed(0)}%</span></div>`
                ).join('');
            } else {
                distressList.innerHTML = '<p class="no-data">No distress events</p>';
            }

            // Alerts
            const alertsList = document.getElementById('alerts-list');
            if (data.alerts && data.alerts.length > 0) {
                alertsList.innerHTML = data.alerts.map(a =>
                    `<div class="alert-item alert-${a.level}"><strong>${a.level.toUpperCase()}</strong> — Person #${a.track_id} (${(a.distress_score * 100).toFixed(0)}%)</div>`
                ).join('');
            } else {
                alertsList.innerHTML = '<p class="no-data">No alerts</p>';
            }
        };
        ws.onclose = function() { console.log('WebSocket closed'); };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            with _state_lock:
                state_copy = dict(_dashboard_state)
            await websocket.send_text(json.dumps(state_copy, default=str))
            await asyncio.sleep(0.1)  # ~10 updates/sec to browser
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)


def update_dashboard_state(key: str, value: Any) -> None:
    """Thread-safe update of dashboard state from the pipeline thread."""
    with _state_lock:
        _dashboard_state[key] = value


def encode_frame_to_b64(frame: np.ndarray, scale: float = 0.5) -> str:
    """Encode an OpenCV frame to base64 JPEG for WebSocket transport."""
    if scale != 1.0:
        frame = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("ascii")


class DashboardServer:
    """Runs the FastAPI dashboard in a background thread."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        config = uvicorn.Config(
            app, host=self._host, port=self._port, log_level="warning"
        )
        server = uvicorn.Server(config)
        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()
        logger.info("Dashboard running at http://%s:%d", self._host, self._port)

    def update_frame(self, frame: np.ndarray) -> None:
        b64 = encode_frame_to_b64(frame)
        update_dashboard_state("frame_jpeg_b64", b64)

    def update_tracks(self, tracks: list) -> None:
        update_dashboard_state(
            "tracks", [t.to_dict() for t in tracks]
        )

    def update_distress(self, events: list) -> None:
        update_dashboard_state(
            "distress_events", [e.to_dict() for e in events]
        )

    def update_alerts(self, alerts: list) -> None:
        update_dashboard_state(
            "alerts", [a.to_dict() for a in alerts]
        )

    def update_controller_state(self, state: str) -> None:
        update_dashboard_state("controller_state", state)

    def update_fps(self, fps: float) -> None:
        update_dashboard_state("fps", fps)
