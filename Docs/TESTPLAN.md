# Project Doe — Test Plan

Two ways to verify the system:

1. **Test video** — proves walking does NOT false-trigger the model.
2. **Live webcam** — proves SOS gestures DO trigger a CRITICAL distress signal.

---

## Pre-flight (one-time) This is for Parth Arnie and Jernigan

```powershell
cd C:\Users\aamir\Coding\Project-Doe
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_model.py        # fetches yolov8n.pt + yolov8n-pose.pt
python scripts/download_test_video.py   # fetches test_videos/people-detection.mp4
python -m pytest tests/ -v              # expect: 42 passed
```

If any unit test fails, stop and fix before running the demos.

---

## Test A — Test video (negative control) 

The test video is **`test_videos/people-detection.mp4`** — an Intel
MIT-licensed clip of people walking down a street. There is no SOS
gesture in it, so this test proves that **ordinary walking does not
fire a distress alert**.

### Run it

```powershell
python scripts/run_test_video.py
```

### Pass conditions

- Console shows `Pipeline started` and processes the full video.
- Console shows `End of video stream` at the end.
- **No `CRITICAL ALERT` lines appear** (since nobody is signalling).
- Possibly a few `WARNING` lines for very brief motion-irregularity
  spikes — these are fine; they do not deploy.
- Pipeline exits cleanly.

### What this proves

The model's gesture classifier is silent on plain walking. If you see
`[SOS_*]` or `[DROWNING_*]` labels in the logs from this video, that's
a false positive bug.

### Optional: same test, but POST every alert to a real HTTP endpoint

```powershell
python scripts/run_test_video.py --webhook-url https://httpbin.org/post
```

Useful to confirm the webhook delivery path works against a live HTTP
server. (`httpbin.org/post` is a free echo service that returns 200.)

---

## Test B — Live webcam (positive tests)

Run:

```powershell
python scripts/run_live_demo.py
```

A preview window opens showing your webcam. The operator dashboard is at
`http://localhost:8000`.

### Test B1 — Walking around does NOT fire

1. Walk around in front of the camera, arms at your sides, ~30 s.
2. **Expected:** bounding box stays **green**, no `WARNING` or
   `CRITICAL ALERT` lines, no `[SOS_*]` label.

### Test B2 — Both arms overhead → SOS

1. Stand still in frame.
2. Raise **both arms straight up** so both wrists are clearly above
   your head. Hold ~1 second.
3. **Expected within ~0.5–1s:**
   - Box turns **red**, label reads `ID:0 D:95% [SOS_BOTH_ARMS]`.
   - Console: `CRITICAL ALERT: track_id=0 …` and
     `DEPLOY command sent for track_id=0`.

### Test B3 — Crossed arms overhead (universal "X" help signal)

1. Wait ~15 s for cooldown.
2. Raise both arms overhead and **cross them at the wrists** in an X.
3. **Expected:** label reads `[CROSSED_ARMS_OVERHEAD]`, CRITICAL fires.

### Test B4 — Single-arm wave

1. Wait for cooldown.
2. Raise **one arm** overhead and wave it side-to-side.
3. **Expected:** label reads `[SOS_SINGLE_ARM]`, CRITICAL fires.
4. Note: just holding one arm still overhead does **not** fire — the
   single-arm rule requires lateral oscillation (so stretching doesn't
   misfire).

### Test B5 — Drowning posture

1. Wait for cooldown.
2. Lean far forward so your head drops to/below shoulder height in
   frame, and shake your hands rapidly side-to-side at hip level.
3. **Expected:** label reads `[DROWNING_POSTURE]`, CRITICAL fires.

### Test B6 — Webhook delivery (live)

```powershell
python scripts/run_live_demo.py --webhook-url https://httpbin.org/post
```

Trigger any of B2–B5. **Expected console line per CRITICAL:**

```
Distress signal delivered: track=0 status=200
```

When your website is live, replace `https://httpbin.org/post` with its
endpoint (set `webhook.enabled: true` and `webhook.url:` in
[config/default.yaml](../config/default.yaml)).

---

## Preview window keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `a` | Arm the deployment system |
| `d` | Disarm |

---

## Tuning if results are off

In [config/default.yaml](../config/default.yaml) under `gesture:`:

| Symptom | Knob | Direction |
|---------|------|-----------|
| False alarms on normal motion | `min_hold_frames` | Raise (6 → 10) |
| False alarms on hand gestures | `wave_min_amplitude_px` | Raise (25 → 40) |
| Real SOS not detected | `min_hold_frames` | Lower (6 → 3) |
| Real SOS not detected | `detection_confidence` | Lower (0.4 → 0.3) |
| Real SOS not detected | Lighting / framing | Get full upper body in frame |

---

## Where to look afterwards

- `logs\events_*.jsonl` — one JSON per event, includes `gesture` field on alerts.
- `logs\snapshots\` — JPEGs captured at the moment of each alert.
- `logs\system.log` — full system log.
