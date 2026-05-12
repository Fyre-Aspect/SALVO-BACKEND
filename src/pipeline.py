from __future__ import annotations

import logging
import time

import cv2

from src.alerting.alert_manager import AlertConfig, AlertManager
from src.control.controller import DeployController
from src.control.serial_comm import BaseSerialComm, MockSerialComm
from src.dashboard.server import DashboardServer
from src.distress.analyzer import DistressAnalyzer, DistressConfig
from src.distress.face_emotion import FaceEmotionAnalyzer, FaceEmotionConfig
from src.distress.pose_classifier import GestureConfig, PoseGestureClassifier
from src.event_bus import EventBus
from src.logging_.event_logger import EventLogger
from src.models.schemas import Alert, AlertLevel, FrameData
from src.notifier.webhook import WebhookConfig, WebhookNotifier
from src.tracking.tracker import PersonTracker
from src.vision.camera import FrameGrabber
from src.vision.detector import PersonDetector

logger = logging.getLogger(__name__)


class Pipeline:
    """Main processing pipeline — orchestrates all modules frame-by-frame."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._running = False

        # Event bus
        self._event_bus = EventBus()

        # Vision
        source = config.get("camera", {}).get("source", 0)
        self._grabber = FrameGrabber(source)

        model_path = config.get("detector", {}).get("model_path", "yolov8n.pt")
        confidence = config.get("detector", {}).get("confidence_threshold", 0.5)
        device = config.get("detector", {}).get("device", "cpu")
        self._detector = PersonDetector(model_path, confidence, device)

        # Tracking
        max_disappeared = config.get("tracker", {}).get("max_disappeared", 30)
        max_distance = config.get("tracker", {}).get("max_distance", 80.0)
        self._tracker = PersonTracker(max_disappeared, max_distance)

        # Distress
        distress_cfg = config.get("distress", {})
        self._analyzer = DistressAnalyzer(
            DistressConfig(
                motion_weight=distress_cfg.get("motion_weight", 0.4),
                submersion_weight=distress_cfg.get("submersion_weight", 0.4),
                stationary_weight=distress_cfg.get("stationary_weight", 0.2),
                face_emotion_weight=distress_cfg.get("face_emotion_weight", 0.0),
                min_track_seconds=distress_cfg.get("min_track_seconds", 1.0),
                fps=distress_cfg.get("fps", 15.0),
                gesture_score_floor=distress_cfg.get(
                    "gesture_score_floor", 0.95
                ),
            )
        )

        # Facial expression distress analyzer (requires deepface — optional)
        face_cfg = config.get("face_emotion", {})
        self._face_emotion = FaceEmotionAnalyzer(
            FaceEmotionConfig(
                enabled=face_cfg.get("enabled", True),
                analyze_every_n_frames=face_cfg.get("analyze_every_n_frames", 5),
                face_crop_top_ratio=face_cfg.get("face_crop_top_ratio", 0.28),
                min_face_dimension_px=face_cfg.get("min_face_dimension_px", 32),
                enforce_detection=face_cfg.get("enforce_detection", False),
            )
        )

        # Pose-based gesture classifier (SOS, drowning posture, etc.)
        gesture_cfg = config.get("gesture", {})
        self._gesture_enabled = gesture_cfg.get("enabled", True)
        self._gesture_classifier: PoseGestureClassifier | None = None
        if self._gesture_enabled:
            self._gesture_classifier = PoseGestureClassifier(
                GestureConfig(
                    model_path=gesture_cfg.get(
                        "model_path", "yolov8n-pose.pt"
                    ),
                    device=gesture_cfg.get("device", "cpu"),
                    detection_confidence=gesture_cfg.get(
                        "detection_confidence", 0.4
                    ),
                    min_hold_frames=gesture_cfg.get("min_hold_frames", 6),
                    wave_history_frames=gesture_cfg.get(
                        "wave_history_frames", 12
                    ),
                    wave_min_amplitude_px=gesture_cfg.get(
                        "wave_min_amplitude_px", 25.0
                    ),
                    drowning_head_drop_px=gesture_cfg.get(
                        "drowning_head_drop_px", 15.0
                    ),
                    drowning_motion_px=gesture_cfg.get(
                        "drowning_motion_px", 60.0
                    ),
                )
            )

        # Alerting
        alert_cfg = config.get("alerting", {})
        self._alert_manager = AlertManager(
            AlertConfig(
                warning_threshold=alert_cfg.get("warning_threshold", 0.4),
                critical_threshold=alert_cfg.get("critical_threshold", 0.7),
                warning_frames=alert_cfg.get("warning_frames", 5),
                critical_frames=alert_cfg.get("critical_frames", 10),
                cooldown_seconds=alert_cfg.get("cooldown_seconds", 30.0),
            )
        )

        # Control
        serial_cfg = config.get("serial", {})
        serial_comm: BaseSerialComm
        if serial_cfg.get("enabled", False):
            from src.control.serial_comm import SerialComm

            serial_comm = SerialComm(
                port=serial_cfg["port"],
                baud_rate=serial_cfg.get("baud_rate", 115200),
            )
        else:
            serial_comm = MockSerialComm()
        self._controller = DeployController(serial_comm)
        self._serial_comm = serial_comm

        # Dashboard
        dash_cfg = config.get("dashboard", {})
        self._dashboard: DashboardServer | None = None
        if dash_cfg.get("enabled", True):
            self._dashboard = DashboardServer(
                host=dash_cfg.get("host", "0.0.0.0"),
                port=dash_cfg.get("port", 8000),
            )

        # Webhook notifier — delivers distress signal to the website backend
        webhook_cfg = config.get("webhook", {})
        self._notifier = WebhookNotifier(
            WebhookConfig(
                enabled=webhook_cfg.get("enabled", False),
                url=webhook_cfg.get("url", ""),
                auth_token=webhook_cfg.get("auth_token", ""),
                timeout_seconds=webhook_cfg.get("timeout_seconds", 5.0),
                include_snapshot=webhook_cfg.get("include_snapshot", True),
                snapshot_scale=webhook_cfg.get("snapshot_scale", 0.5),
                only_critical=webhook_cfg.get("only_critical", True),
                source_id=webhook_cfg.get("source_id", "drone-1"),
            )
        )

        # Logging
        log_cfg = config.get("logging", {})
        self._event_logger = EventLogger(
            event_bus=self._event_bus,
            log_dir=log_cfg.get("log_dir", "logs"),
            snapshot_on_alert=log_cfg.get("snapshot_on_alert", True),
        )

        # Pipeline settings
        self._target_fps = config.get("pipeline", {}).get("target_fps", 15)
        self._show_preview = config.get("pipeline", {}).get("show_preview", True)

    def run(self) -> None:
        """Main loop — runs until stopped or video ends."""
        self._running = True
        frame_interval = 1.0 / self._target_fps

        # Start dashboard
        if self._dashboard:
            self._dashboard.start()

        # Auto-arm if configured
        if self._config.get("control", {}).get("auto_arm", False):
            self._controller.arm()

        logger.info("Pipeline started (target FPS: %d)", self._target_fps)
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        try:
            while self._running:
                loop_start = time.time()

                # 1. Capture frame
                frame_data = self._grabber.read()
                if frame_data is None:
                    logger.info("End of video stream")
                    break

                self._event_bus.publish("frame.captured", frame_data)

                # 2. Detect persons
                detections = self._detector.detect(frame_data)
                self._event_bus.publish("detection.complete", detections)

                # 3. Track persons
                tracked = self._tracker.update(detections)
                self._event_bus.publish("track.updated", tracked)

                # 4a. Pose-based gesture classification
                gestures = {}
                if self._gesture_classifier is not None:
                    gestures = self._gesture_classifier.classify(
                        frame_data, tracked
                    )

                # 4b. Facial expression distress scoring (optional — needs deepface)
                face_emotions: dict[int, float] = {}
                if self._face_emotion.available:
                    for person in tracked:
                        if person.frames_since_seen == 0:
                            b = person.bbox
                            face_emotions[person.track_id] = self._face_emotion.score(
                                frame_data.frame,
                                person.track_id,
                                b.x1, b.y1, b.x2, b.y2,
                            )

                # 4c. Analyze distress (gesture takes priority over heuristic)
                distress_events = self._analyzer.analyze(
                    tracked, frame_data.frame_id, gestures, face_emotions
                )
                self._event_bus.publish("distress.scored", distress_events)

                # 5. Evaluate alerts
                alerts = self._alert_manager.evaluate(distress_events)

                # Attach bounding boxes + gesture type to alerts
                track_map = {t.track_id: t for t in tracked}
                event_map = {e.track_id: e for e in distress_events}
                for alert in alerts:
                    if alert.track_id in track_map:
                        alert.bbox = track_map[alert.track_id].bbox
                    if alert.track_id in event_map:
                        alert.gesture = event_map[alert.track_id].gesture

                if alerts:
                    self._event_bus.publish("alert.triggered", alerts)

                # 6. Notify website backend on distress (non-blocking)
                for alert in alerts:
                    if alert.level == AlertLevel.CRITICAL:
                        self._notifier.notify(alert, frame_data.frame)

                # 7. Handle deployment
                for alert in alerts:
                    cmd = self._controller.handle_alert(alert)
                    if cmd:
                        self._event_bus.publish("deploy.commanded", cmd)

                # Cleanup stale alert and face emotion states
                active_ids = {t.track_id for t in tracked}
                self._alert_manager.cleanup_stale(active_ids)
                for tid in list(face_emotions):
                    if tid not in active_ids:
                        self._face_emotion.cleanup_track(tid)

                # FPS calculation
                fps_counter += 1
                elapsed_fps = time.time() - fps_timer
                if elapsed_fps >= 1.0:
                    current_fps = fps_counter / elapsed_fps
                    fps_counter = 0
                    fps_timer = time.time()

                # Update dashboard
                if self._dashboard:
                    annotated = self._annotate_frame(
                        frame_data, detections, tracked, distress_events, alerts
                    )
                    self._dashboard.update_frame(annotated)
                    self._dashboard.update_tracks(tracked)
                    self._dashboard.update_distress(distress_events)
                    self._dashboard.update_alerts(alerts)
                    self._dashboard.update_controller_state(
                        self._controller.state.value
                    )
                    self._dashboard.update_fps(current_fps)

                # Show local preview window
                if self._show_preview:
                    preview = self._annotate_frame(
                        frame_data, detections, tracked, distress_events, alerts
                    )
                    cv2.imshow("Project Doe", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("a"):
                        self._controller.arm()
                    elif key == ord("d"):
                        self._controller.disarm()

                # FPS limiting
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._grabber.release()
        self._serial_comm.close()
        self._notifier.close()
        self._event_logger.close()
        if self._show_preview:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped")

    def _annotate_frame(
        self, frame_data, detections, tracked, distress_events, alerts
    ):
        """Draw bounding boxes and labels on the frame for display."""
        frame = frame_data.frame.copy()

        # Build distress score lookup
        distress_map = {e.track_id: e for e in distress_events}
        alert_ids = {a.track_id for a in alerts}

        for person in tracked:
            bbox = person.bbox
            tid = person.track_id

            # Color based on distress
            color = (0, 255, 0)  # Green = normal
            if tid in alert_ids:
                color = (0, 0, 255)  # Red = alert
            elif tid in distress_map and distress_map[tid].distress_score > 0.4:
                color = (0, 165, 255)  # Orange = warning

            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                2,
            )

            label = f"ID:{tid}"
            if tid in distress_map:
                evt = distress_map[tid]
                label += f" D:{evt.distress_score:.0%}"
                if evt.gesture.value != "none":
                    label += f" [{evt.gesture.value.upper()}]"

            cv2.putText(
                frame,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Controller state overlay
        state_text = f"System: {self._controller.state.value}"
        cv2.putText(
            frame, state_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        return frame
