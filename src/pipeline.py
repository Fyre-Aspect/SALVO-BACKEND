from __future__ import annotations

import logging
import time

import cv2

from src.alerting.alert_manager import AlertConfig, AlertManager
from src.control.controller import DeployController
from src.control.serial_comm import BaseSerialComm, MockSerialComm
from src.dashboard.server import DashboardServer
from src.distress.analyzer import DistressAnalyzer, DistressConfig
from src.event_bus import EventBus
from src.logging_.event_logger import EventLogger
from src.models.schemas import Alert, FrameData
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
                min_track_seconds=distress_cfg.get("min_track_seconds", 1.0),
                fps=distress_cfg.get("fps", 15.0),
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

                # 4. Analyze distress
                distress_events = self._analyzer.analyze(
                    tracked, frame_data.frame_id
                )
                self._event_bus.publish("distress.scored", distress_events)

                # 5. Evaluate alerts
                alerts = self._alert_manager.evaluate(distress_events)

                # Attach bounding boxes to alerts
                track_map = {t.track_id: t for t in tracked}
                for alert in alerts:
                    if alert.track_id in track_map:
                        alert.bbox = track_map[alert.track_id].bbox

                if alerts:
                    self._event_bus.publish("alert.triggered", alerts)

                # 6. Handle deployment
                for alert in alerts:
                    cmd = self._controller.handle_alert(alert)
                    if cmd:
                        self._event_bus.publish("deploy.commanded", cmd)

                # Cleanup stale alert states
                active_ids = {t.track_id for t in tracked}
                self._alert_manager.cleanup_stale(active_ids)

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
                score = distress_map[tid].distress_score
                label += f" D:{score:.0%}"

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
