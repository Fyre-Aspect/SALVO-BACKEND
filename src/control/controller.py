from __future__ import annotations

import logging
import time

from src.control.serial_comm import BaseSerialComm
from src.models.schemas import (
    Alert,
    AlertLevel,
    CommandType,
    ControllerState,
    DeployCommand,
)

logger = logging.getLogger(__name__)


class DeployController:
    """Manages deployment safety state and issues serial commands."""

    def __init__(self, serial_comm: BaseSerialComm) -> None:
        self._serial = serial_comm
        self._state = ControllerState.DISARMED
        self._last_deploy_track: int | None = None

    @property
    def state(self) -> ControllerState:
        return self._state

    def arm(self) -> bool:
        if self._state != ControllerState.DISARMED:
            logger.warning("Cannot arm: current state is %s", self._state.value)
            return False
        if self._serial.send_command("ARM"):
            self._state = ControllerState.ARMED
            logger.info("System ARMED")
            return True
        return False

    def disarm(self) -> bool:
        if self._serial.send_command("DISARM"):
            self._state = ControllerState.DISARMED
            self._last_deploy_track = None
            logger.info("System DISARMED")
            return True
        return False

    def handle_alert(self, alert: Alert) -> DeployCommand | None:
        """Process an alert and decide whether to deploy."""
        if alert.level != AlertLevel.CRITICAL:
            return None

        if self._state != ControllerState.ARMED:
            logger.info(
                "Alert received but system not armed (state=%s)", self._state.value
            )
            return None

        # Prevent double-deploy for the same person
        if self._last_deploy_track == alert.track_id:
            logger.info("Already deployed for track_id=%d", alert.track_id)
            return None

        # Deploy
        if self._serial.send_command("DEPLOY"):
            self._state = ControllerState.DEPLOYING
            self._last_deploy_track = alert.track_id
            cmd = DeployCommand(
                command=CommandType.DEPLOY,
                target_track_id=alert.track_id,
                timestamp=time.time(),
            )
            logger.warning(
                "DEPLOY command sent for track_id=%d", alert.track_id
            )
            return cmd

        logger.error("DEPLOY command failed for track_id=%d", alert.track_id)
        return None

    def stop(self) -> bool:
        """Emergency stop."""
        if self._serial.send_command("STOP"):
            self._state = ControllerState.DISARMED
            logger.warning("EMERGENCY STOP executed")
            return True
        return False

    def reset(self) -> bool:
        """Reset from DEPLOYED/DEPLOYING back to DISARMED."""
        return self.disarm()
