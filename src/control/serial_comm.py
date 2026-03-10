from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import serial

logger = logging.getLogger(__name__)

ACK_TIMEOUT = 2.0  # seconds


class BaseSerialComm(ABC):
    """Abstract serial communication interface."""

    @abstractmethod
    def send_command(self, cmd: str) -> bool:
        """Send a command and return True if ACK received."""
        ...

    @abstractmethod
    def read_status(self) -> str:
        """Query device status."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class SerialComm(BaseSerialComm):
    """Real serial communication with ESP32/Arduino."""

    def __init__(self, port: str, baud_rate: int = 115200) -> None:
        self._ser = serial.Serial(port, baud_rate, timeout=ACK_TIMEOUT)
        time.sleep(2)  # Wait for Arduino reset
        logger.info("SerialComm connected: %s @ %d", port, baud_rate)

    def send_command(self, cmd: str) -> bool:
        cmd_clean = cmd.strip().upper()
        self._ser.write(f"{cmd_clean}\n".encode("ascii"))
        self._ser.flush()
        logger.debug("Sent: %s", cmd_clean)

        # Wait for ACK
        response = self._ser.readline().decode("ascii").strip()
        expected_ack = f"ACK:{cmd_clean}"
        if response == expected_ack:
            logger.debug("ACK received: %s", response)
            return True

        logger.warning("Unexpected response: '%s' (expected '%s')", response, expected_ack)
        return False

    def read_status(self) -> str:
        self._ser.write(b"STATUS\n")
        self._ser.flush()
        response = self._ser.readline().decode("ascii").strip()
        # Expected format: STATUS:ARMED
        if response.startswith("STATUS:"):
            return response[7:]
        return response

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()
            logger.info("SerialComm closed")


class MockSerialComm(BaseSerialComm):
    """Mock serial for testing without hardware."""

    def __init__(self) -> None:
        self._state = "DISARMED"
        self._command_log: list[str] = []
        logger.info("MockSerialComm initialized")

    def send_command(self, cmd: str) -> bool:
        cmd_clean = cmd.strip().upper()
        self._command_log.append(cmd_clean)
        logger.info("MockSerial: %s", cmd_clean)

        if cmd_clean == "ARM":
            self._state = "ARMED"
        elif cmd_clean == "DISARM":
            self._state = "DISARMED"
        elif cmd_clean == "DEPLOY":
            if self._state == "ARMED":
                self._state = "DEPLOYING"
            else:
                logger.warning("MockSerial: Cannot deploy — not armed")
                return False
        elif cmd_clean == "STOP":
            self._state = "DISARMED"

        return True

    def read_status(self) -> str:
        return self._state

    @property
    def command_log(self) -> list[str]:
        return list(self._command_log)

    def close(self) -> None:
        logger.info("MockSerialComm closed")
