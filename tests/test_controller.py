"""Tests for DeployController with MockSerialComm."""

from src.control.controller import DeployController
from src.control.serial_comm import MockSerialComm
from src.models.schemas import (
    Alert,
    AlertLevel,
    BoundingBox,
    CommandType,
    ControllerState,
)


def _make_alert(track_id=1, level=AlertLevel.CRITICAL):
    return Alert(
        track_id=track_id,
        level=level,
        distress_score=0.9,
        timestamp=1000.0,
        frame_id=42,
        bbox=BoundingBox(100, 50, 200, 300),
    )


def test_arm_disarm():
    comm = MockSerialComm()
    ctrl = DeployController(comm)

    assert ctrl.state == ControllerState.DISARMED
    assert ctrl.arm()
    assert ctrl.state == ControllerState.ARMED
    assert ctrl.disarm()
    assert ctrl.state == ControllerState.DISARMED


def test_deploy_when_armed():
    comm = MockSerialComm()
    ctrl = DeployController(comm)
    ctrl.arm()

    cmd = ctrl.handle_alert(_make_alert())
    assert cmd is not None
    assert cmd.command == CommandType.DEPLOY
    assert ctrl.state == ControllerState.DEPLOYING


def test_no_deploy_when_disarmed():
    comm = MockSerialComm()
    ctrl = DeployController(comm)

    cmd = ctrl.handle_alert(_make_alert())
    assert cmd is None


def test_no_deploy_for_warning():
    comm = MockSerialComm()
    ctrl = DeployController(comm)
    ctrl.arm()

    cmd = ctrl.handle_alert(_make_alert(level=AlertLevel.WARNING))
    assert cmd is None


def test_no_double_deploy_same_track():
    comm = MockSerialComm()
    ctrl = DeployController(comm)
    ctrl.arm()

    cmd1 = ctrl.handle_alert(_make_alert(track_id=5))
    cmd2 = ctrl.handle_alert(_make_alert(track_id=5))

    assert cmd1 is not None
    assert cmd2 is None  # Already deployed for this track


def test_emergency_stop():
    comm = MockSerialComm()
    ctrl = DeployController(comm)
    ctrl.arm()

    assert ctrl.stop()
    assert ctrl.state == ControllerState.DISARMED
