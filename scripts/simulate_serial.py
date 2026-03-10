"""Simulate a serial device for testing without hardware.

Listens on a virtual serial port and responds to commands with ACKs,
mimicking the ESP32/Arduino deployment controller.

Usage:
    pip install pyserial
    python scripts/simulate_serial.py --port COM4

On Windows, you need a virtual COM port pair (e.g., com0com).
On Linux, you can use socat:
    socat -d -d pty,raw,echo=0 pty,raw,echo=0
"""

from __future__ import annotations

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Serial device simulator")
    parser.add_argument("--port", type=str, required=True, help="Serial port to listen on")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("Install pyserial: pip install pyserial")
        sys.exit(1)

    state = "DISARMED"

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Cannot open {args.port}: {e}")
        sys.exit(1)

    print(f"Simulator listening on {args.port} @ {args.baud} baud")
    print(f"State: {state}")

    try:
        while True:
            line = ser.readline().decode("ascii").strip()
            if not line:
                continue

            print(f"Received: {line}")
            cmd = line.upper()

            if cmd == "ARM":
                state = "ARMED"
                ser.write(b"ACK:ARM\n")
            elif cmd == "DISARM":
                state = "DISARMED"
                ser.write(b"ACK:DISARM\n")
            elif cmd == "DEPLOY":
                if state == "ARMED":
                    state = "DEPLOYING"
                    ser.write(b"ACK:DEPLOY\n")
                    time.sleep(0.5)
                    state = "DEPLOYED"
                else:
                    ser.write(b"ERR:NOT_ARMED\n")
            elif cmd == "STOP":
                state = "DISARMED"
                ser.write(b"ACK:STOP\n")
            elif cmd == "STATUS":
                ser.write(f"STATUS:{state}\n".encode("ascii"))
            else:
                ser.write(b"ERR:UNKNOWN\n")

            print(f"State: {state}")

    except KeyboardInterrupt:
        print("\nSimulator stopped")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
