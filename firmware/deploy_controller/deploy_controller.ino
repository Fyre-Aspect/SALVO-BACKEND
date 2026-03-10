/*
 * Project Doe — Deployment Controller Firmware
 * Target: ESP32 or Arduino (Uno/Mega)
 *
 * Receives serial commands from the laptop and controls
 * a servo or relay to deploy a flotation device.
 *
 * Commands (newline-terminated ASCII):
 *   ARM      → Enable deployment capability
 *   DISARM   → Disable deployment
 *   DEPLOY   → Fire the deployment mechanism (only if armed)
 *   STOP     → Emergency halt
 *   STATUS   → Report current state
 *
 * Responses:
 *   ACK:ARM, ACK:DISARM, ACK:DEPLOY, ACK:STOP
 *   STATUS:{state}
 *   ERR:NOT_ARMED, ERR:UNKNOWN
 *
 * Safety:
 *   - Will not deploy unless armed
 *   - Watchdog: auto-disarms after 5 seconds without heartbeat
 *   - Single deploy per arm cycle
 */

#include <Servo.h>

// Pin assignments — adjust for your board
const int SERVO_PIN = 9;
const int STATUS_LED = 13;
const int ARMED_LED = 12;    // Optional: LED to show armed state

// Servo positions
const int SERVO_LOCKED = 0;
const int SERVO_RELEASED = 90;

// Watchdog timeout (ms) — auto-disarm if no commands received
const unsigned long WATCHDOG_TIMEOUT = 5000;

enum State {
  STATE_DISARMED,
  STATE_ARMED,
  STATE_DEPLOYING,
  STATE_DEPLOYED
};

Servo deployServo;
State currentState = STATE_DISARMED;
unsigned long lastCommandTime = 0;
String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  deployServo.attach(SERVO_PIN);
  deployServo.write(SERVO_LOCKED);

  pinMode(STATUS_LED, OUTPUT);
  pinMode(ARMED_LED, OUTPUT);

  digitalWrite(STATUS_LED, HIGH);
  digitalWrite(ARMED_LED, LOW);

  Serial.println("STATUS:DISARMED");
}

void loop() {
  // Read serial input
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }

  // Watchdog check
  if (currentState == STATE_ARMED) {
    if (millis() - lastCommandTime > WATCHDOG_TIMEOUT) {
      currentState = STATE_DISARMED;
      deployServo.write(SERVO_LOCKED);
      digitalWrite(ARMED_LED, LOW);
      Serial.println("STATUS:WATCHDOG_DISARM");
    }
  }

  // Status LED blink pattern
  updateStatusLED();

  delay(10);
}

void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  lastCommandTime = millis();

  if (cmd == "ARM") {
    if (currentState == STATE_DISARMED) {
      currentState = STATE_ARMED;
      deployServo.write(SERVO_LOCKED);
      digitalWrite(ARMED_LED, HIGH);
      Serial.println("ACK:ARM");
    } else {
      Serial.println("ACK:ARM");  // Idempotent
    }

  } else if (cmd == "DISARM") {
    currentState = STATE_DISARMED;
    deployServo.write(SERVO_LOCKED);
    digitalWrite(ARMED_LED, LOW);
    Serial.println("ACK:DISARM");

  } else if (cmd == "DEPLOY") {
    if (currentState == STATE_ARMED) {
      currentState = STATE_DEPLOYING;
      deployServo.write(SERVO_RELEASED);
      Serial.println("ACK:DEPLOY");
      delay(500);
      currentState = STATE_DEPLOYED;
    } else {
      Serial.println("ERR:NOT_ARMED");
    }

  } else if (cmd == "STOP") {
    currentState = STATE_DISARMED;
    deployServo.write(SERVO_LOCKED);
    digitalWrite(ARMED_LED, LOW);
    Serial.println("ACK:STOP");

  } else if (cmd == "STATUS") {
    Serial.print("STATUS:");
    switch (currentState) {
      case STATE_DISARMED:  Serial.println("DISARMED"); break;
      case STATE_ARMED:     Serial.println("ARMED"); break;
      case STATE_DEPLOYING: Serial.println("DEPLOYING"); break;
      case STATE_DEPLOYED:  Serial.println("DEPLOYED"); break;
    }

  } else {
    Serial.println("ERR:UNKNOWN");
  }
}

void updateStatusLED() {
  static unsigned long lastBlink = 0;
  static bool ledState = false;

  unsigned long interval;
  switch (currentState) {
    case STATE_DISARMED:  interval = 2000; break;  // Slow blink
    case STATE_ARMED:     interval = 500;  break;  // Fast blink
    case STATE_DEPLOYING: interval = 100;  break;  // Very fast
    case STATE_DEPLOYED:  interval = 1000; break;  // Medium
    default:              interval = 2000; break;
  }

  if (millis() - lastBlink >= interval) {
    ledState = !ledState;
    digitalWrite(STATUS_LED, ledState ? HIGH : LOW);
    lastBlink = millis();
  }
}
