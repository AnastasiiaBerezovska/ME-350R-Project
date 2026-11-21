// Arduino UNO R3 + one L298N module with independently controlled channels
// Motor outputs require D2 connected to GND and an explicit ARM command
// An independent 300 ms serial watchdog disarms the controller on lost commands
// This file is updated with new pins. 
#include <Arduino.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <string.h>

const uint8_t LEFT_IN1 = 3, LEFT_IN2 = 4;
const uint8_t RIGHT_IN1 = 5, RIGHT_IN2 = 6;
const uint8_t LEFT_ENABLE = 2, RIGHT_ENABLE = 7;
const uint8_t ENABLE_INPUT = 8;
const int MAX_PWM = 120;
const unsigned long WATCHDOG_MS = 300;
bool armed = false;
unsigned long session = 0, sequence = 0, lastCommand = 0;
char lineBuffer[96];
uint8_t used = 0;
bool overflowing = false;

void driveMotor(uint8_t enable, uint8_t in1, uint8_t in2, int value) {
  // The L298N coasts with EN low. Disable the bridge before setting direction.
  analogWrite(enable, 0);
  digitalWrite(in1, value > 0 ? HIGH : LOW);
  digitalWrite(in2, value < 0 ? HIGH : LOW);
  if (value != 0) analogWrite(enable, abs(value));
}

void coast() {
  driveMotor(LEFT_ENABLE, LEFT_IN1, LEFT_IN2, 0);
  driveMotor(RIGHT_ENABLE, RIGHT_IN1, RIGHT_IN2, 0);
  digitalWrite(LED_BUILTIN, LOW);
}

void disarm() { coast(); armed = false; session = 0; sequence = 0; }

bool positiveNumber(const char *s, unsigned long &out) {
  if (!s || !*s) return false;
  for (const char *p = s; *p; p++) if (*p < '0' || *p > '9') return false;
  char *end; errno = 0; out = strtoul(s, &end, 10);
  return !errno && !*end && out > 0;
}

bool pwmNumber(const char *s, int &out) {
  if (!s || !*s) return false;
  const char *p = s; if (*p == '-') p++;
  if (!*p) return false;
  for (; *p; p++) if (*p < '0' || *p > '9') return false;
  char *end; errno = 0; long value = strtol(s, &end, 10);
  if (errno || *end || value < -MAX_PWM || value > MAX_PWM) return false;
  out = (int)value; return true;
}

void error(const __FlashStringHelper *message) {
  disarm(); Serial.print(F("ERR ")); Serial.println(message);
}

void process(char *line) {
  char *tokens[7]; uint8_t count = 0;
  char *part = strtok(line, " \t\r");
  while (part && count < 7) { tokens[count++] = part; part = strtok(NULL, " \t\r"); }
  if (!count) return;
  if (!strcmp(tokens[0], "HALT") && count == 1) {
    disarm(); Serial.println(F("ACK HALT")); return;
  }
  if (!strcmp(tokens[0], "HELLO") && count == 1) {
    Serial.print(F("READY WALKER 1 "));
    Serial.print(digitalRead(ENABLE_INPUT) == LOW ? 1 : 0);
    Serial.print(' '); Serial.println(MAX_PWM); return;
  }
  unsigned long wanted = 0, next = 0;
  if (!strcmp(tokens[0], "ARM") && count == 2 && positiveNumber(tokens[1], wanted)) {
    disarm();
    if (digitalRead(ENABLE_INPUT) != LOW) { error(F("ENABLE_OPEN")); return; }
    armed = true; session = wanted; lastCommand = millis();
    Serial.print(F("ACK ARM ")); Serial.println(session); return;
  }
  int left = 0, right = 0;
  if (!strcmp(tokens[0], "DRIVE") && count == 5 && positiveNumber(tokens[1], wanted)
      && positiveNumber(tokens[2], next) && pwmNumber(tokens[3], left) && pwmNumber(tokens[4], right)) {
    if (!armed || wanted != session) { error(F("NOT_ARMED")); return; }
    if (digitalRead(ENABLE_INPUT) != LOW) { error(F("ENABLE_OPEN")); return; }
    if (next <= sequence) { error(F("OLD_SEQUENCE")); return; }
    sequence = next; lastCommand = millis();
    driveMotor(LEFT_ENABLE, LEFT_IN1, LEFT_IN2, left);
    driveMotor(RIGHT_ENABLE, RIGHT_IN1, RIGHT_IN2, right);
    digitalWrite(LED_BUILTIN, left || right ? HIGH : LOW);
    Serial.print(F("ACK DRIVE ")); Serial.print(session); Serial.print(' '); Serial.println(sequence);
    return;
  }
  error(F("MALFORMED"));
}

void setup() {
  digitalWrite(LEFT_ENABLE, LOW);
  digitalWrite(RIGHT_ENABLE, LOW);
  pinMode(LEFT_ENABLE, OUTPUT);
  pinMode(RIGHT_ENABLE, OUTPUT);
  pinMode(LEFT_IN1, OUTPUT); pinMode(LEFT_IN2, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT); pinMode(RIGHT_IN2, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT); pinMode(ENABLE_INPUT, INPUT_PULLUP);
  disarm(); Serial.begin(115200);
}

void loop() {
  if (armed && (digitalRead(ENABLE_INPUT) != LOW || millis() - lastCommand > WATCHDOG_MS)) {
    disarm(); Serial.println(F("ERR WATCHDOG_OR_ENABLE"));
  }
  // Bound serial work 
  uint8_t budget = 32;
  while (Serial.available() && budget--) {
    char c = (char)Serial.read();
    if (c == '\n') {
      if (overflowing) error(F("LINE_TOO_LONG"));
      else { lineBuffer[used] = 0; process(lineBuffer); }
      used = 0; overflowing = false;
    } else if (!overflowing) {
      if (c == 0 || used >= sizeof(lineBuffer) - 1) {
        disarm(); overflowing = true;
      } else lineBuffer[used++] = c;
    }
  }
}