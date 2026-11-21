#include "Arduino.h"
#include <cassert>
#include <iostream>

unsigned long fakeTime = 0;
int pins[32] = {};
FakeSerial Serial;

#include "../firmware/walker_usb/walker_usb.ino"

void send(const std::string &s) {
  for (char c : s) {
    Serial.input.push_back(c);
  }
  while (Serial.available()) {
    loop();
  }
}

bool stopped() {
  return pins[LEFT_ENABLE] == 0 && pins[RIGHT_ENABLE] == 0
      && pins[LEFT_IN1] == LOW && pins[LEFT_IN2] == LOW
      && pins[RIGHT_IN1] == LOW && pins[RIGHT_IN2] == LOW && !armed;
}

int main() {
  setup();
  assert(stopped());

  pins[ENABLE_INPUT] = HIGH;
  send("ARM 12\n");
  assert(stopped());

  pins[ENABLE_INPUT] = LOW;
  send("ARM 12\nDRIVE 12 1 70 -80\n");
  assert(armed && pins[LEFT_ENABLE] == 70 && pins[RIGHT_ENABLE] == 80);
  assert(pins[LEFT_IN1] == HIGH && pins[LEFT_IN2] == LOW);
  assert(pins[RIGHT_IN1] == LOW && pins[RIGHT_IN2] == HIGH);

  fakeTime = 301;
  loop();
  assert(stopped());
  send("DRIVE 12 2 70 70\n");
  assert(stopped());

  send("ARM 13\nDRIVE 13 1 70 70\n");
  assert(armed);
  send("DRIVE 13 1 80 80\n");
  assert(stopped());

  send("ARM 14\nDRIVE 14 1 121 80\n");
  assert(stopped());

  send("ARM 15\nDRIVE 15 1 80 80\n");
  pins[ENABLE_INPUT] = HIGH;
  loop();
  assert(stopped());

  pins[ENABLE_INPUT] = LOW;
  send("ARM 16\nDRIVE 16 1 80 80\nHALT\n");
  assert(stopped());

  send("ARM 17\nDRIVE 17 1 80 80 extra\n");
  assert(stopped());
  send("ARM 18\n" + std::string(200, 'x') + "\n");
  assert(stopped());

  send("ARM 19\nDRIVE 19 1 80 80\n");
  fakeTime += 301;
  for (int i = 0; i < 200; i++) {
    Serial.input.push_back('x');
  }
  loop();
  assert(stopped());

  Serial.input.clear();
  used = 0;
  overflowing = false;
  send("ARM 20\nDRIVE 20 1 -70 0\n");
  assert(pins[LEFT_ENABLE] == 70 && pins[LEFT_IN1] == LOW && pins[LEFT_IN2] == HIGH);
  assert(pins[RIGHT_ENABLE] == 0);

  send("DRIVE 20 2 70 0\n");
  assert(pins[LEFT_ENABLE] == 70 && pins[LEFT_IN1] == HIGH && pins[LEFT_IN2] == LOW);
  assert(pins[RIGHT_ENABLE] == 0);

  send("DRIVE 20 3 0 0\n");
  assert(armed && pins[LEFT_ENABLE] == 0 && pins[RIGHT_ENABLE] == 0);
  send("HALT\n");
  assert(stopped());

  std::cout << "Firmware host tests: boot, enable, signed PWM, watchdog, stale session, "
            << "replay, limits, HALT, malformed, overflow, flood, reversal PASS\n";
}
