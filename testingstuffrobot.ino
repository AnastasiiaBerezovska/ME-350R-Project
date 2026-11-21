// L298N motor test for Raspberry Pi Pico (arduino-pico core)
// Pins are GP numbers, same as the wiring table

const int ENA = 2, IN1 = 3, IN2 = 4;   // Motor A (left)
const int ENB = 7, IN3 = 5, IN4 = 6;   // Motor B (right)

// Flip these if a motor spins the wrong way
const bool INVERT_LEFT  = false;
const bool INVERT_RIGHT = false;

// speed: -100 to 100 (percent)
void runMotor(int en, int inA, int inB, int speed, bool invert) {
  if (invert) speed = -speed;
  speed = constrain(speed, -100, 100);
  if (speed > 0)      { digitalWrite(inA, HIGH); digitalWrite(inB, LOW);  }
  else if (speed < 0) { digitalWrite(inA, LOW);  digitalWrite(inB, HIGH); }
  else                { digitalWrite(inA, LOW);  digitalWrite(inB, LOW);  }
  analogWrite(en, map(abs(speed), 0, 100, 0, 255));
}

void left(int s)  { runMotor(ENA, IN1, IN2, s, INVERT_LEFT);  }
void right(int s) { runMotor(ENB, IN3, IN4, s, INVERT_RIGHT); }

void setup() {
  Serial.begin(115200);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  analogWriteFreq(1000);    // 1 kHz PWM
  analogWriteRange(255);    // 0-255 duty range
  left(0); right(0);
  delay(3000);              // time to open Serial Monitor
}

void testMotor(const char* name, void (*m)(int), int speed) {
  Serial.print(name); Serial.println(" forward");
  m(speed);  delay(2000);
  m(0);      delay(1000);
  Serial.print(name); Serial.println(" reverse");
  m(-speed); delay(2000);
  m(0);      delay(1000);
}

void loop() {
  testMotor("left", left, 60);
  testMotor("right", right, 60);

  Serial.println("both forward");
  left(60); right(60); delay(2000);

  Serial.println("ramp");
  for (int s = 0; s <= 100; s += 10) { left(s); right(s); delay(300); }
  for (int s = 100; s >= 0; s -= 10) { left(s); right(s); delay(300); }

  left(0); right(0);
  Serial.println("done, pausing 5 s");
  delay(5000);
}