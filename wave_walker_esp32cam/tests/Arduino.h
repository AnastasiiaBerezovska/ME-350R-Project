// I'm using stand-in Arduino functions for the host tests this is a test for me.
#pragma once

#include <cstdint>
#include <deque>
#include <sstream>
#include <string>

using uint8_t = std::uint8_t;
struct __FlashStringHelper {};

#define F(x) reinterpret_cast<const __FlashStringHelper *>(x)
#define LOW 0
#define HIGH 1
#define OUTPUT 1
#define INPUT_PULLUP 2
#define LED_BUILTIN 13

extern unsigned long fakeTime;
extern int pins[32];

inline unsigned long millis() {
  return fakeTime;
}

inline void pinMode(int, int) {}

inline void analogWrite(int p, int v) {
  pins[p] = v;
}

inline void digitalWrite(int p, int v) {
  pins[p] = v;
}

inline int digitalRead(int p) {
  return pins[p];
}

struct FakeSerial {
  std::deque<char> input;
  std::string output;

  void begin(int) {}

  int available() {
    return input.size();
  }

  int read() {
    char c = input.front();
    input.pop_front();
    return c;
  }

  void print(const __FlashStringHelper *v) {
    output += reinterpret_cast<const char *>(v);
  }

  void print(const char *v) {
    output += v;
  }

  void print(char v) {
    output += v;
  }

  template<class T>
  void print(T v) {
    std::ostringstream s;
    s << v;
    output += s.str();
  }

  template<class T>
  void println(T v) {
    print(v);
    output += '\n';
  }
};

extern FakeSerial Serial;
