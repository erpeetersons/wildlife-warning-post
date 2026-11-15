// Wildlife warning sign on the UNO Q 8x13 LED matrix + Modulino Vibro (Qwiic) haptic alert.
// Linux calls Bridge "set_state" with: 0 idle, 1 motion, 2 checking, 3 animal, 4 person, 5 unknown/caution
// and "vibrate" with: 1 test pulse, 3 animal pattern, 4 person pattern.
// The vibro buzzes automatically when the state changes to animal (3 long pulses) or person (2 short pulses).
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Modulino.h>

Arduino_LED_Matrix matrix;
const int ROWS = 8, COLS = 13;
uint8_t frame[ROWS * COLS];
volatile int state = 0;
unsigned long stateSince = 0;

// ---- vibration (I2C is only touched from loop(); Bridge handlers just set flags)
ModulinoVibro vibro;
ModulinoBuzzer buzzer;
bool vibroOk = false, buzzerOk = false;
int pulseFreq = 1000;
volatile int pendingVibe = 0;   // pattern requested by Linux
int lastState = 0;
int pulsesLeft = 0, pulseLen = 0, pulseGap = 0, pulsePower = MAXIMUM;
unsigned long nextPulse = 0;

void startPattern(int kind) {
  if (kind == 3) { pulsesLeft = 3; pulseLen = 400; pulseGap = 200; pulsePower = MAXIMUM; pulseFreq = 880; }       // animal: 3 long low
  else if (kind == 4) { pulsesLeft = 2; pulseLen = 150; pulseGap = 150; pulsePower = MAXIMUM; pulseFreq = 1320; } // person: 2 short high
  else { pulsesLeft = 1; pulseLen = 300; pulseGap = 0; pulsePower = MAXIMUM; pulseFreq = 1000; }                  // test
  nextPulse = millis();
}

void runVibro() {
  if (!vibroOk && !buzzerOk) return;
  if (pendingVibe) { startPattern(pendingVibe); pendingVibe = 0; }
  if (pulsesLeft > 0 && (long)(millis() - nextPulse) >= 0) {
    if (vibroOk) vibro.on(pulseLen, false, pulsePower);   // modules time the pulse themselves
    if (buzzerOk) buzzer.tone(pulseFreq, pulseLen);
    pulsesLeft--;
    nextPulse = millis() + pulseLen + pulseGap;
  }
}

void vibrate(int kind) { pendingVibe = kind <= 0 ? 1 : kind; }
int vibro_status() { return (vibroOk ? 1 : 0) | (buzzerOk ? 2 : 0); }   // bit0 vibro, bit1 buzzer

// Diagnostics: list I2C addresses on Wire1 (Qwiic) and Wire (header SDA/SCL). Runs in loop() thread (provide_safe).
String scanBus(TwoWire &w) {
  String out;
  for (uint8_t a = 1; a < 127; a++) {
    w.beginTransmission(a);
    if (w.endTransmission() == 0) { out += "0x"; out += String(a, HEX); out += " "; }
  }
  return out.length() ? out : String("none");
}
String i2c_scan() {
  Wire.begin();
  return String("Wire1(Qwiic): ") + scanBus(Wire1) + String("| Wire(header): ") + scanBus(Wire);
}

unsigned long lastProbe = 0;
void probeVibro() {            // hot-plug: keep looking for the module if it was not found at boot
  if ((vibroOk && buzzerOk) || millis() - lastProbe < 3000) return;
  lastProbe = millis();
  if (!vibroOk && (vibroOk = vibro.begin())) vibro.on(150, false, GENTLE);
  if (!buzzerOk && (buzzerOk = buzzer.begin())) buzzer.tone(2000, 60);
}

const char *ANIMAL[ROWS] = {
  ".........#.#.",
  "..........##.",
  "..........###",
  ".##########..",
  "##########...",
  ".########....",
  ".#.#...#.#...",
  ".#.#...#.#..."};
const char *PERSON[ROWS] = {
  "......#......",
  "....#####....",
  "......#......",
  "......#......",
  ".....#.#.....",
  ".....#.#.....",
  "....#...#....",
  "....#...#...."};
const char *TRIANGLE[ROWS] = {
  "......#......",
  ".....#.#.....",
  "....#.#.#....",
  "....#.#.#....",
  "...#..#..#...",
  "..#.......#..",
  ".#....#....#.",
  "#############"};

void clearFrame() { memset(frame, 0, sizeof(frame)); }
void px(int r, int c, uint8_t v) { if (r >= 0 && r < ROWS && c >= 0 && c < COLS) frame[r * COLS + c] = v; }
void icon(const char *img[], uint8_t v) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (img[r][c] == '#') px(r, c, v);
}

void set_state(int s) {
  if (s < 0 || s > 5) return;
  if (s != state) { state = s; stateSince = millis(); }
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);   // brightness 0..7
  matrix.clear();
  Modulino.begin();
  vibroOk = vibro.begin();
  buzzerOk = buzzer.begin();
  Bridge.begin();
  Bridge.provide("set_state", set_state);
  Bridge.provide("vibrate", vibrate);
  Bridge.provide("vibro_status", vibro_status);
  Bridge.provide_safe("i2c_scan", i2c_scan);
  if (vibroOk) { vibro.on(150, false, GENTLE); }   // short "hello" at boot = module found
  if (buzzerOk) { buzzer.tone(2000, 60); }
}

void loop() {
  unsigned long t = millis();
  unsigned long dt = t - stateSince;
  int s = state;
  if (s != lastState) {                 // new detection -> buzz
    if (s == 3 || s == 4) startPattern(s);
    lastState = s;
  }
  probeVibro();
  runVibro();
  clearFrame();
  switch (state) {
    case 0: {  // idle: one dim dot sweeping the bottom row, like a radar scan
      int c = (t / 150) % (2 * COLS - 2);
      if (c >= COLS) c = 2 * COLS - 2 - c;
      px(7, c, 2);
      break;
    }
    case 1: {  // motion: dim pulsing centre
      uint8_t v = 1 + ((t / 200) % 3);
      px(3, 6, v); px(4, 6, v); px(3, 5, v - 1); px(4, 7, v - 1);
      break;
    }
    case 2: {  // checking: bar bouncing left/right
      int c = (t / 80) % (2 * COLS - 4);
      if (c >= COLS - 2) c = 2 * COLS - 4 - c;
      for (int r = 2; r < 6; r++) { px(r, c, 6); px(r, c + 1, 6); }
      break;
    }
    case 3:    // animal: animal icon alternating with warning triangle
      if ((dt / 500) % 2 == 0) icon(ANIMAL, 7); else icon(TRIANGLE, 7);
      break;
    case 4:    // person: steady person icon with a blinking frame corner
      icon(PERSON, 7);
      if ((dt / 400) % 2 == 0) { px(0, 0, 4); px(0, 12, 4); px(7, 0, 4); px(7, 12, 4); }
      break;
    case 5:    // unknown: blinking warning triangle
      if ((dt / 350) % 2 == 0) icon(TRIANGLE, 7);
      break;
  }
  matrix.draw(frame);
  delay(30);
}
