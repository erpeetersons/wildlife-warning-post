# Wildlife Warning Post

**Roadside poles that light up only when a moose is really there.**

A radar on each pole watches the road all the time. When a slow, persistent target heads for the road, the pole wakes a camera, an on-device AI model confirms that it's a large animal (or a person), and amber lights warn the driver. Cars, rain, snow and bushes are filtered out, so the warning keeps its meaning.

This repository contains the code of the working prototype built at **Junction 2026** on an **Arduino UNO Q**: the detection pipeline and dashboard that run on the board's Linux side, and the sketch that runs on its microcontroller.

---

## Contents

- [Why](#why)
- [How it works](#how-it-works)
- [What the prototype does](#what-the-prototype-does)
- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Using it](#using-it)
- [Interfaces](#interfaces)
- [Measured results](#measured-results)
- [Limitations and next steps](#limitations-and-next-steps)
- [Credits and licences](#credits-and-licences)

---

## Why

- **1,656** moose collisions in Finland in 2024, almost 10% more than in the two years before ([Fintraffic](https://www.fintraffic.fi/en/news/fintraffics-road-traffic-centre-elks-move-again-caution-recommended-especially-morning-and)).
- **13,758** road accidents involving game animals in Finland in 2022 ([Statistics Finland via Yle](https://yle.fi/a/74-20185168)).
- **42%** of Finland's moose collisions happen in September–November, mostly at dusk and in the dark.
- Collisions with moose, deer, roe deer and wild boar cost Sweden about **SEK 10 bn (≈ €0.9 bn) a year** ([Ruralis](https://ruralis.no/en/?p=18902)).

Existing measures all have a catch:

| Measure | Problem | Price |
|---|---|---|
| Static moose signs | Always on, so drivers stop reacting | cheap |
| Wildlife fences + overpasses | Expensive; animals still get in at fence ends | €25–50k / km, overpasses cost millions |
| Flashing detection posts (AniMot, Calstrom) | React to anything that moves; often need mains power and cleared verges | ≈ €11–800k / km |
| Animal scarers (Flox) | Move animals away; don't warn drivers | ≈ €3,200 / unit |
| Car night vision | Premium cars only | ≈ €2,100–2,500 per car |

Prices come from the team's competitor research (Nov 2026), converted to euros at approximate rates.

## How it works

```
 STM32 microcontroller (always on, ~mW)          Linux on the QRB2210 (wakes on demand)
 ─────────────────────────────────────           ─────────────────────────────────────────
 radar ── distance · angle · speed               camera frame
   │                                               │
   ▼  filter: slow? persistent? heading for road?  ▼  zoom on the radar bearing
   │  cars, rain, bushes → ignored                 ▼  YOLO11n on the board's CPU
   ▼                                               ▼  radar + camera agree → "Animal ahead"
 wake Linux, IR light on  ───────────────────►    radar only (fog) → "Caution"
                                                   │
 LED sign · amber lights · buzzer  ◄── set_state ──┘  + snapshot and log entry
```

Design rules:

1. **Radar decides when and where to look; the camera decides what it is.** Radar works in darkness, fog, rain and snow.
2. **The camera never cancels a warning, it only makes it more specific.** If the camera is blind, the radar-only caution still goes out.
3. **The expensive parts sleep.** Camera, IR light and AI only run while a filtered target exists, which is what makes battery operation possible.

Planned range for the product is about 100 m at night with a 60 GHz radar and an IR floodlight (≈ 4.5 s of warning at 80 km/h), and about 0.37 W while waiting, roughly 3 months on a 100 Ah battery. These are estimates from component specs, not field measurements.

## What the prototype does

The prototype has the camera, AI, filtering, warning output and dashboard running on the board. The radar, IR light, PIR and power behaviour are **simulated** because that hardware wasn't available; the radar input is a real interface that a radar reader can feed later.

**Running on the UNO Q:**

- **Detection**, in two modes you can switch on the dashboard:
  - **Full frame (default):** YOLO11n runs on the whole 1920×1080 camera image, scaled to 512×288, back to back. Detections are tracked between runs by box overlap.
  - **Motion + zoom:** cheap motion detection on every frame, tracking, filters (flicker, fast movers, exclusion zones, wiggling in place), then YOLO11n at 320×192 only on a full-resolution crop around each suspicious track (up to ~4.3× digital zoom). An optional **patrol scan** covers the full frame plus a 4×3 grid of ~3× zoom tiles when YOLO is idle.
- **Classes:** person; large animal (COCO's cat, dog, horse, sheep, cow, elephant, bear, zebra and giraffe, all grouped as "animal"); vehicles are recognised and ignored. COCO has no moose, deer or wolf class, but those animals are detected under the nearest class (moose as "cow", deer as "sheep", wolf as "dog"), which is all a warning needs.
- **Thresholds with hysteresis:** a new person needs score ≥ 0.30, a new animal ≥ 0.20; keeping an already-confirmed target needs ≥ 0.20.
- **Warnings:** *Animal ahead*, *Pedestrian ahead* or *Caution* (radar target the camera can't confirm). Each warning saves a snapshot to `~/yolo/data/events/` (the last 300 are kept).
- **Warning sign on the board:** the STM32 drives the 8×13 LED matrix (animal icon ↔ warning triangle, person icon, blinking "!", scanning dot when idle) and an optional **Modulino Buzzer** (3 low beeps for an animal, 2 high beeps for a person) and/or **Modulino Vibro** on the Qwiic connector, detected automatically, hot-pluggable.
- **Web dashboard** on port 8080: live camera with clean detection boxes (debug overlays optional), the exact 512×288 or 320×192 image YOLO receives, road-sign state, simulated hardware and power, radar plot, tracks, performance and event snapshots.
- **Starts at boot** as a systemd service; the sketch is the board's default Arduino app.

**Simulated (controlled from the dashboard):**

| Part | Behaviour |
|---|---|
| Radar | Off (camera-only), Empty road, Moose (approaches from 90 m), Person, Car (80 km/h), Rain (random echoes); bearing slider ±35° to aim at something real in view |
| Night | the real camera image is darkened to ~5% and only becomes bright (greyscale, IR look) once the simulated IR light switches on after a radar wake-up |
| Fog | the real camera image is blurred and greyed, so the camera can't confirm and the radar-only caution takes over |
| PIR | triggers for warm targets (moose, person) within 15 m |
| Power | modelled power per component, average W, Wh/day and battery life for 20/100/200 Ah (modelled, not measured) |

## Hardware

**Used in the prototype:**

| Part | Notes |
|---|---|
| Arduino UNO Q (2 GB) | Qualcomm QRB2210 (4× Cortex-A53, Debian 13) + STM32U585 microcontroller |
| USB webcam (UVC, MJPEG) | tested with a 16 MP sensor at 1920×1080 |
| **Powered** USB-C hub (PD, ≥ 15 W) | the board's only USB-C port must act as host for the camera; an unpowered hub caused reboots |
| Modulino Buzzer and/or Modulino Vibro (optional) | on the Qwiic connector |

**Proposed for the product** (estimates): a 60 GHz radar that also sees standing animals (e.g. TI IWR6843), an IR-capable day/night camera, an 850 nm IR floodlight switched by the STM32, amber LED lights, a 12 V LiFePO4 battery, IP65 enclosure. Estimated cost of a pole layout: **€15–20k per km installed** (parts €6.4–13.2k per km).

## Repository layout

```
board/                     Linux side of the UNO Q
  wildwatch.py             detection pipeline + web server (the main program)
  wildwatch.html           dashboard page served on :8080
  radar_sim.py             command-line radar simulator (UDP)
  wildwatch.service        systemd unit (runs wildwatch.py at boot)
  run.sh                   status / restart / stop / start / logs
  requirements.txt         Python packages, as tested
  legacy/person_detect.py  the first, simpler detector (kept for reference)
mcu/warning-sign/          Arduino App: STM32 sketch for LED matrix + buzzer/vibro
scripts/
  export_models.sh         export YOLO11n to ncnn at the sizes used (run on a PC)
  install_board.sh         copy everything to the board and install it
tools/                     development tools (run on the board)
  bench_backends.py        ncnn GPU vs ncnn CPU vs onnxruntime
  bench_input_sizes.py     speed of each exported model input size
  profile_pipeline.py      where one frame's time goes (camera, prep, inference, encode)
  setup-vnc.sh             optional remote desktop (TigerVNC, Xfce on :1)
LICENSE                    MIT
```

## Setup

### 1. Prepare the board

1. Do the UNO Q's first-time setup with **Arduino App Lab**: set the `arduino` user's password and connect it to Wi-Fi.
2. Connect the webcam (and optionally the Modulino buzzer/vibro) through a **powered** USB-C hub.
3. Make sure you can log in: `ssh arduino@<board-ip>` (or `adb shell` over USB). The board also answers as `<board-name>.local` on most networks.

### 2. Export the models (on a PC)

The models aren't stored in the repo (about 10 MB each). Export them once:

```bash
./scripts/export_models.sh models
```

This installs Ultralytics in a local virtual environment, downloads YOLO11n and exports it to ncnn as `models/yolo11n_512x288_ncnn` (full-frame mode), `yolo11n_320x192_ncnn` (motion + zoom crops) and `yolo11n_640x352_ncnn` (slower, more detail).

### 3. Install on the board

```bash
./scripts/install_board.sh <board-ip> models
```

It copies the pipeline, models and sketch, installs the Python packages into `~/yolo-env`, adds the `arduino` user to the `arduino-router` group, installs and starts `wildwatch.service`, and flashes the sketch as the default Arduino app. It asks for the board's password for the `sudo` steps.

<details>
<summary>Manual installation</summary>

```bash
# on the board
sudo apt-get install -y python3-venv python3-pip
sudo usermod -aG arduino-router arduino            # access to the Arduino Router socket
python3 -m venv ~/yolo-env
~/yolo-env/bin/pip install -r ~/yolo/requirements.txt
sudo cp ~/yolo/wildwatch.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now wildwatch
arduino-app-cli properties set default ~/ArduinoApps/warning-sign
arduino-app-cli app start ~/ArduinoApps/warning-sign
```

Files go to `~/yolo/` (`wildwatch.py`, `wildwatch.html`, `radar_sim.py`, `run.sh`, `requirements.txt` and the `yolo11n_*_ncnn` model folders) and `~/ArduinoApps/warning-sign/`.
</details>

Notes:

- The camera is found by its stable USB ID (`/dev/v4l/by-id/…`), because device numbers change between boots (the Qualcomm video codec can take `/dev/video0`).
- `pip install` needs Debian's `python3-venv`; without it, `python3 -m venv` creates an environment without pip.

## Using it

Open **`http://<board-ip>:8080`**.

| Control | What it does |
|---|---|
| Mode | Full frame (YOLO on the whole image) ↔ Motion + zoom |
| Patrol scan | In motion mode: scan the frame in zoomed tiles when YOLO is idle |
| Debug overlays | Show every pipeline stage (motion blobs, ignored tracks, crop box, radar bearings) instead of only confirmed detections |
| Draw exclusion zone | Drag a box over areas to ignore (bushes, a TV, a window) |
| Test alert | One beep/buzz from the Modulino |
| Sim radar target / Environment / Bearing | The simulation described above |

On the board:

```bash
~/yolo/run.sh status | restart | stop | start | logs
~/yolo-env/bin/python ~/yolo/radar_sim.py moose --angle -10   # scripted radar target over UDP
```

Command-line options of `wildwatch.py` (set them in `ExecStart` of the service): `--mode full|motion`, `--full-imgsz 512 288`, `--imgsz 320 192`, `--conf`, `--animal-conf`, `--keep-conf`, `--hold`, `--patrol`, `--patrol-grid 4x3`, `--width/--height/--fps`, `--camera`, `--port`. Run `wildwatch.py --help` for the full list.

**Demo tips**

- Hold an animal photo or video on a phone in front of the camera, or play a road video on a screen; full-frame mode handles screens best.
- For the radar story: set **Night** and **Empty road** (everything asleep, ~0.37 W), aim the bearing at the photo and click **Moose**: radar wakes the IR light, the image lights up, the camera confirms the animal, the sign and buzzer go off.

## Interfaces

**Radar input (UDP, `127.0.0.1:5005`, ~10 Hz):**

```json
{"targets": [{"id": 1, "range_m": 72.0, "angle_deg": -8.0, "speed_mps": -1.5}]}
```

`angle_deg` negative = left of the camera centre; `speed_mps` negative = approaching. Targets faster than 8 m/s are treated as vehicles; targets seen for less than 0.6 s are clutter. The radar angle maps to an image column (70° camera field of view assumed) and the distance sets the zoom. A target is confirmed when YOLO finds an animal or person inside its bearing window; with no confirmation after 2–3 attempts it becomes a radar-only caution. A reader for a real radar (LD2410/LD2451/IWR6843) only has to send this format.

**HTTP API (port 8080):**

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | dashboard |
| GET | `/stream` | MJPEG stream with detections drawn |
| GET | `/stats` | JSON: decision, tracks, radar, simulated hardware, performance |
| GET | `/events`, `/snap/<file>` | warning log and snapshots |
| GET | `/crop.jpg` | last image sent to YOLO |
| POST | `/zones` | exclusion zones, `[[x0,y0,x1,y1], …]` in 0–1 |
| POST | `/view` | `{"mode": "full"\|"motion", "debug": bool, "patrol": bool}` |
| POST | `/sim` | `{"target": "off"\|"empty"\|"moose"\|"person"\|"car"\|"rain", "env": "day"\|"night"\|"fog", "angle": deg}` |
| POST | `/vibrate` | test alert |

**Linux → microcontroller (Arduino Router, msgpack-RPC on `/var/run/arduino-router.sock`):**

| Method | Meaning |
|---|---|
| `set_state(int)` | 0 idle, 1 motion, 2 checking, 3 animal, 4 person, 5 caution; the sketch beeps/buzzes when entering 3 or 4 |
| `vibrate(int)` | 1 test, 3 animal pattern, 4 person pattern |
| `vibro_status()` | bit 0 = Vibro found, bit 1 = Buzzer found |
| `i2c_scan()` | addresses on the Qwiic and header I²C buses (diagnostics) |

## Measured results

All on the UNO Q unless noted.

| What | Result |
|---|---|
| Full-frame detection (512×288) | ~260 ms per frame live → **~3.4 detections per second** |
| Crop detection (320×192) | ~115–125 ms per crop |
| Camera loop | 1920×1080 MJPEG at ~7.5 frames/s indoors |
| CPU load | ~42% of 4 cores in full-frame mode (at 640×352); ~27% in motion mode while detecting |
| Temperature | 45–59 °C under load, no throttling (cores stay at 2.0 GHz) |
| Wildlife photo test (moose, reindeer, red/roe/white-tailed/fallow deer, boar, wolf, bear, lynx) | **10 of 10 detected as animal** |

Engineering findings:

| Test | Result |
|---|---|
| Backend at 320×320 | ncnn CPU **9.4 FPS** · onnxruntime CPU 6.0 · ncnn on the Adreno 702 GPU (Vulkan) 1.8 |
| Full-frame input size (service paused) | 512×288: 182 ms (5.5 FPS) · 640×352: 325 ms (3.1) · 768×448: 456 ms (2.2) |
| Small target (person ~40 px tall in 1080p) | missed at 1× and 1.8× zoom; found at 3× (0.28) and 4.5× (0.57) |

The QRB2210 has no NPU, and its Hexagon DSP isn't usable for user models on this board, so all inference runs on the CPU with ncnn.

## Reproducing the benchmarks

The numbers above come from the scripts in `tools/`. `scripts/install_board.sh` copies them to `~/yolo/tools/` on the board. Stop the service first so it doesn't compete for the CPU:

```bash
sudo systemctl stop wildwatch
~/yolo-env/bin/python ~/yolo/tools/bench_backends.py      # GPU vs CPU vs onnxruntime (320x320)
~/yolo-env/bin/python ~/yolo/tools/bench_input_sizes.py   # every yolo11n_<W>x<H>_ncnn in ~/yolo
~/yolo-env/bin/python ~/yolo/tools/profile_pipeline.py    # time per pipeline stage
sudo systemctl start wildwatch
```

`bench_backends.py` and `profile_pipeline.py` use the 320×320 ncnn model and the ONNX model, which `scripts/export_models.sh` also exports. Optional remote desktop: `sudo bash tools/setup-vnc.sh` on the board, then connect a VNC viewer to `<board-ip>:1`.

## Limitations and next steps

- Desk scale only: no real radar, IR light or long lens yet, so road distances are untested.
- The COCO model often names Nordic animals wrong (moose → "cow"); the warning is still right because all large animals count. Night, distance and IR performance need training data.
- Power figures are modelled; the board doesn't suspend between events yet.

Next:

1. Radar reader for a real radar (LD2451 for a road-scale demo, IWR6843 for standing animals).
2. Fine-tune YOLO11n on Nordic wildlife footage, including IR night images.
3. Suspend Linux between events, wake it from the STM32, and measure real battery life.
4. Field pilot: a short stretch of poles at a known moose hotspot, logging every warning with a photo.
5. Send warnings to traffic services and navigation apps.

## Credits and licences

- Built at Junction 2026 on the Arduino UNO Q.
- **YOLO11n** by [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). The exported models are not included in this repo; `scripts/export_models.sh` downloads and converts them.
- [ncnn](https://github.com/Tencent/ncnn) (BSD-3-Clause), [OpenCV](https://opencv.org/) (Apache-2.0), Arduino libraries `Arduino_RouterBridge`, `Arduino_LED_Matrix`, `Arduino_Modulino`.
- Statistics: Fintraffic, Statistics Finland (via Yle), Ruralis; competitor prices from the team's own research.
- Code in this repository is released under the [MIT License](LICENSE).
