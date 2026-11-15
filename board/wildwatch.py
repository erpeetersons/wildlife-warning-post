#!/usr/bin/env python3
"""Wildlife warning post: camera-only prototype for the Arduino UNO Q.

Pipeline (camera motion stands in for the radar):
  1. Motion    - cheap background subtraction on a 320x180 copy of every frame
  2. Tracking  - motion blobs are linked into tracks (age, speed, position)
  3. Filters   - ignore flicker (short-lived), fast movers (vehicles) and exclusion zones
  4. Zoom      - crop the track's region from the full-resolution frame (digital zoom)
  5. Classify  - YOLO11n (ncnn, CPU) runs ONLY on those crops
  6. Decide    - animal / person warning, caution for unknown persistent objects

Web UI on http://<board>:8080
"""
import argparse
import collections
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, nargs=2, default=[320, 192], metavar=("W", "H"),
                help="model input size; a matching export yolo11n_<W>x<H>_ncnn must exist")
ap.add_argument("--full-imgsz", type=int, nargs=2, default=[512, 288], metavar=("W", "H"),
                help="model input for full-frame mode (512x288 runs ~4-5 fps, 640x352 ~3 fps on the UNO Q)")
ap.add_argument("--mode", choices=["full", "motion"], default="full",
                help="full = YOLO on the whole frame continuously; motion = motion trigger + zoomed crops")
ap.add_argument("--ncnn-model", default=None, help="defaults to yolo11n_<W>x<H>_ncnn/model.ncnn next to this script")
ap.add_argument("--camera", default="auto", help="video device, or 'auto' = first USB camera (stable across reboots)")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--fps", type=int, default=15, help="camera frame rate to request")
ap.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold for people and vehicles")
ap.add_argument("--keep-conf", type=float, default=0.20,
                help="lower threshold to KEEP an already-confirmed target (hysteresis)")
ap.add_argument("--animal-conf", type=float, default=0.20,
                help="lower threshold for animals: crops are already pre-filtered by motion, and deer/moose score low on COCO")
ap.add_argument("--min-age", type=float, default=0.6, help="seconds a track must persist before it counts")
ap.add_argument("--fast", type=float, default=0.45, help="frame-widths per second above which a mover is 'fast' (vehicle-like)")
ap.add_argument("--hold", type=float, default=10.0, help="seconds a warning stays on after the last confirmation")
ap.add_argument("--no-caution", action="store_true", help="do not raise CAUTION for unclassified persistent objects")
ap.add_argument("--quality", type=int, default=80)
ap.add_argument("--stream-fps", type=float, default=8.0)
ap.add_argument("--threads", type=int, default=3, help="ncnn CPU threads")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--data", default=os.path.join(HERE, "data"))
ap.add_argument("--patrol", type=float, default=0.4,
                help="when YOLO is idle, scan the full frame + a grid of zoomed tiles, one step every N seconds (0 = off)")
ap.add_argument("--patrol-grid", default="4x3", help="patrol tile grid COLSxROWS (4x3 = ~3x zoom tiles)")
ap.add_argument("--radar-port", type=int, default=5005, help="UDP port for radar target lists (JSON)")
ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view in degrees (maps radar angle to image column)")
ap.add_argument("--horizon", type=float, default=0.55, help="image height fraction where radar targets are assumed to be")
ap.add_argument("--radar-fast", type=float, default=8.0, help="radar targets faster than this (m/s) are treated as vehicles")
args = ap.parse_args()

os.makedirs(os.path.join(args.data, "events"), exist_ok=True)
ZONES_FILE = os.path.join(args.data, "zones.json")
IW, IH = args.imgsz  # model input (must match the exported model)
if args.ncnn_model is None:
    args.ncnn_model = os.path.join(HERE, f"yolo11n_{IW}x{IH}_ncnn/model.ncnn")

# COCO classes we care about
ANIMALS = {15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
           20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
NAMES = {0: "person", **ANIMALS, **VEHICLES}
INTEREST = np.array(sorted(NAMES))
ANIMAL_IDS = np.array(sorted(ANIMALS))


def group_of(c):
    return "person" if c == 0 else "animal" if c in ANIMALS else "vehicle"


def confirm_thr(c):
    return args.animal_conf if c in ANIMALS else args.conf


# ---------------------------------------------------------------- model
net = ncnn.Net()
net.opt.num_threads = args.threads
net.load_param(args.ncnn_model + ".param")
net.load_model(args.ncnn_model + ".bin")


FW, FH = args.full_imgsz
net_full = ncnn.Net()
net_full.opt.num_threads = 4
net_full.load_param(os.path.join(HERE, f"yolo11n_{FW}x{FH}_ncnn/model.ncnn.param"))
net_full.load_model(os.path.join(HERE, f"yolo11n_{FW}x{FH}_ncnn/model.ncnn.bin"))


def infer(img, model=None):
    blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    ex = (model or net).create_extractor()
    ex.input("in0", ncnn.Mat(blob))
    _, out = ex.extract("out0")
    return np.array(out)


def decode(out):
    sub = out[4:][INTEREST]                       # (k, N) scores for classes we care about
    best = sub.argmax(0)
    conf = sub[best, np.arange(sub.shape[1])]
    cls_all = INTEREST[best]
    keep = conf > min(args.keep_conf, args.animal_conf, args.conf)
    if not keep.any():
        return []
    cx, cy, w, h = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    cls, cf = cls_all[keep], conf[keep]
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), cf.tolist(), min(args.keep_conf, args.animal_conf, args.conf), 0.45)
    return [(int(cls[i]), float(cf[i]), *boxes[i][:2], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3])
            for i in np.array(idx).flatten()]


# ---------------------------------------------------------------- camera
def find_camera():
    """Device numbers change between boots (the Qualcomm codec can take /dev/video0); use the stable by-id link."""
    import glob
    links = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    return os.path.realpath(links[0]) if links else "/dev/video0"


if args.camera == "auto":
    args.camera = find_camera()
cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {args.camera}")
CAM_W, CAM_H = int(cap.get(3)), int(cap.get(4))
print(f"camera {args.camera} {CAM_W}x{CAM_H}, model {IW}x{IH}", flush=True)

# ---------------------------------------------------------------- simulated hardware (no real radar / PIR / IR light yet)
view = {"debug": False, "patrol": True, "mode": args.mode}
sim = {"target": "off", "env": "day", "angle": 0.0, "t0": time.time()}
hw = {"ir": False, "pir": False, "awake": False, "awake_until": 0.0, "mode": "camera",
      "power_w": 0.0, "energy_wh": 0.0, "since": time.time(), "parts": {}}
# modelled power draw in watts (typical figures, not measured on this board)
POWER = {"radar": 0.3, "mcu": 0.02, "linux_sleep": 0.05, "linux_idle": 1.5, "motion": 0.5,
         "yolo": 2.0, "camera": 1.0, "ir": 30.0, "led": 0.2}
BATTERY_WH = {"20 Ah": 230, "100 Ah": 1100, "200 Ah": 2200}   # usable LiFePO4 energy, 12 V


def apply_env(f):
    env = sim["env"]
    if env == "fog":
        h, w = f.shape[:2]
        blur = cv2.resize(cv2.resize(f, (w // 12, h // 12), interpolation=cv2.INTER_AREA), (w, h))
        return cv2.convertScaleAbs(blur, alpha=0.18, beta=125)
    if env == "night":
        if hw["ir"]:   # IR floodlight on: monochrome, well lit
            g = cv2.convertScaleAbs(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), alpha=0.95, beta=8)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return cv2.convertScaleAbs(f, alpha=0.05, beta=0)
    return f


frame_cv = threading.Condition()
latest = {"frame": None, "id": 0, "t": 0.0}


def grab():
    while True:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        f = apply_env(f)
        with frame_cv:
            latest.update(frame=f, id=latest["id"] + 1, t=time.time())
            frame_cv.notify_all()


# ---------------------------------------------------------------- zones
def load_zones():
    try:
        with open(ZONES_FILE) as f:
            z = json.load(f)
        return [[float(v) for v in r] for r in z if len(r) == 4]
    except (OSError, ValueError):
        return []


zones = load_zones()


def in_zone(cx, cy):
    nx, ny = cx / CAM_W, cy / CAM_H
    return any(x0 <= nx <= x1 and y0 <= ny <= y1 for x0, y0, x1, y1 in zones)


# ---------------------------------------------------------------- tracking
def merge_boxes(boxes, gap):
    """Merge motion boxes that overlap or lie within `gap` px (one body often splits into several blobs)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def inside(pt, box, pad=0.0):
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - px <= pt[0] <= x1 + px and y0 - py <= pt[1] <= y1 + py



class Track:
    def __init__(self, tid, box, t):
        self.id = tid
        self.hist = collections.deque([(t, box)], maxlen=90)
        self.first = self.last = t
        self.hits = 1
        self.group = None      # animal / person / vehicle / unknown
        self.label = None
        self.conf = 0.0
        self.det = None        # last YOLO box in frame coords
        self.tries = 0
        self.last_try = 0.0
        self.cls_t = 0.0
        self.pending = False
        self.status = "new"
        self.counted = set()   # which ignore-reasons were already counted
        self.sc = self.center()  # smoothed motion centre, updated every frame
        self.anchor = None       # motion centre at the moment the current YOLO box was measured
        self.job_center = None   # motion centre when the running YOLO job's crop was taken

    @property
    def box(self):
        return self.hist[-1][1]

    def center(self):
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    def display_box(self):
        """YOLO's box, shifted by how far the tracked motion has moved since YOLO measured it."""
        if self.det is None:
            return self.box
        if self.anchor is None:
            return self.det
        dx, dy = self.sc[0] - self.anchor[0], self.sc[1] - self.anchor[1]
        x0, y0, x1, y1 = self.det
        return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    def moving(self, now):
        return now - self.last < 0.5 and self.speed() > 0.03

    def union(self, n=6):
        bs = [b for _, b in list(self.hist)[-n:]]
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))

    def speed(self):
        """frame-widths per second over the last ~0.6 s"""
        pts = [(t, b) for t, b in self.hist if self.last - t <= 0.6]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.15:
            return 0.0
        (t0, a), (t1, b) = pts[0], pts[-1]
        dx = ((b[0] + b[2]) - (a[0] + a[2])) / 2
        dy = ((b[1] + b[3]) - (a[1] + a[3])) / 2
        return float(np.hypot(dx, dy) / CAM_W / (t1 - t0))


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes, t):
        free = set(self.tracks)
        for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best, bd = None, 1e9
            for tid in free:
                tr = self.tracks[tid]
                tx, ty = tr.center()
                x0, y0, x1, y1 = tr.box
                lim = max(0.12 * CAM_W, 0.6 * np.hypot(x1 - x0, y1 - y0))
                d = np.hypot(cx - tx, cy - ty)
                if d < lim and d < bd:
                    best, bd = tid, d
            if best is None:
                for tid, tr in self.tracks.items():
                    if tr.group in ("animal", "person") and tr.det is not None and t - tr.cls_t < 3 and inside((cx, cy), tr.display_box(), 0.15):
                        best = tid
                        break
            if best is None:
                self.tracks[self.next_id] = Track(self.next_id, b, t)
                self.next_id += 1
            else:
                tr = self.tracks[best]
                tr.hist.append((t, b))
                tr.last = t
                tr.hits += 1
                c = tr.center()
                tr.sc = (0.6 * tr.sc[0] + 0.4 * c[0], 0.6 * tr.sc[1] + 0.4 * c[1])
                free.discard(best)
        dead = []
        for tid, tr in list(self.tracks.items()):
            # classified tracks survive a little longer (target may stand still)
            if tr.group in ("animal", "person"):
                gone = t - max(tr.last, tr.cls_t) > 10.0    # no motion AND no fresh YOLO confirmation (> one patrol cycle)
            else:
                gone = t - tr.last > 1.0
            if gone:
                dead.append(tr)
                del self.tracks[tid]
        return dead


tracker = Tracker()

# ---------------------------------------------------------------- radar input
class RadarTarget:
    """A radar track as reported by the radar (or the simulator): range m, angle deg (negative = left), speed m/s (negative = approaching)."""
    def __init__(self, rid, t):
        self.id = rid
        self.hist = collections.deque(maxlen=50)
        self.first = self.last = t
        self.group = self.label = self.det = None
        self.conf, self.tries, self.last_try, self.cls_t = 0.0, 0, 0.0, 0.0
        self.pending, self.status, self.counted = False, "new", set()
        self.range = self.angle = self.speed = 0.0

    def update(self, rng, ang, spd, t):
        self.range, self.angle, self.speed, self.last = rng, ang, spd, t
        self.hist.append((t, rng, ang, spd))

    def column(self):
        return CAM_W / 2 + (self.angle / (args.hfov / 2)) * CAM_W / 2

    def crop_box(self):
        zoom = min(max(self.range / 25.0, 1.0), CAM_W / IW)
        cw = CAM_W / zoom
        ch = cw * (IH / IW)
        x0 = int(min(max(self.column() - cw / 2, 0), CAM_W - cw))
        y0 = int(min(max(args.horizon * CAM_H - ch / 2, 0), CAM_H - ch))
        return x0, y0, int(x0 + cw), int(y0 + ch)


radar = {}
radar_info = {"packets": 0, "last": 0.0}


def radar_listener():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.radar_port))
    while True:
        data, _ = sock.recvfrom(65535)
        try:
            msg = json.loads(data)
            tgts = msg.get("targets", [])
        except (ValueError, AttributeError):
            continue
        now = time.time()
        with lock:
            radar_info["packets"] += 1
            radar_info["last"] = now
            for tg in tgts:
                try:
                    rid = "R%d" % int(tg["id"])
                    rng, ang, spd = float(tg["range_m"]), float(tg["angle_deg"]), float(tg.get("speed_mps", 0.0))
                except (KeyError, ValueError, TypeError):
                    continue
                rt = radar.get(rid)
                if rt is None:
                    rt = radar[rid] = RadarTarget(rid, now)
                rt.update(rng, ang, spd, now)


def sim_radar():
    """Built-in radar simulator, driven from the web page. Feeds the same radar dict as a real radar would."""
    nid = 100000
    while True:
        time.sleep(0.1)
        tg, ang = sim["target"], sim["angle"]
        t = time.time() - sim["t0"]
        targets = []
        if tg == "moose":
            cyc = t % 60
            targets.append((1, max(8.0, 90 - 1.6 * cyc), ang, -1.6 if cyc < 51 else 0.0))
        elif tg == "person":
            targets.append((2, max(5.0, 20 - 0.3 * t), ang, -0.3))
        elif tg == "car":
            r = 140 - 22 * (t % 7)
            if r > 5:
                targets.append((3 + int(t // 7), r, ang + 3.0, -22.0))
        elif tg == "rain":
            for _ in range(20):
                nid += 1
                targets.append((nid, np.random.uniform(5, 135), np.random.uniform(-28, 28), np.random.uniform(-3, 3)))
        # simulated PIR: warm body within 15 m (moose / person)
        hw["pir"] = tg in ("moose", "person") and any(r < 15 for _, r, _, _ in targets)
        if tg == "off":
            continue
        now = time.time()
        with lock:
            radar_info["packets"] += 1      # radar is on: empty target lists still count as a heartbeat
            radar_info["last"] = now
            for i, r, a, v in targets:
                rid = "R%d" % i
                rt = radar.get(rid)
                if rt is None:
                    rt = radar[rid] = RadarTarget(rid, now)
                rt.update(float(r), float(a), float(v), now)


# ---------------------------------------------------------------- shared state
lock = threading.Lock()
jobs = queue.Queue(maxsize=1)
results = []
state = {
    "decision": "idle", "decision_label": "", "decision_conf": 0.0, "decision_since": time.time(),
    "loop_fps": 0.0, "motion_ms": 0.0, "yolo_ms": 0.0, "zoom": 1.0, "crop_box": None, "crop_tid": None,
    "crop_ts": 0.0, "motion_frac": 0.0, "lighting": False, "busy": False,
}
yolo_runs = collections.deque()
ignored = collections.deque()          # (t, reason)
last_seen = {"animal": None, "person": None, "unknown": None, "radar": None}   # (t, label, conf, tid)
events = collections.deque(maxlen=60)
crop_jpeg = {"data": None}
stream_jpeg = {"data": None}
START = time.time()


# ---------------------------------------------------------------- classifier thread
def classifier():
    while True:
        tid, (x0, y0, x1, y1), crop, t = jobs.get()
        t1 = time.time()
        mw, mh, model = (FW, FH, net_full) if tid == "F" else (IW, IH, net)
        interp = cv2.INTER_AREA if crop.shape[1] > mw else cv2.INTER_LINEAR
        img = cv2.resize(crop, (mw, mh), interpolation=interp)
        dets = decode(infer(img, model))
        ms = (time.time() - t1) * 1000
        sx, sy = (x1 - x0) / mw, (y1 - y0) / mh
        fdets = [(c, s, x0 + a * sx, y0 + b * sy, x0 + cc * sx, y0 + d * sy) for c, s, a, b, cc, d in dets]
        vis = img.copy()
        for c, s, a, b, cc, d in dets:
            col = COLORS[group_of(c)]
            cv2.rectangle(vis, (int(a), int(b)), (int(cc), int(d)), col, 2)
            put_label(vis, f"{NAMES[c]} {s:.2f}", int(a), int(b), col, 0.5, 1)
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with lock:
            results.append((tid, fdets, t))
            state["yolo_ms"] = ms if state["yolo_ms"] == 0 else 0.8 * state["yolo_ms"] + 0.2 * ms
            yolo_runs.append(time.time())
            if ok:
                crop_jpeg["data"] = jpg.tobytes()
                state["crop_ts"] = time.time()
            state["busy"] = False


# ---------------------------------------------------------------- drawing
COLORS = {  # BGR
    "motion": (195, 207, 70), "candidate": (195, 207, 70), "new": (150, 160, 170),
    "fast": (154, 131, 113), "zone": (154, 131, 113), "vehicle": (154, 131, 113), "jitter": (154, 131, 113),
    "animal": (67, 196, 245), "person": (120, 220, 120), "unknown": (92, 157, 255),
    "crop": (255, 123, 194), "zonefill": (138, 106, 74),
}


def put_label(img, text, x, y, col, scale=0.8, th=2):
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y = max(y, tht + 8)
    cv2.rectangle(img, (x, y - tht - 8), (x + tw + 8, y), col, -1)
    cv2.putText(img, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (17, 17, 11), th, cv2.LINE_AA)


STATUS_TEXT = {"new": "new", "fast": "IGNORED fast", "zone": "IGNORED zone", "vehicle": "IGNORED vehicle",
               "jitter": "IGNORED no movement",
               "candidate": "checking", "unknown": "unknown", "animal": "ANIMAL", "person": "PERSON"}
BANNER = {"animal": ("WARNING  ANIMAL", (67, 196, 245)), "person": ("WARNING  PERSON", (120, 220, 120)),
          "unknown": ("CAUTION  OBJECT", (92, 157, 255)), "checking": ("CHECKING", (195, 207, 70)),
          "motion": ("MOTION", (150, 160, 170)), "idle": ("MONITORING", (150, 160, 170))}


def draw_target(img, box, text, col):
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
    put_label(img, text, x0, y0, col, 0.8, 2)


def render(frame, motion_boxes, debug=None):
    """Clean view: only confirmed detections + banner. Debug view: every pipeline stage."""
    debug = view["debug"] if debug is None else debug
    img = frame.copy()
    H, W = img.shape[:2]
    with lock:
        rts = list(radar.values())
    if debug:
        for x0, y0, x1, y1 in zones:
            p0, p1 = (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H))
            roi = img[p0[1]:p1[1], p0[0]:p1[0]]
            if roi.size:
                roi[:] = cv2.addWeighted(roi, 0.6, np.full_like(roi, COLORS["zonefill"]), 0.4, 0)
            cv2.rectangle(img, p0, p1, (200, 160, 110), 2)
            cv2.putText(img, "exclusion zone", (p0[0] + 6, p0[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 200, 160), 2, cv2.LINE_AA)
        for x0, y0, x1, y1 in motion_boxes:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), COLORS["motion"], 1)
        cb = state["crop_box"]
        if cb and (time.time() - state["crop_ts"] < 1.5 or state["busy"]):
            cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), COLORS["crop"], 3)
            put_label(img, f"zoom {state['zoom']:.1f}x -> YOLO", cb[0], cb[1], COLORS["crop"], 0.7, 2)
        for rt in rts:
            u = int(rt.column())
            col = {"vehicle": COLORS["vehicle"], "new": COLORS["new"], "animal": COLORS["animal"],
                   "person": COLORS["person"], "radar-only": COLORS["unknown"]}.get(rt.status, COLORS["crop"])
            for y in range(0, H, 28):
                cv2.line(img, (u, y), (u, y + 14), col, 2)
            tag = {"vehicle": "IGNORED vehicle", "new": "new", "candidate": "checking", "radar-only": "RADAR ONLY"}.get(rt.status, rt.status.upper())
            put_label(img, f"{rt.id} {rt.range:.0f} m {abs(rt.speed) * 3.6:.0f} km/h {tag}", max(0, u - 160), H - 20, col, 0.7, 2)
    for tr in list(tracker.tracks.values()):
        st = tr.status
        confirmed = st in ("animal", "person", "unknown")
        if not debug and not confirmed:
            continue
        col = COLORS.get(st, COLORS["candidate"])
        box = tr.display_box() if st in ("animal", "person") else tr.box
        if confirmed:
            txt = f"{tr.label} {tr.conf:.2f}" if st in ("animal", "person") else "unknown object"
            if st == "animal":
                txt = f"ANIMAL ({txt})"
            elif st == "person":
                txt = f"PERSON ({txt})"
            draw_target(img, box, txt, col)
        else:
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            put_label(img, f"#{tr.id} {STATUS_TEXT.get(st, st)}", x0, y0, col, 0.75, 2)
    for rt in rts:   # radar targets confirmed by the camera
        if rt.status in ("animal", "person") and rt.det is not None:
            draw_target(img, rt.det, f"{rt.status.upper()} ({rt.label} {rt.conf:.2f}) · radar {rt.range:.0f} m",
                        COLORS[rt.status])
    text, col = BANNER.get(state["decision"], BANNER["idle"])
    put_label(img, text, 16, 56, col, 1.3, 3)
    if debug and state["lighting"]:
        put_label(img, "lighting change - ignored", 16, 110, (154, 131, 113), 0.8, 2)
    tags = [t for t, on in (("IR LIGHT ON", hw["ir"]), ("PIR: WARM BODY", hw["pir"]),
                            ("NIGHT (sim)", sim["env"] == "night"), ("FOG (sim)", sim["env"] == "fog")) if on]
    x = W - 16
    for t in tags:
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x -= tw + 18
        put_label(img, t, x, 56, COLORS["crop"] if t.startswith("IR") else (200, 200, 200), 0.8, 2)
    return img

last_event = {}


def save_event(kind, label, conf, tid, frame, motion_boxes):
    # one event per kind while its warning is active (a person moving around can split into several tracks)
    now = time.time()
    if now - last_event.get(kind, 0) < args.hold + 2:
        last_event[kind] = now
        return
    last_event[kind] = now
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{kind}_{tid}.jpg"
    snap = render(frame, motion_boxes, debug=False)
    snap = cv2.resize(snap, (960, int(960 * snap.shape[0] / snap.shape[1])))
    cv2.imwrite(os.path.join(args.data, "events", name), snap, [cv2.IMWRITE_JPEG_QUALITY, 80])
    files = sorted(os.listdir(os.path.join(args.data, "events")))
    for old in files[:-300]:
        try:
            os.remove(os.path.join(args.data, "events", old))
        except OSError:
            pass
    events.appendleft({"t": time.time(), "time": time.strftime("%H:%M:%S"), "kind": kind,
                       "label": label, "conf": round(conf, 2), "track": tid, "snap": name})
    print(f"EVENT {kind} {label} {conf:.2f} track {tid}", flush=True)


# ---------------------------------------------------------------- MCU: LED matrix sign over the Arduino Router
MCU_STATES = {"idle": 0, "motion": 1, "checking": 2, "animal": 3, "person": 4, "unknown": 5}
mcu = {"want": 0, "ok": False, "err": "", "cmds": collections.deque(maxlen=5), "vibrations": 0, "vibro": None}
mcu_cv = threading.Condition()


def notify_mcu(decision):
    with mcu_cv:
        mcu["want"] = MCU_STATES.get(decision, 0)
        if mcu["want"] in (3, 4):
            mcu["vibrations"] += 1      # the sketch buzzes by itself on entering animal/person
        mcu_cv.notify()


def mcu_command(method, *params):
    with mcu_cv:
        mcu["cmds"].append((method, list(params)))
        mcu_cv.notify()


def mcu_worker():
    """Sends the current state to the sketch's set_state(); resends every 5 s so a reset MCU recovers."""
    import socket
    import msgpack
    sock, unp, msgid = None, None, 0
    while True:
        with mcu_cv:
            mcu_cv.wait(timeout=5)
            want = mcu["want"]
        try:
            if sock is None:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect("/var/run/arduino-router.sock")
                unp = msgpack.Unpacker(raw=False)
            calls = []
            with mcu_cv:
                while mcu["cmds"]:
                    calls.append(mcu["cmds"].popleft())
            calls.append(("set_state", [want]))
            for method, params in calls:
                msgid += 1
                reply = rpc(sock, unp, msgid, method, params)
            msgid += 1
            st = rpc(sock, unp, msgid, "vibro_status", [])
            mcu["vibro"] = None if st[2] is not None else int(st[3])   # bit0 vibro, bit1 buzzer
            mcu["ok"], mcu["err"] = reply[2] is None, "" if reply[2] is None else str(reply[2])
        except (OSError, ValueError, ConnectionError) as e:
            mcu["ok"], mcu["err"] = False, str(e)
            try:
                sock and sock.close()
            except OSError:
                pass
            sock = None


def rpc(sock, unp, msgid, method, params):
    """One msgpack-RPC call to the sketch through the Arduino Router."""
    sock.sendall(__import__("msgpack").packb([0, msgid, method, params]))
    while True:
        for m in unp:
            if isinstance(m, list) and len(m) == 4 and m[0] == 1 and m[1] == msgid:
                return m
        data = sock.recv(4096)
        if not data:
            raise ConnectionError("router closed the connection")
        unp.feed(data)


# ---------------------------------------------------------------- main loop
def crop_for(box):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * 1.5, (y1 - y0) * 1.5
    cw = max(bw, bh * (IW / IH), IW * 1.4)   # never zoom in further than ~4.3x: tighter crops lose context
    cw = min(cw, CAM_W, CAM_H * (IW / IH))
    ch = cw * (IH / IW)
    x0 = int(min(max(cx - cw / 2, 0), CAM_W - cw))
    y0 = int(min(max(cy - ch / 2, 0), CAM_H - ch))
    return x0, y0, int(x0 + cw), int(y0 + ch)


PATROL = {"step": 0, "last": 0.0}


PCOLS, PROWS = (int(v) for v in args.patrol_grid.lower().split("x"))
PATROL_STEPS = 1 + PCOLS * PROWS


def patrol_box(step):
    """Step 0: full frame. Then a COLSxROWS grid of overlapping tiles (4x3 -> ~3x zoom), so small/far targets get pixels."""
    if step == 0:
        return crop_for((0, 0, CAM_W, CAM_H))
    tw = int(min(CAM_W, CAM_W / PCOLS * 1.3))
    th = int(tw * IH / IW)
    if th > CAM_H:
        th, tw = CAM_H, int(CAM_H * IW / IH)
    c, r = (step - 1) % PCOLS, (step - 1) // PCOLS
    x0 = int((CAM_W - tw) * c / max(PCOLS - 1, 1))
    y0 = int((CAM_H - th) * r / max(PROWS - 1, 1))
    return x0, y0, x0 + tw, y0 + th


def patrol_result(dets, now, frame, boxes):
    """Detections from a patrol scan become (or refresh) confirmed tracks."""
    for c, conf, x0, y0, x1, y1 in dets:
        g = group_of(c)
        if g == "vehicle" or conf < confirm_thr(c):
            continue
        box = (x0, y0, x1, y1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        tr = next((q for q in tracker.tracks.values()
                   if inside((cx, cy), q.display_box(), 0.2) or inside(q.center(), box, 0.1)), None)
        first = tr is None or tr.group != g
        if tr is None:
            tr = Track(tracker.next_id, box, now)
            tracker.tracks[tracker.next_id] = tr
            tracker.next_id += 1
            tr.hits = 4
        tr.group, tr.label, tr.conf, tr.det, tr.cls_t, tr.tries = g, NAMES[c], conf, box, now, 0
        tr.anchor = tr.sc
        last_seen[g] = (now, tr.label, tr.conf, tr.id)
        if first:
            save_event(g, tr.label, tr.conf, tr.id, frame, boxes)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def full_result(dets, now, frame):
    """Full-frame mode: YOLO detections ARE the tracks (matched by overlap between runs)."""
    used = set()
    for c, conf, x0, y0, x1, y1 in sorted(dets, key=lambda d: -d[1]):
        g = group_of(c)
        box = (x0, y0, x1, y1)
        best, bi = None, 0.0
        for q in tracker.tracks.values():
            if q.id not in used:
                v = iou(box, q.display_box())
                if v > bi:
                    best, bi = q, v
        if best is None or bi < 0.15:
            if conf < confirm_thr(c):
                continue
            best = Track(tracker.next_id, box, now)
            tracker.tracks[tracker.next_id] = best
            tracker.next_id += 1
            first = True
        else:
            if conf < (args.keep_conf if best.group == g else confirm_thr(c)):
                continue
            first = best.group != g
            best.hist.append((now, box))
            best.last = now
            best.hits += 1
        best.group, best.label, best.conf, best.det, best.cls_t = g, NAMES[c], conf, box, now
        best.anchor, best.sc, best.hits = None, ((x0 + x1) / 2, (y0 + y1) / 2), max(best.hits, 4)
        used.add(best.id)
        if g in ("animal", "person"):
            last_seen[g] = (now, best.label, best.conf, best.id)
            if first:
                save_event(g, best.label, best.conf, best.id, frame, [])
    # radar fusion: a radar target is confirmed if a detection lies within its bearing window
    with lock:
        rts = list(radar.values())
    for rt in rts:
        if rt.status not in ("candidate", "radar-only", "animal", "person"):
            continue
        bx0, _, bx1, _ = rt.crop_box()
        match = [d for d in dets if group_of(d[0]) in ("animal", "person") and bx0 <= (d[2] + d[4]) / 2 <= bx1
                 and d[1] >= (args.keep_conf if rt.group == group_of(d[0]) else confirm_thr(d[0]))]
        if match:
            d = max(match, key=lambda q: q[1])
            g = group_of(d[0])
            first = rt.group != g
            rt.group, rt.label, rt.conf, rt.det, rt.cls_t, rt.tries = g, NAMES[d[0]], d[1], d[2:], now, 0
            last_seen[g] = (now, rt.label, rt.conf, rt.id)
            if first:
                save_event(g, rt.label, rt.conf, rt.id, frame, [])
        else:
            rt.tries += 1
            if rt.tries >= 3 and rt.group in (None, "animal", "person"):
                if rt.group is None:
                    save_event("radar", "radar only", 0.0, rt.id, frame, [])
                rt.group, rt.det, rt.cls_t = "radar-only", None, now


def prune_full(now):
    for tid in [q.id for q in tracker.tracks.values() if now - q.cls_t > 1.2]:
        del tracker.tracks[tid]
    return []


def recheck_interval(tr, now):
    """Confirmed targets: re-check often while they move (keeps the box accurate), rarely when still."""
    if isinstance(tr, RadarTarget):
        return 2.5
    return 0.7 if tr.moving(now) else 2.5


def main_loop():
    seen, last_render, t_prev = 0, 0.0, time.time()
    SW, SH = 320, 180
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while True:
        with frame_cv:
            while latest["id"] == seen:
                frame_cv.wait()
            frame, seen, ft = latest["frame"], latest["id"], latest["t"]
        t0 = time.time()

        # 1. motion (only in motion mode)
        full = view["mode"] == "full"
        frac, lighting, boxes = 0.0, False, []
        if not full:
            small = cv2.resize(frame, (SW, SH), interpolation=cv2.INTER_AREA)
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            mask = mog.apply(gray)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=3)
            frac = cv2.countNonZero(mask) / (SW * SH)
            lighting = frac > 0.45 or time.time() - START < 3
            boxes = []
            if not lighting:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sx, sy = CAM_W / SW, CAM_H / SH
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h >= SW * SH * 0.003:
                        boxes.append((x * sx, y * sy, (x + w) * sx, (y + h) * sy))
                boxes = merge_boxes(boxes, 0.04 * CAM_W)
        motion_ms = (time.time() - t0) * 1000

        # 2. tracking
        dead = prune_full(time.time()) if full else tracker.update(boxes, ft)
        now = time.time()
        for tr in dead:
            if tr.last - tr.first < args.min_age and tr.group is None:
                ignored.append((now, "flicker"))

        # merge classifier results
        with lock:
            res, results[:] = results[:], []
        for tid, dets, _ in res:
            if tid == "F":
                if full:
                    full_result(dets, now, frame)
                continue
            if isinstance(tid, str) and tid.startswith("P"):
                patrol_result(dets, now, frame, boxes)
                continue
            is_radar = isinstance(tid, str)
            with lock:
                tr = radar.get(tid) if is_radar else tracker.tracks.get(tid)
            if tr is None:
                continue
            tr.pending = False
            tr.last_try = now
            pick = None
            for g in ("animal", "person", "vehicle"):
                # hysteresis: a target already confirmed as g is kept with a lower score
                cand = [d for d in dets if group_of(d[0]) == g and
                        d[1] >= (args.keep_conf if tr.group == g else confirm_thr(d[0]))]
                if cand:
                    pick = max(cand, key=lambda d: d[1])
                    break
            if pick:
                g = group_of(pick[0])
                first = tr.group != g
                tr.group, tr.label, tr.conf, tr.det, tr.cls_t, tr.tries = g, NAMES[pick[0]], pick[1], pick[2:], now, 0
                if not is_radar:
                    tr.anchor = tr.job_center or tr.sc
                if g in ("animal", "person"):
                    last_seen[g] = (now, tr.label, tr.conf, tid)
                    # absorb duplicate tracks that sit inside the confirmed box
                    for oid, other in list(tracker.tracks.items()):
                        if oid != tid and other.group in (None, "unknown", g) and inside(other.center(), tr.display_box(), 0.15):
                            del tracker.tracks[oid]
                    if first:
                        save_event(g, tr.label, tr.conf, tid, frame, boxes)
                elif first:
                    ignored.append((now, "vehicle"))
            else:
                tr.tries += 1
                if is_radar:
                    if tr.group in ("animal", "person") and tr.tries >= 2:
                        tr.group, tr.det = None, None
                    if tr.tries >= 2 and tr.group is None:
                        tr.group, tr.cls_t = "radar-only", now
                        save_event("radar", "radar only", 0.0, tid, frame, boxes)
                    continue
                if tr.group in ("animal", "person"):
                    if tr.tries >= 3:
                        tr.group, tr.det = None, None
                    continue
                (fx, fy), (lx, ly) = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (tr.hist[0][1], tr.hist[-1][1])]
                travelled = np.hypot(lx - fx, ly - fy) / CAM_W
                confirmed_active = any(v and now - v[0] < args.hold for k, v in last_seen.items() if k != "unknown")
                if tr.tries >= 4 and tr.group is None and (travelled < 0.05 or confirmed_active):
                    tr.group, tr.cls_t = "jitter", now
                    ignored.append((now, "jitter"))
                elif tr.tries >= 4 and tr.group is None:
                    tr.group, tr.cls_t = "unknown", now
                    last_seen["unknown"] = (now, "unknown", 0.0, tid)
                    if not args.no_caution:
                        save_event("unknown", "unknown", 0.0, tid, frame, boxes)
                elif tr.group == "unknown":
                    last_seen["unknown"] = (now, "unknown", 0.0, tid)

        # 3. filters / status
        for tr in tracker.tracks.values():
            age = tr.last - tr.first
            cx, cy = tr.center()
            if tr.group:
                tr.status = tr.group
            elif age < args.min_age or tr.hits < 4:
                tr.status = "new"
            elif tr.speed() > args.fast:
                tr.status = "fast"
            elif in_zone(cx, cy):
                tr.status = "zone"
            else:
                tr.status = "candidate"
            if tr.status in ("fast", "zone") and tr.status not in tr.counted:
                tr.counted.add(tr.status)
                ignored.append((now, tr.status))

        # radar targets: expire, filter, keep radar-only caution alive
        with lock:
            for rid, rt in list(radar.items()):
                age = rt.last - rt.first
                if now - rt.last > 0.6:
                    if age < args.min_age and rt.group is None:
                        ignored.append((now, "radar clutter"))
                    del radar[rid]
                    continue
                if rt.group == "vehicle" or abs(rt.speed) > args.radar_fast:
                    rt.status = "vehicle"
                elif rt.group:
                    rt.status = rt.group
                elif age < args.min_age:
                    rt.status = "new"
                else:
                    rt.status = "candidate"
                if rt.status == "vehicle" and "vehicle" not in rt.counted:
                    rt.counted.add("vehicle")
                    ignored.append((now, "radar vehicle"))
                if rt.status == "radar-only":
                    last_seen["radar"] = (now, "radar only", 0.0, rid)
            radar_list = list(radar.values())

        # 4. zoom + schedule classification (one crop at a time)
        with lock:
            busy = state["busy"]
        if not busy and full:      # full-frame mode: YOLO back-to-back on the whole image
            box = (0, 0, CAM_W, CAM_H)
            with lock:
                state.update(busy=True, crop_box=box, crop_tid="F", zoom=1.0)
            jobs.put(("F", box, frame.copy(), now))
        elif not busy:
            pool = [tr for tr in list(tracker.tracks.values()) + radar_list
                    if now - tr.last_try > 0.6 and (
                        tr.status == "candidate" or
                        (tr.status in ("animal", "person", "unknown", "radar-only") and now - tr.cls_t > recheck_interval(tr, now)))]
            if pool:
                tr = min(pool, key=lambda q: (q.group is not None, q.last_try))
                if isinstance(tr, RadarTarget):
                    box = tr.crop_box()
                else:
                    box = crop_for(tr.display_box() if tr.group in ("animal", "person") and tr.det is not None else tr.union())
                    tr.job_center = tr.sc
                crop = frame[box[1]:box[3], box[0]:box[2]].copy()
                tr.pending, tr.last_try = True, now
                with lock:
                    state.update(busy=True, crop_box=box, crop_tid=tr.id, zoom=CAM_W / (box[2] - box[0]))
                jobs.put((tr.id, box, crop, now))
            elif view["patrol"] and args.patrol > 0 and now - PATROL["last"] >= args.patrol:
                box = patrol_box(PATROL["step"])
                PATROL["step"] = (PATROL["step"] + 1) % PATROL_STEPS
                PATROL["last"] = now
                crop = frame[box[1]:box[3], box[0]:box[2]].copy()
                with lock:
                    state.update(busy=True, crop_box=box, crop_tid="P", zoom=CAM_W / (box[2] - box[0]))
                jobs.put(("P", box, crop, now))

        # 6. decision
        hold = 3.0 if full else args.hold
        dec, lab, cf = "idle", "", 0.0
        if last_seen["animal"] and now - last_seen["animal"][0] < hold:
            dec, lab, cf = "animal", last_seen["animal"][1], last_seen["animal"][2]
        elif last_seen["person"] and now - last_seen["person"][0] < hold:
            dec, lab, cf = "person", last_seen["person"][1], last_seen["person"][2]
        elif last_seen["radar"] and now - last_seen["radar"][0] < 2:
            dec, lab = "unknown", "radar only"
        elif not args.no_caution and last_seen["unknown"] and now - last_seen["unknown"][0] < 4:
            dec = "unknown"
        elif any(tr.status == "candidate" or tr.pending for tr in list(tracker.tracks.values()) + radar_list):
            dec = "checking"
        elif boxes:
            dec = "motion"
        if dec != state["decision"]:
            state["decision_since"] = now
            notify_mcu(dec)
        state.update(decision=dec, decision_label=lab, decision_conf=cf, lighting=lighting and now - START > 3)

        # simulated hardware: wake state, IR light, power model
        active = any(q.status in ("candidate", "animal", "person", "unknown", "radar-only") or q.pending
                     for q in list(tracker.tracks.values()) + radar_list)
        if active:
            hw["awake_until"] = now + 5.0
        hw["awake"] = now < hw["awake_until"]
        hw["ir"] = hw["awake"] and sim["env"] == "night"
        hw["mode"] = "radar" if now - radar_info["last"] < 2 else "camera"
        parts = {"MCU": POWER["mcu"]}
        if hw["mode"] == "radar":      # radar watches; Linux + camera sleep until woken
            parts["Radar"] = POWER["radar"]
            parts["Linux"] = POWER["linux_idle"] if hw["awake"] else POWER["linux_sleep"]
            if hw["awake"]:
                parts["Camera"] = POWER["camera"]
        else:                           # camera-only mode: camera + motion detection run all the time
            parts["Camera"] = POWER["camera"]
            parts["Linux"] = POWER["linux_idle"] + POWER["motion"]
        if state["busy"]:
            parts["YOLO"] = POWER["yolo"]
        if hw["ir"]:
            parts["IR light"] = POWER["ir"]
        if state["decision"] in ("animal", "person", "unknown"):
            parts["LED sign"] = POWER["led"]
        pw = sum(parts.values())
        hw["energy_wh"] += pw * (now - t_prev) / 3600
        hw["power_w"], hw["parts"] = pw, parts

        # stats
        dt = max(now - t_prev, 1e-6)
        t_prev = now
        state["loop_fps"] = 0.9 * state["loop_fps"] + 0.1 / dt if state["loop_fps"] else 1 / dt
        state["motion_ms"] = 0.9 * state["motion_ms"] + 0.1 * motion_ms
        state["motion_frac"] = frac
        while yolo_runs and now - yolo_runs[0] > 60:
            yolo_runs.popleft()
        while ignored and now - ignored[0][0] > 60:
            ignored.popleft()

        # render stream at a limited rate
        if now - last_render >= 1 / args.stream_fps:
            last_render = now
            ok, jpg = cv2.imencode(".jpg", render(frame, boxes), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if ok:
                stream_jpeg["data"] = jpg.tobytes()


# ---------------------------------------------------------------- web
_cpu_prev = [0, 0]


def system_stats():
    def read(p):
        try:
            with open(p) as f:
                return f.read()
        except OSError:
            return ""
    temps = [int(v) / 1000 for v in (read(f"/sys/class/thermal/thermal_zone{i}/temp").strip() for i in range(16)) if v]
    cpu = [int(v) for v in read("/proc/stat").split("\n")[0].split()[1:]]
    idle, total = cpu[3] + cpu[4], sum(cpu)
    d_idle, d_total = idle - _cpu_prev[0], total - _cpu_prev[1]
    _cpu_prev[:] = [idle, total]
    mem = {l.split(":")[0]: int(l.split()[1]) for l in read("/proc/meminfo").splitlines()}
    return {"cpu_pct": round(100 * (1 - d_idle / d_total), 1) if d_total else 0,
            "temp_c": round(max(temps), 1) if temps else None,
            "mem_used_mb": (mem["MemTotal"] - mem["MemAvailable"]) // 1024,
            "mem_total_mb": mem["MemTotal"] // 1024}


def stats_json():
    now = time.time()
    with lock:
        rlist = list(radar.values())
    tracks = [{"id": tr.id, "status": "classifying" if tr.pending else tr.status, "label": tr.label,
               "conf": round(tr.conf, 2), "age": round(tr.last - tr.first, 1), "speed": round(tr.speed(), 2)}
              for tr in sorted(tracker.tracks.values(), key=lambda q: q.id)]
    ign = collections.Counter(r for t, r in ignored if now - t < 60)
    return {
        "decision": state["decision"], "label": state["decision_label"], "conf": round(state["decision_conf"], 2),
        "since": round(now - state["decision_since"], 1),
        "loop_fps": round(state["loop_fps"], 1), "motion_ms": round(state["motion_ms"], 1),
        "yolo_ms": round(state["yolo_ms"]), "yolo_per_min": len(yolo_runs), "zoom": round(state["zoom"], 1),
        "busy": state["busy"], "crop_ts": state["crop_ts"], "lighting": state["lighting"],
        "motion_pct": round(state["motion_frac"] * 100, 1), "tracks": tracks, "ignored": dict(ign),
        "zones": zones, "camera": f"{CAM_W}x{CAM_H}", "model": f"YOLO11n {IW}x{IH} ncnn",
        "uptime_s": round(now - START), "system": system_stats(),
        "led": {"ok": mcu["ok"], "state": mcu["want"], "err": mcu["err"], "vibrations": mcu["vibrations"], "vibro": mcu["vibro"]},
        "sim": {k: sim[k] for k in ("target", "env", "angle")},
        "debug": view["debug"], "patrol": view["patrol"], "mode": view["mode"], "full_model": f"{FW}x{FH}", "crop_src": state["crop_tid"] if isinstance(state["crop_tid"], str) else "track",
        "hw": {"mode": hw["mode"], "awake": hw["awake"], "ir": hw["ir"], "pir": hw["pir"],
               "power_w": round(hw["power_w"], 2), "parts": {k: round(v, 2) for k, v in hw["parts"].items()},
               "avg_w": round(hw["energy_wh"] * 3600 / max(now - hw["since"], 1), 2),
               "batteries": BATTERY_WH},
        "radar": {"connected": now - radar_info["last"] < 2, "packets": radar_info["packets"],
                  "targets": [{"id": r.id, "range": round(r.range, 1), "angle": round(r.angle, 1),
                               "speed": round(r.speed, 2), "status": "classifying" if r.pending else r.status,
                               "label": r.label} for r in rlist]},
    }


with open(os.path.join(HERE, "wildwatch.html"), "rb") as f:
    PAGE = f.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE)
        if p == "/stats":
            return self._send(200, "application/json", json.dumps(stats_json()).encode())
        if p == "/events":
            return self._send(200, "application/json", json.dumps(list(events)).encode())
        if p == "/crop.jpg":
            d = crop_jpeg["data"]
            return self._send(200, "image/jpeg", d) if d else self._send(204, "text/plain", b"")
        if p.startswith("/snap/"):
            name = os.path.basename(p[6:])
            fp = os.path.join(args.data, "events", name)
            if name.endswith(".jpg") and os.path.isfile(fp):
                with open(fp, "rb") as f:
                    return self._send(200, "image/jpeg", f.read())
            return self._send(404, "text/plain", b"not found")
        if p == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    jpg = stream_jpeg["data"]
                    if jpg is not None and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        global zones
        if self.path == "/vibrate":
            mcu_command("vibrate", 1)
            return self._send(200, "application/json", b'{"ok": true}')
        if self.path == "/view":
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
                for k in ("debug", "patrol"):
                    if k in d:
                        view[k] = bool(d[k])
                if d.get("mode") in ("full", "motion") and d["mode"] != view["mode"]:
                    view["mode"] = d["mode"]
                    tracker.tracks.clear()
            except (ValueError, TypeError, AttributeError):
                return self._send(400, "text/plain", b"bad view request")
            return self._send(200, "application/json", json.dumps(view).encode())
        if self.path == "/sim":
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
                if d.get("target") in ("off", "empty", "moose", "person", "car", "rain") and d["target"] != sim["target"]:
                    sim["target"], sim["t0"] = d["target"], time.time()
                if d.get("env") in ("day", "night", "fog"):
                    sim["env"] = d["env"]
                if "angle" in d:
                    sim["angle"] = min(max(float(d["angle"]), -args.hfov / 2), args.hfov / 2)
                if d.get("reset_energy"):
                    hw["energy_wh"], hw["since"] = 0.0, time.time()
            except (ValueError, TypeError):
                return self._send(400, "text/plain", b"bad sim request")
            return self._send(200, "application/json", json.dumps({k: sim[k] for k in ("target", "env", "angle")}).encode())
        if self.path != "/zones":
            return self._send(404, "text/plain", b"not found")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"[]")
            new = []
            for r in data[:20]:
                x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in r)
                if x1 - x0 > 0.01 and y1 - y0 > 0.01:
                    new.append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)])
        except (ValueError, TypeError):
            return self._send(400, "text/plain", b"expected a list of [x0,y0,x1,y1] in 0..1")
        zones = new
        with open(ZONES_FILE, "w") as f:
            json.dump(zones, f)
        self._send(200, "application/json", json.dumps(zones).encode())


threading.Thread(target=grab, daemon=True).start()
threading.Thread(target=classifier, daemon=True).start()
threading.Thread(target=mcu_worker, daemon=True).start()
threading.Thread(target=radar_listener, daemon=True).start()
threading.Thread(target=sim_radar, daemon=True).start()
threading.Thread(target=main_loop, daemon=True).start()
print(f"Serving on http://0.0.0.0:{args.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
