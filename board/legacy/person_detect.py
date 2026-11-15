#!/usr/bin/env python3
"""YOLO11n person detection on a USB camera, streamed as MJPEG over HTTP.

Open http://<board-ip>:8080 in a browser.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["ncnn", "ncnn-gpu", "onnx"], default="ncnn",
                help="ncnn = CPU (fastest), ncnn-gpu = Adreno 702 via Vulkan, onnx = onnxruntime CPU")
ap.add_argument("--ncnn-model", default="/home/arduino/yolo/yolo11n_512x288_ncnn/model.ncnn")
ap.add_argument("--onnx-model", default="/home/arduino/yolo/yolo11n.onnx")
ap.add_argument("--imgsz", type=int, nargs=2, default=[512, 288], metavar=("W", "H"),
                help="model input size; must match the exported model")
ap.add_argument("--camera", default="/dev/video0")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--iou", type=float, default=0.45)
ap.add_argument("--quality", type=int, default=80, help="stream JPEG quality")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()

IW, IH = args.imgsz

if args.backend == "onnx":
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(args.onnx_model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    IH, IW = inp.shape[2], inp.shape[3]

    def infer(blob):
        return sess.run(None, {inp.name: blob[None]})[0][0]
else:
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = args.backend == "ncnn-gpu"
    if net.opt.use_vulkan_compute:
        net.set_vulkan_device(0)  # Turnip Adreno 702
    net.opt.num_threads = 4
    net.load_param(args.ncnn_model + ".param")
    net.load_model(args.ncnn_model + ".bin")

    def infer(blob):
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(blob))
        _, out = ex.extract("out0")
        return np.array(out)

print(f"backend={args.backend} model_input={IW}x{IH} camera={args.width}x{args.height}")

cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {args.camera}")
print(f"camera actual: {int(cap.get(3))}x{int(cap.get(4))}")

# --- camera thread: always holds the newest frame -------------------------
frame_lock = threading.Condition()
latest_frame, frame_id = None, 0


def grab():
    global latest_frame, frame_id
    while True:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame, frame_id = f, frame_id + 1
            frame_lock.notify_all()


# --- detection ------------------------------------------------------------
def letterbox(img):
    h, w = img.shape[:2]
    r = min(IW / w, IH / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((IH, IW, 3), 114, np.uint8)
    left, top = (IW - nw) // 2, (IH - nh) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, r, left, top


def detect(frame):
    img, r, dx, dy = letterbox(frame)
    blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    out = infer(blob)  # (84, N); row 4 = person score
    scores = out[4]
    keep = scores > args.conf
    if not keep.any():
        return []
    cx, cy, w, h = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
    s = scores[keep]
    boxes = np.stack([(cx - w / 2 - dx) / r, (cy - h / 2 - dy) / r, w / r, h / r], 1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), s.tolist(), args.conf, args.iou)
    return [(boxes[i], float(s[i])) for i in np.array(idx).flatten()]


latest_jpeg = None
jpeg_lock = threading.Lock()
START = time.time()
stats = {"fps": 0.0, "people": 0, "scores": [], "detect_ms": 0.0, "encode_ms": 0.0,
         "frame_ms": 0.0, "jpeg_kb": 0.0, "frames": 0}


def ema(old, new, a=0.15):
    return new if old == 0 else (1 - a) * old + a * new


def loop():
    global latest_jpeg
    t_prev, seen = time.time(), 0
    scale = args.height / 720
    th = max(2, int(3 * scale))
    while True:
        with frame_lock:
            while frame_id == seen:
                frame_lock.wait()
            frame, seen = latest_frame, frame_id
        t0 = time.time()
        dets = detect(frame)
        t1 = time.time()
        for (x, y, w, h), s in dets:
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), th)
            cv2.putText(frame, f"person {s:.2f}", (p1[0], max(p1[1] - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 255, 0), th)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        t2 = time.time()
        if ok:
            with jpeg_lock:
                latest_jpeg = jpg.tobytes()
        stats["fps"] = ema(stats["fps"], 1 / max(t2 - t_prev, 1e-6))
        stats["frame_ms"] = ema(stats["frame_ms"], (t2 - t_prev) * 1000)
        stats["detect_ms"] = ema(stats["detect_ms"], (t1 - t0) * 1000)
        stats["encode_ms"] = ema(stats["encode_ms"], (t2 - t1) * 1000)
        stats["jpeg_kb"] = ema(stats["jpeg_kb"], len(jpg) / 1024 if ok else 0)
        stats["people"] = len(dets)
        stats["scores"] = sorted((round(s, 2) for _, s in dets), reverse=True)
        stats["frames"] += 1
        t_prev = t2


_cpu_prev = [0, 0]


def system_stats():
    def read(p, default=""):
        try:
            with open(p) as f:
                return f.read()
        except OSError:
            return default
    temps = []
    for i in range(16):
        t = read(f"/sys/class/thermal/thermal_zone{i}/temp").strip()
        if t:
            temps.append(int(t) / 1000)
    cpu = [int(v) for v in read("/proc/stat").split("\n")[0].split()[1:]]
    idle, total = cpu[3] + cpu[4], sum(cpu)
    d_idle, d_total = idle - _cpu_prev[0], total - _cpu_prev[1]
    _cpu_prev[:] = [idle, total]
    mem = {l.split(":")[0]: int(l.split()[1]) for l in read("/proc/meminfo").splitlines()}
    freq = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").strip()
    return {
        "cpu_pct": round(100 * (1 - d_idle / d_total), 1) if d_total else 0,
        "temp_c": round(max(temps), 1) if temps else None,
        "cpu_mhz": int(freq) // 1000 if freq else None,
        "load": read("/proc/loadavg").split()[:3],
        "mem_used_mb": (mem["MemTotal"] - mem["MemAvailable"]) // 1024,
        "mem_total_mb": mem["MemTotal"] // 1024,
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UNO Q Person Detection</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--line:#262a33;--text:#e6e8eb;--muted:#8b919c;--accent:#3ddc84}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
  font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{display:flex;gap:16px;padding:16px;height:100vh}
.video{flex:1;min-width:0;display:flex;align-items:center;justify-content:center;
  background:#000;border-radius:10px;overflow:hidden}
.video img{max-width:100%;max-height:100%;object-fit:contain;display:block}
aside{width:260px;flex:none;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:16px;overflow:auto}
h1{font-size:15px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12px;margin-bottom:14px}
.big{display:flex;gap:10px;margin-bottom:14px}
.big div{flex:1;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px}
.big b{display:block;font-size:26px;font-variant-numeric:tabular-nums}
.big span{color:var(--muted);font-size:12px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:14px 0 6px}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 12px}
dt{color:var(--muted)} dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:6px}
.off{background:#e5534b}
@media (max-width:800px){.wrap{flex-direction:column;height:auto}.video{height:auto}aside{width:auto}}
</style></head><body><div class="wrap">
<div class="video"><img src="/stream" alt="live camera"></div>
<aside>
<h1><span class="dot" id="dot"></span>Person detection</h1>
<div class="sub" id="sub">Arduino UNO Q</div>
<div class="big"><div><b id="people">-</b><span>people</span></div><div><b id="fps">-</b><span>FPS</span></div></div>
<h2>Pipeline</h2><dl>
<dt>Detect (model)</dt><dd id="detect">-</dd>
<dt>Draw + encode</dt><dd id="encode">-</dd>
<dt>Frame time</dt><dd id="frame">-</dd>
<dt>JPEG size</dt><dd id="jpeg">-</dd>
<dt>Confidence</dt><dd id="scores">-</dd>
<dt>Frames</dt><dd id="frames">-</dd></dl>
<h2>Config</h2><dl id="cfg"></dl>
<h2>Board</h2><dl>
<dt>CPU</dt><dd id="cpu">-</dd>
<dt>CPU clock</dt><dd id="mhz">-</dd>
<dt>Temperature</dt><dd id="temp">-</dd>
<dt>Load avg</dt><dd id="load">-</dd>
<dt>Memory</dt><dd id="mem">-</dd>
<dt>Uptime</dt><dd id="up">-</dd></dl>
</aside></div>
<script>
const $=id=>document.getElementById(id);
function dur(s){s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return (h?h+'h ':'')+m+'m '+(s%60)+'s'}
let cfgDone=false;
async function tick(){
  try{
    const r=await fetch('/stats',{cache:'no-store'}); const d=await r.json();
    $('dot').className='dot';
    $('people').textContent=d.people; $('fps').textContent=d.fps.toFixed(1);
    $('detect').textContent=d.detect_ms.toFixed(0)+' ms';
    $('encode').textContent=d.encode_ms.toFixed(0)+' ms';
    $('frame').textContent=d.frame_ms.toFixed(0)+' ms';
    $('jpeg').textContent=d.jpeg_kb.toFixed(0)+' KB';
    $('scores').textContent=d.scores.length?d.scores.join(', '):'-';
    $('frames').textContent=d.frames.toLocaleString();
    const s=d.system;
    $('cpu').textContent=s.cpu_pct+' %'; $('mhz').textContent=s.cpu_mhz?s.cpu_mhz+' MHz':'-';
    $('temp').textContent=s.temp_c!=null?s.temp_c+' °C':'-'; $('load').textContent=s.load.join(' ');
    $('mem').textContent=s.mem_used_mb+' / '+s.mem_total_mb+' MB'; $('up').textContent=dur(d.uptime_s);
    if(!cfgDone){cfgDone=true;
      $('cfg').innerHTML=Object.entries(d.config).map(([k,v])=>'<dt>'+k+'</dt><dd>'+v+'</dd>').join('');}
  }catch(e){$('dot').className='dot off';}
}
tick(); setInterval(tick,1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stats":
            body = json.dumps({
                **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()},
                "uptime_s": round(time.time() - START),
                "system": system_stats(),
                "config": {
                    "Model": "YOLO11n (person)",
                    "Backend": args.backend,
                    "Model input": f"{IW}×{IH}",
                    "Camera": f"{int(cap.get(3))}×{int(cap.get(4))}",
                    "Threshold": args.conf,
                    "JPEG quality": args.quality,
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last = None
        try:
            while True:
                with jpeg_lock:
                    jpg = latest_jpeg
                if jpg is not None and jpg is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass


threading.Thread(target=grab, daemon=True).start()
threading.Thread(target=loop, daemon=True).start()
print(f"Streaming on http://0.0.0.0:{args.port}")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
