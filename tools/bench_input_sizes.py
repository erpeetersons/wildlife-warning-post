#!/usr/bin/env python3
"""Time every exported YOLO11n ncnn model (yolo11n_<W>x<H>_ncnn) at its own input size, on a real camera
frame resized to that size. ncnn exports have a fixed input size, so each size needs its own export.

Results on our board (service stopped): 512x288 182 ms (5.5 FPS), 640x352 325 ms (3.1), 768x448 456 ms (2.2).

    sudo systemctl stop wildwatch
    ~/yolo-env/bin/python bench_input_sizes.py [--models ~/yolo] [--runs 12]
"""
import argparse
import glob
import os
import re
import time

import cv2
import ncnn
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/yolo"), help="folder containing yolo11n_<W>x<H>_ncnn folders")
ap.add_argument("--runs", type=int, default=12)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()

cams = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
frame = None
if cams:
    cap = cv2.VideoCapture(os.path.realpath(cams[0]), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(3, 1920)
    cap.set(4, 1080)
    for _ in range(5):
        ok, f = cap.read()
        frame = f if ok else frame
    cap.release()
if frame is None:
    print("no camera frame, using random data")
    frame = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)


def size_of(path):
    m = re.search(r"yolo11n_(\d+)x(\d+)_ncnn", path)
    return int(m.group(1)) * int(m.group(2)) if m else 0


for d in sorted(glob.glob(os.path.join(args.models, "yolo11n_*x*_ncnn")), key=size_of):
    w, h = map(int, re.search(r"yolo11n_(\d+)x(\d+)_ncnn", d).groups())
    net = ncnn.Net()
    net.opt.num_threads = args.threads
    net.load_param(os.path.join(d, "model.ncnn.param"))
    net.load_model(os.path.join(d, "model.ncnn.bin"))

    def run():
        im = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255))
        return np.array(ex.extract("out0")[1])
    for _ in range(4):
        run()
    t0 = time.time()
    for _ in range(args.runs):
        run()
    dt = (time.time() - t0) / args.runs
    print(f"{w}x{h}: {dt * 1000:5.0f} ms -> {1 / dt:4.1f} FPS", flush=True)
