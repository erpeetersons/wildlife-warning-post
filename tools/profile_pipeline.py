#!/usr/bin/env python3
"""Profile where one frame's time goes: camera read, preprocessing, YOLO inference, post-processing,
drawing + JPEG encoding. 640x480 camera, 320x320 model.

Result on our board: inference was ~87% of the frame time; the camera delivered ~12.6 FPS on its own.

    sudo systemctl stop wildwatch
    ~/yolo-env/bin/python profile_pipeline.py [--model PATH] [--frames 40]
"""
import argparse
import glob
import os
import time

import cv2
import ncnn
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/yolo/yolo11n_320x320_ncnn/model.ncnn"), help="ncnn model path without .param/.bin")
ap.add_argument("--frames", type=int, default=40)
args = ap.parse_args()

net = ncnn.Net()
net.opt.num_threads = 4
net.load_param(args.model + ".param")
net.load_model(args.model + ".bin")


def infer(blob):
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(blob))
    return np.array(ex.extract("out0")[1])


cams = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
cap = cv2.VideoCapture(os.path.realpath(cams[0]) if cams else "/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(10):
    cap.read()

t0, n = time.time(), 60
for _ in range(n):
    cap.read()
print(f"camera alone: {n / (time.time() - t0):.1f} FPS")

stages = dict.fromkeys(["read", "prep", "infer", "post", "draw+enc"], 0.0)
for _ in range(args.frames):
    t0 = time.time()
    ok, fr = cap.read()
    t1 = time.time()
    canvas = np.full((320, 320, 3), 114, np.uint8)
    canvas[40:280] = cv2.resize(fr, (320, 240))
    blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    t2 = time.time()
    out = infer(blob)
    t3 = time.time()
    keep = out[4] > 0.4
    boxes = np.stack([out[0][keep], out[1][keep], out[2][keep], out[3][keep]], 1)
    cv2.dnn.NMSBoxes(boxes.tolist(), out[4][keep].tolist(), 0.4, 0.45)
    t4 = time.time()
    cv2.rectangle(fr, (10, 10), (100, 100), (0, 255, 0), 2)
    cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    t5 = time.time()
    for k, d in zip(stages, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]):
        stages[k] += d
total = sum(stages.values())
for k, v in stages.items():
    print(f"  {k:9s} {v / args.frames * 1000:6.1f} ms  ({v / total * 100:4.1f}%)")
print(f"  total     {total / args.frames * 1000:6.1f} ms -> {args.frames / total:.1f} FPS")
