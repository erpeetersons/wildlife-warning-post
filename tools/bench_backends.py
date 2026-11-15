#!/usr/bin/env python3
"""Compare YOLO11n inference backends on the UNO Q: ncnn on the Adreno GPU (Vulkan, fp16/fp32),
ncnn on the CPU, and onnxruntime on the CPU. Input 320x320, random data.

Results on our board: ncnn CPU 9.4 FPS, onnxruntime CPU 6.0 FPS, ncnn GPU 1.8 FPS.
Stop wildwatch first for clean numbers:  sudo systemctl stop wildwatch

    ~/yolo-env/bin/python bench_backends.py [--ncnn PATH] [--onnx PATH]
"""
import argparse
import os
import time

import ncnn
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ncnn", default=os.path.expanduser("~/yolo/yolo11n_320x320_ncnn/model.ncnn"), help="ncnn model path without .param/.bin")
ap.add_argument("--onnx", default=os.path.expanduser("~/yolo/yolo11n_320.onnx"), help="ONNX model (skipped if missing)")
ap.add_argument("--runs", type=int, default=20)
args = ap.parse_args()

x = np.random.rand(3, 320, 320).astype(np.float32)


def cpu_time():
    return sum(os.times()[:2])


def bench_ncnn(gpu, fp16=True, threads=4):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = gpu
    if gpu:
        net.set_vulkan_device(0)
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
    net.opt.num_threads = threads
    net.load_param(args.ncnn + ".param")
    net.load_model(args.ncnn + ".bin")

    def run():
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(x))
        return np.array(ex.extract("out0")[1])
    for _ in range(3):
        run()
    c0, t0 = cpu_time(), time.time()
    for _ in range(args.runs):
        out = run()
    dt = (time.time() - t0) / args.runs
    return dt, (cpu_time() - c0) / args.runs, out.shape


for name, gpu, fp16 in [("ncnn GPU fp16", True, True), ("ncnn GPU fp32", True, False), ("ncnn CPU", False, True)]:
    dt, cpu, shape = bench_ncnn(gpu, fp16)
    print(f"{name:16s} {dt * 1000:6.0f} ms  {1 / dt:5.1f} FPS  cpu-s/frame {cpu:.2f}  out {shape}", flush=True)

if os.path.exists(args.onnx):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    for _ in range(3):
        s.run(None, {name: x[None]})
    c0, t0 = cpu_time(), time.time()
    for _ in range(args.runs):
        s.run(None, {name: x[None]})
    dt = (time.time() - t0) / args.runs
    print(f"{'onnxruntime CPU':16s} {dt * 1000:6.0f} ms  {1 / dt:5.1f} FPS  cpu-s/frame {(cpu_time() - c0) / args.runs:.2f}")
else:
    print(f"onnxruntime: skipped ({args.onnx} not found)")
