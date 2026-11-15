#!/usr/bin/env python3
"""Radar simulator: sends radar target lists to wildwatch over UDP, like a real radar reader would.

  python radar_sim.py moose  --angle -10   # slow target approaching from 90 m
  python radar_sim.py person --angle 5     # slow target at 20 m (point --angle at a real person/photo for camera confirmation)
  python radar_sim.py car                  # fast target along the road -> ignored
  python radar_sim.py rain                 # random short-lived echoes -> ignored
  python radar_sim.py fog  --angle -10     # like moose; with nothing real at that bearing the camera can't confirm -> radar-only caution

Message format (10 Hz): {"targets": [{"id": 1, "range_m": 72.0, "angle_deg": -8.0, "speed_mps": -1.5}]}
angle: negative = left of the camera centre; speed: negative = approaching.
"""
import argparse
import json
import random
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("scenario", choices=["moose", "person", "car", "rain", "fog"])
ap.add_argument("--angle", type=float, default=-10.0, help="bearing of the target in degrees")
ap.add_argument("--seconds", type=float, default=40.0)
ap.add_argument("--port", type=int, default=5005)
a = ap.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t0, nid = time.time(), 1000
while (t := time.time() - t0) < a.seconds:
    targets = []
    if a.scenario in ("moose", "fog"):
        targets.append({"id": 1, "range_m": max(8.0, 90 - 1.6 * t), "angle_deg": a.angle, "speed_mps": -1.6})
    elif a.scenario == "person":
        targets.append({"id": 2, "range_m": max(5.0, 20 - 0.3 * t), "angle_deg": a.angle, "speed_mps": -0.3})
    elif a.scenario == "car":
        r = 140 - 22 * (t % 7)
        if r > 5:
            targets.append({"id": 3 + int(t // 7), "range_m": r, "angle_deg": 3.0, "speed_mps": -22.0})
    elif a.scenario == "rain":
        for _ in range(20):
            nid += 1
            targets.append({"id": nid, "range_m": random.uniform(5, 135), "angle_deg": random.uniform(-28, 28),
                            "speed_mps": random.uniform(-3, 3)})
    sock.sendto(json.dumps({"targets": targets}).encode(), ("127.0.0.1", a.port))
    time.sleep(0.1)
print("done")
