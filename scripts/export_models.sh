#!/bin/bash
# Export YOLO11n to ncnn at the input sizes the pipeline uses. Run on a PC (needs ~2 GB for PyTorch).
#   ./scripts/export_models.sh [output-dir]      (default: ./models)
# Copy the resulting yolo11n_<W>x<H>_ncnn folders next to wildwatch.py on the board (~/yolo/).
set -e
OUT=$(realpath "${1:-models}")
mkdir -p "$OUT"; cd "$OUT"
python3 -m venv .export-env
.export-env/bin/pip install -q --upgrade pip
.export-env/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
.export-env/bin/pip install -q ultralytics onnx onnxslim
# W x H pairs: 512x288 = full-frame mode (default), 320x192 = motion+zoom crops, 640x352 = slower, more detail
for WH in 512x288 320x192 640x352; do
  W=${WH%x*}; H=${WH#*x}
  rm -rf yolo11n_ncnn_model
  .export-env/bin/yolo export model=yolo11n.pt format=ncnn imgsz=$H,$W >/dev/null
  rm -rf "yolo11n_${W}x${H}_ncnn" && mv yolo11n_ncnn_model "yolo11n_${W}x${H}_ncnn"
  echo "exported yolo11n_${W}x${H}_ncnn"
done
