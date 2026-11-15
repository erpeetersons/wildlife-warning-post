#!/bin/bash
# Install the pipeline on an Arduino UNO Q from a PC on the same network.
#   ./scripts/install_board.sh <board-ip> [models-dir]
# Prerequisites: SSH access as user "arduino"; models exported with scripts/export_models.sh.
# Commands that need sudo prompt for the board's password.
set -e
B=arduino@${1:?usage: $0 <board-ip> [models-dir]}
M=${2:-models}
cd "$(dirname "$0")/.."
ssh "$B" 'mkdir -p ~/yolo ~/ArduinoApps'
scp board/wildwatch.py board/wildwatch.html board/radar_sim.py board/run.sh board/requirements.txt "$B":~/yolo/
scp -r "$M"/yolo11n_*_ncnn "$B":~/yolo/
scp -r mcu/warning-sign "$B":~/ArduinoApps/
scp board/wildwatch.service "$B":/tmp/
ssh -t "$B" '
  set -e
  sudo apt-get install -y python3-venv python3-pip
  sudo usermod -aG arduino-router arduino
  [ -d ~/yolo-env ] || python3 -m venv ~/yolo-env
  ~/yolo-env/bin/pip install -q -r ~/yolo/requirements.txt
  sudo cp /tmp/wildwatch.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now wildwatch
  arduino-app-cli properties set default ~/ArduinoApps/warning-sign
  arduino-app-cli app start ~/ArduinoApps/warning-sign
'
echo "Done. Open http://${1}:8080"
