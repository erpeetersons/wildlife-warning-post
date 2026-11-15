# The Linux-side logic lives in ~/yolo/wildwatch.py, which calls set_state over the Router.
# This app only exists so arduino-app-cli builds and flashes the sketch.
from arduino.app_utils import App
App.run()
