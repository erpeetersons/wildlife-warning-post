#!/bin/bash
# wildwatch runs as a systemd service (starts at boot, restarts on crash).
#   ./run.sh status | restart | stop | start | logs
case "$1" in
  logs) journalctl -u wildwatch -f ;;
  status) systemctl status wildwatch --no-pager | head -12 ;;
  start|stop|restart) sudo systemctl "$1" wildwatch ;;
  *) echo "usage: $0 status|restart|stop|start|logs" ;;
esac
