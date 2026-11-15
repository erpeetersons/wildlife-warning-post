#!/bin/bash
# Sets up a TigerVNC Xfce desktop on :1 (port 5901) that starts at boot.
set -e
[ "$(id -u)" = 0 ] || { echo "Run with: sudo bash $0"; exit 1; }
U=arduino
H=/home/$U
CFG=$H/.config/tigervnc

apt-get update
apt-get install -y tigervnc-standalone-server tigervnc-tools dbus-x11

# Map display :1 to the arduino user
touch /etc/tigervnc/vncserver.users
grep -q '^:1=' /etc/tigervnc/vncserver.users || echo ":1=$U" >> /etc/tigervnc/vncserver.users

# Per-user VNC config
sudo -u $U mkdir -p "$CFG"
cat > "$CFG/config" <<EOF
session=xfce
geometry=1920x1080
localhost=no
alwaysshared
EOF
chown $U:$U "$CFG/config"

# VNC password (asked once)
if [ ! -s "$CFG/passwd" ]; then
  echo ">>> Choose a VNC password (6-8 chars). Answer 'n' to the view-only question."
  sudo -u $U vncpasswd "$CFG/passwd"
fi
sudo -u $U ln -sfn "$CFG/passwd" "$H/.vnc/passwd" 2>/dev/null || { sudo -u $U mkdir -p "$H/.vnc"; sudo -u $U ln -sfn "$CFG/passwd" "$H/.vnc/passwd"; }

UNIT=$(systemctl list-unit-files --no-legend 'tigervncserver@*' 'vncserver@*' | awk '{print $1}' | head -1)
UNIT=${UNIT%@.service}
echo "Using unit ${UNIT}@:1"
systemctl enable --now "${UNIT}@:1"
sleep 3
systemctl --no-pager status "${UNIT}@:1" | head -5
ss -tln | grep 5901 && echo ">>> VNC is listening on port 5901"
