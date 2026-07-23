#!/usr/bin/env bash
# NOTE: do NOT `set -u` -- ROS setup.bash references unbound vars.
set -e
source /opt/ros/humble/setup.bash
if [ -f /farm_nav_ws/install/setup.bash ]; then
  source /farm_nav_ws/install/setup.bash
fi
exec "$@"
