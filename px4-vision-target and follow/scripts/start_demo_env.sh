#!/usr/bin/env bash
# Brings up everything needed for a search_and_follow demo recording EXCEPT
# the actual mission launch -- run this off-camera first, wait for
# "Environment ready.", then start recording and run the mission launch
# command (printed at the end) on camera so its log output is what shows up
# in the recording.
#
# Usage: PX4_AUTOPILOT=~/PX4-Autopilot ./scripts/start_demo_env.sh
set -euo pipefail

PX4_AUTOPILOT="${PX4_AUTOPILOT:-$HOME/PX4-Autopilot}"
PX4_ROS2_WS="${PX4_ROS2_WS:-$HOME/px4_ros2_ws}"

echo "==> Stopping any previous PX4/Gazebo/Agent instances"
pkill -9 -f "build/px4_sitl_default/bin/px4" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
sleep 2

echo "==> Launching PX4 SITL + Gazebo (gz_x500_mono_cam)"
(cd "$PX4_AUTOPILOT" && nohup make px4_sitl gz_x500_mono_cam < /dev/null > /dev/null 2>&1 &)

echo "==> Waiting for PX4/Gazebo to come up"
for i in $(seq 1 60); do
    if pgrep -f "build/px4_sitl_default/bin/px4" > /dev/null && pgrep -f "gz sim -g" > /dev/null; then
        break
    fi
    sleep 1
done
sleep 3

echo "==> Launching Micro XRCE-DDS Agent"
nohup MicroXRCEAgent udp4 -p 8888 < /dev/null > /dev/null 2>&1 &
sleep 2

echo "==> Spawning target_person"
gz service -s /world/default/create \
  --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
  --timeout 2000 \
  --req "sdf_filename: \"$PX4_AUTOPILOT/Tools/simulation/gz/models/target_person/model.sdf\""

echo ""
echo "Environment ready."
echo ""
echo "Now, in a NEW terminal (this is the one to have visible/recording):"
echo ""
echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $PX4_ROS2_WS/install/setup.bash"
echo "  ros2 launch sim/launch/search_and_follow.launch.py \\"
echo "    mission:=mission/mission_example.json \\"
echo "    vision_config:=config/target_config.yaml"
