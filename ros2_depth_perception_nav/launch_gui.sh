#!/usr/bin/env bash
# GUI run: full stack with the Gazebo client + RViz visible (for the live demo /
# screen capture). Builds the image and workspace, then launches.
set -e
WS="$(cd "$(dirname "$0")" && pwd)"
IMAGE=farm_nav:humble
DISPLAY_NUM="${DISPLAY:-:0}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> building docker image $IMAGE"
  docker build -t "$IMAGE" -f "$WS/docker/Dockerfile" "$WS/docker"
fi

mkdir -p "$WS/output"
xhost +local:root || true

docker run --rm -it \
  --gpus all \
  \
  -e DISPLAY="$DISPLAY_NUM" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -v "$WS/src:/farm_nav_ws/src:rw" \
  -v "$WS/output:/farm_nav_ws/output:rw" \
  "$IMAGE" \
  bash -lc '
    source /opt/ros/humble/setup.bash
    cd /farm_nav_ws
    colcon build --symlink-install
    source install/setup.bash
    ros2 launch farm_bringup bringup.launch.py \
      use_rviz:=true use_slam:=true capture:=true headless:=false
  '
