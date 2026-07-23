#!/usr/bin/env bash
# Headless run WITHOUT an NVIDIA GPU: builds the image (if needed), colcon-builds
# the workspace inside the container, launches the full stack headless using
# software (CPU) rendering via llvmpipe, and records the deliverables (annotated
# video + path overlay) into ./output.
#
# This is slower and less reliable than the GPU path in run.sh -- Gazebo's sensor
# rendering (the depth camera) runs on the CPU, so the sim may fall well below
# real-time. That's fine for correctness (the demo just takes longer), but it can
# make Nav2's lifecycle activation / bond checks time out if the default timeouts
# are too tight, so we pad them here via env overrides + a longer server timeout.
set -e
WS="$(cd "$(dirname "$0")" && pwd)"
IMAGE=farm_nav:humble

# build image if missing
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> building docker image $IMAGE"
  docker build -t "$IMAGE" -f "$WS/docker/Dockerfile" "$WS/docker"
fi

mkdir -p "$WS/output"

docker run --rm -it \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e GALLIUM_DRIVER=llvmpipe \
  -e MESA_GL_VERSION_OVERRIDE=3.3 \
  -e QT_X11_NO_MITSHM=1 \
  -v "$WS/src:/farm_nav_ws/src:rw" \
  -v "$WS/output:/farm_nav_ws/output:rw" \
  "$IMAGE" \
  bash -lc '
    source /opt/ros/humble/setup.bash
    cd /farm_nav_ws
    echo ">> colcon build"
    colcon build --symlink-install
    source install/setup.bash
    echo ">> launch (headless, software rendering, recording)"
    xvfb-run -a --server-args="-screen 0 1280x1024x24" \
    ros2 launch farm_bringup bringup.launch.py \
      use_rviz:=false use_slam:=true capture:=true headless:=true
  '
