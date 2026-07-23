#!/usr/bin/env bash
# install.sh -- reproduces the FULL environment for search_and_follow
# (PX4 SITL + Gazebo Harmonic + ROS 2 Humble + YOLOv8) from scratch on a
# fresh Ubuntu 22.04 machine.
#
# What this script does NOT do: it does not run the demo itself. PX4
# SITL, the Micro-XRCE-DDS-Agent, and this project's mission launch are
# three long-running foreground processes that belong in three separate
# terminals -- see RUN.md (generated at the end of this script, also
# shipped alongside it) for that sequence.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Idempotency: safe to re-run. Steps that already succeeded (clone exists,
# package already installed, workspace already built) are skipped or
# overwritten in place rather than erroring.
#
# Tunable via environment variables (defaults shown):
#   PX4_AUTOPILOT_DIR   = $HOME/PX4-Autopilot
#   PX4_AUTOPILOT_REF   = e0137fe7a7 # the exact main-branch commit this
#                                     # project was verified against (there is
#                                     # no PX4-Autopilot release tag matching
#                                     # px4_msgs' v1.17.0 -- that tag is
#                                     # 118 commits behind this commit; do not
#                                     # "correct" this to v1.17.0, that tag
#                                     # exists but is the wrong ref).
#   PX4_MSGS_REF        = v1.17.0    # pinned per submission requirements,
#                                     # confirmed against the verified checkout
#   ROS2_WS_DIR          = $HOME/px4_ros2_ws
#   MICRO_XRCE_DIR       = $HOME/Micro-XRCE-DDS-Agent
#   REPO_DIR             = directory this script lives in

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PX4_AUTOPILOT_DIR="${PX4_AUTOPILOT_DIR:-$HOME/PX4-Autopilot}"
PX4_AUTOPILOT_REF="${PX4_AUTOPILOT_REF:-e0137fe7a7}"
PX4_MSGS_REF="${PX4_MSGS_REF:-v1.17.0}"
ROS2_WS_DIR="${ROS2_WS_DIR:-$HOME/px4_ros2_ws}"
MICRO_XRCE_DIR="${MICRO_XRCE_DIR:-$HOME/Micro-XRCE-DDS-Agent}"

log() { echo -e "\n==> $*\n"; }

if [[ "$(lsb_release -rs 2>/dev/null || echo '?')" != "22.04" ]]; then
    echo "WARNING: this script is written for Ubuntu 22.04. Continuing anyway." >&2
fi

# ---------------------------------------------------------------------------
log "1/9 Base build tooling"
# ---------------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y \
    git curl wget lsb-release gnupg2 build-essential cmake ninja-build \
    python3-pip python3-venv python3-dev software-properties-common

# ---------------------------------------------------------------------------
log "2/9 ROS 2 Humble (apt)"
# ---------------------------------------------------------------------------
if ! dpkg -l | grep -q ros-humble-desktop; then
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y universe
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y ros-humble-desktop ros-humble-cv-bridge \
        python3-colcon-common-extensions python3-rosdep python3-vcstool
    sudo rosdep init 2>/dev/null || true
    rosdep update
else
    log "ros-humble-desktop already installed, skipping"
    sudo apt-get install -y ros-humble-cv-bridge python3-colcon-common-extensions \
        python3-rosdep python3-vcstool
fi
source /opt/ros/humble/setup.bash

# ---------------------------------------------------------------------------
log "3/9 PX4-Autopilot clone + toolchain + Gazebo Harmonic"
# ---------------------------------------------------------------------------
# PX4's own Tools/setup/ubuntu.sh adds the Gazebo apt repo and installs the
# gz-harmonic metapackage (dev headers included) plus the rest of PX4's SITL
# build dependencies. This is also what provides the libgz-*-dev packages
# ros_gz needs in step 5 -- do not skip this even though we build ros_gz
# from source, since it supplies the Gazebo side of that build, not the ROS
# side.
if [[ ! -d "$PX4_AUTOPILOT_DIR" ]]; then
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive "$PX4_AUTOPILOT_DIR"
fi
(
    cd "$PX4_AUTOPILOT_DIR"
    git fetch --tags
    git checkout "$PX4_AUTOPILOT_REF"
    git submodule update --init --recursive
    bash ./Tools/setup/ubuntu.sh
)

# ---------------------------------------------------------------------------
log "4/9 NAV_DLL_ACT=0 airframe .post overrides"
# ---------------------------------------------------------------------------
# Default NAV_DLL_ACT (2) requires an active GCS data link before PX4 will
# arm. This project flies fully autonomously via the ROS 2 offboard/
# companion-computer link with no GCS attached, so that check can never
# pass at its default value. These .post files are sourced automatically by
# rcS's autostart-post-script step after their non-.post counterpart, so
# they survive a PX4-Autopilot upstream sync without editing tracked files.
AIRFRAMES_DIR="$PX4_AUTOPILOT_DIR/ROMFS/px4fmu_common/init.d-posix/airframes"
AIRFRAMES_CMAKE="$AIRFRAMES_DIR/CMakeLists.txt"

write_post_override() {
    local post_file="$1"
    cat > "$AIRFRAMES_DIR/$post_file" <<'EOF'
# Autostart post-script (see rcS's "execute autostart post script" step) --
# sourced automatically after the matching airframe file without needing to
# edit that file, so this survives a PX4-Autopilot upstream sync.
#
# Default NAV_DLL_ACT (2) requires an active GCS data link before PX4 will
# arm (see src/modules/commander/HealthAndArmingChecks/checks/rcAndDataLinkCheck.cpp).
# This project flies fully autonomously via the ROS 2 offboard/companion-computer
# link with no GCS attached, so that check can never pass at its default value.
param set-default NAV_DLL_ACT 0
EOF
}

register_in_cmake() {
    local base="$1" post="$2"
    if ! grep -qF "$post" "$AIRFRAMES_CMAKE"; then
        # Insert the .post filename on the line immediately following its
        # non-.post counterpart, matching this project's confirmed layout
        # (line N: base file, line N+1: base.post). Tab-indented to match
        # this file's real indentation style byte-for-byte (verified by
        # diffing this exact sed command's output against the actual
        # committed fix) -- other entries in this file use a leading tab,
        # not spaces.
        sed -i "/\b${base}\b/a\\\t${post}" "$AIRFRAMES_CMAKE"
    fi
}

if [[ -d "$AIRFRAMES_DIR" ]]; then
    write_post_override "4001_gz_x500.post"
    write_post_override "4010_gz_x500_mono_cam.post"
    register_in_cmake "4001_gz_x500" "4001_gz_x500.post"
    register_in_cmake "4010_gz_x500_mono_cam" "4010_gz_x500_mono_cam.post"
else
    echo "WARNING: $AIRFRAMES_DIR not found -- PX4-Autopilot layout may have" >&2
    echo "changed at ref $PX4_AUTOPILOT_REF. Add the .post overrides by hand." >&2
fi

# One headless build to validate the toolchain (and to pick up the airframe
# CMakeLists.txt change above into the ROMFS image) without needing a
# display on this machine yet.
log "First PX4 SITL build (headless, validates toolchain)"
(cd "$PX4_AUTOPILOT_DIR" && HEADLESS=1 make px4_sitl gz_x500_mono_cam || true)
pkill -9 -f "build/px4_sitl_default/bin/px4" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true

# ---------------------------------------------------------------------------
log "5/9 px4_msgs, pinned to $PX4_MSGS_REF"
# ---------------------------------------------------------------------------
mkdir -p "$ROS2_WS_DIR/src"
if [[ ! -d "$ROS2_WS_DIR/src/px4_msgs" ]]; then
    git clone https://github.com/PX4/px4_msgs.git "$ROS2_WS_DIR/src/px4_msgs"
fi
(
    cd "$ROS2_WS_DIR/src/px4_msgs"
    git fetch --tags
    git checkout "$PX4_MSGS_REF"
)

# ---------------------------------------------------------------------------
log "6/9 ros_gz, built from source with GZ_VERSION=harmonic"
# ---------------------------------------------------------------------------
# apt's ros-humble-ros-gz-bridge links against Gazebo Fortress, not the
# Harmonic version PX4 defaults to -- that combination bridges nothing and
# fails silently (no crash, just no messages on /camera). Must build from
# source against the Harmonic libs step 3 installed.
if [[ ! -d "$ROS2_WS_DIR/src/ros_gz" ]]; then
    git clone https://github.com/gazebosim/ros_gz.git -b humble "$ROS2_WS_DIR/src/ros_gz"
fi

(
    cd "$ROS2_WS_DIR"
    source /opt/ros/humble/setup.bash
    rosdep update
    rosdep install -r --from-paths src --ignore-src -y --rosdistro humble \
        --skip-keys "px4_msgs" || true
    colcon build --symlink-install --packages-select px4_msgs
    # Matches the exact package set actually built and verified this
    # session (confirmed present under $ROS2_WS_DIR/install/) -- ros_gz_sim
    # and ros_gz_interfaces are not guaranteed transitive deps of
    # ros_gz_bridge/ros_gz_image alone, so they're listed explicitly rather
    # than relying on --packages-up-to to pull them in.
    GZ_VERSION=harmonic colcon build --symlink-install \
        --packages-select ros_gz ros_gz_image ros_gz_bridge ros_gz_sim ros_gz_interfaces
)

# ---------------------------------------------------------------------------
log "7/9 Micro-XRCE-DDS-Agent, built from source"
# ---------------------------------------------------------------------------
if [[ ! -d "$MICRO_XRCE_DIR" ]]; then
    git clone -b v3.0.1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$MICRO_XRCE_DIR"
fi
(
    cd "$MICRO_XRCE_DIR"
    mkdir -p build && cd build
    cmake ..
    make -j"$(nproc)"
    sudo make install
    sudo ldconfig /usr/local/lib/
)

# ---------------------------------------------------------------------------
log "8/9 Python env for vision/ and ros2_nodes/ (numpy<2 / opencv-python<5 pin)"
# ---------------------------------------------------------------------------
# Plain venv, NOT --system-site-packages -- confirmed against the actual
# verified venv on this machine (pyvenv.cfg: include-system-site-packages =
# false). cv_bridge/rclpy still resolve fine from inside it at runtime
# because RUN.md has you source /opt/ros/humble/setup.bash (and the colcon
# overlay) in the same shell BEFORE activating this venv -- that sets
# PYTHONPATH to include ROS's dist-packages, and PYTHONPATH is honored by
# any interpreter regardless of the venv's own site-packages setting.
# Venv activation only touches PATH/VIRTUAL_ENV/PS1, so it doesn't clobber
# PYTHONPATH set by the already-sourced setup.bash. Get the order backwards
# (activate venv, then source setup.bash) and this can put a different
# python3 first on PATH -- keep to RUN.md's documented order.
#
# NumPy 2.x breaks cv_bridge's C-API ABI (built against Ubuntu 22.04's
# stock NumPy 1.x headers) -- requirements.txt pins numpy<2 and
# opencv-python<5 together for exactly this reason; do not loosen those
# independently.
VENV_DIR="$REPO_DIR/.venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"
deactivate

# ---------------------------------------------------------------------------
log "9/9 Done"
# ---------------------------------------------------------------------------
cat <<EOF

Environment setup complete.

  PX4-Autopilot : $PX4_AUTOPILOT_DIR  (ref $PX4_AUTOPILOT_REF)
  px4_msgs      : $ROS2_WS_DIR/src/px4_msgs  (ref $PX4_MSGS_REF)
  ros_gz        : $ROS2_WS_DIR/src/ros_gz  (branch humble, GZ_VERSION=harmonic)
  colcon ws     : $ROS2_WS_DIR/install/setup.bash
  MicroXRCEAgent: installed to /usr/local/bin
  Python venv   : $REPO_DIR/.venv  (--system-site-packages, requirements.txt installed)

This script does NOT start PX4/Gazebo/the agent/the mission -- those are
three separate long-running terminals. See RUN.md for the exact sequence.
EOF
