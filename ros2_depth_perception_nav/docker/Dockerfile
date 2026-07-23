# ROS 2 Humble + Gazebo Classic + Nav2 + RTAB-Map for the farm coverage mission.
# Host is Ubuntu 24.04 (Humble needs 22.04) -> everything runs in this container.
FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    ros-humble-rtabmap-ros \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-message-filters \
    ros-humble-sensor-msgs-py \
    ros-humble-tf-transformations \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    python3-colcon-common-extensions \
    xvfb \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# GPU rendering (host: NVIDIA). Do NOT force software GL or Nav2 lifecycle times out.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV QT_X11_NO_MITSHM=1

WORKDIR /farm_nav_ws
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
