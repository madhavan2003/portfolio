# Running the demo (after install.sh)

This is the part `install.sh` cannot do for you: PX4 SITL, the PX4<->ROS 2
bridge, and this project's mission launch are three separate long-running
foreground processes. Open three terminals.

Run these once per new shell (all three terminals need it):

```bash
source /opt/ros/humble/setup.bash
source $HOME/px4_ros2_ws/install/setup.bash    # must come after humble's setup.bash --
                                                # this overlays the correct Harmonic-built
                                                # ros_gz_image and the v1.17.0-pinned
                                                # px4_msgs ahead of any apt-installed version
```

## Terminal 1 -- PX4 SITL + Gazebo

```bash
cd $HOME/PX4-Autopilot
make px4_sitl gz_x500_mono_cam
```

Wait for the Gazebo GUI window and the PX4 shell prompt (`pxh>`) before
moving on.

## Terminal 2 -- PX4 <-> ROS 2 bridge

```bash
MicroXRCEAgent udp4 -p 8888
```

Wait for repeated `session established` / client-connect log lines.

## Terminal 3 -- this project's deliverable

```bash
cd <this repo>
source .venv/bin/activate
ros2 launch sim/launch/search_and_follow.launch.py \
  mission:=mission/mission_example.json \
  vision_config:=config/target_config.yaml
```

## Spawn a target for the drone to see

`sim/worlds/target_actor.sdf` is an early scaffold, superseded by
`target_person` (a life-size, `grace_hopper.jpg`-textured box shipped in
this project's own PX4-Autopilot checkout, confirmed YOLO-detectable both
as a standalone image and live from the drone's onboard camera -- see
`readme.pdf`, "Known limitations"). Spawn that one instead. Needed once per
Gazebo session -- it's lost on a Gazebo restart, not a PX4-only restart:

```bash
gz service -s /world/default/create \
  --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
  --timeout 2000 \
  --req 'sdf_filename: "<PX4-Autopilot>/Tools/simulation/gz/models/target_person/model.sdf"'
```

## What to expect

This is the actual logged output from a verified run (`readme.pdf`,
"What to expect on screen"); paths and timestamps will differ, everything
else should match:

```
[INFO] [launch]: All log files can be found below /home/<you>/.ros/log/...
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [image_bridge-1]: process started with pid [...]
[INFO] [python3-2]: process started with pid [...]
[python3-2] [INFO] [...] [mission_executor]: step 0: success=True message=reached takeoff altitude 5.0m
[python3-2] [INFO] [...] [mission_executor]: step 1: success=True message=search_and_follow completed after 30.0s (final state: tracking)
[python3-2] [INFO] [...] [mission_executor]: step 2: success=True message=landed
[python3-2] [operator-notify:noop] target acquired -> snapshots/<timestamp>_person.jpg (no operator_endpoint configured, see snapshot_sender._notify_operator)
[INFO] [python3-2]: process has finished cleanly [pid ...]
```

Concretely: the drone arms and climbs to 5m (`takeoff`), then
`search_and_follow` runs while it searches; as soon as YOLO detects
`target_class` in the bridged camera feed, a snapshot lands in `snapshots/`
(filename pattern `<YYYYMMDD-HHMMSS>_<class_name>.jpg`) and the drone begins
adjusting yaw/altitude/forward speed to center the target and hold standoff
distance -- visible as real, changing velocity in
`/fmu/in/trajectory_setpoint` correlated with real, changing
`/fmu/out/vehicle_local_position_v1`, and as the vehicle turning/approaching
in the Gazebo GUI. `final state: tracking` means the target was still being
actively tracked, not lost, when `return_after_seconds` elapsed. After that
it hovers and the executor moves on to `land`.

## No-sim smoke test (skip all of the above)

If you just want to confirm the vision/control code runs before touching
PX4/Gazebo at all:

```bash
source .venv/bin/activate
python3 tests/mock_demo.py --source 0 --target-class person
python3 -m pytest tests/test_follow_controller.py tests/test_detector.py tests/test_executor.py -v
```
