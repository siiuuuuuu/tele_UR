#!/usr/bin/env bash
#bash replay_trajectory.sh


# Edit these settings before replaying a trajectory on the real robot.
# Keep the robot, workspace, initial pose, dt, and hand settings consistent
# with the values used during data collection.
data_file="${HOME}/dp_data/tast_task1_expertdata/demo_20260609_120148.h5"
ur_host="192.168.3.6"
workspace_x=(-1.5 1.5)
workspace_y=(-1.5 1.5)
workspace_z=(-0.5 1.5)
initial_pose=(0.248 0.1212 0.3978 1.16 1.25 1.28)
dt="0.04"
hand_port="/dev/ttyUSB0"
hand_baudrate="115200"
speed="1"
# Reject playback if the first target is farther than this distance, in meters.
max_initial_distance="0.10"

sudo chmod 666 "${hand_port}"

python traj_valid.py \
  --data_file "${data_file}" \
  --ur_host "${ur_host}" \
  --workspace_x "${workspace_x[@]}" \
  --workspace_y "${workspace_y[@]}" \
  --workspace_z "${workspace_z[@]}" \
  --initial_pose "${initial_pose[@]}" \
  --dt "${dt}" \
  --hand_port "${hand_port}" \
  --hand_baudrate "${hand_baudrate}" \
  --speed "${speed}" \
  --max_initial_distance "${max_initial_distance}"
