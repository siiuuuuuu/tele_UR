#!/usr/bin/env bash
# bash replay_trajectory_mpc.sh

# Edit these settings before replaying a trajectory on the real robot.
# This script replays xyz through task-space MPC and keeps recorded rotation
# and hand commands unchanged.
data_file="${HOME}/dp_data/task3_demo/demo_20260701_121230.h5"
python_bin="${PYTHON_BIN:-/home/lrz/miniconda3/envs/tele/bin/python}"
ur_host="192.168.3.6"
workspace_x=(-1.0 1.0)
workspace_y=(-1.0 1.0)
workspace_z=(0.0 1.0)
initial_pose=(0.248 0.1212 0.3978 1.16 1.25 1.28)
dt="0.0333"
servo_frequency="30"
hand_port="/dev/ttyUSB0"
hand_baudrate="115200"
speed="1.0"
max_initial_distance="0.10"

# Low-level UR servoL parameters.
servo_speed="0.005"
servo_acceleration="0.005"
servo_lookahead_time="0.2"
servo_gain="500"

# MPC parameters. tau=0.12 was estimated from replay tracking CSV.
mpc_horizon="15"
mpc_tau="0.20"
mpc_iterations="24"
mpc_w_track="10.0"
mpc_track_decay="1.0"
mpc_w_cmd="0.0"
mpc_w_yx="0.0"
mpc_w_dy="0.0"
mpc_w_ddy="20.0"
mpc_max_cmd_actual_gap="0.0"
mpc_max_velocity="0.7"
mpc_max_acceleration="6.0"

# Print per-frame errors if you want live detail. Summary is always printed.
print_tracking_error="false"
# Leave empty to disable CSV output.
tracking_error_csv="${data_file%.h5}_mpc_replay_tracking_error.csv"

tracking_error_args=()
if [[ "${print_tracking_error}" == "true" ]]; then
  tracking_error_args+=(--print_tracking_error)
fi
if [[ -n "${tracking_error_csv}" ]]; then
  tracking_error_args+=(--tracking_error_csv "${tracking_error_csv}")
fi

sudo chmod 666 "${hand_port}"

"${python_bin}" traj_valid_mpc.py \
  --data_file "${data_file}" \
  --ur_host "${ur_host}" \
  --workspace_x "${workspace_x[@]}" \
  --workspace_y "${workspace_y[@]}" \
  --workspace_z "${workspace_z[@]}" \
  --initial_pose "${initial_pose[@]}" \
  --dt "${dt}" \
  --servo_frequency "${servo_frequency}" \
  --servo_speed "${servo_speed}" \
  --servo_acceleration "${servo_acceleration}" \
  --servo_lookahead_time "${servo_lookahead_time}" \
  --servo_gain "${servo_gain}" \
  --hand_port "${hand_port}" \
  --hand_baudrate "${hand_baudrate}" \
  --speed "${speed}" \
  --max_initial_distance "${max_initial_distance}" \
  --mpc_horizon "${mpc_horizon}" \
  --mpc_tau "${mpc_tau}" \
  --mpc_iterations "${mpc_iterations}" \
  --mpc_w_track "${mpc_w_track}" \
  --mpc_track_decay "${mpc_track_decay}" \
  --mpc_w_cmd "${mpc_w_cmd}" \
  --mpc_w_yx "${mpc_w_yx}" \
  --mpc_w_dy "${mpc_w_dy}" \
  --mpc_w_ddy "${mpc_w_ddy}" \
  --mpc_max_cmd_actual_gap "${mpc_max_cmd_actual_gap}" \
  --mpc_max_velocity "${mpc_max_velocity}" \
  --mpc_max_acceleration "${mpc_max_acceleration}" \
  "${tracking_error_args[@]}"
