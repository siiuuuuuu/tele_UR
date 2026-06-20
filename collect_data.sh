#!/usr/bin/env bash
#bash collect_data.sh


# Edit these settings before collecting data.
demo_dir="${HOME}/dp_data/test_demo"
ur_host="192.168.3.6"
workspace_x=(-1.5 1.5)
workspace_y=(-1.5 1.5)
workspace_z=(-0.5 1.5)
initial_pose=(0.248 0.1212 0.3978 1.16 1.25 1.28)
dt="0.04"
tracker_frequency="60"
servo_frequency="120"
tracker_timeout="0.25"
hand_frequency="120"
robot_state_frequency="125"
manus_timeout="0.25"
manus_zmq_endpoint="tcp://127.0.0.1:2044"
manus_zmq_rcvhwm="1"
manus_zmq_conflate="true"
manus_zmq_poll_timeout_ms="100"
manus_control_threshold="10"
manus_scale_factor="15"
hand_smoothing_omega="25"
hand_smoothing_damping="0.8"
hand_input_alpha="0.6"
alignment_tolerance_ms="25"
max_length="1000"
hand_port="/dev/ttyUSB0"
hand_baudrate="115200"
use_wrist_img="true"

sudo chmod 666 "${hand_port}"

python servoL.py \
  --demo_dir "${demo_dir}" \
  --ur_host "${ur_host}" \
  --workspace_x "${workspace_x[@]}" \
  --workspace_y "${workspace_y[@]}" \
  --workspace_z "${workspace_z[@]}" \
  --initial_pose "${initial_pose[@]}" \
  --dt "${dt}" \
  --tracker_frequency "${tracker_frequency}" \
  --servo_frequency "${servo_frequency}" \
  --tracker_timeout "${tracker_timeout}" \
  --hand_frequency "${hand_frequency}" \
  --robot_state_frequency "${robot_state_frequency}" \
  --manus_timeout "${manus_timeout}" \
  --manus_zmq_endpoint "${manus_zmq_endpoint}" \
  --manus_zmq_rcvhwm "${manus_zmq_rcvhwm}" \
  --manus_zmq_conflate "${manus_zmq_conflate}" \
  --manus_zmq_poll_timeout_ms "${manus_zmq_poll_timeout_ms}" \
  --manus_control_threshold "${manus_control_threshold}" \
  --manus_scale_factor "${manus_scale_factor}" \
  --hand_smoothing_omega "${hand_smoothing_omega}" \
  --hand_smoothing_damping "${hand_smoothing_damping}" \
  --hand_input_alpha "${hand_input_alpha}" \
  --alignment_tolerance_ms "${alignment_tolerance_ms}" \
  --max_length "${max_length}" \
  --hand_port "${hand_port}" \
  --hand_baudrate "${hand_baudrate}" \
  --use_wrist_img "${use_wrist_img}"
