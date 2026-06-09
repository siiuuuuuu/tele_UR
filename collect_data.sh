#!/usr/bin/env bash
#bash collect_data.sh


# Edit these settings before collecting data.
demo_dir="${HOME}/dp_data/tast_task1_expertdata"
ur_host="192.168.3.6"
workspace_x=(-1.5 1.5)
workspace_y=(-1.5 1.5)
workspace_z=(-0.5 1.5)
initial_pose=(0.248 0.1212 0.3978 1.16 1.25 1.28)
dt="0.04"
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
  --max_length "${max_length}" \
  --hand_port "${hand_port}" \
  --hand_baudrate "${hand_baudrate}" \
  --use_wrist_img "${use_wrist_img}"
