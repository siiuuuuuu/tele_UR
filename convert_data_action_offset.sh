#!/usr/bin/env bash
# Convert task3 H5 demos to Zarr with future action-label offsets.
#
# Usage:
#   bash convert_data_action_offset.sh        # converts offsets +1 +2 +3
#   bash convert_data_action_offset.sh 2      # converts only offset +2
#   bash convert_data_action_offset.sh 1 3    # converts offsets +1 and +3

set -euo pipefail

demo_dir="${HOME}/dp_data/task4_demo"

# Use 1 to include an observation type in Zarr, or 0 to skip it.
save_img="1"
save_wrist_img="1"
save_depth="0"
save_cloud="0"

if [ "$#" -gt 0 ]; then
  offsets=("$@")
else
  offsets=("1" "2" "3")
fi

for action_offset_frames in "${offsets[@]}"; do
  save_dir="${HOME}/dp_data/test_zarr_task4_action_t_plus_${action_offset_frames}"

  python convert_demos.py \
    --demo_dir "${demo_dir}" \
    --save_dir "${save_dir}" \
    --save_img "${save_img}" \
    --save_wrist_img "${save_wrist_img}" \
    --save_depth "${save_depth}" \
    --save_cloud "${save_cloud}" \
    --action_offset_frames "${action_offset_frames}"
done
