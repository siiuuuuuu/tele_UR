#!/usr/bin/env bash
# Merge rollout/expert H5 data into Zarr with future action-label offsets.
#
# Usage:
#   bash convert_rollout_data_action_offset.sh        # offsets +1 +2 +3
#   bash convert_rollout_data_action_offset.sh 2      # only offset +2
#   bash convert_rollout_data_action_offset.sh 1 3    # offsets +1 and +3

set -euo pipefail

# Edit these settings before conversion.
demo_dirs=(
  "${HOME}/dp_data/task1_expertdata"
  "${HOME}/dp_data/task1_rollout"
)
save_dir_prefix="${HOME}/dp_data/task1_Recap_iter2_action_t_plus"

# Use 1 to include an observation type in Zarr, or 0 to skip it.
save_img="1"
save_wrist_img="1"
save_depth="0"
save_cloud="0"

# Value used when an H5 file has no intervention dataset:
# 0=non-intervention, 1=intervention.
default_intervention="0"

if [[ "$#" -gt 0 ]]; then
  offsets=("$@")
else
  offsets=("1" "2" "3")
fi

for action_offset_frames in "${offsets[@]}"; do
  save_dir="${save_dir_prefix}_${action_offset_frames}"

  python convert_demos_rollout.py \
    --demo_dirs "${demo_dirs[@]}" \
    --save_dir "${save_dir}" \
    --overwrite \
    --save_img "${save_img}" \
    --save_wrist_img "${save_wrist_img}" \
    --save_depth "${save_depth}" \
    --save_cloud "${save_cloud}" \
    --default_intervention "${default_intervention}" \
    --action_offset_frames "${action_offset_frames}"
done
