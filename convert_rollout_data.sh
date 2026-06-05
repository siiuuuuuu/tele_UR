#!/usr/bin/env bash

# Edit these settings before merging and converting data with
# success/intervention metadata.
demo_dirs=(
  "${HOME}/dp_data/task1_expertdata"
  "${HOME}/dp_data/task1_rollout"
)
# This directory is overwritten if it already exists.
save_dir="${HOME}/dp_data/task1_Recap_iter2"
# Use 1 to include an observation type in Zarr, or 0 to skip it.
save_img="1"
save_wrist_img="1"
save_depth="0"
save_cloud="0"

python convert_demos_rollout.py \
  --demo_dirs "${demo_dirs[@]}" \
  --save_dir "${save_dir}" \
  --save_img "${save_img}" \
  --save_wrist_img "${save_wrist_img}" \
  --save_depth "${save_depth}" \
  --save_cloud "${save_cloud}"
