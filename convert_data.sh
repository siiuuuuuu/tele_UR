#!/usr/bin/env bash
#bash convert_data.sh

# Edit these settings before converting regular teleoperation data.
# Source H5 files may contain a raw-only "timestamps" group; the training
# Zarr+memmap dataset intentionally drops it.
demo_dir="${HOME}/dp_data/task3_test_demo"
# This directory is overwritten if it already exists.
save_dir="${HOME}/dp_data/zarr_task3_test"
# Use 1 to include an observation type, or 0 to skip it. Visual arrays are
# written as NPY memmaps; state/action/meta remain in data.zarr.
save_img="1"
save_wrist_img="1"
save_depth="0"
save_cloud="0"
# Default training label: observation[i] -> action[i + 1].
action_offset_frames="1"

python convert_demos.py \
  --demo_dir "${demo_dir}" \
  --save_dir "${save_dir}" \
  --overwrite \
  --save_img "${save_img}" \
  --save_wrist_img "${save_wrist_img}" \
  --save_depth "${save_depth}" \
  --save_cloud "${save_cloud}" \
  --action_offset_frames "${action_offset_frames}"
