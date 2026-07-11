#!/usr/bin/env bash
#bash convert_data.sh

# Edit these settings before converting regular teleoperation data.
# Source H5 files may contain a raw-only "timestamps" group; training Zarr
# intentionally drops it and keeps only image/state/action/meta fields.
demo_dir="${HOME}/dp_data/task4_demo"
# This directory is overwritten if it already exists.
save_dir="${HOME}/dp_data/test_zarr_task4"
# Use 1 to include an observation type in Zarr, or 0 to skip it.
save_img="1"
save_wrist_img="1"
save_depth="0"
save_cloud="0"

python convert_demos.py \
  --demo_dir "${demo_dir}" \
  --save_dir "${save_dir}" \
  --save_img "${save_img}" \
  --save_wrist_img "${save_wrist_img}" \
  --save_depth "${save_depth}" \
  --save_cloud "${save_cloud}"
