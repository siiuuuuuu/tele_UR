#!/usr/bin/env bash
#bash convert_rollout_data.sh
# Edit these settings before merging and converting rollout/demo data.
# Source H5 files may contain a raw-only "timestamps" group; the training
# Zarr+memmap dataset intentionally drops it.
demo_dirs=(
  #"${HOME}/dp_data/task2_expertdata"
  #"${HOME}/dp_data/second_task3_expertdata"
  #"${HOME}/dp_data/offlineRL_data/second_task3_iter1"
  #"${HOME}/dp_data/offlineRL_data/second_task3_iter2"
  "${HOME}/dp_data/offlineRL_data/task2_testset"

)
# This directory is overwritten if it already exists.
save_dir="${HOME}/dp_data/test_task2_dataset"
# Use 1 to include an observation type, or 0 to skip it. Visual arrays are
# written as NPY memmaps; state/action/intervention/meta remain in data.zarr.
save_img="1"
save_wrist_img="1"
save_depth="0"
save_cloud="0"
# Default training label: observation[i] -> action/intervention[i + 1].
action_offset_frames="1"
# Value for H5 files without an "intervention" dataset: 0=non-intervention, 1=intervention.
default_intervention="0"

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
