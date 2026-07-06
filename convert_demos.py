import os
import argparse
import pickle
import numpy as np
import random
import time
from termcolor import colored
import h5py
import zarr
from termcolor import cprint
from tqdm import tqdm
import argparse


RAW_ONLY_H5_KEYS = ("timestamps",)
TRAINING_OUTPUT_DATA_KEYS = ("img", "wrist_img", "depth", "cloud", "state", "action")
TRAINING_OUTPUT_META_KEYS = ("episode_ends", "success")


def report_ignored_raw_only_keys(h5_data, file_name):
    ignored = [key for key in RAW_ONLY_H5_KEYS if key in h5_data]
    if ignored:
        cprint(
            f"ignore raw-only H5 keys for training conversion in {file_name}: "
            f"{', '.join(ignored)}",
            "cyan",
        )


def validate_training_zarr_schema(zarr_data, zarr_meta):
    data_keys = set(zarr_data.keys())
    meta_keys = set(zarr_meta.keys())
    raw_only = set(RAW_ONLY_H5_KEYS)

    leaked_keys = sorted((data_keys | meta_keys) & raw_only)
    if leaked_keys:
        raise RuntimeError(
            "raw-only timestamp fields leaked into training zarr: "
            + ", ".join(leaked_keys)
        )

    unexpected_data_keys = sorted(data_keys - set(TRAINING_OUTPUT_DATA_KEYS))
    unexpected_meta_keys = sorted(meta_keys - set(TRAINING_OUTPUT_META_KEYS))
    if unexpected_data_keys or unexpected_meta_keys:
        raise RuntimeError(
            "unexpected training zarr schema: "
            f"data={unexpected_data_keys}, meta={unexpected_meta_keys}"
        )


def convert_dataset(args):
    demo_dir = args.demo_dir
    save_dir = args.save_dir
    action_offset_frames = int(args.action_offset_frames)
    if action_offset_frames < 0:
        raise ValueError("--action_offset_frames must be >= 0")
    
    save_img = args.save_img
    save_wrist_img = args.save_wrist_img
    save_depth = args.save_depth
    save_cloud = args.save_cloud
    
    # create dir to save demonstrations
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".h5")]
    demo_files = sorted(demo_files)
    
    
    total_count = 0
    color_arrays = []
    wrist_color_arrays = []
    depth_arrays = []
    cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    success_arrays = []
    intervention_arrays = []
    
    skipped_short_episodes = 0

    for demo_file in demo_files:
        # load file (h5)
        file_name = os.path.join(demo_dir, demo_file)
        print("process:", file_name)

        with h5py.File(file_name, "r") as data:
            report_ignored_raw_only_keys(data, file_name)

            if save_img:
                color_array = data["color"][:]
            if save_wrist_img:
                wrist_color_array = data["wrist_color"][:]

            if save_depth:
                depth_array = data["depth"][:]
            if save_cloud:  
                cloud_array = data["cloud"][:]

            action_array = data["action"][:]
            proprioception_array = data["env_qpos_proprioception"][:]

            lengths = [len(action_array), len(proprioception_array)]
            if save_img:
                lengths.append(len(color_array))
            if save_wrist_img:
                lengths.append(len(wrist_color_array))
            if save_depth:
                lengths.append(len(depth_array))
            if save_cloud:
                lengths.append(len(cloud_array))

            raw_length = min(lengths)
            length = raw_length - action_offset_frames
            if length <= 0:
                skipped_short_episodes += 1
                cprint(
                    f"skip {file_name}: raw length {raw_length} <= "
                    f"action offset {action_offset_frames}",
                    "yellow",
                )
                continue

            obs_slice = slice(0, length)
            action_slice = slice(action_offset_frames, action_offset_frames + length)

            if save_img:
                color_array = color_array[obs_slice]
                color_array = [color_array[i] for i in range(length)]
            if save_wrist_img:
                wrist_color_array = wrist_color_array[obs_slice]
                wrist_color_array = [wrist_color_array[i] for i in range(length)]
              
            if save_depth:
                depth_array = depth_array[obs_slice]
                depth_array = [depth_array[i] for i in range(length)]
            if save_cloud:
                cloud_array = cloud_array[obs_slice]

            proprioception_array = proprioception_array[obs_slice]
            action_array = action_array[action_slice]

            proprioception_array = [proprioception_array[i] for i in range(length)]
            action_array = [action_array[i] for i in range(length)]
         
    
        total_count += len(action_array)
        if save_cloud:
            cloud_arrays.extend(cloud_array)
       
        if save_img:
            color_arrays.extend(color_array)
        if save_wrist_img:
            wrist_color_arrays.extend(wrist_color_array)
        if save_depth:
            depth_arrays.extend(depth_array)
            
        state_arrays.extend(proprioception_array)
        action_arrays.extend(action_array)
        episode_ends_arrays.append(total_count)

    if total_count == 0:
        raise RuntimeError(
            "no usable frames found; check demo_dir and --action_offset_frames"
        )

    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_root.attrs["action_offset_frames"] = action_offset_frames
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    if save_img:
        color_arrays = np.stack(color_arrays, axis=0)
        if color_arrays.shape[1] == 3: # make channel last
            color_arrays = np.transpose(color_arrays, (0,2,3,1))
    if save_wrist_img:
        wrist_color_arrays = np.stack(wrist_color_arrays, axis=0)
        if wrist_color_arrays.shape[1] == 3: # make channel last
            wrist_color_arrays = np.transpose(wrist_color_arrays, (0,2,3,1))
       
    if save_depth:
        depth_arrays = np.stack(depth_arrays, axis=0)
       
    if save_cloud:  
        cloud_arrays = np.stack(cloud_arrays, axis=0)

    state_arrays = np.stack(state_arrays, axis=0)      
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    success_arrays = np.ones_like(episode_ends_arrays, dtype=bool) #标记遥操数据全为成功

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    single_size = 500
    state_chunk_size = (single_size, state_arrays.shape[1])
    if save_cloud:
        point_cloud_chunk_size = (single_size, cloud_arrays.shape[1], cloud_arrays.shape[2])
    action_chunk_size = (single_size, action_arrays.shape[1])
    if save_img:
        img_chunk_size = (single_size, color_arrays.shape[1], color_arrays.shape[2], color_arrays.shape[3])
        zarr_data.create_dataset('img', data=color_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    if save_wrist_img:
        wrist_img_chunk_size = (single_size, wrist_color_arrays.shape[1], wrist_color_arrays.shape[2], wrist_color_arrays.shape[3])
        zarr_data.create_dataset('wrist_img', data=wrist_color_arrays, chunks=wrist_img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    if save_depth:
        depth_chunk_size = (single_size, depth_arrays.shape[1], depth_arrays.shape[2])
        zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        
    if save_cloud:  
        zarr_data.create_dataset('cloud', data=cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('success', data=success_arrays, dtype='bool', overwrite=True, compressor=compressor)
    validate_training_zarr_schema(zarr_data, zarr_meta)
    #包含每个episode结束时的全局索引，用于区分不同episode
    cprint(f'episode nums: {episode_ends_arrays.shape}', 'green')
    cprint(f'action offset frames: +{action_offset_frames}', 'green')
    if skipped_short_episodes:
        cprint(f'skipped short episodes: {skipped_short_episodes}', 'yellow')

    # print shape
    if save_img:
        cprint(f'color shape: {color_arrays.shape}, range: [{np.min(color_arrays)}, {np.max(color_arrays)}]', 'green')
    if save_wrist_img:
        cprint(f'wrist_img shape: {wrist_color_arrays.shape}, range: [{np.min(wrist_color_arrays)}, {np.max(wrist_color_arrays)}]', 'green')
    if save_depth:
        cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    if save_cloud:
        cprint(f'cloud shape: {cloud_arrays.shape}, range: [{np.min(cloud_arrays)}, {np.max(cloud_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # count file size
    total_size = 0
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    cprint(f"Total size: {total_size/1e6} MB", "green")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, default=os.path.expanduser("~/dp_data/new_task1_expertdata"))
    parser.add_argument("--save_dir", type=str, default=os.path.expanduser("~/dp_data/zarr_task1"))
    parser.add_argument("--save_img", type=int, default=1)
    parser.add_argument("--save_wrist_img", type=int, default=1)#是否保存手腕相机图像
    parser.add_argument("--save_depth", type=int, default=0)
    parser.add_argument("--save_cloud", type=int, default=0)
    parser.add_argument(
        "--action_offset_frames",
        type=int,
        default=0,
        help=(
            "Use future action labels inside each episode: obs/state/image[i] -> "
            "action[i + offset]. The last offset frames of each episode are dropped."
        ),
    )
    
    args = parser.parse_args()
    
    convert_dataset(args)
    
