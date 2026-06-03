import os
import argparse
import shutil

import h5py
import numpy as np
import zarr
from termcolor import cprint


DEFAULT_DEMO_DIR = os.path.expanduser("~/dp_data/task1_expertdata")


def parse_demo_dirs(args):
    raw_demo_dirs = []
    if args.demo_dir:
        raw_demo_dirs.append(args.demo_dir)
    if args.demo_dirs:
        raw_demo_dirs.extend(args.demo_dirs)

    if not raw_demo_dirs:
        raw_demo_dirs.append(DEFAULT_DEMO_DIR)

    demo_dirs = []
    for item in raw_demo_dirs:
        for part in item.split(","):
            part = part.strip()
            if part:
                demo_dirs.append(os.path.expanduser(part))

    # remove duplicates while preserving order
    demo_dirs = list(dict.fromkeys(demo_dirs))
    if not demo_dirs:
        raise ValueError("No valid demo directory is provided.")
    return demo_dirs


def to_channel_last_uint8(img_array, key, file_name):
    img_array = np.asarray(img_array)
    if img_array.ndim != 4:
        raise ValueError(
            f"{key} should be 4D [T,H,W,C] or [T,C,H,W] in {file_name}, "
            f"got shape={img_array.shape}"
        )

    # Support both channel-last and channel-first image layouts.
    if img_array.shape[1] == 3 and img_array.shape[-1] != 3:
        img_array = np.transpose(img_array, (0, 2, 3, 1))

    if img_array.shape[-1] != 3:
        raise ValueError(
            f"{key} last dim should be 3 after normalization in {file_name}, "
            f"got shape={img_array.shape}"
        )
    return np.asarray(img_array, dtype=np.uint8)


def create_stream_dataset(zarr_group, name, sample_batch, single_size, dtype, compressor):
    trailing_shape = sample_batch.shape[1:]
    chunks = (single_size,) + trailing_shape
    return zarr_group.create_dataset(
        name,
        shape=(0,) + trailing_shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=True,
        compressor=compressor,
    )


def append_rows(dataset, batch):
    old_rows = dataset.shape[0]
    new_rows = old_rows + batch.shape[0]
    dataset.resize((new_rows,) + dataset.shape[1:])
    dataset[old_rows:new_rows] = batch


def update_min_max(stats, key, array):
    if array.size == 0:
        return
    cur_min = np.min(array)
    cur_max = np.max(array)
    if stats[key]["min"] is None or cur_min < stats[key]["min"]:
        stats[key]["min"] = cur_min
    if stats[key]["max"] is None or cur_max > stats[key]["max"]:
        stats[key]["max"] = cur_max


def format_range(stats, key):
    min_v = stats[key]["min"]
    max_v = stats[key]["max"]
    if isinstance(min_v, np.generic):
        min_v = min_v.item()
    if isinstance(max_v, np.generic):
        max_v = max_v.item()
    return min_v, max_v


def convert_dataset(args):
    demo_dirs = parse_demo_dirs(args)
    save_dir = args.save_dir

    save_img = bool(args.save_img)
    save_wrist_img = bool(args.save_wrist_img)
    save_depth = bool(args.save_depth)
    save_cloud = bool(args.save_cloud)

    # create dir to save demonstrations
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            else:
                os.remove(save_dir)
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    single_size = 500

    img_ds = None
    wrist_img_ds = None
    depth_ds = None
    cloud_ds = None
    state_ds = None
    action_ds = None
    intervention_ds = None

    stats = {
        "img": {"min": None, "max": None},
        "wrist_img": {"min": None, "max": None},
        "depth": {"min": None, "max": None},
        "cloud": {"min": None, "max": None},
        "state": {"min": None, "max": None},
        "action": {"min": None, "max": None},
    }

    total_count = 0
    processed_files = 0
    episode_ends_arrays = []
    success_arrays = []

    for demo_dir in demo_dirs:
        if not os.path.isdir(demo_dir):
            cprint(f"Skip invalid demo_dir: {demo_dir}", "yellow")
            continue

        demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".h5")]
        demo_files = sorted(demo_files)
        cprint(f"Loading {len(demo_files)} files from {demo_dir}", "cyan")

        for demo_file in demo_files:
            # load file (h5)
            file_name = os.path.join(demo_dir, demo_file)
            print("process:", file_name)

            with h5py.File(file_name, "r") as data:
                action_array = np.asarray(data["action"][:], dtype=np.float32)
                length = action_array.shape[0]

                state_array = np.asarray(data["env_qpos_proprioception"][:], dtype=np.float32)
                if state_array.shape[0] != length:
                    raise ValueError(
                        f"state length mismatch in {file_name}: "
                        f"action length={length}, state shape={state_array.shape}"
                    )

                if "intervention" in data:
                    intervention_array = np.asarray(data["intervention"][:], dtype=bool)
                else:
                    # 遥操数据不记录干预时，按时间维默认全为专家干预
                    intervention_array = np.ones((length,), dtype=bool)

                # 统一成 (T,) 的每步干预标记，避免和 action 维度绑定导致后续 stack 失败
                if intervention_array.ndim == 0:
                    intervention_array = np.full((length,), bool(intervention_array), dtype=bool)
                elif intervention_array.ndim > 1:
                    if intervention_array.shape[0] != length:
                        raise ValueError(
                            f"intervention first dim mismatch in {file_name}: "
                            f"action length={length}, intervention shape={intervention_array.shape}"
                        )
                    reduction_axes = tuple(range(1, intervention_array.ndim))
                    intervention_array = np.any(intervention_array, axis=reduction_axes)

                if intervention_array.shape[0] != length:
                    raise ValueError(
                        f"intervention length mismatch in {file_name}: "
                        f"action length={length}, intervention shape={intervention_array.shape}"
                    )

                if "success" in data.attrs:
                    success_attr = np.asarray(data.attrs["success"])
                    if success_attr.size != 1:
                        raise ValueError(
                            f"success attr should be a scalar bool in {file_name}, "
                            f"got shape={success_attr.shape}"
                        )
                    success = bool(success_attr.reshape(-1)[0])
                else:
                    success = True

                if save_img:
                    color_array = to_channel_last_uint8(data["color"][:], "color", file_name)
                    if color_array.shape[0] != length:
                        raise ValueError(
                            f"color length mismatch in {file_name}: "
                            f"action length={length}, color shape={color_array.shape}"
                        )
                if save_wrist_img:
                    wrist_color_array = to_channel_last_uint8(data["wrist_color"][:], "wrist_color", file_name)
                    if wrist_color_array.shape[0] != length:
                        raise ValueError(
                            f"wrist_color length mismatch in {file_name}: "
                            f"action length={length}, wrist_color shape={wrist_color_array.shape}"
                        )
                if save_depth:
                    depth_array = np.asarray(data["depth"][:], dtype=np.float32)
                    if depth_array.shape[0] != length:
                        raise ValueError(
                            f"depth length mismatch in {file_name}: "
                            f"action length={length}, depth shape={depth_array.shape}"
                        )
                if save_cloud:
                    cloud_array = np.asarray(data["cloud"][:], dtype=np.float32)
                    if cloud_array.shape[0] != length:
                        raise ValueError(
                            f"cloud length mismatch in {file_name}: "
                            f"action length={length}, cloud shape={cloud_array.shape}"
                        )

            if save_img and img_ds is None:
                img_ds = create_stream_dataset(zarr_data, 'img', color_array, single_size, 'uint8', compressor)
            if save_wrist_img and wrist_img_ds is None:
                wrist_img_ds = create_stream_dataset(zarr_data, 'wrist_img', wrist_color_array, single_size, 'uint8', compressor)
            if save_depth and depth_ds is None:
                depth_ds = create_stream_dataset(zarr_data, 'depth', depth_array, single_size, 'float32', compressor)
            if save_cloud and cloud_ds is None:
                cloud_ds = create_stream_dataset(zarr_data, 'cloud', cloud_array, single_size, 'float32', compressor)
            if state_ds is None:
                state_ds = create_stream_dataset(zarr_data, 'state', state_array, single_size, 'float32', compressor)
            if action_ds is None:
                action_ds = create_stream_dataset(zarr_data, 'action', action_array, single_size, 'float32', compressor)
            if intervention_ds is None:
                intervention_ds = create_stream_dataset(zarr_data, 'intervention', intervention_array, single_size, 'bool', compressor)

            if save_img:
                append_rows(img_ds, color_array)
                update_min_max(stats, "img", color_array)
            if save_wrist_img:
                append_rows(wrist_img_ds, wrist_color_array)
                update_min_max(stats, "wrist_img", wrist_color_array)
            if save_depth:
                append_rows(depth_ds, depth_array)
                update_min_max(stats, "depth", depth_array)
            if save_cloud:
                append_rows(cloud_ds, cloud_array)
                update_min_max(stats, "cloud", cloud_array)

            append_rows(state_ds, state_array)
            append_rows(action_ds, action_array)
            append_rows(intervention_ds, intervention_array)
            update_min_max(stats, "state", state_array)
            update_min_max(stats, "action", action_array)

            total_count += length
            episode_ends_arrays.append(total_count)
            success_arrays.append(success)
            processed_files += 1

    if processed_files == 0:
        raise RuntimeError(f"No valid demo data found in demo_dirs={demo_dirs}")

    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)
    if len(success_arrays) != len(episode_ends_arrays):
        raise ValueError(
            f"success length mismatch: success={len(success_arrays)}, "
            f"episodes={len(episode_ends_arrays)}"
        )
    success_arrays = np.asarray(success_arrays, dtype=bool)

    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('success', data=success_arrays, dtype='bool', overwrite=True, compressor=compressor)
    # 包含每个episode结束时的全局索引，用于区分不同episode
    cprint(f'episode nums: {episode_ends_arrays.shape}', 'green')

    # print shape
    if save_img:
        color_min, color_max = format_range(stats, "img")
        cprint(f'color shape: {img_ds.shape}, range: [{color_min}, {color_max}]', 'green')
    if save_wrist_img:
        wrist_min, wrist_max = format_range(stats, "wrist_img")
        cprint(f'wrist_img shape: {wrist_img_ds.shape}, range: [{wrist_min}, {wrist_max}]', 'green')
    if save_depth:
        depth_min, depth_max = format_range(stats, "depth")
        cprint(f'depth shape: {depth_ds.shape}, range: [{depth_min}, {depth_max}]', 'green')
    if save_cloud:
        cloud_min, cloud_max = format_range(stats, "cloud")
        cprint(f'cloud shape: {cloud_ds.shape}, range: [{cloud_min}, {cloud_max}]', 'green')

    state_min, state_max = format_range(stats, "state")
    action_min, action_max = format_range(stats, "action")
    cprint(f'state shape: {state_ds.shape}, range: [{state_min}, {state_max}]', 'green')
    cprint(f'action shape: {action_ds.shape}, range: [{action_min}, {action_max}]', 'green')
    cprint(f'intervention shape: {intervention_ds.shape}', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # count file size
    total_size = 0
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    cprint(f"Total size: {total_size/1e6} MB", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #/home/lrz/dp_data/task1_expertdata
    parser.add_argument("--demo_dir", type=str, default=os.path.expanduser("None"))  # 为空时回落到 DEFAULT_DEMO_DIR
    parser.add_argument("--demo_dirs", type=str, nargs="+", default=None,
                        help="Multiple demo dirs, e.g. --demo_dirs /a /b or --demo_dirs /a,/b") # 加载多个数据集示例（会合并上面单个数据集）
    parser.add_argument("--save_dir", type=str, default=os.path.expanduser("~/dp_data/task1_Recap_iter2"))
    parser.add_argument("--save_img", type=int, default=1)
    parser.add_argument("--save_wrist_img", type=int, default=1) # 是否保存手腕相机图像
    parser.add_argument("--save_depth", type=int, default=0)
    parser.add_argument("--save_cloud", type=int, default=0)

    args = parser.parse_args()

    convert_dataset(args)
