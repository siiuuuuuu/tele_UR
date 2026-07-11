import os
import argparse
import shutil

import h5py
import numpy as np
import zarr
from termcolor import cprint


DEFAULT_DEMO_DIR = os.path.expanduser("~/dp_data/task1_expertdata")
RAW_ONLY_H5_KEYS = ("timestamps",)
TRAINING_OUTPUT_DATA_KEYS = (
    "img",
    "wrist_img",
    "depth",
    "cloud",
    "state",
    "action",
    "intervention",
)
TRAINING_OUTPUT_META_KEYS = ("episode_ends", "success")


def report_ignored_raw_only_keys(h5_data, file_name):
    ignored = [key for key in RAW_ONLY_H5_KEYS if key in h5_data]
    if ignored:
        cprint(
            f"ignore raw-only H5 keys for training conversion in {file_name}: "
            f"{', '.join(ignored)}",
            "cyan",
        )


def read_recorded_action_offset_frames(h5_data, override=None):
    if override is not None:
        return float(override)

    value = h5_data.attrs.get("action_alignment_offset_frames", 0.0)
    return float(np.asarray(value).item())


def common_recorded_action_offset(offset_values):
    if not offset_values:
        return 0.0

    first = float(offset_values[0])
    for value in offset_values[1:]:
        if not np.isclose(float(value), first, rtol=0.0, atol=1e-6):
            raise ValueError(
                "mixed recorded action offsets in one zarr conversion: "
                f"{offset_values}"
            )
    return first


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

    action_offset_frames = int(args.action_offset_frames)
    if action_offset_frames < 0:
        raise ValueError("--action_offset_frames must be >= 0")
    recorded_action_offset_override = args.recorded_action_offset_frames

    save_img = bool(args.save_img)
    save_wrist_img = bool(args.save_wrist_img)
    save_depth = bool(args.save_depth)
    save_cloud = bool(args.save_cloud)
    default_intervention = bool(args.default_intervention)

    # create dir to save demonstrations
    if os.path.exists(save_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"output already exists: {save_dir}; pass --overwrite to replace it"
            )
        cprint('Overwriting {}'.format(save_dir), 'red')
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        else:
            os.remove(save_dir)
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
    skipped_short_episodes = 0
    recorded_action_offset_values = []
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
                report_ignored_raw_only_keys(data, file_name)

                recorded_action_offset_values.append(
                    read_recorded_action_offset_frames(
                        data,
                        override=recorded_action_offset_override,
                    )
                )

                action_array = np.asarray(data["action"][:], dtype=np.float32)
                raw_length = action_array.shape[0]

                state_array = np.asarray(data["env_qpos_proprioception"][:], dtype=np.float32)
                if state_array.shape[0] != raw_length:
                    raise ValueError(
                        f"state length mismatch in {file_name}: "
                        f"action length={raw_length}, state shape={state_array.shape}"
                    )

                if "intervention" in data:
                    intervention_array = np.asarray(data["intervention"][:], dtype=bool)
                else:
                    # H5 缺失 intervention 时，使用命令行指定的默认标签。
                    intervention_array = np.full(
                        (raw_length,), default_intervention, dtype=bool
                    )

                # 统一成 (T,) 的每步干预标记，避免和 action 维度绑定导致后续 stack 失败
                if intervention_array.ndim == 0:
                    intervention_array = np.full(
                        (raw_length,), bool(intervention_array), dtype=bool
                    )
                elif intervention_array.ndim > 1:
                    if intervention_array.shape[0] != raw_length:
                        raise ValueError(
                            f"intervention first dim mismatch in {file_name}: "
                            f"action length={raw_length}, "
                            f"intervention shape={intervention_array.shape}"
                        )
                    reduction_axes = tuple(range(1, intervention_array.ndim))
                    intervention_array = np.any(intervention_array, axis=reduction_axes)

                if intervention_array.shape[0] != raw_length:
                    raise ValueError(
                        f"intervention length mismatch in {file_name}: "
                        f"action length={raw_length}, "
                        f"intervention shape={intervention_array.shape}"
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
                    if color_array.shape[0] != raw_length:
                        raise ValueError(
                            f"color length mismatch in {file_name}: "
                            f"action length={raw_length}, color shape={color_array.shape}"
                        )
                if save_wrist_img:
                    wrist_color_array = to_channel_last_uint8(data["wrist_color"][:], "wrist_color", file_name)
                    if wrist_color_array.shape[0] != raw_length:
                        raise ValueError(
                            f"wrist_color length mismatch in {file_name}: "
                            f"action length={raw_length}, "
                            f"wrist_color shape={wrist_color_array.shape}"
                        )
                if save_depth:
                    depth_array = np.asarray(data["depth"][:], dtype=np.float32)
                    if depth_array.shape[0] != raw_length:
                        raise ValueError(
                            f"depth length mismatch in {file_name}: "
                            f"action length={raw_length}, depth shape={depth_array.shape}"
                        )
                if save_cloud:
                    cloud_array = np.asarray(data["cloud"][:], dtype=np.float32)
                    if cloud_array.shape[0] != raw_length:
                        raise ValueError(
                            f"cloud length mismatch in {file_name}: "
                            f"action length={raw_length}, cloud shape={cloud_array.shape}"
                        )

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
                action_slice = slice(
                    action_offset_frames,
                    action_offset_frames + length,
                )
                state_array = state_array[obs_slice]
                action_array = action_array[action_slice]
                # Intervention describes who produced the action, so it must
                # stay aligned with the shifted future action label.
                intervention_array = intervention_array[action_slice]
                if save_img:
                    color_array = color_array[obs_slice]
                if save_wrist_img:
                    wrist_color_array = wrist_color_array[obs_slice]
                if save_depth:
                    depth_array = depth_array[obs_slice]
                if save_cloud:
                    cloud_array = cloud_array[obs_slice]

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

    recorded_action_offset_frames = common_recorded_action_offset(
        recorded_action_offset_values
    )
    effective_action_offset_frames = (
        recorded_action_offset_frames + action_offset_frames
    )
    if np.isclose(
        effective_action_offset_frames,
        round(effective_action_offset_frames),
        rtol=0.0,
        atol=1e-6,
    ):
        action_offset_attr = int(round(effective_action_offset_frames))
    else:
        action_offset_attr = float(effective_action_offset_frames)
    zarr_root.attrs["action_offset_frames"] = action_offset_attr
    zarr_root.attrs["recorded_action_offset_frames"] = float(
        recorded_action_offset_frames
    )
    zarr_root.attrs["action_index_offset_frames"] = int(action_offset_frames)

    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('success', data=success_arrays, dtype='bool', overwrite=True, compressor=compressor)
    validate_training_zarr_schema(zarr_data, zarr_meta)
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
    cprint(
        f'action offset frames: +{effective_action_offset_frames:g} '
        f'(recorded +{recorded_action_offset_frames:g}, '
        f'index +{action_offset_frames})',
        'green',
    )
    if skipped_short_episodes:
        cprint(f'skipped short episodes: {skipped_short_episodes}', 'yellow')
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace save_dir if it already exists.",
    )
    parser.add_argument("--default_intervention", type=int, choices=(0, 1), default=0,
                        help="Value for H5 files without intervention: 0=non-intervention, 1=intervention")
    parser.add_argument(
        "--action_offset_frames",
        type=int,
        default=1,
        help=(
            "Additional future action-label offset inside each episode: "
            "obs/state/image[i] -> action/intervention[i + offset]. The last "
            "offset frames of every episode are dropped. This is added to any "
            "action_alignment_offset_frames stored in the source H5 files. "
            "Default: 1 frame, compensating the measured RealSense image-time "
            "offset."
        ),
    )
    parser.add_argument(
        "--recorded_action_offset_frames",
        type=float,
        default=None,
        help=(
            "Override the H5 action_alignment_offset_frames metadata. Use this "
            "only for old H5 files whose action-label timing is known but not "
            "stored in attrs."
        ),
    )

    args = parser.parse_args()

    convert_dataset(args)
