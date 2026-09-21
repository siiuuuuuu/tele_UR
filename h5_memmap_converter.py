#!/usr/bin/env python3
"""Shared HDF5 to DexPIE Zarr+memmap dataset conversion utilities."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from termcolor import cprint
import zarr


MEMMAP_ZARR_DIR = "data.zarr"
FORMAT_ATTR = "_dexpie_memmap_format"
VISUAL_KEYS = ("img", "wrist_img", "depth")
RAW_ONLY_H5_KEYS = ("timestamps",)
CHUNK_ROWS = 500


@dataclass(frozen=True)
class EpisodePlan:
    path: Path
    raw_length: int
    length: int
    success: bool


def _report_ignored_raw_only_keys(h5_data, file_name):
    ignored = [key for key in RAW_ONLY_H5_KEYS if key in h5_data]
    if ignored:
        cprint(
            f"ignore raw-only H5 keys for training conversion in {file_name}: "
            f"{', '.join(ignored)}",
            "cyan",
        )


def _read_recorded_action_offset_frames(h5_data, override=None):
    if override is not None:
        return float(override)
    value = h5_data.attrs.get("action_alignment_offset_frames", 0.0)
    return float(np.asarray(value).item())


def _common_recorded_action_offset(offset_values):
    if not offset_values:
        return 0.0

    first = float(offset_values[0])
    for value in offset_values[1:]:
        if not np.isclose(float(value), first, rtol=0.0, atol=1e-6):
            raise ValueError(
                "mixed recorded action offsets in one dataset conversion: "
                f"{offset_values}"
            )
    return first


def _read_success(h5_data, file_name, use_h5_success):
    if not use_h5_success or "success" not in h5_data.attrs:
        return True

    success_attr = np.asarray(h5_data.attrs["success"])
    if success_attr.size != 1:
        raise ValueError(
            f"success attr should be a scalar bool in {file_name}, "
            f"got shape={success_attr.shape}"
        )
    return bool(success_attr.reshape(-1)[0])


def _channel_last_shape(dataset, key, file_name):
    shape = tuple(dataset.shape)
    if len(shape) != 4:
        raise ValueError(
            f"{key} should be 4D [T,H,W,C] or [T,C,H,W] in {file_name}, "
            f"got shape={shape}"
        )
    if shape[1] == 3 and shape[-1] != 3:
        shape = (shape[0], shape[2], shape[3], shape[1])
    if shape[-1] != 3:
        raise ValueError(
            f"{key} last dim should be 3 after normalization in {file_name}, "
            f"got shape={shape}"
        )
    return shape


def _to_channel_last_uint8(img_array, key, file_name):
    img_array = np.asarray(img_array)
    if img_array.ndim != 4:
        raise ValueError(
            f"{key} should be 4D [T,H,W,C] or [T,C,H,W] in {file_name}, "
            f"got shape={img_array.shape}"
        )
    if img_array.shape[1] == 3 and img_array.shape[-1] != 3:
        img_array = np.transpose(img_array, (0, 2, 3, 1))
    if img_array.shape[-1] != 3:
        raise ValueError(
            f"{key} last dim should be 3 after normalization in {file_name}, "
            f"got shape={img_array.shape}"
        )
    return np.asarray(img_array, dtype=np.uint8)


def _check_required_dataset(h5_data, key, file_name):
    if key not in h5_data:
        raise KeyError(f"H5 dataset {key!r} is missing in {file_name}")
    dataset = h5_data[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"H5 key {key!r} is not a dataset in {file_name}")
    if dataset.ndim < 1:
        raise ValueError(
            f"H5 dataset {key!r} must have a time dimension in {file_name}, "
            f"got shape={dataset.shape}"
        )
    return dataset


def _add_spec(specs, key, trailing_shape, dtype, file_name):
    spec = (tuple(trailing_shape), np.dtype(dtype))
    previous = specs.get(key)
    if previous is not None and previous != spec:
        raise ValueError(
            f"inconsistent output shape/dtype for {key!r} in {file_name}: "
            f"{spec} != {previous}"
        )
    specs[key] = spec


def _scan_episodes(
    demo_files,
    *,
    save_img,
    save_wrist_img,
    save_depth,
    save_cloud,
    include_intervention,
    use_h5_success,
    action_offset_frames,
    recorded_action_offset_override,
    strict_lengths,
):
    plans = []
    specs = {}
    recorded_offsets = []
    skipped_short_episodes = 0

    enabled_inputs = {
        "state": "env_qpos_proprioception",
        "action": "action",
    }
    if save_img:
        enabled_inputs["img"] = "color"
    if save_wrist_img:
        enabled_inputs["wrist_img"] = "wrist_color"
    if save_depth:
        enabled_inputs["depth"] = "depth"
    if save_cloud:
        enabled_inputs["cloud"] = "cloud"

    for file_path in demo_files:
        file_name = str(file_path)
        print("scan:", file_name)
        with h5py.File(file_name, "r") as data:
            _report_ignored_raw_only_keys(data, file_name)
            datasets = {
                output_key: _check_required_dataset(data, input_key, file_name)
                for output_key, input_key in enabled_inputs.items()
            }
            action_dataset = datasets["action"]
            raw_action_length = int(action_dataset.shape[0])

            lengths = {
                key: int(dataset.shape[0])
                for key, dataset in datasets.items()
            }
            if strict_lengths:
                mismatches = {
                    key: length
                    for key, length in lengths.items()
                    if length != raw_action_length
                }
                if mismatches:
                    raise ValueError(
                        f"dataset length mismatch in {file_name}: "
                        f"action={raw_action_length}, others={mismatches}"
                    )
                raw_length = raw_action_length
            else:
                raw_length = min(lengths.values())

            if include_intervention and "intervention" in data:
                intervention = data["intervention"]
                if not isinstance(intervention, h5py.Dataset):
                    raise TypeError(
                        f"H5 key 'intervention' is not a dataset in {file_name}"
                    )
                if intervention.ndim > 0 and intervention.shape[0] != raw_action_length:
                    raise ValueError(
                        f"intervention length mismatch in {file_name}: "
                        f"action={raw_action_length}, "
                        f"intervention={intervention.shape}"
                    )

            recorded_offsets.append(
                _read_recorded_action_offset_frames(
                    data, override=recorded_action_offset_override
                )
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

            _add_spec(
                specs,
                "state",
                datasets["state"].shape[1:],
                np.float32,
                file_name,
            )
            _add_spec(
                specs,
                "action",
                datasets["action"].shape[1:],
                np.float32,
                file_name,
            )
            if save_img:
                shape = _channel_last_shape(datasets["img"], "color", file_name)
                _add_spec(specs, "img", shape[1:], np.uint8, file_name)
            if save_wrist_img:
                shape = _channel_last_shape(
                    datasets["wrist_img"], "wrist_color", file_name
                )
                _add_spec(specs, "wrist_img", shape[1:], np.uint8, file_name)
            if save_depth:
                _add_spec(
                    specs,
                    "depth",
                    datasets["depth"].shape[1:],
                    np.float32,
                    file_name,
                )
            if save_cloud:
                _add_spec(
                    specs,
                    "cloud",
                    datasets["cloud"].shape[1:],
                    np.float32,
                    file_name,
                )
            if include_intervention:
                _add_spec(specs, "intervention", (), np.bool_, file_name)

            plans.append(
                EpisodePlan(
                    path=file_path,
                    raw_length=raw_length,
                    length=length,
                    success=_read_success(data, file_name, use_h5_success),
                )
            )

    if not plans:
        raise RuntimeError("no usable H5 episodes found")

    return (
        plans,
        specs,
        _common_recorded_action_offset(recorded_offsets),
        skipped_short_episodes,
    )


def _create_zarr_array(group, key, total_steps, spec, compressor):
    trailing_shape, dtype = spec
    chunks = (min(CHUNK_ROWS, total_steps),) + trailing_shape
    return group.create_dataset(
        key,
        shape=(total_steps,) + trailing_shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=True,
        compressor=compressor,
    )


def _update_min_max(stats, key, array):
    if array.size == 0:
        return
    current_min = np.min(array)
    current_max = np.max(array)
    if stats[key]["min"] is None or current_min < stats[key]["min"]:
        stats[key]["min"] = current_min
    if stats[key]["max"] is None or current_max > stats[key]["max"]:
        stats[key]["max"] = current_max


def _format_range(stats, key):
    minimum = stats[key]["min"]
    maximum = stats[key]["max"]
    if isinstance(minimum, np.generic):
        minimum = minimum.item()
    if isinstance(maximum, np.generic):
        maximum = maximum.item()
    return minimum, maximum


def _read_intervention_batch(
    data, start, stop, raw_length, default_intervention, file_name
):
    if "intervention" not in data:
        return np.full((stop - start,), default_intervention, dtype=bool)

    dataset = data["intervention"]
    if dataset.ndim == 0:
        return np.full((stop - start,), bool(dataset[()]), dtype=bool)

    array = np.asarray(dataset[start:stop], dtype=bool)
    if array.shape[0] != stop - start:
        raise ValueError(
            f"intervention batch length mismatch in {file_name}: "
            f"expected={stop - start}, got={array.shape}"
        )
    if array.ndim > 1:
        array = np.any(array, axis=tuple(range(1, array.ndim)))
    if dataset.shape[0] != raw_length:
        raise ValueError(
            f"intervention length mismatch in {file_name}: "
            f"expected={raw_length}, got={dataset.shape}"
        )
    return array


def _close_memmaps(memmaps):
    for array in memmaps.values():
        try:
            array.flush()
        finally:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _validate_output(
    output_dir, *, total_steps, specs, visual_keys, nonvisual_keys, episode_ends
):
    zarr_path = output_dir / MEMMAP_ZARR_DIR
    root = zarr.open(str(zarr_path), mode="r")
    if set(root.group_keys()) != {"data", "meta"}:
        raise RuntimeError(
            f"unexpected groups in {zarr_path}: {sorted(root.group_keys())}"
        )
    if set(root["data"].array_keys()) != set(nonvisual_keys):
        raise RuntimeError(
            "unexpected nonvisual arrays in data.zarr/data: "
            f"{sorted(root['data'].array_keys())}"
        )
    if set(root["meta"].array_keys()) != {"episode_ends", "success"}:
        raise RuntimeError(
            "unexpected arrays in data.zarr/meta: "
            f"{sorted(root['meta'].array_keys())}"
        )

    format_attr = dict(root.attrs.get(FORMAT_ATTR, {}))
    expected_format = {"version": 1, "visual_keys": list(visual_keys)}
    if format_attr != expected_format:
        raise RuntimeError(
            f"invalid {FORMAT_ATTR}: {format_attr} != {expected_format}"
        )

    stored_episode_ends = np.asarray(root["meta"]["episode_ends"][:])
    np.testing.assert_array_equal(stored_episode_ends, episode_ends)
    if int(stored_episode_ends[-1]) != total_steps:
        raise RuntimeError(
            f"episode_ends final value is not total steps: "
            f"{stored_episode_ends[-1]} != {total_steps}"
        )

    for key in nonvisual_keys:
        expected_shape = (total_steps,) + specs[key][0]
        array = root["data"][key]
        if array.shape != expected_shape or array.dtype != specs[key][1]:
            raise RuntimeError(
                f"data/{key} has shape/dtype {array.shape}/{array.dtype}, "
                f"expected {expected_shape}/{specs[key][1]}"
            )

    for key in visual_keys:
        path = output_dir / f"{key}.npy"
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
        try:
            expected_shape = (total_steps,) + specs[key][0]
            if array.shape != expected_shape or array.dtype != specs[key][1]:
                raise RuntimeError(
                    f"{path.name} has shape/dtype {array.shape}/{array.dtype}, "
                    f"expected {expected_shape}/{specs[key][1]}"
                )
        finally:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _remove_existing_output(path):
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def convert_h5_dataset(
    *,
    demo_dirs,
    save_dir,
    save_img=True,
    save_wrist_img=True,
    save_depth=False,
    save_cloud=False,
    include_intervention=False,
    use_h5_success=False,
    default_intervention=False,
    action_offset_frames=1,
    recorded_action_offset_override=None,
    overwrite=False,
    batch_size=64,
    strict_lengths=True,
):
    """Convert H5 episodes directly into the DexPIE Zarr+NPY layout."""
    action_offset_frames = int(action_offset_frames)
    batch_size = int(batch_size)
    if action_offset_frames < 0:
        raise ValueError("--action_offset_frames must be >= 0")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    save_dir = Path(os.path.expanduser(str(save_dir))).resolve()
    demo_dirs = [Path(os.path.expanduser(str(path))).resolve() for path in demo_dirs]
    if save_dir in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"refusing to use broad directory as save_dir: {save_dir}")
    if any(save_dir == demo_dir for demo_dir in demo_dirs):
        raise ValueError("save_dir must not be the same directory as an H5 source")
    if save_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {save_dir}; pass --overwrite to replace it"
        )

    demo_files = []
    for demo_dir in demo_dirs:
        if not demo_dir.is_dir():
            cprint(f"Skip invalid demo_dir: {demo_dir}", "yellow")
            continue
        files = sorted(demo_dir.glob("*.h5"))
        cprint(f"Found {len(files)} H5 files in {demo_dir}", "cyan")
        demo_files.extend(files)

    plans, specs, recorded_action_offset_frames, skipped_short_episodes = (
        _scan_episodes(
            demo_files,
            save_img=bool(save_img),
            save_wrist_img=bool(save_wrist_img),
            save_depth=bool(save_depth),
            save_cloud=bool(save_cloud),
            include_intervention=bool(include_intervention),
            use_h5_success=bool(use_h5_success),
            action_offset_frames=action_offset_frames,
            recorded_action_offset_override=recorded_action_offset_override,
            strict_lengths=bool(strict_lengths),
        )
    )

    total_steps = sum(plan.length for plan in plans)
    episode_ends = np.cumsum(
        np.asarray([plan.length for plan in plans], dtype=np.int64)
    )
    success = np.asarray([plan.success for plan in plans], dtype=bool)
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

    visual_keys = [key for key in VISUAL_KEYS if key in specs]
    nonvisual_keys = [
        key
        for key in ("cloud", "state", "action", "intervention")
        if key in specs
    ]
    stats = {
        key: {"min": None, "max": None}
        for key in specs
        if key != "intervention"
    }

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{save_dir.name}.tmp-", dir=str(save_dir.parent)
        )
    )
    memmaps = {}
    committed = False
    try:
        root = zarr.open(str(temporary_dir / MEMMAP_ZARR_DIR), mode="w")
        zarr_data = root.create_group("data")
        zarr_meta = root.create_group("meta")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

        zarr_arrays = {
            key: _create_zarr_array(
                zarr_data, key, total_steps, specs[key], compressor
            )
            for key in nonvisual_keys
        }
        for key in visual_keys:
            trailing_shape, dtype = specs[key]
            memmaps[key] = open_memmap(
                str(temporary_dir / f"{key}.npy"),
                mode="w+",
                dtype=dtype,
                shape=(total_steps,) + trailing_shape,
            )

        cursor = 0
        for plan in plans:
            file_name = str(plan.path)
            print("write:", file_name)
            with h5py.File(file_name, "r") as data:
                for local_start in range(0, plan.length, batch_size):
                    local_stop = min(local_start + batch_size, plan.length)
                    output_slice = slice(
                        cursor + local_start, cursor + local_stop
                    )
                    obs_slice = slice(local_start, local_stop)
                    action_slice = slice(
                        local_start + action_offset_frames,
                        local_stop + action_offset_frames,
                    )

                    state = np.asarray(
                        data["env_qpos_proprioception"][obs_slice],
                        dtype=np.float32,
                    )
                    action = np.asarray(
                        data["action"][action_slice], dtype=np.float32
                    )
                    zarr_arrays["state"][output_slice] = state
                    zarr_arrays["action"][output_slice] = action
                    _update_min_max(stats, "state", state)
                    _update_min_max(stats, "action", action)

                    if save_img:
                        image = _to_channel_last_uint8(
                            data["color"][obs_slice], "color", file_name
                        )
                        memmaps["img"][output_slice] = image
                        _update_min_max(stats, "img", image)
                    if save_wrist_img:
                        wrist_image = _to_channel_last_uint8(
                            data["wrist_color"][obs_slice],
                            "wrist_color",
                            file_name,
                        )
                        memmaps["wrist_img"][output_slice] = wrist_image
                        _update_min_max(stats, "wrist_img", wrist_image)
                    if save_depth:
                        depth = np.asarray(
                            data["depth"][obs_slice], dtype=np.float32
                        )
                        memmaps["depth"][output_slice] = depth
                        _update_min_max(stats, "depth", depth)
                    if save_cloud:
                        cloud = np.asarray(
                            data["cloud"][obs_slice], dtype=np.float32
                        )
                        zarr_arrays["cloud"][output_slice] = cloud
                        _update_min_max(stats, "cloud", cloud)
                    if include_intervention:
                        intervention = _read_intervention_batch(
                            data,
                            action_slice.start,
                            action_slice.stop,
                            plan.raw_length,
                            bool(default_intervention),
                            file_name,
                        )
                        zarr_arrays["intervention"][output_slice] = intervention
            cursor += plan.length

        if cursor != total_steps:
            raise RuntimeError(f"wrote {cursor} steps, expected {total_steps}")

        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "success",
            data=success,
            dtype="bool",
            overwrite=True,
            compressor=compressor,
        )
        root.attrs["action_offset_frames"] = action_offset_attr
        root.attrs["recorded_action_offset_frames"] = float(
            recorded_action_offset_frames
        )
        root.attrs["action_index_offset_frames"] = action_offset_frames
        root.attrs[FORMAT_ATTR] = {
            "version": 1,
            "visual_keys": visual_keys,
        }

        _close_memmaps(memmaps)
        memmaps = {}
        _validate_output(
            temporary_dir,
            total_steps=total_steps,
            specs=specs,
            visual_keys=visual_keys,
            nonvisual_keys=nonvisual_keys,
            episode_ends=episode_ends,
        )

        if save_dir.exists():
            cprint(f"Overwriting {save_dir}", "red")
            _remove_existing_output(save_dir)
        os.replace(str(temporary_dir), str(save_dir))
        committed = True
    finally:
        if memmaps:
            _close_memmaps(memmaps)
        if not committed:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    cprint(f"episode nums: {episode_ends.shape}", "green")
    for key in ("img", "wrist_img", "depth", "cloud", "state", "action"):
        if key not in specs:
            continue
        minimum, maximum = _format_range(stats, key)
        shape = (total_steps,) + specs[key][0]
        cprint(
            f"{key} shape: {shape}, range: [{minimum}, {maximum}]", "green"
        )
    if include_intervention:
        cprint(f"intervention shape: {(total_steps,)}", "green")
    cprint(
        f"action offset frames: +{effective_action_offset_frames:g} "
        f"(recorded +{recorded_action_offset_frames:g}, "
        f"index +{action_offset_frames})",
        "green",
    )
    if skipped_short_episodes:
        cprint(f"skipped short episodes: {skipped_short_episodes}", "yellow")
    cprint(f"Saved Zarr+memmap dataset to {save_dir}", "green")

    total_size = sum(
        path.stat().st_size for path in save_dir.rglob("*") if path.is_file()
    )
    cprint(f"Total size: {total_size / 1e6} MB", "green")
