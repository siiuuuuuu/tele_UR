#!/usr/bin/env python3
"""Convert regular teleoperation H5 files to DexPIE Zarr+memmap format."""

import argparse
import os

from h5_memmap_converter import convert_h5_dataset


def convert_dataset(args):
    convert_h5_dataset(
        demo_dirs=[args.demo_dir],
        save_dir=args.save_dir,
        save_img=bool(args.save_img),
        save_wrist_img=bool(args.save_wrist_img),
        save_depth=bool(args.save_depth),
        save_cloud=bool(args.save_cloud),
        include_intervention=False,
        use_h5_success=False,
        action_offset_frames=args.action_offset_frames,
        recorded_action_offset_override=args.recorded_action_offset_frames,
        overwrite=args.overwrite,
        batch_size=getattr(args, "batch_size", 64),
        # Preserve the regular converter's existing shortest-stream behavior.
        strict_lengths=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert regular H5 demonstrations directly to the DexPIE "
            "data.zarr + visual NPY memmap layout."
        )
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        default=os.path.expanduser("~/dp_data/new_task1_expertdata"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.expanduser("~/dp_data/zarr_task1"),
    )
    parser.add_argument("--save_img", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--save_wrist_img", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--save_depth", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save_cloud", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace save_dir after the new dataset has been validated.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Frames read from each H5 stream per batch (default: 64).",
    )
    parser.add_argument(
        "--action_offset_frames",
        type=int,
        default=1,
        help=(
            "Additional conversion-time future action labels inside each "
            "episode: obs/state/image[i] -> action[i + offset]. The final "
            "offset frames are dropped. Default: 1."
        ),
    )
    parser.add_argument(
        "--recorded_action_offset_frames",
        type=float,
        default=None,
        help=(
            "Override action_alignment_offset_frames from H5 metadata. "
            "Use only for old files whose offset is known but not stored."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args())
