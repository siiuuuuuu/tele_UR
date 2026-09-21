#!/usr/bin/env python3
"""Convert rollout/expert H5 files to DexPIE Zarr+memmap format."""

import argparse
import os

from h5_memmap_converter import convert_h5_dataset


DEFAULT_DEMO_DIR = os.path.expanduser("~/dp_data/task1_expertdata")


def parse_demo_dirs(args):
    raw_demo_dirs = []
    if args.demo_dir and args.demo_dir != "None":
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
    demo_dirs = list(dict.fromkeys(demo_dirs))
    if not demo_dirs:
        raise ValueError("No valid demo directory is provided.")
    return demo_dirs


def convert_dataset(args):
    convert_h5_dataset(
        demo_dirs=parse_demo_dirs(args),
        save_dir=args.save_dir,
        save_img=bool(args.save_img),
        save_wrist_img=bool(args.save_wrist_img),
        save_depth=bool(args.save_depth),
        save_cloud=bool(args.save_cloud),
        include_intervention=True,
        use_h5_success=True,
        default_intervention=bool(args.default_intervention),
        action_offset_frames=args.action_offset_frames,
        recorded_action_offset_override=args.recorded_action_offset_frames,
        overwrite=args.overwrite,
        batch_size=getattr(args, "batch_size", 64),
        strict_lengths=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge rollout/expert H5 demonstrations directly into the DexPIE "
            "data.zarr + visual NPY memmap layout."
        )
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        default=None,
        help="One H5 directory; may be combined with --demo_dirs.",
    )
    parser.add_argument(
        "--demo_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple H5 directories, separated by spaces and/or commas.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.expanduser("~/dp_data/task1_Recap_iter2"),
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
        "--default_intervention",
        type=int,
        choices=(0, 1),
        default=0,
        help="Value for H5 files without intervention: 0 or 1.",
    )
    parser.add_argument(
        "--action_offset_frames",
        type=int,
        default=1,
        help=(
            "Additional future action-label offset: obs/state/image[i] -> "
            "action/intervention[i + offset]. The final offset frames are "
            "dropped. Default: 1."
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
