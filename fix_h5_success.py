import argparse
import os

import h5py
import numpy as np


def parse_bool(value):
    value = value.lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("success must be true, false, 1, or 0")


def read_success(file_path):
    with h5py.File(file_path, "r") as h5_file:
        return h5_file.attrs.get("success")


def main():
    parser = argparse.ArgumentParser(
        description="Set the success attribute of one H5 file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_h5_success.py ~/dp_data/offlineRL_data/task1_iter1/demo_20260507_114630.h5 false
  python fix_h5_success.py ~/dp_data/offlineRL_data/task1_iter1/demo_20260507_114630.h5 true
""",
    )
    parser.add_argument("file", help="Absolute path to the H5 file.")
    parser.add_argument("success", type=parse_bool, help="New value: true or false.")
    args = parser.parse_args()

    if not os.path.isabs(args.file):
        parser.error("file must be an absolute path")
    if not os.path.isfile(args.file):
        parser.error(f"file does not exist: {args.file}")

    old_success = read_success(args.file)
    print(f"File: {args.file}")
    print(f"Old success: {old_success}")

    if isinstance(old_success, (bool, np.bool_)) and bool(old_success) == args.success:
        print(f"Success is already {args.success}; no change needed.")
        return

    with h5py.File(args.file, "r+") as h5_file:
        h5_file.attrs["success"] = np.bool_(args.success)

    new_success = read_success(args.file)
    if not isinstance(new_success, (bool, np.bool_)) or bool(new_success) != args.success:
        raise RuntimeError(f"failed to verify written success value: {new_success}")

    print(f"New success: {new_success}")


if __name__ == "__main__":
    main()
