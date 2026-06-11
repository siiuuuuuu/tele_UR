#!/usr/bin/env python3
"""Replay a trajectory recorded by servoL.py."""

import argparse
import os
import time

import h5py
import numpy as np

from joint_smoother import JointSmoother
from teleop_interfaces import InspireHandController, URArmInterface
from tools import MATHTOOLS


DEFAULT_UR_HOST = "192.168.3.6"
DEFAULT_WORKSPACE_X = [-1.5, 1.5]
DEFAULT_WORKSPACE_Y = [-1.5, 1.5]
DEFAULT_WORKSPACE_Z = [-0.5, 1.5]
DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_DT = 1.0 / 25.0
DEFAULT_HAND_PORT = "/dev/ttyUSB0"
DEFAULT_HAND_BAUDRATE = 115200
DEFAULT_HAND_RESET_COMMAND = [1000, 1000, 1000, 1000, 1000, 1000]
DEFAULT_HAND_SMOOTHER_HZ = 100.0
DEFAULT_HAND_SMOOTHER_W = 25.0
DEFAULT_HAND_SMOOTHER_Z = 0.8

ARM_ACTION_DIM = 9
HAND_ACTION_DIM = 6
ACTION_DIM = ARM_ACTION_DIM + HAND_ACTION_DIM


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive float")
    return value


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def validate_workspace_limit(axis, limit):
    if limit[0] > limit[1]:
        raise ValueError(f"workspace_{axis} lower bound must be <= upper bound")
    return limit


def load_action_data(data_file):
    with h5py.File(data_file, "r") as h5_file:
        if "action" not in h5_file:
            raise ValueError("H5 file does not contain an 'action' dataset")
        action_data = np.asarray(h5_file["action"][:], dtype=np.float64)

    if action_data.ndim != 2 or action_data.shape[1] != ACTION_DIM:
        raise ValueError(
            f"expected action shape [T, {ACTION_DIM}], got {action_data.shape}"
        )
    if len(action_data) == 0:
        raise ValueError("action dataset is empty")
    if not np.all(np.isfinite(action_data)):
        raise ValueError("action dataset contains NaN or infinite values")
    return action_data


def arm_action_to_target_pose(tool, arm_action):
    """Convert the absolute xyz + 6D rotation action saved by servoL.py."""
    arm_action = np.asarray(arm_action, dtype=np.float64)
    target_matrix = tool.xyz_6drot_to_mat(arm_action[np.newaxis, :])[0]
    return tool.mat2xyz_rotvec(target_matrix)


def playback_trajectory(args):
    action_data = load_action_data(args.data_file)
    workspace_limits = {
        "x": validate_workspace_limit("x", args.workspace_x),
        "y": validate_workspace_limit("y", args.workspace_y),
        "z": validate_workspace_limit("z", args.workspace_z),
    }
    dt = args.dt / args.speed
    tool = MATHTOOLS()
    robot = None
    hand_controller = None
    hand_smoother = None

    print("Loaded trajectory data:")
    print(f"  Action shape: {action_data.shape}")
    print(f"  Total frames: {len(action_data)}")
    print(f"  Playback speed: {args.speed}x")
    print(f"  Effective frequency: {1 / dt:.2f} Hz")
    print(
        "  Hand smoother: "
        f"{args.hand_smoother_hz:.1f} Hz, "
        f"w={args.hand_smoother_w:.2f}, "
        f"z={args.hand_smoother_z:.2f}"
    )

    first_target = arm_action_to_target_pose(tool, action_data[0, :ARM_ACTION_DIM])
    initial_distance = np.linalg.norm(
        np.asarray(first_target[:3]) - np.asarray(args.initial_pose[:3])
    )
    if initial_distance > args.max_initial_distance:
        raise RuntimeError(
            "first recorded target is too far from the configured initial pose: "
            f"{initial_distance:.3f} m > {args.max_initial_distance:.3f} m"
        )

    try:
        robot = URArmInterface(
            args.ur_host,
            workspace_limits,
            servo_speed=0.005,
            servo_acceleration=0.005,
            servo_dt=dt / 2,
            lookahead_time=0.2,
            gain=500,
        )
        hand_controller = InspireHandController(args.hand_port, args.hand_baudrate)
        hand_smoother = JointSmoother(
            send_callback=hand_controller.apply,
            hz=args.hand_smoother_hz,
            w=args.hand_smoother_w,
            z=args.hand_smoother_z,
            dim=HAND_ACTION_DIM,
        )

        print(f"Current robot pose: {robot.get_tcp_pose()}")
        print("Moving to initial position")
        robot.move_l(args.initial_pose, 0.3, 0.3)
        hand_smoother.reset_state(DEFAULT_HAND_RESET_COMMAND)
        hand_smoother.start()

        print("\nStarting trajectory playback")
        print("Press Ctrl+C to stop playback early")

        completed = True
        for frame_idx, action in enumerate(action_data):
            start_time = time.monotonic()
            if not robot.is_ready():
                print("Robot is stopped (protective or emergency). Stopping playback.")
                completed = False
                break

            target_pose = arm_action_to_target_pose(tool, action[:ARM_ACTION_DIM])
            hand_command = np.clip(
                action[ARM_ACTION_DIM:ACTION_DIM] * 1000.0,
                0,
                1000,
            ).astype(np.float32)
            hand_smoother.update(hand_command)
            robot.servo(target_pose)

            sleep_time = dt - (time.monotonic() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        robot.stop_servo()
        hand_smoother.stop()
        if completed:
            print("\nTrajectory playback completed successfully.")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        print("Stopping servo control and disconnecting.")
        if hand_smoother is not None:
            hand_smoother.stop()
        if hand_controller is not None:
            hand_controller.close()
        if robot is not None:
            robot.close(stop_script=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Replay an H5 trajectory recorded by servoL.py."
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Path to an HDF5 file containing recorded trajectory data.",
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="Playback speed multiplier.",
    )
    parser.add_argument("--ur_host", default=DEFAULT_UR_HOST)
    parser.add_argument(
        "--workspace_x",
        type=float,
        nargs=2,
        default=DEFAULT_WORKSPACE_X,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--workspace_y",
        type=float,
        nargs=2,
        default=DEFAULT_WORKSPACE_Y,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--workspace_z",
        type=float,
        nargs=2,
        default=DEFAULT_WORKSPACE_Z,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--initial_pose",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_POSE,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    parser.add_argument(
        "--dt",
        type=positive_float,
        default=DEFAULT_DT,
        help="Recording control period before applying --speed.",
    )
    parser.add_argument("--hand_port", default=DEFAULT_HAND_PORT)
    parser.add_argument(
        "--hand_baudrate",
        type=positive_int,
        default=DEFAULT_HAND_BAUDRATE,
    )
    parser.add_argument(
        "--hand_smoother_hz",
        type=positive_float,
        default=DEFAULT_HAND_SMOOTHER_HZ,
        help="High-frequency sender rate for smoothed hand playback.",
    )
    parser.add_argument(
        "--hand_smoother_w",
        type=positive_float,
        default=DEFAULT_HAND_SMOOTHER_W,
        help="Natural frequency for the 2nd-order hand joint smoother.",
    )
    parser.add_argument(
        "--hand_smoother_z",
        type=positive_float,
        default=DEFAULT_HAND_SMOOTHER_Z,
        help="Damping ratio for the 2nd-order hand joint smoother.",
    )
    parser.add_argument(
        "--max_initial_distance",
        type=positive_float,
        default=0.10,
        help="Maximum allowed xyz distance from initial_pose to the first target.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    return parser


def main(args):
    args.data_file = os.path.abspath(os.path.expanduser(args.data_file))
    if not os.path.isfile(args.data_file):
        raise SystemExit(f"Data file does not exist: {args.data_file}")

    try:
        action_data = load_action_data(args.data_file)
        first_target = arm_action_to_target_pose(MATHTOOLS(), action_data[0, :ARM_ACTION_DIM])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid trajectory file: {exc}") from exc

    print(f"Preparing to play back trajectory from: {args.data_file}")
    print(f"First recorded target pose: {np.asarray(first_target)}")
    print(f"Configured initial pose:    {np.asarray(args.initial_pose)}")
    print(f"Playback speed: {args.speed}x")

    if not args.yes:
        confirmation = input("Start trajectory playback? (y/N): ")
        if confirmation.lower() not in ("y", "yes"):
            print("Playback cancelled.")
            return

    try:
        playback_trajectory(args)
    except Exception as exc:
        raise SystemExit(f"Playback failed: {exc}") from exc


if __name__ == "__main__":
    main(build_parser().parse_args())
