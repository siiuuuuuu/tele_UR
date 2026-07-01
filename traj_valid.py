#!/usr/bin/env python3
"""Replay a trajectory recorded by servoL.py."""

import argparse
import csv
import os
import time

import h5py
import numpy as np

from teleop_interfaces import InspireHandController, URArmInterface
from tools import MATHTOOLS


DEFAULT_UR_HOST = "192.168.3.6"
DEFAULT_WORKSPACE_X = [-1.5, 1.5]
DEFAULT_WORKSPACE_Y = [-1.5, 1.5]
DEFAULT_WORKSPACE_Z = [-0.5, 1.5]
DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_DT = 1.0 / 60
DEFAULT_SERVO_FREQUENCY = 60
DEFAULT_HAND_PORT = "/dev/ttyUSB0"
DEFAULT_HAND_BAUDRATE = 115200

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


def read_robot_state(robot):
    obs = robot.get_obs()
    if obs is None or "state" not in obs:
        return None
    state = np.asarray(obs["state"], dtype=np.float64).reshape(-1)
    if state.shape[0] < 12:
        raise RuntimeError(
            f"expected robot state with at least 12 values, got {state.shape}"
        )
    return state


def build_tracking_error_record(
    action_frame,
    state_frame,
    action_time_ns,
    state_time_ns,
    target_pose,
    observed_state,
):
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(6)
    observed_tcp_pose = np.asarray(observed_state[-6:], dtype=np.float64).reshape(6)
    error = target_pose - observed_tcp_pose
    return {
        "action_frame": int(action_frame),
        "state_frame": int(state_frame),
        "action_time_ns": int(action_time_ns),
        "state_time_ns": int(state_time_ns),
        "delay_ms": (int(state_time_ns) - int(action_time_ns)) / 1e6,
        "target_pose": target_pose,
        "observed_tcp_pose": observed_tcp_pose,
        "error": error,
        "position_error_m": float(np.linalg.norm(error[:3])),
        "rotvec_error_rad": float(np.linalg.norm(error[3:])),
    }


def format_vector(values):
    return np.array2string(
        np.asarray(values),
        precision=5,
        suppress_small=False,
        separator=", ",
    )


def print_tracking_error(record):
    print(
        "Tracking error "
        f"action[{record['action_frame']}] -> state[{record['state_frame']}]: "
        f"pos={record['position_error_m'] * 1000.0:.2f} mm, "
        f"rotvec={record['rotvec_error_rad']:.5f} rad, "
        f"delay={record['delay_ms']:.2f} ms, "
        f"diff={format_vector(record['error'])}"
    )


def append_tracking_error(
    records,
    robot,
    action_frame,
    state_frame,
    action_time_ns,
    target_pose,
    print_each,
):
    observed_state = read_robot_state(robot)
    if observed_state is None:
        print(
            f"Tracking error action[{action_frame}] -> state[{state_frame}] skipped: "
            "failed to read robot state."
        )
        return

    record = build_tracking_error_record(
        action_frame=action_frame,
        state_frame=state_frame,
        action_time_ns=action_time_ns,
        state_time_ns=time.monotonic_ns(),
        target_pose=target_pose,
        observed_state=observed_state,
    )
    records.append(record)
    if print_each:
        print_tracking_error(record)


def write_tracking_error_csv(csv_path, records):
    if not csv_path or not records:
        return

    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = [
        "action_frame",
        "state_frame",
        "action_time_ns",
        "state_time_ns",
        "delay_ms",
        "target_x",
        "target_y",
        "target_z",
        "target_rx",
        "target_ry",
        "target_rz",
        "observed_x",
        "observed_y",
        "observed_z",
        "observed_rx",
        "observed_ry",
        "observed_rz",
        "error_x",
        "error_y",
        "error_z",
        "error_rx",
        "error_ry",
        "error_rz",
        "position_error_m",
        "rotvec_error_rad",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            target_pose = record["target_pose"]
            observed_pose = record["observed_tcp_pose"]
            error = record["error"]
            writer.writerow(
                {
                    "action_frame": record["action_frame"],
                    "state_frame": record["state_frame"],
                    "action_time_ns": record["action_time_ns"],
                    "state_time_ns": record["state_time_ns"],
                    "delay_ms": record["delay_ms"],
                    "target_x": target_pose[0],
                    "target_y": target_pose[1],
                    "target_z": target_pose[2],
                    "target_rx": target_pose[3],
                    "target_ry": target_pose[4],
                    "target_rz": target_pose[5],
                    "observed_x": observed_pose[0],
                    "observed_y": observed_pose[1],
                    "observed_z": observed_pose[2],
                    "observed_rx": observed_pose[3],
                    "observed_ry": observed_pose[4],
                    "observed_rz": observed_pose[5],
                    "error_x": error[0],
                    "error_y": error[1],
                    "error_z": error[2],
                    "error_rx": error[3],
                    "error_ry": error[4],
                    "error_rz": error[5],
                    "position_error_m": record["position_error_m"],
                    "rotvec_error_rad": record["rotvec_error_rad"],
                }
            )

    print(f"Saved tracking error CSV: {csv_path}")


def print_tracking_error_summary(records):
    if not records:
        print("No tracking error samples were recorded.")
        return

    position_errors = np.asarray([record["position_error_m"] for record in records])
    rotvec_errors = np.asarray([record["rotvec_error_rad"] for record in records])
    print("\nTracking error summary (target TCP pose - next observed TCP state):")
    print(
        "  Position error: "
        f"mean={np.mean(position_errors) * 1000.0:.2f} mm, "
        f"p95={np.percentile(position_errors, 95) * 1000.0:.2f} mm, "
        f"max={np.max(position_errors) * 1000.0:.2f} mm"
    )
    print(
        "  Rotvec error: "
        f"mean={np.mean(rotvec_errors):.5f} rad, "
        f"p95={np.percentile(rotvec_errors, 95):.5f} rad, "
        f"max={np.max(rotvec_errors):.5f} rad"
    )


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
    tracking_enabled = args.print_tracking_error or args.tracking_error_csv is not None
    tracking_records = []
    tracking_reported = False
    previous_target_pose = None
    previous_action_frame = None
    previous_action_time_ns = None

    print("Loaded trajectory data:")
    print(f"  Action shape: {action_data.shape}")
    print(f"  Total frames: {len(action_data)}")
    print(f"  Playback speed: {args.speed}x")
    print(f"  Effective frequency: {1 / dt:.2f} Hz")
    if tracking_enabled:
        print("  Tracking error: target TCP pose(frame i) - observed TCP state(frame i+1)")

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
            servo_dt=1.0 / args.servo_frequency,
            lookahead_time=0.1,
            gain=500,
            control_frequency=args.servo_frequency,
        )
        hand_controller = InspireHandController(args.hand_port, args.hand_baudrate)

        print(f"Current robot pose: {robot.get_tcp_pose()}")
        print("Moving to initial position")
        robot.move_l(args.initial_pose, 0.3, 0.3)

        print("\nStarting trajectory playback")
        print("Press Ctrl+C to stop playback early")

        completed = True
        for frame_idx, action in enumerate(action_data):
            start_time = time.monotonic()
            if tracking_enabled and previous_target_pose is not None:
                append_tracking_error(
                    tracking_records,
                    robot=robot,
                    action_frame=previous_action_frame,
                    state_frame=frame_idx,
                    action_time_ns=previous_action_time_ns,
                    target_pose=previous_target_pose,
                    print_each=args.print_tracking_error,
                )
                previous_target_pose = None
                previous_action_frame = None
                previous_action_time_ns = None

            if not robot.is_ready():
                print("Robot is stopped (protective or emergency). Stopping playback.")
                completed = False
                break

            target_pose = arm_action_to_target_pose(tool, action[:ARM_ACTION_DIM])
            hand_controller.apply(action[ARM_ACTION_DIM:ACTION_DIM] * 1000.0)
            previous_action_time_ns = time.monotonic_ns()
            robot.servo(target_pose)
            previous_target_pose = np.asarray(target_pose, dtype=np.float64)
            previous_action_frame = frame_idx

            sleep_time = dt - (time.monotonic() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if tracking_enabled and previous_target_pose is not None:
            append_tracking_error(
                tracking_records,
                robot=robot,
                action_frame=previous_action_frame,
                state_frame=len(action_data),
                action_time_ns=previous_action_time_ns,
                target_pose=previous_target_pose,
                print_each=args.print_tracking_error,
            )
            previous_target_pose = None
            previous_action_frame = None
            previous_action_time_ns = None
        if tracking_enabled:
            print_tracking_error_summary(tracking_records)
            write_tracking_error_csv(args.tracking_error_csv, tracking_records)
            tracking_reported = True

        robot.stop_servo()
        if completed:
            print("\nTrajectory playback completed successfully.")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        if tracking_enabled and not tracking_reported:
            if previous_target_pose is not None and robot is not None:
                try:
                    append_tracking_error(
                        tracking_records,
                        robot=robot,
                        action_frame=previous_action_frame,
                        state_frame=previous_action_frame + 1,
                        action_time_ns=previous_action_time_ns,
                        target_pose=previous_target_pose,
                        print_each=args.print_tracking_error,
                    )
                except Exception as exc:
                    print(f"Final tracking error sample skipped: {exc}")
            if tracking_records:
                print_tracking_error_summary(tracking_records)
                write_tracking_error_csv(args.tracking_error_csv, tracking_records)
        print("Stopping servo control and disconnecting.")
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
    parser.add_argument(
        "--servo_frequency",
        type=positive_float,
        default=DEFAULT_SERVO_FREQUENCY,
        help="RTDE servoL control frequency used during playback.",
    )
    parser.add_argument("--hand_port", default=DEFAULT_HAND_PORT)
    parser.add_argument(
        "--hand_baudrate",
        type=positive_int,
        default=DEFAULT_HAND_BAUDRATE,
    )
    parser.add_argument(
        "--max_initial_distance",
        type=positive_float,
        default=0.10,
        help="Maximum allowed xyz distance from initial_pose to the first target.",
    )
    parser.add_argument(
        "--print_tracking_error",
        action="store_true",
        help=(
            "Print per-frame TCP tracking error. The error is target TCP pose at "
            "action frame i minus observed TCP state at frame i+1."
        ),
    )
    parser.add_argument(
        "--tracking_error_csv",
        default=None,
        help="Optional CSV path for per-frame TCP tracking errors.",
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
