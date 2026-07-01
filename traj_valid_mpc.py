#!/usr/bin/env python3
"""Replay a trajectory with xyz task-space MPC compensation."""

import argparse
import csv
import os
import time

import numpy as np

from task_space_mpc import TaskSpaceMPCConfig, TaskSpaceMPCTracker
from teleop_interfaces import InspireHandController, URArmInterface
from tools import MATHTOOLS
from traj_valid import (
    ACTION_DIM,
    ARM_ACTION_DIM,
    DEFAULT_DT,
    DEFAULT_HAND_BAUDRATE,
    DEFAULT_HAND_PORT,
    DEFAULT_INITIAL_POSE,
    DEFAULT_SERVO_FREQUENCY,
    DEFAULT_UR_HOST,
    DEFAULT_WORKSPACE_X,
    DEFAULT_WORKSPACE_Y,
    DEFAULT_WORKSPACE_Z,
    arm_action_to_target_pose,
    format_vector,
    load_action_data,
    positive_float,
    positive_int,
    read_robot_state,
    validate_workspace_limit,
)


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative float")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def action_data_to_target_poses(tool, action_data):
    return np.asarray(
        [
            arm_action_to_target_pose(tool, action[:ARM_ACTION_DIM])
            for action in action_data
        ],
        dtype=np.float64,
    )


def build_mpc_config(args):
    return TaskSpaceMPCConfig(
        horizon=args.mpc_horizon,
        tau=args.mpc_tau,
        iterations=args.mpc_iterations,
        w_track=args.mpc_w_track,
        track_decay=args.mpc_track_decay,
        w_cmd=args.mpc_w_cmd,
        w_yx=args.mpc_w_yx,
        w_dy=args.mpc_w_dy,
        w_ddy=args.mpc_w_ddy,
        max_cmd_actual_gap=args.mpc_max_cmd_actual_gap,
        max_velocity=args.mpc_max_velocity,
        max_acceleration=args.mpc_max_acceleration,
    )


def build_mpc_tracking_record(
    action_frame,
    state_frame,
    action_time_ns,
    state_time_ns,
    reference_pose,
    command_pose,
    observed_state,
    planning_actual_xyz,
    mpc_result,
):
    reference_pose = np.asarray(reference_pose, dtype=np.float64).reshape(6)
    command_pose = np.asarray(command_pose, dtype=np.float64).reshape(6)
    observed_tcp_pose = np.asarray(observed_state[-6:], dtype=np.float64).reshape(6)
    planning_actual_xyz = np.asarray(planning_actual_xyz, dtype=np.float64).reshape(3)
    reference_error = reference_pose - observed_tcp_pose
    command_error = command_pose - observed_tcp_pose
    return {
        "action_frame": int(action_frame),
        "state_frame": int(state_frame),
        "action_time_ns": int(action_time_ns),
        "state_time_ns": int(state_time_ns),
        "delay_ms": (int(state_time_ns) - int(action_time_ns)) / 1e6,
        "reference_pose": reference_pose,
        "command_pose": command_pose,
        "observed_tcp_pose": observed_tcp_pose,
        "planning_actual_xyz": planning_actual_xyz,
        "reference_error": reference_error,
        "command_error": command_error,
        "reference_position_error_m": float(np.linalg.norm(reference_error[:3])),
        "command_position_error_m": float(np.linalg.norm(command_error[:3])),
        "rotvec_error_rad": float(np.linalg.norm(reference_error[3:])),
        "mpc_reference_gap_m": float(mpc_result.reference_gap),
        "mpc_command_gap_m": float(mpc_result.command_gap),
        "mpc_predicted_tracking_error_m": float(mpc_result.tracking_error),
        "mpc_cost": float(mpc_result.cost),
        "mpc_status": int(mpc_result.status),
        "mpc_iterations": int(mpc_result.iterations),
    }


def append_mpc_tracking_record(
    records,
    robot,
    action_frame,
    state_frame,
    action_time_ns,
    reference_pose,
    command_pose,
    planning_actual_xyz,
    mpc_result,
    print_each,
    observed_state=None,
):
    if observed_state is None:
        observed_state = read_robot_state(robot)
    if observed_state is None:
        print(
            f"MPC tracking action[{action_frame}] -> state[{state_frame}] skipped: "
            "failed to read robot state."
        )
        return None

    record = build_mpc_tracking_record(
        action_frame=action_frame,
        state_frame=state_frame,
        action_time_ns=action_time_ns,
        state_time_ns=time.monotonic_ns(),
        reference_pose=reference_pose,
        command_pose=command_pose,
        observed_state=observed_state,
        planning_actual_xyz=planning_actual_xyz,
        mpc_result=mpc_result,
    )
    records.append(record)
    if print_each:
        print_mpc_tracking_error(record)
    return observed_state


def print_mpc_tracking_error(record):
    print(
        "MPC tracking "
        f"action[{record['action_frame']}] -> state[{record['state_frame']}]: "
        f"ref_pos={record['reference_position_error_m'] * 1000.0:.2f} mm, "
        f"cmd_pos={record['command_position_error_m'] * 1000.0:.2f} mm, "
        f"rotvec={record['rotvec_error_rad']:.5f} rad, "
        f"delay={record['delay_ms']:.2f} ms, "
        f"ref_diff={format_vector(record['reference_error'])}, "
        f"cmd_diff={format_vector(record['command_error'])}"
    )


def prefixed_pose_fields(prefix):
    return [
        f"{prefix}_x",
        f"{prefix}_y",
        f"{prefix}_z",
        f"{prefix}_rx",
        f"{prefix}_ry",
        f"{prefix}_rz",
    ]


def prefixed_xyz_fields(prefix):
    return [
        f"{prefix}_x",
        f"{prefix}_y",
        f"{prefix}_z",
    ]


def update_pose_row(row, prefix, values):
    for key, value in zip(prefixed_pose_fields(prefix), values):
        row[key] = value


def update_xyz_row(row, prefix, values):
    for key, value in zip(prefixed_xyz_fields(prefix), values):
        row[key] = value


def write_mpc_tracking_csv(csv_path, records):
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
        *prefixed_pose_fields("reference"),
        *prefixed_pose_fields("command"),
        *prefixed_pose_fields("observed"),
        *prefixed_xyz_fields("planning_actual"),
        *prefixed_pose_fields("reference_error"),
        *prefixed_pose_fields("command_error"),
        "reference_position_error_m",
        "command_position_error_m",
        "rotvec_error_rad",
        "mpc_reference_gap_m",
        "mpc_command_gap_m",
        "mpc_predicted_tracking_error_m",
        "mpc_cost",
        "mpc_status",
        "mpc_iterations",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "action_frame": record["action_frame"],
                "state_frame": record["state_frame"],
                "action_time_ns": record["action_time_ns"],
                "state_time_ns": record["state_time_ns"],
                "delay_ms": record["delay_ms"],
                "reference_position_error_m": record["reference_position_error_m"],
                "command_position_error_m": record["command_position_error_m"],
                "rotvec_error_rad": record["rotvec_error_rad"],
                "mpc_reference_gap_m": record["mpc_reference_gap_m"],
                "mpc_command_gap_m": record["mpc_command_gap_m"],
                "mpc_predicted_tracking_error_m": (
                    record["mpc_predicted_tracking_error_m"]
                ),
                "mpc_cost": record["mpc_cost"],
                "mpc_status": record["mpc_status"],
                "mpc_iterations": record["mpc_iterations"],
            }
            update_pose_row(row, "reference", record["reference_pose"])
            update_pose_row(row, "command", record["command_pose"])
            update_pose_row(row, "observed", record["observed_tcp_pose"])
            update_xyz_row(row, "planning_actual", record["planning_actual_xyz"])
            update_pose_row(row, "reference_error", record["reference_error"])
            update_pose_row(row, "command_error", record["command_error"])
            writer.writerow(row)

    print(f"Saved MPC tracking CSV: {csv_path}")


def print_scaled_summary(name, values, scale=1.0, unit=""):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"  {name}: "
        f"mean={np.mean(values) * scale:.2f}{suffix}, "
        f"p95={np.percentile(values, 95) * scale:.2f}{suffix}, "
        f"max={np.max(values) * scale:.2f}{suffix}"
    )


def print_smoothness_summary(name, xyz_values, dt):
    xyz_values = np.asarray(xyz_values, dtype=np.float64)
    if xyz_values.shape[0] < 2:
        return
    step = np.linalg.norm(np.diff(xyz_values, axis=0), axis=1)
    velocity = step / dt
    print_scaled_summary(f"{name} xyz step", step, scale=1000.0, unit="mm")
    print_scaled_summary(f"{name} xyz velocity", velocity, unit="m/s")
    if xyz_values.shape[0] >= 3:
        acceleration = np.linalg.norm(
            np.diff(xyz_values, n=2, axis=0),
            axis=1,
        ) / (dt * dt)
        print_scaled_summary(f"{name} xyz acceleration", acceleration, unit="m/s^2")


def print_mpc_tracking_summary(records, dt):
    if not records:
        print("No MPC tracking samples were recorded.")
        return

    reference_errors = np.asarray(
        [record["reference_position_error_m"] for record in records],
        dtype=np.float64,
    )
    command_errors = np.asarray(
        [record["command_position_error_m"] for record in records],
        dtype=np.float64,
    )
    rotvec_errors = np.asarray(
        [record["rotvec_error_rad"] for record in records],
        dtype=np.float64,
    )
    delays = np.asarray([record["delay_ms"] for record in records], dtype=np.float64)
    reference_xyz = np.asarray([record["reference_pose"][:3] for record in records])
    command_xyz = np.asarray([record["command_pose"][:3] for record in records])
    command_reference_gap = np.linalg.norm(command_xyz - reference_xyz, axis=1)

    print("\nMPC tracking summary:")
    print_scaled_summary(
        "Reference position error",
        reference_errors,
        scale=1000.0,
        unit="mm",
    )
    print_scaled_summary(
        "Command position error",
        command_errors,
        scale=1000.0,
        unit="mm",
    )
    print_scaled_summary("Rotvec error", rotvec_errors, unit="rad")
    print_scaled_summary("Command-reference gap", command_reference_gap, scale=1000.0, unit="mm")
    print_scaled_summary("State read delay", delays, unit="ms")

    print("\nSmoothness summary:")
    print_smoothness_summary("Raw reference", reference_xyz, dt)
    print_smoothness_summary("MPC command", command_xyz, dt)


def playback_trajectory(args):
    action_data = load_action_data(args.data_file)
    workspace_limits = {
        "x": validate_workspace_limit("x", args.workspace_x),
        "y": validate_workspace_limit("y", args.workspace_y),
        "z": validate_workspace_limit("z", args.workspace_z),
    }
    dt = args.dt / args.speed
    tool = MATHTOOLS()
    target_poses = action_data_to_target_poses(tool, action_data)
    mpc_config = build_mpc_config(args)
    mpc_tracker = TaskSpaceMPCTracker(
        config=mpc_config,
        dt=dt,
        workspace_limits=workspace_limits,
    )
    robot = None
    hand_controller = None
    tracking_records = []
    tracking_reported = False
    tracking_summary_printed = False
    previous_reference_pose = None
    previous_command_pose = None
    previous_action_frame = None
    previous_action_time_ns = None
    previous_planning_actual_xyz = None
    previous_mpc_result = None

    print("Loaded trajectory data:")
    print(f"  Action shape: {action_data.shape}")
    print(f"  Total frames: {len(action_data)}")
    print(f"  Playback speed: {args.speed}x")
    print(f"  Effective frequency: {1 / dt:.2f} Hz")
    print(
        "  MPC: "
        f"horizon={mpc_config.horizon}, tau={mpc_config.tau:.3f}s, "
        f"max_velocity={mpc_config.max_velocity:.3f} m/s, "
        f"max_acceleration={mpc_config.max_acceleration:.3f} m/s^2"
    )

    first_target = target_poses[0]
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
            servo_speed=args.servo_speed,
            servo_acceleration=args.servo_acceleration,
            servo_dt=1.0 / args.servo_frequency,
            lookahead_time=args.servo_lookahead_time,
            gain=args.servo_gain,
            control_frequency=args.servo_frequency,
        )
        hand_controller = InspireHandController(args.hand_port, args.hand_baudrate)

        print(f"Current robot pose: {robot.get_tcp_pose()}")
        print("Moving to initial position")
        robot.move_l(args.initial_pose, 0.3, 0.3)
        mpc_tracker.reset()

        print("\nStarting MPC trajectory playback")
        print("Press Ctrl+C to stop playback early")

        completed = True
        for frame_idx, action in enumerate(action_data):
            start_time = time.monotonic()
            current_state = None

            if previous_command_pose is not None:
                current_state = read_robot_state(robot)
                append_mpc_tracking_record(
                    tracking_records,
                    robot=robot,
                    action_frame=previous_action_frame,
                    state_frame=frame_idx,
                    action_time_ns=previous_action_time_ns,
                    reference_pose=previous_reference_pose,
                    command_pose=previous_command_pose,
                    planning_actual_xyz=previous_planning_actual_xyz,
                    mpc_result=previous_mpc_result,
                    print_each=args.print_tracking_error,
                    observed_state=current_state,
                )
                previous_reference_pose = None
                previous_command_pose = None
                previous_action_frame = None
                previous_action_time_ns = None
                previous_planning_actual_xyz = None
                previous_mpc_result = None

            if not robot.is_ready():
                print("Robot is stopped (protective or emergency). Stopping playback.")
                completed = False
                break

            if current_state is None:
                current_state = read_robot_state(robot)
            if current_state is None:
                raise RuntimeError("failed to read robot state for MPC planning")

            reference_pose = target_poses[frame_idx].copy()
            references_xyz = target_poses[
                frame_idx: frame_idx + max(1, int(mpc_config.horizon)),
                :3,
            ]
            planning_actual_xyz = np.asarray(current_state[-6:-3], dtype=np.float64)
            mpc_result = mpc_tracker.plan(references_xyz, planning_actual_xyz)

            command_pose = reference_pose.copy()
            command_pose[:3] = mpc_result.command

            hand_controller.apply(action[ARM_ACTION_DIM:ACTION_DIM] * 1000.0)
            previous_action_time_ns = time.monotonic_ns()
            if robot.servo(command_pose) is False:
                raise RuntimeError("servoL command was rejected")

            previous_reference_pose = reference_pose
            previous_command_pose = command_pose
            previous_action_frame = frame_idx
            previous_planning_actual_xyz = planning_actual_xyz
            previous_mpc_result = mpc_result

            sleep_time = dt - (time.monotonic() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if previous_command_pose is not None:
            append_mpc_tracking_record(
                tracking_records,
                robot=robot,
                action_frame=previous_action_frame,
                state_frame=len(action_data),
                action_time_ns=previous_action_time_ns,
                reference_pose=previous_reference_pose,
                command_pose=previous_command_pose,
                planning_actual_xyz=previous_planning_actual_xyz,
                mpc_result=previous_mpc_result,
                print_each=args.print_tracking_error,
            )
            previous_command_pose = None

        print_mpc_tracking_summary(tracking_records, dt)
        tracking_summary_printed = True
        write_mpc_tracking_csv(args.tracking_error_csv, tracking_records)
        tracking_reported = True

        robot.stop_servo()
        if completed:
            print("\nMPC trajectory playback completed successfully.")

    except KeyboardInterrupt:
        print("\nMPC playback interrupted by user.")
    finally:
        if not tracking_reported:
            if previous_command_pose is not None and robot is not None:
                try:
                    append_mpc_tracking_record(
                        tracking_records,
                        robot=robot,
                        action_frame=previous_action_frame,
                        state_frame=previous_action_frame + 1,
                        action_time_ns=previous_action_time_ns,
                        reference_pose=previous_reference_pose,
                        command_pose=previous_command_pose,
                        planning_actual_xyz=previous_planning_actual_xyz,
                        mpc_result=previous_mpc_result,
                        print_each=args.print_tracking_error,
                    )
                except Exception as exc:
                    print(f"Final MPC tracking sample skipped: {exc}")
            if tracking_records and not tracking_summary_printed:
                print_mpc_tracking_summary(tracking_records, dt)
                tracking_summary_printed = True
            if tracking_records:
                write_mpc_tracking_csv(args.tracking_error_csv, tracking_records)
        print("Stopping servo control and disconnecting.")
        if hand_controller is not None:
            hand_controller.close()
        if robot is not None:
            robot.close(stop_script=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Replay an H5 trajectory with xyz task-space MPC."
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
    parser.add_argument("--servo_speed", type=positive_float, default=0.005)
    parser.add_argument("--servo_acceleration", type=positive_float, default=0.005)
    parser.add_argument("--servo_lookahead_time", type=positive_float, default=0.2)
    parser.add_argument("--servo_gain", type=positive_float, default=500.0)
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
    parser.add_argument("--mpc_horizon", type=positive_int, default=15)
    parser.add_argument("--mpc_tau", type=positive_float, default=0.12)
    parser.add_argument("--mpc_iterations", type=nonnegative_int, default=24)
    parser.add_argument("--mpc_w_track", type=nonnegative_float, default=10.0)
    parser.add_argument("--mpc_track_decay", type=positive_float, default=1.0)
    parser.add_argument("--mpc_w_cmd", type=nonnegative_float, default=0.0)
    parser.add_argument("--mpc_w_yx", type=nonnegative_float, default=0.0)
    parser.add_argument("--mpc_w_dy", type=nonnegative_float, default=0.0)
    parser.add_argument("--mpc_w_ddy", type=nonnegative_float, default=20.0)
    parser.add_argument(
        "--mpc_max_cmd_actual_gap",
        type=nonnegative_float,
        default=0.0,
        help="0 lets the MPC use tau * max_velocity.",
    )
    parser.add_argument("--mpc_max_velocity", type=nonnegative_float, default=0.5)
    parser.add_argument("--mpc_max_acceleration", type=nonnegative_float, default=6.0)
    parser.add_argument(
        "--print_tracking_error",
        action="store_true",
        help="Print per-frame MPC reference and command tracking errors.",
    )
    parser.add_argument(
        "--tracking_error_csv",
        default=None,
        help="Optional CSV path for per-frame MPC tracking errors.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    return parser


def main(args):
    args.data_file = os.path.abspath(os.path.expanduser(args.data_file))
    if not os.path.isfile(args.data_file):
        raise SystemExit(f"Data file does not exist: {args.data_file}")

    try:
        action_data = load_action_data(args.data_file)
        first_target = action_data_to_target_poses(
            MATHTOOLS(),
            action_data[:1],
        )[0]
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid trajectory file: {exc}") from exc

    print(f"Preparing MPC playback from: {args.data_file}")
    print(f"First recorded target pose: {np.asarray(first_target)}")
    print(f"Configured initial pose:    {np.asarray(args.initial_pose)}")
    print(f"Playback speed: {args.speed}x")
    print(f"MPC tau: {args.mpc_tau:.3f}s")

    if not args.yes:
        confirmation = input("Start MPC trajectory playback? (y/N): ")
        if confirmation.lower() not in ("y", "yes"):
            print("MPC playback cancelled.")
            return

    try:
        playback_trajectory(args)
    except Exception as exc:
        raise SystemExit(f"MPC playback failed: {exc}") from exc


if __name__ == "__main__":
    main(build_parser().parse_args())
