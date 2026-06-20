#!/usr/bin/env python3
"""
UR5 Robot RTDE Control Program
Uses OpenVR tracker to control UR5 robot motion
Modified version: Uses RTDE interface with servoL for smooth teleoperation
"""

import time
import sys
import numpy as np
import triad_openvr
from episode_buffer import EpisodeBuffer
from keyboard_control import KeyboardControl
from SM_multi_realsense import MultiRealSense
from tools import MATHTOOLS
import os
from termcolor import cprint
import argparse
from datetime import datetime
from arm_servo_controller import HighRateArmController
from hand_control_controller import HighRateHandController
from robot_state_reader import HighRateRobotStateReader
from SM_inspire_manus_zmq import inspire_Manus
from teleop_interfaces import InspireHandController, URArmInterface
from tracker_pose_processor import TrackerPoseProcessor

DEFAULT_UR_HOST = "192.168.3.6" 
DEFAULT_WORKSPACE_X = [-1.5, 1.5]
DEFAULT_WORKSPACE_Y = [-1.5, 1.5]
DEFAULT_WORKSPACE_Z = [-0.5, 1.5]
DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_DT = 1.0 / 25.0
DEFAULT_TRACKER_FREQUENCY = 60.0
DEFAULT_SERVO_FREQUENCY = 120.0
DEFAULT_TRACKER_TIMEOUT = 0.25
DEFAULT_HAND_FREQUENCY = 60.0
DEFAULT_ROBOT_STATE_FREQUENCY = 125.0
DEFAULT_MANUS_TIMEOUT = 0.25
DEFAULT_MANUS_ZMQ_ENDPOINT = "tcp://127.0.0.1:2044"
DEFAULT_MANUS_ZMQ_RCVHWM = 1
DEFAULT_MANUS_ZMQ_CONFLATE = True
DEFAULT_MANUS_ZMQ_POLL_TIMEOUT_MS = 100
DEFAULT_MANUS_CONTROL_THRESHOLD = 10.0
DEFAULT_MANUS_SCALE_FACTOR = 15.0
DEFAULT_HAND_SMOOTHING_OMEGA = 25.0
DEFAULT_HAND_SMOOTHING_DAMPING = 0.8
DEFAULT_HAND_INPUT_ALPHA = 0.6
DEFAULT_ALIGNMENT_TOLERANCE_MS = 25.0
DEFAULT_MAX_LENGTH = 1000
DEFAULT_HAND_PORT = "/dev/ttyUSB0"
DEFAULT_HAND_BAUDRATE = 115200

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive float")
    return value


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative float")
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


def scalar_int(value, default=-1):
    if value is None:
        return default
    return int(np.asarray(value).item())


def scalar_float(value, default=np.nan):
    if value is None:
        return default
    return float(np.asarray(value).item())


def time_delta_ms(value_ns, anchor_ns, default=np.nan):
    if value_ns is None or anchor_ns is None:
        return default
    return (scalar_int(value_ns) - scalar_int(anchor_ns)) / 1e6


def main(args):
    # Workspace safety limits (meters)
    workspace_limits = {
        'x': validate_workspace_limit("x", args.workspace_x),
        'y': validate_workspace_limit("y", args.workspace_y),
        'z': validate_workspace_limit("z", args.workspace_z),
    }
    dt = args.dt              
    tool=MATHTOOLS()
    max_length=args.max_length
    # Initialize RTDE connection
    try:
        robot=URArmInterface(
            args.ur_host,
            workspace_limits,
            servo_speed=0.005,
            servo_acceleration=0.005,
            servo_dt=1.0 / args.servo_frequency,
            lookahead_time=0.2,
            gain=500,
            control_frequency=args.servo_frequency,
        )
    except Exception:
        print("Cannot connect to UR5, exiting")
        sys.exit(1)
    #init data saving  
    data_dir = args.demo_dir
    use_wrist_img=args.use_wrist_img
    os.makedirs(data_dir, exist_ok=True)

    #initialize inspire_Manus
    hand_Manus = inspire_Manus(
        use_right_hand=True,
        use_left_hand=False,
        zmq_endpoint=args.manus_zmq_endpoint,
        control_threshold=args.manus_control_threshold,
        scale_factor=args.manus_scale_factor,
        rcvhwm=args.manus_zmq_rcvhwm,
        conflate=args.manus_zmq_conflate,
        poll_timeout_ms=args.manus_zmq_poll_timeout_ms,
    )
    hand_controller=InspireHandController(args.hand_port, args.hand_baudrate)

    # initialize realsense
    cam_context=MultiRealSense(use_right_cam=use_wrist_img, front_num_points=20000, 
                         use_grid_sampling=True, use_crop=False,img_size=256)

    hand_Manus.start()
    cam_context.start()

    current_pose = robot.get_tcp_pose()
    print(current_pose)
    initial_pose = args.initial_pose
    print("Moving to initial position")
    robot.move_l(initial_pose, 0.3, 0.3)
    tracker_processor = TrackerPoseProcessor(tool, initial_pose)

    # Initialize OpenVR
    print("Initializing OpenVR")
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()

    print("\n" + "="*50)
    print("Press 'a' to stop the data collection loop, including while waiting for 's'...")
    print("Press 's' to start recording; press 's' again to stop recording")
    print("="*50 + "\n")
    #time.sleep(1)
    keyboard_control = KeyboardControl()
    keyboard_control.start()# Start listener thread
    time.sleep(0.1)
    arm_controller = HighRateArmController(
        robot=robot,
        tracker_processor=tracker_processor,
        tracker_device=v.devices["tracker_1"],
        tracker_frequency=args.tracker_frequency,
        servo_frequency=args.servo_frequency,
        tracker_timeout=args.tracker_timeout,
    )
    hand_control_worker = HighRateHandController(
        manus_source=hand_Manus,
        hand_controller=hand_controller,
        control_frequency=args.hand_frequency,
        manus_timeout=args.manus_timeout,
        smoothing_natural_frequency=args.hand_smoothing_omega,
        smoothing_damping_ratio=args.hand_smoothing_damping,
        smoothing_input_alpha=args.hand_input_alpha,
    )
    robot_state_reader = HighRateRobotStateReader(
        robot=robot,
        read_frequency=args.robot_state_frequency,
    )
    control_workers_running = False

    try:
        while not keyboard_control.should_stop_collection():

            # Wait for recording while the environment is reset and the operator's hand is aligned.
            if not keyboard_control.wait_recording():
                break
            print("Resetting Inspire hand before recording...\r")
            #hand_controller.reset(settle_time=0.2)
            if (
                keyboard_control.should_stop_collection()
                or not keyboard_control.is_recording()
            ):
                continue
            episode = EpisodeBuffer(use_wrist_img=use_wrist_img)
            arm_controller.start()
            control_workers_running = True
            hand_control_worker.start()
            robot_state_reader.start()
            episode_start_ns = time.monotonic_ns()
            print(f"Recording frequency set to: {1/dt:.2f} Hz\r")
            print(f"Tracker frequency set to: {args.tracker_frequency:.2f} Hz\r")
            print(f"Servo frequency set to: {args.servo_frequency:.2f} Hz\r")
            print(f"Hand control frequency set to: {args.hand_frequency:.2f} Hz\r")
            print(f"Robot state frequency set to: {args.robot_state_frequency:.2f} Hz\r")
            print(
                "Hand smoother set to: "
                f"omega={args.hand_smoothing_omega:.2f}, "
                f"damping={args.hand_smoothing_damping:.2f}, "
                f"input_alpha={args.hand_input_alpha:.2f}\r"
            )
            print("Starting recording tele-operation...\r")
            print("Recording started. Press 's' to stop recording.\r")    
            step = 0
            try:
                while (
                    step < max_length
                    and keyboard_control.is_recording()
                    and not keyboard_control.should_stop_collection()
                ):
                    record_start_ns = time.monotonic_ns()
                    start_time = time.monotonic()
                    # Check robot status
                    if robot.is_ready():
                        cam_dict=cam_context()
                        t_camera_read_ns = time.monotonic_ns()
                        front_meta = cam_dict.get("front_meta", {})
                        wrist_meta = cam_dict.get("right_meta", {})
                        t_anchor_ns = front_meta.get("t_host_ns")
                        if t_anchor_ns is None or t_anchor_ns < episode_start_ns:
                            time.sleep(0.001)
                            continue

                        motion = arm_controller.motion_at_time_ns(t_anchor_ns)
                        t_arm_read_ns = time.monotonic_ns()
                        if motion is None:
                            time.sleep(0.001)
                            continue
                        robot_obs = robot_state_reader.obs_at_time_ns(t_anchor_ns)
                        if robot_obs is None:
                            time.sleep(0.001)
                            continue
                        hand_sample = hand_control_worker.command_at_time_ns(t_anchor_ns)
                        t_hand_read_ns = time.monotonic_ns()
                        if hand_sample is None:
                            time.sleep(0.001)
                            continue

                        t_arm_action_ns = scalar_int(
                            motion.get("t_arm_action_host_ns")
                        )
                        t_robot_obs_host_ns = scalar_int(
                            robot_obs.get("t_robot_obs_host_ns")
                        )
                        t_hand_action_ns = scalar_int(
                            hand_sample.get("t_hand_action_host_ns")
                        )
                        arm_sync_delta_ms = time_delta_ms(
                            t_arm_action_ns,
                            t_anchor_ns,
                        )
                        robot_sync_delta_ms = time_delta_ms(
                            t_robot_obs_host_ns,
                            t_anchor_ns,
                        )
                        hand_sync_delta_ms = time_delta_ms(
                            t_hand_action_ns,
                            t_anchor_ns,
                        )
                        if (
                            abs(arm_sync_delta_ms) > args.alignment_tolerance_ms
                            or abs(robot_sync_delta_ms) > args.alignment_tolerance_ms
                            or abs(hand_sync_delta_ms) > args.alignment_tolerance_ms
                        ):
                            time.sleep(0.001)
                            continue

                        hand_command = hand_sample["command"]
                        hand_action_array=hand_command.astype(np.float32) / 1000.0
                        arm_action=motion["arm_action"]# Absolute target pose as xyz + 6D rotation
                        action=np.concatenate((arm_action,hand_action_array))
                        timestamps = {
                            "t_record_start_ns": record_start_ns,
                            "t_anchor_ns": t_anchor_ns,
                            "t_arm_read_ns": t_arm_read_ns,
                            "t_arm_action_host_ns": t_arm_action_ns,
                            "t_arm_servo_host_ns": scalar_int(
                                motion.get("t_arm_servo_host_ns")
                            ),
                            "t_arm_target_host_ns": scalar_int(
                                motion.get("t_arm_target_host_ns")
                            ),
                            "t_tracker0_host_ns": scalar_int(
                                motion.get("t_tracker0_host_ns")
                            ),
                            "t_tracker1_host_ns": scalar_int(
                                motion.get("t_tracker1_host_ns")
                            ),
                            "t_tracker_latest_host_ns": scalar_int(
                                motion.get("t_tracker_latest_host_ns")
                            ),
                            "interpolation_alpha": scalar_float(
                                motion.get("interpolation_alpha")
                            ),
                            "t_robot_obs_host_ns": t_robot_obs_host_ns,
                            "t_camera_read_ns": t_camera_read_ns,
                            "t_front_camera_host_ns": front_meta.get("t_host_ns"),
                            "front_camera_dev_ts": front_meta.get("t_dev_ts"),
                            "front_camera_seq": front_meta.get("seq"),
                            "front_camera_frame_no": front_meta.get("frame_no"),
                            "t_wrist_camera_host_ns": wrist_meta.get("t_host_ns"),
                            "wrist_camera_dev_ts": wrist_meta.get("t_dev_ts"),
                            "wrist_camera_seq": wrist_meta.get("seq"),
                            "wrist_camera_frame_no": wrist_meta.get("frame_no"),
                            "t_hand_read_ns": t_hand_read_ns,
                            "t_hand_action_host_ns": t_hand_action_ns,
                            "t_hand_command_host_ns": hand_sample.get(
                                "t_hand_command_host_ns"
                            ),
                            "t_manus_sample_host_ns": hand_sample.get(
                                "t_manus_sample_host_ns"
                            ),
                            "manus_seq": hand_sample.get("manus_seq"),
                            "t_aligned_arm_action_ns": t_arm_action_ns,
                            "t_aligned_robot_obs_ns": t_robot_obs_host_ns,
                            "t_aligned_hand_action_ns": t_hand_action_ns,
                            "sync_delta_arm_action_ms": arm_sync_delta_ms,
                            "sync_delta_robot_obs_ms": robot_sync_delta_ms,
                            "sync_delta_hand_action_ms": hand_sync_delta_ms,
                            "sync_delta_wrist_camera_ms": time_delta_ms(
                                wrist_meta.get("t_host_ns"),
                                t_anchor_ns,
                            ),
                        }
                        timestamps["t_record_end_ns"] = time.monotonic_ns()
                        episode.append(
                            robot_obs["state"],
                            cam_dict,
                            action,
                            timestamps,
                        )
                        step += 1
                    else:
                        print("Robot is stopped (protective or emergency).")
                        break # Exit loop

                    # Maintain the recording loop frequency
                    elapsed = time.monotonic() - start_time
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                hand_control_worker.stop()
                arm_controller.stop()
                robot_state_reader.stop()
                control_workers_running = False
                keyboard_control.clear_recording()
                print("Resetting Inspire hand after recording...\r")
                hand_controller.reset(settle_time=0.2)
                hand_control_worker.raise_if_failed()
                arm_controller.raise_if_failed()
                robot_state_reader.raise_if_failed()

            # Save the episode data
            if len(episode)>0:
                user_choice = input("Save recorded data? (y/n): ").lower().strip()
                if user_choice=='y':
                    record_file_name = os.path.join(data_dir, datetime.now().strftime("demo_%Y%m%d_%H%M%S")+".h5")
                    print("Data recording")
                    summary = episode.save_h5(record_file_name)    
                    print("Data recording done.")
                    cprint(f"color shape: {summary['color_shape']}", "yellow")
                    if use_wrist_img:
                        cprint(f"wrist_color shape: {summary['wrist_color_shape']}", "yellow")
                    cprint(f"action shape: {summary['action_shape']}", "yellow")
                    cprint(f"env_qpos shape: {summary['env_qpos_shape']}", "yellow")
                    cprint(f"save data at step: {summary['seq_length']} in {summary['record_file_name']}", "yellow")
                else:
                    print("Data not saved")
            else:
                print("No data to save")

            # Return the UR to its initial position for a blocking reset.
            robot.move_l(initial_pose, 0.3, 0.3)

            # Check whether to continue after completing an episode
            if keyboard_control.should_stop_collection():
                break

        print("collecting completed.\r")
        keyboard_control.stop()
        cam_context.finalize()
        hand_Manus.finalize()
        hand_controller.close()
        print("Stopping servo control and disconnecting.\r")
        robot.close(stop_script=False)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping control.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping servo control and disconnecting.")
        if control_workers_running:
            hand_control_worker.stop()
            arm_controller.stop()
            robot_state_reader.stop()
        keyboard_control.stop()
        cam_context.finalize()
        hand_Manus.finalize()
        hand_controller.close()
        robot.close(stop_script=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, default=os.path.expanduser("~/dp_data/test_demo"))
    parser.add_argument("--use_wrist_img", type=str2bool, default=True)
    parser.add_argument("--ur_host", type=str, default=DEFAULT_UR_HOST)
    parser.add_argument("--workspace_x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X, metavar=("MIN", "MAX"))
    parser.add_argument("--workspace_y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y, metavar=("MIN", "MAX"))
    parser.add_argument("--workspace_z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z, metavar=("MIN", "MAX"))
    parser.add_argument("--initial_pose", type=float, nargs=6, default=DEFAULT_INITIAL_POSE, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--dt", type=positive_float, default=DEFAULT_DT)
    parser.add_argument("--tracker_frequency", type=positive_float, default=DEFAULT_TRACKER_FREQUENCY)
    parser.add_argument("--servo_frequency", type=positive_float, default=DEFAULT_SERVO_FREQUENCY)
    parser.add_argument("--tracker_timeout", type=positive_float, default=DEFAULT_TRACKER_TIMEOUT)
    parser.add_argument("--hand_frequency", type=positive_float, default=DEFAULT_HAND_FREQUENCY)
    parser.add_argument("--robot_state_frequency", type=positive_float, default=DEFAULT_ROBOT_STATE_FREQUENCY)
    parser.add_argument("--manus_timeout", type=positive_float, default=DEFAULT_MANUS_TIMEOUT)
    parser.add_argument(
        "--manus_zmq_endpoint",
        type=str,
        default=DEFAULT_MANUS_ZMQ_ENDPOINT,
        help="ZMQ SUB endpoint for MANUS protobuf frames.",
    )
    parser.add_argument(
        "--manus_zmq_rcvhwm",
        type=positive_int,
        default=DEFAULT_MANUS_ZMQ_RCVHWM,
        help="ZMQ receive high-water mark; keep low to avoid stale hand frames.",
    )
    parser.add_argument(
        "--manus_zmq_conflate",
        type=str2bool,
        default=DEFAULT_MANUS_ZMQ_CONFLATE,
        help="Keep only the latest queued ZMQ frame when receiver falls behind.",
    )
    parser.add_argument(
        "--manus_zmq_poll_timeout_ms",
        type=positive_int,
        default=DEFAULT_MANUS_ZMQ_POLL_TIMEOUT_MS,
        help="Receiver thread poll timeout in milliseconds.",
    )
    parser.add_argument(
        "--manus_control_threshold",
        type=nonnegative_float,
        default=DEFAULT_MANUS_CONTROL_THRESHOLD,
        help="Minimum calibrated MANUS motion before updating Inspire command.",
    )
    parser.add_argument(
        "--manus_scale_factor",
        type=positive_float,
        default=DEFAULT_MANUS_SCALE_FACTOR,
        help="Scale factor from MANUS finger-angle deltas to Inspire commands.",
    )
    parser.add_argument("--hand_smoothing_omega", type=positive_float, default=DEFAULT_HAND_SMOOTHING_OMEGA)
    parser.add_argument("--hand_smoothing_damping", type=positive_float, default=DEFAULT_HAND_SMOOTHING_DAMPING)
    parser.add_argument("--hand_input_alpha", type=positive_float, default=DEFAULT_HAND_INPUT_ALPHA)
    parser.add_argument("--alignment_tolerance_ms", type=positive_float, default=DEFAULT_ALIGNMENT_TOLERANCE_MS)
    parser.add_argument("--max_length", type=positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--hand_port", type=str, default=DEFAULT_HAND_PORT)
    parser.add_argument("--hand_baudrate", type=positive_int, default=DEFAULT_HAND_BAUDRATE)
    args = parser.parse_args()
    main(args)
