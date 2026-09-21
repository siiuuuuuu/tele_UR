#!/usr/bin/env python3
"""
UR5 Robot RTDE Control Program
Uses OpenVR tracker to control UR5 robot motion
Modified version: Uses RTDE interface with servoL for smooth teleoperation
"""

import time
import sys
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
from teleop_recording import (
    AlignedSampleProvider,
    TeleopRuntime,
    TimestampBuilder,
)

DEFAULT_UR_HOST = "192.168.3.6" 
DEFAULT_WORKSPACE_X = [-1.5, 1.5]
DEFAULT_WORKSPACE_Y = [-1.5, 1.5]
DEFAULT_WORKSPACE_Z = [-0.5, 1.5]
DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_DT = 1.0 / 30.0
DEFAULT_TRACKER_FREQUENCY = 60.0
DEFAULT_SERVO_FREQUENCY = 120.0
DEFAULT_TRACKER_TIMEOUT = 0.25
DEFAULT_INTERPOLATION_DELAY = None
DEFAULT_HAND_FREQUENCY = 120.0
DEFAULT_ROBOT_STATE_FREQUENCY = 125.0
DEFAULT_MANUS_TIMEOUT = 0.25
DEFAULT_MANUS_ZMQ_ENDPOINT = "tcp://127.0.0.1:2044"
DEFAULT_MANUS_ZMQ_RCVHWM = 1
DEFAULT_MANUS_ZMQ_CONFLATE = True
DEFAULT_MANUS_ZMQ_POLL_TIMEOUT_MS = 100
DEFAULT_MANUS_CONTROL_THRESHOLD = 10.0
DEFAULT_MANUS_SCALE_FACTOR = 15.0
DEFAULT_HAND_SMOOTHING_OMEGA = 30.0
DEFAULT_HAND_SMOOTHING_DAMPING = 0.85
DEFAULT_HAND_INPUT_ALPHA = 0.84
DEFAULT_ALIGNMENT_TOLERANCE_MS = 25.0
DEFAULT_HISTORY_WAIT_TIMEOUT_MS = 5.0
DEFAULT_ACTION_ALIGNMENT_OFFSET_FRAMES = 0
DEFAULT_FRONT_CAMERA_FPS = 30
DEFAULT_WRIST_CAMERA_FPS = 60
DEFAULT_CAMERA_SYNC_WAIT_TIMEOUT_MS = 5
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


def main(args):
    # Workspace safety limits (meters)
    workspace_limits = {
        'x': validate_workspace_limit("x", args.workspace_x),
        'y': validate_workspace_limit("y", args.workspace_y),
        'z': validate_workspace_limit("z", args.workspace_z),
    }
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
    front_camera_fps = args.camera_fps or args.front_camera_fps
    wrist_camera_fps = args.camera_fps or args.wrist_camera_fps
    action_alignment_offset_s = (
        float(args.action_alignment_offset_frames) / float(front_camera_fps)
    )
    cam_context=MultiRealSense(use_right_cam=use_wrist_img, front_num_points=20000, 
                         use_grid_sampling=True, use_crop=False,img_size=256,
                         front_camera_fps=front_camera_fps,
                         wrist_camera_fps=wrist_camera_fps,
                         sync_right_to_front=True,
                         sync_wait_timeout_ms=args.camera_sync_wait_timeout_ms)

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
        interpolation_delay=args.interpolation_delay,
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
    runtime = TeleopRuntime(
        robot=robot,
        camera=cam_context,
        hand_manus=hand_Manus,
        hand_controller=hand_controller,
        keyboard_control=keyboard_control,
        arm_controller=arm_controller,
        hand_control_worker=hand_control_worker,
        robot_state_reader=robot_state_reader,
    )
    sample_provider = AlignedSampleProvider(
        camera=cam_context,
        arm_controller=arm_controller,
        hand_control_worker=hand_control_worker,
        robot_state_reader=robot_state_reader,
        use_wrist_img=use_wrist_img,
        alignment_tolerance_ms=args.alignment_tolerance_ms,
        timestamp_builder=TimestampBuilder(),
        history_wait_timeout_ms=args.history_wait_timeout_ms,
        action_alignment_offset_s=action_alignment_offset_s,
    )
    closed_normally = False

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
            episode = EpisodeBuffer(
                use_wrist_img=use_wrist_img,
                metadata={
                    "action_alignment_policy": "front_anchor_plus_offset_frames",
                    "action_alignment_offset_frames": float(
                        args.action_alignment_offset_frames
                    ),
                    "action_alignment_offset_seconds": float(
                        action_alignment_offset_s
                    ),
                    "front_camera_fps": float(front_camera_fps),
                    "wrist_camera_fps": float(wrist_camera_fps),
                },
            )
            runtime.start_episode_workers()
            sample_provider.reset_episode()
            episode_start_ns = time.monotonic_ns()
            print(
                "Recording paced by front camera: "
                f"{front_camera_fps:.2f} Hz\r"
            )
            if use_wrist_img:
                print(f"Wrist camera frequency set to: {wrist_camera_fps:.2f} Hz\r")
            print(
                "Camera timestamp source: RealSense get_timestamp() in the "
                "global-time domain, mapped to host monotonic time\r"
            )
            print(f"Tracker frequency set to: {args.tracker_frequency:.2f} Hz\r")
            print(f"Servo frequency set to: {args.servo_frequency:.2f} Hz\r")
            if args.interpolation_delay is None:
                print("Interpolation delay set to: auto (1 tracker period)\r")
            else:
                print(
                    "Interpolation delay set to: "
                    f"{args.interpolation_delay * 1000.0:.2f} ms\r"
                )
            print(f"Hand control frequency set to: {args.hand_frequency:.2f} Hz\r")
            print(f"Robot state frequency set to: {args.robot_state_frequency:.2f} Hz\r")
            print(
                "History wait timeout set to: "
                f"{args.history_wait_timeout_ms:.2f} ms\r"
            )
            print(
                "Action alignment offset set to: "
                f"+{args.action_alignment_offset_frames:.3f} front frame(s) "
                f"({action_alignment_offset_s * 1000.0:.2f} ms)\r"
            )
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
                    # Check robot status
                    if robot.is_ready():
                        sample = sample_provider.read(episode_start_ns)
                        if sample is None:
                            time.sleep(0.001)
                            continue

                        episode.append(
                            sample.robot_state,
                            sample.cam_dict,
                            sample.action,
                            sample.timestamps,
                        )
                        step += 1
                    else:
                        print("Robot is stopped (protective or emergency).")
                        break # Exit loop
            finally:
                runtime.stop_episode_workers()

            # Save the episode data
            if len(episode)>0:
                user_input = input("Save recorded data? (y/n): ").lower().strip()
                user_choice = next(
                    (char for char in reversed(user_input) if char in ("y", "n")),
                    None,
                )
                if user_choice == "y":
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
        print("Stopping servo control and disconnecting.\r")
        runtime.close(stop_script=False)
        closed_normally = True

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping control.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if not closed_normally:
            print("Stopping servo control and disconnecting.")
            runtime.close(stop_script=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, default=os.path.expanduser("~/dp_data/test_demo"))
    parser.add_argument("--use_wrist_img", type=str2bool, default=True)
    parser.add_argument("--ur_host", type=str, default=DEFAULT_UR_HOST)
    parser.add_argument("--workspace_x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X, metavar=("MIN", "MAX"))
    parser.add_argument("--workspace_y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y, metavar=("MIN", "MAX"))
    parser.add_argument("--workspace_z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z, metavar=("MIN", "MAX"))
    parser.add_argument("--initial_pose", type=float, nargs=6, default=DEFAULT_INITIAL_POSE, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument(
        "--dt",
        type=positive_float,
        default=DEFAULT_DT,
        help=(
            "Legacy timer period. Recording is paced by the front camera; "
            "this value is kept for script compatibility."
        ),
    )
    parser.add_argument("--tracker_frequency", type=positive_float, default=DEFAULT_TRACKER_FREQUENCY)
    parser.add_argument("--servo_frequency", type=positive_float, default=DEFAULT_SERVO_FREQUENCY)
    parser.add_argument("--tracker_timeout", type=positive_float, default=DEFAULT_TRACKER_TIMEOUT)
    parser.add_argument(
        "--interpolation_delay",
        type=nonnegative_float,
        default=DEFAULT_INTERPOLATION_DELAY,
        help=(
            "Seconds to look back when interpolating tracker targets. "
            "Use 0 to follow the latest tracker sample without delay; "
            "omit for one tracker period."
        ),
    )
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
    parser.add_argument("--history_wait_timeout_ms", type=nonnegative_float, default=DEFAULT_HISTORY_WAIT_TIMEOUT_MS)
    parser.add_argument(
        "--action_alignment_offset_frames",
        type=nonnegative_float,
        default=DEFAULT_ACTION_ALIGNMENT_OFFSET_FRAMES,
        help=(
            "Record actions nearest to front_camera_anchor + this many front "
            "camera frames while observations stay aligned to the anchor. "
            "Use 0 for the legacy anchor-time action labels."
        ),
    )
    parser.add_argument("--front_camera_fps", type=positive_int, default=DEFAULT_FRONT_CAMERA_FPS)
    parser.add_argument("--wrist_camera_fps", type=positive_int, default=DEFAULT_WRIST_CAMERA_FPS)
    parser.add_argument("--camera_fps", type=positive_int, default=None,
                        help="Legacy option: set both front and wrist camera FPS to the same value.")
    parser.add_argument("--camera_sync_wait_timeout_ms", type=nonnegative_float, default=DEFAULT_CAMERA_SYNC_WAIT_TIMEOUT_MS)
    parser.add_argument("--max_length", type=positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--hand_port", type=str, default=DEFAULT_HAND_PORT)
    parser.add_argument("--hand_baudrate", type=positive_int, default=DEFAULT_HAND_BAUDRATE)
    args = parser.parse_args()
    main(args)
