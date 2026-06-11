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
from SM_inspire_manus_process import inspire_Manus
from joint_smoother import JointSmoother
from teleop_interfaces import InspireHandController, URArmInterface
from tracker_pose_processor import TrackerPoseProcessor

DEFAULT_UR_HOST = "192.168.3.6" 
DEFAULT_WORKSPACE_X = [-1.5, 1.5]
DEFAULT_WORKSPACE_Y = [-1.5, 1.5]
DEFAULT_WORKSPACE_Z = [-0.5, 1.5]
DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
DEFAULT_DT = 1.0 / 25.0
DEFAULT_MAX_LENGTH = 1000
DEFAULT_HAND_PORT = "/dev/ttyUSB0"
DEFAULT_HAND_BAUDRATE = 115200
DEFAULT_HAND_RESET_COMMAND = [1000, 1000, 1000, 1000, 1000, 1000]
DEFAULT_HAND_SMOOTHER_HZ = 100.0
DEFAULT_HAND_SMOOTHER_W = 25.0
DEFAULT_HAND_SMOOTHER_Z = 0.8

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
            servo_dt=dt,
            lookahead_time=0.2,
            gain=500,
        )
    except Exception:
        print("Cannot connect to UR5, exiting")
        sys.exit(1)
    #init data saving  
    data_dir = args.demo_dir
    use_wrist_img=args.use_wrist_img
    os.makedirs(data_dir, exist_ok=True)

    #initialize inspire_Manus
    hand_Manus=inspire_Manus(use_right_hand=True, use_left_hand=False)
    hand_controller=InspireHandController(args.hand_port, args.hand_baudrate)
    hand_smoother=JointSmoother(
        send_callback=hand_controller.apply,
        hz=args.hand_smoother_hz,
        w=args.hand_smoother_w,
        z=args.hand_smoother_z,
        dim=len(DEFAULT_HAND_RESET_COMMAND),
    )

    # initialize realsense
    cam_context=MultiRealSense(use_right_cam=use_wrist_img, front_num_points=20000, 
                         use_grid_sampling=True, use_crop=False,img_size=256)

    hand_Manus.start()
    cam_context.start()
    hand_smoother.reset_state(DEFAULT_HAND_RESET_COMMAND)
    hand_smoother.start()

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
    print("Press 'c' to stop the data collection loop...")
    print("Press 's' to start recording; press 's' again to stop recording")
    print("="*50 + "\n")
    #time.sleep(1)
    keyboard_control = KeyboardControl()
    keyboard_control.start()# Start listener thread
    time.sleep(0.1)

    try:
        while not keyboard_control.should_stop_collection():

            # Wait for recording while the environment is reset and the operator's hand is aligned.
            if not keyboard_control.wait_recording():
                break
            episode = EpisodeBuffer(use_wrist_img=use_wrist_img)
            tracker_processor.clear_reference()
            print(f"Control frequency set to: {1/dt:.2f} Hz\r")
            print("Starting recording tele-operation...\r")
            print("Recording started. Press 's' to stop recording.\r")    
            step = 0
            while step < max_length and keyboard_control.is_recording():
                start_time = time.time()
                # Check robot status
                if robot.is_ready():
                    current_tracker_mat = tracker_processor.read_tracker_mat(v.devices["tracker_1"])
                    if current_tracker_mat is None:
                        time.sleep(0.01)
                        continue
                    if not tracker_processor.has_reference():
                        tracker_processor.reset_reference(current_tracker_mat)
                        print("Initial tracking position recorded. Starting servo control...")
                        current_pose = robot.get_tcp_pose()
                        robot.servo(current_pose)
                        continue
                    robot_obs=robot.get_obs()
                    if robot_obs is None:
                        time.sleep(0.01)
                        continue
                    cam_dict=cam_context()
                    motion = tracker_processor.compute(current_tracker_mat)
                    hand_action_dict=hand_Manus()
                    hand_action_raw=np.asarray(hand_action_dict['right'], dtype=np.float32)
                    hand_command=np.clip(hand_action_raw, 0, 1000).astype(np.float32)
                    hand_smoother.update(hand_command)
                    hand_action_array=hand_command.astype(np.float32) / 1000.0
                    arm_action=motion["arm_action"]# Absolute target pose as xyz + 6D rotation
                    action=np.concatenate((arm_action,hand_action_array))
                    episode.append(robot_obs["state"], cam_dict, action)
                    target_pose = motion["target_pose"]
                    step += 1
                
                    # Send servoL command; URArmInterface clips xyz outside the workspace limits.
                    robot.servo(target_pose)
                else:
                    print("Robot is stopped (protective or emergency).")
                    break # Exit loop

                # Maintain the control loop frequency
                elapsed = time.time() - start_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            robot.stop_servo() # Stop the servo control thread
            hand_smoother.stop()
            hand_controller.reset()
            hand_smoother.reset_state(DEFAULT_HAND_RESET_COMMAND)
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
            robot.move_l(initial_pose, 0.1, 0.1)
            hand_smoother.start()

            # Check whether to continue after completing an episode
            if keyboard_control.should_stop_collection():
                break

        print("collecting completed.\r")
        keyboard_control.stop()
        cam_context.finalize()
        hand_Manus.finalize()
        hand_smoother.stop()
        hand_controller.close()
        print("Stopping servo control and disconnecting.\r")
        robot.close(stop_script=False)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping control.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping servo control and disconnecting.")
        keyboard_control.stop()
        cam_context.finalize()
        hand_Manus.finalize()
        hand_smoother.stop()
        hand_controller.close()
        robot.close(stop_script=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, default=os.path.expanduser("~/dp_data/new_task1_expertdata"))
    parser.add_argument("--use_wrist_img", type=str2bool, default=True)
    parser.add_argument("--ur_host", type=str, default=DEFAULT_UR_HOST)
    parser.add_argument("--workspace_x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X, metavar=("MIN", "MAX"))
    parser.add_argument("--workspace_y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y, metavar=("MIN", "MAX"))
    parser.add_argument("--workspace_z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z, metavar=("MIN", "MAX"))
    parser.add_argument("--initial_pose", type=float, nargs=6, default=DEFAULT_INITIAL_POSE, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--dt", type=positive_float, default=DEFAULT_DT)
    parser.add_argument("--max_length", type=positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--hand_port", type=str, default=DEFAULT_HAND_PORT)
    parser.add_argument("--hand_baudrate", type=positive_int, default=DEFAULT_HAND_BAUDRATE)
    parser.add_argument("--hand_smoother_hz", type=positive_float, default=DEFAULT_HAND_SMOOTHER_HZ)
    parser.add_argument("--hand_smoother_w", type=positive_float, default=DEFAULT_HAND_SMOOTHER_W)
    parser.add_argument("--hand_smoother_z", type=positive_float, default=DEFAULT_HAND_SMOOTHER_Z)
    args = parser.parse_args()
    main(args)
