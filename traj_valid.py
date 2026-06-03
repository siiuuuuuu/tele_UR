#!/usr/bin/env python3
"""
UR5 Robot Trajectory Playback Program
Plays back previously recorded trajectory data from HDF5 file
"""
#回放轨迹，验证动作数据
import math
import time
import sys
import numpy as np
import rtde_control
import rtde_receive
import h5py
from tools import MATHTOOLS
import os
import argparse
from datetime import datetime
from InspireHandControl_V1 import InspireHand

# UR_HOST = "192.168.1.178"  # UR5 robot IP address
UR_HOST = "192.168.3.6"  # 210


def init_rtde_connection():
    """Initialize RTDE connection"""
    try:
        rtde_c = rtde_control.RTDEControlInterface(UR_HOST)
        rtde_r = rtde_receive.RTDEReceiveInterface(UR_HOST)
        print("RTDE connected")
        return rtde_c, rtde_r
    except Exception as e:
        print(f"RTDE connection failed: {e}")
        return None, None


def is_pose_safe(pose, workspace_limits):
    """Check if target position is within safe workspace"""
    x, y, z = pose[0], pose[1], pose[2]

    if not (workspace_limits['x'][0] <= x <= workspace_limits['x'][1]):
        print(f"X-axis out of bounds: {x:.3f}")
        return False
    if not (workspace_limits['y'][0] <= y <= workspace_limits['y'][1]):
        print(f"Y-axis out of bounds: {y:.3f}")
        return False
    if not (workspace_limits['z'][0] <= z <= workspace_limits['z'][1]):
        print(f"Z-axis out of bounds: {z:.3f}")
        return False

    return True


def playback_trajectory(data_file, playback_speed=1.0):
    """
    Play back trajectory from HDF5 file
    
    Args:
        data_file: Path to the HDF5 file containing trajectory data
        playback_speed: Speed multiplier for playback (1.0 = original speed)
    """
    # Load trajectory data from HDF5 file
    with h5py.File(data_file, 'r') as f:
        action_data = f['action'][:]  # Combined arm and hand actions
        env_qpos_data = f['env_qpos_proprioception'][:]  # Robot joint states
        color_data = f['color'][:]  # Camera images (not used in playback, but loaded for completeness)

    print(f"Loaded trajectory data:")
    print(f"  Action data shape: {action_data.shape}")
    print(f"  Environment qpos data shape: {env_qpos_data.shape}")
    print(f"  Color data shape: {color_data.shape}")
    print(f"  Total frames: {action_data.shape[0]}")

    # Initialize RTDE connection
    rtde_c, rtde_r = init_rtde_connection()
    if rtde_c is None or rtde_r is None:
        print("Cannot connect to UR5, exiting")
        sys.exit(1)

    hand=InspireHand(port="/dev/ttyUSB0", baudrate=115200)
    time.sleep(1)
    hand.reset()
    time.sleep(1)
    # Workspace safety limits (meters)
    workspace_limits = {
        'x': [-0.8, 0.8],
        'y': [-0.8, 0.8],
        'z': [0, 0.8]
    }

    # Set playback parameters
    servo_speed = 0.005          # 机器人最大TCP速度 (m/s)
    servo_acceleration = 0.005 
    base_dt = 1.0 / 25  # Original recording frequency (25 Hz)
    dt = base_dt / playback_speed  # Adjust for playback speed
    servo_dt = dt / 2  # servoL internal control frequency
    lookahead_time = 0.2  # Smooth time (0.03-0.2)
    gain = 500  # Proportional gain (100-2000)

    # Get current robot pose to initialize
    current_pose = rtde_r.getActualTCPPose()
    print(f"Current robot pose: {current_pose}")

    # Calculate initial TCP matrix
    tool = MATHTOOLS()
    initial_pose =[0.248,0.1212,0.3978,1.16,1.25,1.28]#210的
    # initial_pose = current_pose
    print("Moving to initial position")
    rtde_c.moveL(initial_pose, 0.3, 0.3)
    #构建起始TCP位姿的SE3矩阵
    temp_rotvec=np.array([initial_pose[3],initial_pose[4],initial_pose[5]])
    rot_mat = tool.rotvec2mat(temp_rotvec)
    init_TCP_mat=np.eye(4)
    init_TCP_mat[:3,:3]=rot_mat
    init_TCP_mat[0,3]=initial_pose[0]
    init_TCP_mat[1,3]=initial_pose[1]
    init_TCP_mat[2,3]=initial_pose[2]

    # Prepare to playback trajectory
    print(f"\nStarting trajectory playback...")
    print(f"Playback speed: {playback_speed}x")
    print(f"Effective frequency: {1/dt:.2f} Hz")
    print("Press Ctrl+C to stop playback early\n")

    try:
        # Start playback loop
        for frame_idx in range(len(action_data)):
            start_time = time.time()

            # Check robot safety status
            if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                print("Robot is stopped (protective or emergency). Stopping playback.")
                break

            # Extract action from recorded data
            # action contains [x, y, z, rx, ry, rz, hand_joint_1, ..., hand_joint_n]
            action = action_data[frame_idx]
            arm_action = np.expand_dims(action[:9],axis=0)
            hand_action = action[9:]*1000
            hand_action=np.clip(hand_action, 0, 1000)

            pinky_angle = int(hand_action[0])
            ring_angle = int(hand_action[1])
            middle_angle = int(hand_action[2])
            index_angle = int(hand_action[3])
            thumb_angle_2 = int(hand_action[4])
            thumb_angle = int(hand_action[5])
            hand.setangle(pinky_angle,ring_angle,middle_angle,index_angle,thumb_angle_2,thumb_angle)#输出动作

            arm_action = tool.xyz_6drot_to_mat(arm_action)[0]
            arm_action = np.dot(init_TCP_mat, arm_action)
            target_pose = tool.mat2xyz_rotvec(arm_action)

            if is_pose_safe(target_pose, workspace_limits):
                # Send servoL command to robot
                rtde_c.servoL(target_pose, servo_speed, servo_acceleration, servo_dt, lookahead_time, gain)
            else:
                # Clamp target to workspace limits
                target_pose[0] = max(workspace_limits['x'][0], min(workspace_limits['x'][1], target_pose[0]))
                target_pose[1] = max(workspace_limits['y'][0], min(workspace_limits['y'][1], target_pose[1]))
                target_pose[2] = max(workspace_limits['z'][0], min(workspace_limits['z'][1], target_pose[2]))
                print(f"Frame {frame_idx}: Target out of workspace! Clamping to safe limits.")
                rtde_c.servoL(target_pose, servo_speed, servo_acceleration, servo_dt, lookahead_time, gain)

            # Control loop timing
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"\nTrajectory playback completed successfully!")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    except Exception as e:
        print(f"An error occurred during playback: {e}")
    finally:
        # Stop servo control and disconnect
        print("Stopping servo control and disconnecting...")
        if rtde_c and rtde_c.isConnected():
            try:
                rtde_c.servoStop()
                rtde_c.stopScript()
            except Exception as e:
                print(f"Error during stopping robot: {e}")
            finally:
                rtde_c.disconnect()

        if rtde_r and rtde_r.isConnected():
            rtde_r.disconnect()


def main(args):
    # Validate input file
    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' does not exist.")
        sys.exit(1)

    print(f"Preparing to play back trajectory from: {args.data_file}")
    print(f"Playback speed: {args.speed}x")

    # Confirm before starting playback
    if not args.yes:
        confirmation = input("Start trajectory playback? (y/N): ")
        if confirmation.lower() not in ['y', 'yes']:
            print("Playback cancelled.")
            return

    # Perform trajectory playback
    playback_trajectory(args.data_file, args.speed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play back recorded robot trajectory')
    parser.add_argument('--data_file', type=str,default="/home/lrz/dp_data/row_data/demo_20251230_161455.h5",
                        help='Path to HDF5 file containing recorded trajectory data')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')

    args = parser.parse_args()
    main(args)